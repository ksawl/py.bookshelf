# app/services/database_service.py

from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import sessionmaker

from app.schemas.job import JobInfo
from app.models.job import Job, Base
from app.core.database_engine import get_database_engine, get_session_maker
from app.core.config import Settings


class DatabaseService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = get_database_engine(settings.DATABASE_URL)
        self.session_maker = get_session_maker(settings.DATABASE_URL)
        
    async def init_db(self):
        """Initialize database schema"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def get_session(self) -> AsyncSession:
        """Get database session"""
        return self.session_maker()

    async def create_job(self, job_id: str, info: JobInfo) -> None:
        """Create new job record with defaults"""
        job_data = {
            "job_id": job_id,
            "created_at": datetime.utcnow(),
            "status": "queued",
            "progress": 0,
            "processed_chunks": 0,
            **{k: v for k, v in info.items() if v is not None}  # Filter None values
        }
        
        async with self.get_session() as session:
            job = Job(**job_data)
            session.add(job)
            await session.commit()

    async def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job by ID"""
        async with self.get_session() as session:
            result = await session.execute(select(Job).where(Job.job_id == job_id))
            job = result.scalar_one_or_none()
            return job.to_dict() if job else None

    async def get_all_jobs(self) -> List[JobInfo]:
        """Get all jobs"""
        async with self.get_session() as session:
            result = await session.execute(select(Job).order_by(Job.created_at.desc()))
            jobs = result.scalars().all()
            return [job.to_dict() for job in jobs]

    async def update_job(self, job_id: str, patch: Dict[str, Any]) -> None:
        """Update job fields"""
        if not patch:
            return
            
        # Handle status change timestamps
        if patch.get("status") == "processing":
            patch.setdefault("started_at", datetime.utcnow())
        elif patch.get("status") in ("done", "error", "cancelled"):
            patch.setdefault("finished_at", datetime.utcnow())
        
        # Filter None values and remove job_id from patch
        filtered_patch = {k: v for k, v in patch.items() if k != "job_id" and v is not None}
        
        if not filtered_patch:
            return
        
        async with self.get_session() as session:
            await session.execute(
                update(Job).where(Job.job_id == job_id).values(**filtered_patch)
            )
            await session.commit()

    async def increment_processed(self, job_id: str, n: int = 1) -> None:
        """Increment processed_chunks and recalculate progress"""
        async with self.get_session() as session:
            # Get current job
            result = await session.execute(select(Job).where(Job.job_id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                return
            
            new_processed = (job.processed_chunks or 0) + n
            
            # Calculate progress
            progress = 0
            if job.total_chunks and job.total_chunks > 0:
                progress = max(0, min(100, int((new_processed / job.total_chunks) * 100)))
            
            await session.execute(
                update(Job)
                .where(Job.job_id == job_id)
                .values(processed_chunks=new_processed, progress=progress)
            )
            await session.commit()

    async def set_progress(self, job_id: str, progress: int) -> None:
        """Set progress directly"""
        progress = max(0, min(100, int(progress)))
        async with self.get_session() as session:
            await session.execute(
                update(Job).where(Job.job_id == job_id).values(progress=progress)
            )
            await session.commit()

    async def finish_job(self, job_id: str, success: bool = True, error: Optional[str] = None) -> None:
        """Mark job as finished"""
        status = "done" if success else "error"
        patch = {
            "status": status,
            "finished_at": datetime.utcnow(),
        }
        if success:
            patch["progress"] = 100
        if error:
            patch["error"] = error
        
        await self.update_job(job_id, patch)

    async def delete_job(self, job_id: str) -> None:
        """Delete job record"""
        async with self.get_session() as session:
            await session.execute(delete(Job).where(Job.job_id == job_id))
            await session.commit()
