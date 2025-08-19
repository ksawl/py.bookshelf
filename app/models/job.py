# app/models/job.py

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    # Primary key
    job_id = Column(String, primary_key=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    
    # Status and basic info
    status = Column(String, default="queued", nullable=False)
    filename = Column(String, nullable=True)
    path = Column(String, nullable=True)
    doc_title = Column(String, nullable=True)
    namespace = Column(String, nullable=True)
    index_name = Column(String, nullable=True)
    
    # Progress tracking
    total_chunks = Column(Integer, nullable=True)
    processed_chunks = Column(Integer, default=0)
    progress = Column(Integer, default=0)
    
    # Additional fields
    callback_url = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    extra = Column(JSON, nullable=True)  # JSON field for additional data

    def __repr__(self):
        return f"<Job(job_id='{self.job_id}', status='{self.status}', progress={self.progress})>"

    def to_dict(self) -> dict:
        """Convert to dictionary matching JobInfo TypedDict"""
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "filename": self.filename,
            "path": self.path,
            "doc_title": self.doc_title,
            "namespace": self.namespace,
            "index_name": self.index_name,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "progress": self.progress,
            "callback_url": self.callback_url,
            "error": self.error,
            "extra": self.extra,
        }
