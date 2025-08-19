# app/schemas/job.py

from typing import TypedDict, Optional, Literal, Dict, Any
from datetime import datetime


# Job status type
JobStatus = Literal["queued", "processing", "done", "error", "cancelled"]


class JobInfo(TypedDict, total=False):
    """TypedDict for job information - matches SQLAlchemy Job model"""

    # Identifiers / timestamps
    job_id: str
    created_at: datetime
    started_at: datetime
    finished_at: datetime

    # Status
    status: JobStatus

    # Source file metadata
    filename: Optional[str]
    path: Optional[str]  # Temporary file path on disk
    doc_title: Optional[str]  # Human-readable document name
    namespace: Optional[str]  # e.g. "book_<job_id>"
    index_name: Optional[str]

    # Progress and metrics
    total_chunks: Optional[int]
    processed_chunks: Optional[int]
    progress: Optional[int]  # 0..100 (int)

    # Callback / errors / additional data
    callback_url: Optional[str]
    error: Optional[str]
    extra: Optional[Dict[str, Any]]
