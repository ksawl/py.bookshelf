from typing import TypedDict, Optional, Literal, Dict, Any
from datetime import datetime


# Job status type
JobStatus = Literal["queued", "processing", "done", "error", "cancelled"]


class JobInfo(TypedDict, total=False):
    """TypedDict for job information - matches SQLAlchemy Job model"""
    # идентификаторы / времена
    job_id: str
    created_at: datetime
    started_at: datetime
    finished_at: datetime

    # статус
    status: JobStatus

    # исходные метаданные о файле
    filename: Optional[str]
    path: Optional[str]  # временный путь к файлу на диске
    doc_title: Optional[str]  # человекочитаемое имя документа
    namespace: Optional[str]  # например "book_<job_id>"
    index_name: Optional[str]

    # прогресс и метрики
    total_chunks: Optional[int]
    processed_chunks: Optional[int]
    progress: Optional[int]  # 0..100 (int)

    # коллбэк / ошибки / доп. данные
    callback_url: Optional[str]
    error: Optional[str]
    extra: Optional[Dict[str, Any]]
