# app/schemas/book.py

from pydantic import BaseModel
from typing import List, Optional, Any, Dict


class Answer(BaseModel):
    text: str
    sources: List[Dict[str, Any]]
    prompt: str


class MetaBook(BaseModel):
    book_id: str
    namespace: str
    index_name: str
    vector_count: int
    sample_metadata: Optional[Dict[str, Any]]
    raw_stats: Optional[Dict[str, Any]]
    error: Optional[str]
