from pydantic import BaseModel
from typing import List, Optional, Any, Dict


class Answer(BaseModel):
    answer: str
    sources: List[str]  # ссылки на источники


class MetaBook(BaseModel):
    book_id: str
    namespace: str
    index_name: str
    vector_count: int
    sample_metadata: Optional[Dict[str, Any]]
    raw_stats: Optional[Dict[str, Any]]
    error: Optional[str]
