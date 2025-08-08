from pydantic import BaseModel
from typing import List, Optional


class Book(BaseModel):
    id: str
    title: str
    filename: str
    size: Optional[int] = None


class BookCreate(BaseModel):
    pass  # Пока пустой, файл передается как UploadFile


class Answer(BaseModel):
    answer: str
    sources: List[str]  # ссылки на источники
