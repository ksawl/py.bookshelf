from fastapi import UploadFile, HTTPException
from app.schemas.book import Book, Answer
from app.services import pinecone_service
from app.utils.file_processor import process_file
from app.core.config import settings
from typing import Dict, Optional
import uuid

# Хранилище книг (в реальном проекте - БД)
books_storage: Dict[str, Book] = {}


async def add_book(file: UploadFile) -> str:
    # Проверка: если уже есть книга и бесплатный Pinecone - удаляем старую
    if books_storage and settings.PINECONE_FREE_TIER:
        old_book_id = list(books_storage.keys())[0]
        await remove_book(old_book_id)

    book_id: str = str(uuid.uuid4())

    # Сохранение метаинформации
    book = Book(
        id=book_id,
        title=file.filename or f"book_{book_id}",
        filename=file.filename or "unknown",
        size=getattr(file, "size", None),
    )
    books_storage[book_id] = book

    try:
        # Обработка файла
        content = await process_file(file)

        # Индексация в Pinecone
        await pinecone_service.index_book_content(book_id, content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Book processing failed: {str(e)}")

    return book_id


async def get_book(book_id: str) -> Optional[Book]:
    return books_storage.get(book_id)


async def get_all_books() -> list[Book]:
    return list(books_storage.values())


async def ask_book_question(book_id: str, question: str) -> Answer:
    if book_id not in books_storage:
        raise ValueError("Book not found")

    # Получение ответа из Pinecone
    answer, sources = await pinecone_service.query_book(book_id, question)

    return Answer(answer=answer, sources=sources)


async def remove_book(book_id: str) -> bool:
    if book_id in books_storage:
        # Удаление из Pinecone
        await pinecone_service.delete_book_index(book_id)
        # Удаление из локального хранилища
        del books_storage[book_id]
        return True
    return False
