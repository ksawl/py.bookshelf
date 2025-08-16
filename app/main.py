from fastapi import (
    FastAPI,
    UploadFile,
    Query,
    File,
    HTTPException,
    BackgroundTasks,
    Depends,
)
from fastapi.responses import JSONResponse
from functools import lru_cache
from typing import Annotated
from pathlib import Path
import tempfile
import os

from app.services.bookshelf_service import BookshelfService
from app.schemas.book import Answer, MetaBook
from app.core import config as settings

app = FastAPI(title="Bookshelf API")


@lru_cache()
def get_bookshelf_singleton() -> BookshelfService:
    return BookshelfService()


@app.get("/bookshelf", summary="Get Indexed Books")
async def get_books_list(
    service: BookshelfService = Depends(get_bookshelf_singleton),
) -> list[str]:
    return await service.get_all_books()


@app.get("/bookshelf/{book_id}", summary="Get Meta from Book")
async def get_book_info(
    book_id: str,
    service: BookshelfService = Depends(get_bookshelf_singleton),
) -> MetaBook:
    book = await service.get_book(book_id)

    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@app.post("/bookshelf", summary="Add Book")
async def upload_book(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    service: BookshelfService = Depends(get_bookshelf_singleton),
) -> str:
    # basic validation
    if not file:
        raise HTTPException(status_code=400, detail="File is required")

    filename = file.filename
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # save to tmp
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"upload_{os.urandom(6).hex()}_{filename}")
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    # enqueue in service: returns book_id
    book_id = service.enqueue_file(tmp_path, filename)

    # schedule background processing
    background.add_task(service.process_file, book_id)

    return JSONResponse(
        status_code=202, content={"doc_id": book_id, "status": "accepted"}
    )


@app.get("/bookshelf/{book_id}/status")
def get_status(
    book_id: str,
    service: BookshelfService = Depends(get_bookshelf_singleton),
):
    status = service.get_job_status(book_id)
    if not status:
        raise HTTPException(status_code=404, detail="book_id not found")
    return status


@app.post("/bookshelf/{book_id}", summary="Send Question from Book")
async def ask_question(
    book_id: str,
    q: Annotated[str, Query(max_length=300)],
    service: BookshelfService = Depends(get_bookshelf_singleton),
) -> Answer:
    return await service.ask_book_question(book_id, q)


@app.delete("/bookshelf/{book_id}", summary="Remove Book")
async def delete_book(
    book_id: str,
    background_tasks: BackgroundTasks,
    service: BookshelfService = Depends(get_bookshelf_singleton),
):
    background_tasks.add_task(service.remove_book, book_id)
    return {"status": "accepted", "book_id": book_id}
