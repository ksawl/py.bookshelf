import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    UploadFile,
    Query,
    File,
    HTTPException,
    BackgroundTasks,
    Depends,
)
from pathlib import Path
from typing import Annotated
from functools import lru_cache
from fastapi.responses import JSONResponse

from app.services.bookshelf_service import BookshelfService
from app.services.database_service import DatabaseService
from app.services.background_processor import BackgroundProcessor
from app.schemas.book import Answer, MetaBook
from app.core import config as settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan - startup and shutdown"""
    # Startup
    db = _create_database_singleton()
    await db.init_db()
    yield
    # Shutdown (if needed)
    # await db.close_connections() - if needed


app = FastAPI(title="Bookshelf API", lifespan=lifespan)


@lru_cache()
def _create_database_singleton() -> DatabaseService:
    """Keep database singleton for connection pooling"""
    return DatabaseService(settings=settings.get_settings())


def create_bookshelf_service(
    cfg: settings.Settings = Depends(settings.get_settings),
) -> BookshelfService:
    """Create new BookshelfService instance for each request (stateless)"""
    db = _create_database_singleton()
    return BookshelfService(settings=cfg, database=db)


def create_background_processor(
    cfg: settings.Settings = Depends(settings.get_settings),
) -> BackgroundProcessor:
    """Create new BackgroundProcessor instance for background tasks"""
    db = _create_database_singleton()
    return BackgroundProcessor(settings=cfg, database=db)


@app.get("/bookshelf", summary="Get Indexed Books")
async def get_books_list(
    service: BookshelfService = Depends(create_bookshelf_service),
) -> list[str]:
    try:
        return await service.get_all_books()
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving books list: {str(e)}"
        )


@app.get("/bookshelf/{book_id}", summary="Get Meta from Book")
async def get_book_info(
    book_id: str,
    service: BookshelfService = Depends(create_bookshelf_service),
) -> MetaBook:
    try:
        book = await service.get_book(book_id)
        if not book:
            raise HTTPException(status_code=404, detail=f"Book with id '{book_id}' not found")
        return book
    except Exception as e:
        # Если это уже HTTPException, перебрасываем как есть
        if isinstance(e, HTTPException):
            raise
        # Для других ошибок возвращаем 500
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving book metadata: {str(e)}"
        )


@app.post("/bookshelf", summary="Add Book")
async def upload_book(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    service: BookshelfService = Depends(create_bookshelf_service),
    cfg: settings.Settings = Depends(settings.get_settings),
) -> str:
    # basic validation
    if not file:
        raise HTTPException(status_code=400, detail="File is required")

    filename = file.filename
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in cfg.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # save to tmp
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"upload_{os.urandom(6).hex()}_{filename}")
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    # enqueue in service: returns book_id
    book_id = await service.enqueue_file(tmp_path, filename)

    # schedule background processing - используем отдельный BackgroundProcessor
    processor = create_background_processor(cfg)
    background.add_task(processor.process_file, book_id)

    return JSONResponse(
        status_code=202, content={"doc_id": book_id, "status": "accepted"}
    )


@app.get("/bookshelf/{book_id}/status")
async def get_status(
    book_id: str,
    service: BookshelfService = Depends(create_bookshelf_service),
):
    try:
        status = await service.get_job_status(book_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Job with book_id '{book_id}' not found")
        return status
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving job status: {str(e)}"
        )


@app.post("/bookshelf/{book_id}", summary="Send Question from Book")
async def ask_question(
    book_id: str,
    q: Annotated[str, Query(max_length=300)],
    service: BookshelfService = Depends(create_bookshelf_service),
) -> Answer:
    try:
        return await service.ask_book_question(book_id, q)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.delete("/bookshelf/{book_id}", summary="Remove Book")
async def delete_book(
    book_id: str,
    service: BookshelfService = Depends(create_bookshelf_service),
):
    result = await service.remove_book(book_id)
    
    if not result["success"]:
        if "not found" in result["message"]:
            raise HTTPException(status_code=404, detail=result["message"])
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    
    return {"status": "success", "message": result["message"], "book_id": book_id}
