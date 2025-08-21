# app/main.py

import os
import asyncio
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
from fastapi.responses import JSONResponse

from app.services.bookshelf_service import BookshelfService
from app.services.background_processor import BackgroundProcessor
from app.schemas.book import Answer, MetaBook
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging, get_logger
from app.core.dependencies import (
    get_database_service,
    get_bookshelf_service,
    get_background_processor,
)
from app.core.exceptions import (
    ConfigurationError,
    validation_http_error,
    not_found_http_error,
    internal_server_http_error,
)


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan - startup and shutdown"""
    # Startup
    logger.info("Starting Bookshelf API")

    try:
        # Setup logging
        setup_logging(log_level="INFO")

        # Validate configuration
        settings = get_settings()
        settings.validate_runtime_dependencies()
        logger.info("Configuration validated successfully")

        # Initialize database
        db_service = get_database_service(settings)
        await db_service.init_db()
        logger.info("Database initialized successfully")

        yield

    except ConfigurationError as e:
        logger.error("Configuration error during startup", error=str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error during startup", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Bookshelf API")


app = FastAPI(title="Bookshelf API", lifespan=lifespan)


@app.get("/bookshelf", summary="Get Indexed Books")
async def get_books_list(
    service: Annotated[BookshelfService, Depends(get_bookshelf_service)],
) -> list[str]:
    try:
        logger.info("Retrieving books list")
        books = await asyncio.to_thread(service.get_all_books)
        logger.info("Successfully retrieved books list", count=len(books))
        return books
    except Exception as e:
        logger.error("Failed to retrieve books list", error=str(e))
        raise internal_server_http_error("Failed to retrieve books list")


@app.get("/bookshelf/{book_id}", summary="Get Meta from Book")
async def get_book_info(
    book_id: str,
    service: Annotated[BookshelfService, Depends(get_bookshelf_service)],
) -> MetaBook:
    try:
        logger.info("Retrieving book metadata", book_id=book_id)
        book = await asyncio.to_thread(service.get_book, book_id)
        if not book:
            logger.warning("Book not found", book_id=book_id)
            raise not_found_http_error("Book", book_id)

        logger.info("Successfully retrieved book metadata", book_id=book_id)
        return book
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve book metadata", book_id=book_id, error=str(e))
        raise internal_server_http_error("Failed to retrieve book metadata")


@app.post("/bookshelf", summary="Add Book")
async def upload_book(
    background: BackgroundTasks,
    service: Annotated[BookshelfService, Depends(get_bookshelf_service)],
    processor: Annotated[BackgroundProcessor, Depends(get_background_processor)],
    cfg: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
) -> JSONResponse:
    try:
        # Validate file
        if not file or not file.filename:
            logger.warning("Upload attempted without file")
            raise validation_http_error("File is required")

        filename = file.filename
        ext = Path(filename).suffix.lower().lstrip(".")

        if ext not in cfg.ALLOWED_EXTENSIONS:
            logger.warning(
                "Unsupported file type attempted",
                filename=filename,
                extension=ext,
                allowed=list(cfg.ALLOWED_EXTENSIONS),
            )
            raise validation_http_error(
                f"Unsupported file type: {ext}. Allowed: {', '.join(cfg.ALLOWED_EXTENSIONS)}"
            )

        logger.info("Starting file upload", filename=filename, size=file.size)

        # Save to temporary location
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"upload_{os.urandom(6).hex()}_{filename}")

        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        logger.info(
            "File saved to temporary location", tmp_path=tmp_path, size=len(contents)
        )

        # Enqueue processing job
        book_id = await asyncio.to_thread(service.enqueue_file, tmp_path, filename)
        logger.info("File enqueued for processing", book_id=book_id, filename=filename)

        # Schedule background processing
        background.add_task(processor.process_file, book_id)
        logger.info("Background processing scheduled", book_id=book_id)

        return JSONResponse(
            status_code=202,
            content={"doc_id": book_id, "status": "accepted", "filename": filename},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to upload book",
            filename=getattr(file, "filename", "unknown"),
            error=str(e),
        )
        raise internal_server_http_error("Failed to upload book")


@app.get("/bookshelf/{book_id}/status")
async def get_status(
    book_id: str,
    service: Annotated[BookshelfService, Depends(get_bookshelf_service)],
):
    try:
        logger.info("Retrieving job status", book_id=book_id)
        status = await asyncio.to_thread(service.get_job_status, book_id)
        if not status:
            logger.warning("Job not found", book_id=book_id)
            raise not_found_http_error("Job", book_id)

        logger.info(
            "Successfully retrieved job status",
            book_id=book_id,
            status=status.get("status"),
        )
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve job status", book_id=book_id, error=str(e))
        raise internal_server_http_error("Failed to retrieve job status")


@app.post("/bookshelf/{book_id}", summary="Send Question from Book")
async def ask_question(
    book_id: str,
    q: Annotated[str, Query(max_length=300)],
    service: Annotated[BookshelfService, Depends(get_bookshelf_service)],
) -> Answer:
    try:
        if not q.strip():
            logger.warning("Empty question attempted", book_id=book_id)
            raise validation_http_error("Question cannot be empty")

        logger.info("Processing question", book_id=book_id, question_length=len(q))
        answer = await asyncio.to_thread(service.ask_book_question, book_id, q)
        logger.info(
            "Successfully processed question",
            book_id=book_id,
            sources_count=len(answer.sources),
        )
        return answer
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to process question", book_id=book_id, question=q[:50], error=str(e)
        )
        raise internal_server_http_error("Failed to process question")


@app.delete("/bookshelf/{book_id}", summary="Remove Book")
async def delete_book(
    book_id: str,
    service: Annotated[BookshelfService, Depends(get_bookshelf_service)],
):
    try:
        logger.info("Deleting book", book_id=book_id)
        result = await asyncio.to_thread(service.remove_book, book_id)

        if not result["success"]:
            if "not found" in result["message"].lower():
                logger.warning("Book not found for deletion", book_id=book_id)
                raise not_found_http_error("Book", book_id)
            else:
                logger.error(
                    "Failed to delete book", book_id=book_id, message=result["message"]
                )
                raise internal_server_http_error(result["message"])

        logger.info("Successfully deleted book", book_id=book_id)
        return {"status": "success", "message": result["message"], "book_id": book_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete book", book_id=book_id, error=str(e))
        raise internal_server_http_error("Failed to delete book")
