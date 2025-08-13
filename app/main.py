from fastapi import FastAPI, UploadFile, Query, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Annotated
from pathlib import Path
import tempfile
import os

from app.services.bookshelf_service import BookshelfService
from app.schemas.book import Answer, MetaBook
from app.core import config as settings

app = FastAPI(title="Bookshelf API")
controller = BookshelfService()


@app.get("/bookshelf", summary="Get Indexed Books", response_model=list[str])
async def get_books_list():
    return await controller.get_all_books()


@app.get("/bookshelf/{book_id}", summary="Get Meta from Book", response_model=MetaBook)
async def get_book_info(book_id: str):
    book = await controller.get_book(book_id)

    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@app.post("/bookshelf", summary="Add Book", response_model=str)
async def upload_book(background: BackgroundTasks, file: UploadFile = File(...)):
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

    # enqueue in controller: returns book_id
    book_id = controller.enqueue_file(tmp_path, filename)

    # schedule background processing
    background.add_task(controller.process_file, book_id)

    return JSONResponse(
        status_code=202, content={"doc_id": book_id, "status": "accepted"}
    )


@app.get("/bookshelf/{book_id}/status")
def get_status(book_id: str):
    status = controller.get_job_status(book_id)
    if not status:
        raise HTTPException(status_code=404, detail="book_id not found")
    return status


@app.post(
    "/bookshelf/{book_id}", summary="Send Question from Book", response_model=Answer
)
async def ask_question(book_id: str, q: Annotated[str, Query(max_length=300)]):
    return await controller.ask_book_question(book_id, q)


@app.delete("/bookshelf/{book_id}", summary="Remove Book")
async def delete_book(book_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(controller.remove_book, book_id)
    return {"status": "accepted", "book_id": book_id}
