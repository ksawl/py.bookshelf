from fastapi import FastAPI, UploadFile, Query, File, HTTPException
from typing import Annotated
from app.services import bookshelf_service as bs
from app.schemas.book import Book, Answer


app = FastAPI(title="Bookshelf API")


@app.get("/bookshelf", summary="Get Indexed Books", response_model=list[Book])
async def get_books_list():
    return await bs.get_all_books()


@app.get("/bookshelf/{book_id}", summary="Get Meta from Book", response_model=Book)
async def get_book_info(book_id: str):
    book = await bs.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    return book


@app.post("/bookshelf", summary="Add Book", response_model=str)
async def upload_book(file: UploadFile = File(...)):
    return await bs.add_book(file)


@app.post(
    "/bookshelf/{book_id}", summary="Send Question from Book", response_model=Answer
)
async def ask_question(book_id: str, q: Annotated[str, Query(max_length=300)]):
    return await bs.ask_book_question(book_id, q)


@app.delete("/bookshelf/{book_id}", summary="Remove Book")
async def delete_book(book_id: str):
    success = await bs.remove_book(book_id)
    if not success:
        raise HTTPException(status_code=404, detail="Book not found")
    return {"message": "Book deleted successfully"}
