from fastapi import FastAPI, UploadFile, Query, File
from typing import Annotated
import uvicorn

app = FastAPI(title="Bookshelf API")


@app.get("/bookshelf", summary="Get Indexed Books")
async def get_books_list():
    return {"message": "Hello World !!!"}


@app.get("/bookshelf/{book_id}", summary="Get Meta from Book")
async def get_book_info(book_id: int):
    return {"message": "Hello World !!!"}


@app.post("/bookshelf", summary="Add Book")
async def upload_book(file: UploadFile = File(...)):
    return {"message": "Hello World !!!"}


@app.post("/bookshelf/{book_id}", summary="Send Question from Book")
async def ask_question(book_id: int, q: Annotated[str, Query(max_length=300)]):
    return {"message": "Hello World !!!"}


@app.delete("/bookshelf/{book_id}", summary="Remove Book")
async def delete_book(book_id: int):
    return {"message": "Hello World !!!"}


if __name__ == "__main__":
    uvicorn.run("main:app", reload="true")
