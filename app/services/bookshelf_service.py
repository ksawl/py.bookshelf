import uuid
import asyncio
import os
from typing import Optional
from app.schemas.book import Book, Answer

from app.services.pinecone_service import PineconeService
from app.core.jobs import JobStore
from app.utils.book_processor import (
    detect_extension,
    convert_docx_to_markdown,
    extract_headings_from_docx,
    extract_text_from_pdf,
    split_into_token_chunks,
    get_embeddings,
    sanitize_metadata,
)


class BookshelfService:
    def __init__(self):
        self.pinecone = PineconeService()
        self.jobs = JobStore()  # simple in-memory; swap for DB in prod
        # Хранилище книг (в реальном проекте - БД)
        # books_storage: Dict[str, Book] = {}

    def enqueue_file(
        self, file_path: str, filename: str, callback_url: Optional[str] = None
    ) -> str:
        """Register job and return book_id immediately"""
        book_id = str(uuid.uuid4())
        namespace = f"book_{book_id}"
        info = {
            "filename": filename,
            "path": file_path,
            "namespace": namespace,
            "callback_url": callback_url,
        }
        self.jobs.create(book_id, info)
        return book_id

    def get_job_status(self, book_id: str):
        return self.jobs.get(book_id)

    async def get_book(book_id: str) -> Optional[Book]:
        # return books_storage.get(book_id)
        pass

    async def get_all_books() -> list[Book]:
        # return list(books_storage.values())
        pass

    async def ask_book_question(book_id: str, question: str) -> Answer:
        # if book_id not in books_storage:
        #     raise ValueError("Book not found")

        # # Получение ответа из Pinecone
        # answer, sources = await pinecone_service.query_book(book_id, question)

        # return Answer(answer=answer, sources=sources)
        pass

    async def remove_book(book_id: str) -> bool:
        # if book_id in books_storage:
        #     # Удаление из Pinecone
        #     await pinecone_service.delete_book_index(book_id)
        #     # Удаление из локального хранилища
        #     del books_storage[book_id]
        #     return True
        # return False
        pass

    async def process_file(self, book_id: str):
        job = self.jobs.get(book_id)
        if not job:
            return

        self.jobs.update(book_id, {"status": "processing"})
        path = job.get("path")
        filename = job.get("filename")

        try:
            with open(path, "rb") as f:
                raw = f.read()
            ext = detect_extension(filename, raw)

            # Extract text + structural info
            heading_points = []
            pages = None
            if ext == "docx":
                markdown = convert_docx_to_markdown(path)
                full_text, heading_points = extract_headings_from_docx(path)
                text_for_chunk = markdown or full_text
            elif ext == "pdf":
                full_text, pages = extract_text_from_pdf(path)
                text_for_chunk = full_text
                heading_points = []
            else:
                # odt, txt or fallback
                try:
                    text_for_chunk = raw.decode("utf-8")
                except Exception:
                    text_for_chunk = raw.decode("utf-8", errors="ignore")

            # split into chunks (token-based)
            chunks = split_into_token_chunks(text_for_chunk)
            total = len(chunks)
            self.jobs.update(book_id, {"total_chunks": total})

            # We'll upsert in batches. We need to ensure a dense Pinecone index exists
            # Determine index only after we compute the first embeddings batch (to know dimension).
            namespace = job.get("namespace")
            BATCH = 64
            first_batch_done = False

            for i in range(0, total, BATCH):
                window = chunks[i : i + BATCH]
                texts_for_emb = []
                metas = []
                ids = []
                for c in window:
                    # Assign heading_chain reliably later; for now use nearest if available
                    heading_chain = None
                    if heading_points:
                        heading_chain = heading_points[0][1] if heading_points else None

                    intro = f"{heading_chain}" if heading_chain else ""
                    chunk_text = intro + c["text"]
                    texts_for_emb.append(chunk_text)
                    raw_meta = {
                        "doc_id": book_id,
                        "doc_title": filename,
                        "chunk_index": c["chunk_index"],
                        "heading_chain": heading_chain,
                    }
                    metas.append(sanitize_metadata(raw_meta))
                    ids.append(f"{book_id}-{c['chunk_index']}")

                # get embeddings (await)
                embeddings = await get_embeddings(texts_for_emb)

                # On first batch, ensure index exists and is dense with proper dimension
                if not first_batch_done:
                    emb_dim = len(embeddings[0]) if embeddings else None
                    if emb_dim is None:
                        raise RuntimeError("Received empty embeddings from provider")
                    # This will create or switch to a dense-compatible index if needed
                    index = self.pinecone.create_index(book_id)
                    chosen_index = self.pinecone.ensure_index_for_dimension(
                        emb_dim, index
                    )
                    # store chosen index name in job for debugging/inspection
                    self.jobs.update(book_id, {"index_name": chosen_index})
                    first_batch_done = True

                # prepare upsert tuples
                upsert_items = [
                    (ids[j], embeddings[j], metas[j]) for j in range(len(ids))
                ]

                # perform upsert into the shared index under the job's namespace
                self.pinecone.upsert(upsert_items, namespace=namespace)

                # update progress
                self.jobs.increment_processed(book_id, n=len(window))

                # optional: notify callback_url if provided (best-effort, non-blocking)
                cb = job.get("callback_url")
                if cb:
                    # fire-and-forget
                    asyncio.create_task(self._notify_callback(cb, book_id))

            self.jobs.finish(book_id, success=True)
        except Exception as e:
            self.jobs.finish(book_id, success=False, error=str(e))
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    async def _notify_callback(self, callback_url: str, job_id: str):
        """Best-effort notifier. Non-blocking.
        Keep simple: no retries here; in prod use httpx with retry/backoff.
        """
        import httpx

        payload = self.jobs.get(job_id)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json=payload, timeout=5.0)
        except Exception:
            # ignore failures for now
            pass
