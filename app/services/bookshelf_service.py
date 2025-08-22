# app/services/bookshelf_service.py
import uuid
from typing import Optional, List, Dict, Any
from langchain_ollama.llms import OllamaLLM

from app.schemas.book import Answer, MetaBook
from app.services.pinecone_service import PineconeService
from app.services.database_service import SyncDatabaseService
from app.schemas.job import JobInfo
from app.core.config import Settings
from app.utils.book_processor import BookProcessor
from app.core.logging import LoggerMixin


class BookshelfService(LoggerMixin):
    """Синхронный сервис для работы с книжной полкой. 
    Создается для каждого запроса через Depends."""
    
    def __init__(self, settings: Settings, database: SyncDatabaseService):
        self.settings = settings
        self.database = database

        # Create services for each request (stateless)
        self.pinecone = PineconeService(settings=settings)
        self.bp = BookProcessor(settings=settings)

        # Initialize LangChain Ollama
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL, base_url=settings.OLLAMA_API_BASE_URL
        )
        self.logger.info("BookshelfService initialized", llm_model=settings.LLM_MODEL)

    def enqueue_file(
        self, file_path: str, filename: str, callback_url: Optional[str] = None
    ) -> str:
        """Register job and return book_id immediately"""
        book_id = str(uuid.uuid4())
        namespace = f"book_{book_id}"
        info: JobInfo = {
            "filename": filename,
            "path": file_path,
            "namespace": namespace,
            "callback_url": callback_url,
        }
        self.database.create_job(book_id, info)
        return book_id

    def get_job_status(self, book_id: str):
        return self.database.get_job(book_id)

    def get_book(self, book_id: str) -> MetaBook:
        # Note: Pinecone service methods are still async, we'll need to run them in sync
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        book = loop.run_until_complete(self.pinecone.get_book_metadata(book_id))
        return book

    def get_all_books(self) -> list[str]:
        """Get all books with synchronization between Pinecone and database"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Get indexes from Pinecone and jobs from DB
        pinecone_indexes = loop.run_until_complete(self.pinecone.list_indexes())
        all_jobs = self.database.get_all_jobs()

        # Extract book_id from completed jobs
        completed_jobs = [job for job in all_jobs if job.get("status") == "done"]
        db_book_ids = set(job["job_id"] for job in completed_jobs)

        # Extract book_id from Pinecone indexes (format: book-{book_id})
        pinecone_book_ids = set()
        for index_name in pinecone_indexes:
            if index_name.startswith("book-"):
                book_id = index_name.replace("book-", "", 1)
                pinecone_book_ids.add(book_id)

        # Synchronize discrepancies
        self._sync_pinecone_and_db(pinecone_book_ids, db_book_ids, loop)

        # Return actual list of book_id
        return list(pinecone_book_ids - (db_book_ids - pinecone_book_ids))

    def _sync_pinecone_and_db(self, pinecone_book_ids: set, db_book_ids: set, loop):
        """Synchronization between Pinecone and database"""
        # Indexes in Pinecone without DB records - create DB records
        orphaned_pinecone = pinecone_book_ids - db_book_ids
        if orphaned_pinecone:
            for book_id in orphaned_pinecone:
                try:
                    self._create_db_record_from_pinecone(book_id, loop)
                except Exception:
                    pass  # Continue with other records

        # DB records without Pinecone indexes - remove from DB
        orphaned_db = db_book_ids - pinecone_book_ids
        if orphaned_db:
            for book_id in orphaned_db:
                try:
                    self.database.delete_job(book_id)
                except Exception:
                    pass  # Continue with other records

    def _create_db_record_from_pinecone(self, book_id: str, loop):
        """Create DB record based on Pinecone metadata"""
        try:
            book_metadata = loop.run_until_complete(self.pinecone.get_book_metadata(book_id))
            namespace = f"book_{book_id}"
            index_name = self.pinecone.create_index_name(book_id)

            job_info = {
                "filename": book_metadata.get("sample_metadata", {}).get(
                    "doc_title", f"book_{book_id}"
                ),
                "path": None,  # File already processed
                "namespace": namespace,
                "index_name": index_name,
                "status": "done",
                "total_chunks": book_metadata.get("vector_count", 0),
                "processed_chunks": book_metadata.get("vector_count", 0),
                "progress": 100,
            }
            self.database.create_job(book_id, job_info)
        except Exception as e:
            print(f"Error creating DB record for book_id {book_id}: {e}")

    def ask_book_question(self, book_id: str, question: str) -> Answer:
        """Get LLM response based on book context"""
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Get embedding and query Pinecone
            q_emb_list = loop.run_until_complete(self.bp.get_embeddings([question]))
            if not q_emb_list:
                raise RuntimeError("get_embeddings returned empty result")

            q_emb = q_emb_list[0]
            resp = loop.run_until_complete(self.pinecone.query_top_k(
                book_id,
                vector=q_emb,
                top_k=self.settings.TOP_K,
                include_metadata=True,
                include_values=False,
            ))

            # Process search results
            matches = resp.get("matches", []) if isinstance(resp, dict) else []
            context_chunks = self._process_search_matches(matches)

            # Build context and call LLM
            combined_context = self._build_context(context_chunks)
            prompt = self._build_prompt(combined_context, question)

            llm_resp_text = loop.run_until_complete(self.llm.ainvoke(prompt))

            return Answer(
                text=llm_resp_text,
                sources=[
                    {"id": c["id"], "score": c["score"], "metadata": c["metadata"]}
                    for c in context_chunks
                ],
                prompt=prompt,
            )

        except Exception as e:
            print(f"Error in ask_book_question: {e}")
            return Answer(text=f"Error processing request: {e}", sources=[], prompt="")

    def _process_search_matches(
        self, matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process search results from Pinecone"""
        context_chunks = []
        for m in matches:
            mid = m.get("id")
            score = m.get("score") or m.get("distance")
            meta = m.get("metadata") or {}
            text = (
                meta.get("text") or meta.get("content") or meta.get("chunk_text") or ""
            )

            context_chunks.append(
                {"id": mid, "score": score, "text": text, "metadata": meta}
            )
        return context_chunks

    def _build_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Build LLM context from found chunks"""
        parts = []
        for c in context_chunks:
            src = (
                c["metadata"].get("source")
                or c["metadata"].get("page")
                or f"chunk:{c['id']}"
            )
            header = f"[source: {src} | id: {c['id']} | score: {c['score']}]"
            parts.append(f"{header}\n{c['text']}")

        combined_context = "\n\n---\n\n".join(parts)
        if len(combined_context) > self.settings.MAX_CONTEXT_CHARS:
            combined_context = (
                combined_context[: self.settings.MAX_CONTEXT_CHARS]
                + "\n\n...[truncated]..."
            )
        return combined_context

    def _build_prompt(self, context: str, question: str) -> str:
        """Build prompt for LLM"""
        return (
            "You are an assistant. Answer using ONLY the provided fragments and only "
            "based on fragment content, without adding anything extra. If information "
            "is insufficient, honestly say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (briefly):"
        )

    def remove_book(self, book_id: str) -> Dict[str, Any]:
        """Remove book from Pinecone index and delete corresponding DB record"""
        self.logger.info("Removing book", book_id=book_id)

        # Check book existence in database
        job = self.database.get_job(book_id)
        if not job:
            self.logger.warning("Book not found for removal", book_id=book_id)
            return {"success": False, "message": f"Book with id {book_id} not found"}

        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Delete index from Pinecone
            index_name = self.pinecone.create_index_name(book_id)
            loop.run_until_complete(self.pinecone.delete_index(index_name))
            self.logger.info(
                "Pinecone index deleted", book_id=book_id, index_name=index_name
            )

            # Delete record from database
            self.database.delete_job(book_id)
            self.logger.info("Database record deleted", book_id=book_id)

            return {"success": True, "message": f"Book {book_id} successfully removed"}

        except Exception as e:
            self.logger.error("Failed to remove book", book_id=book_id, error=str(e))
            return {
                "success": False,
                "message": f"Error removing book {book_id}: {str(e)}",
            }
