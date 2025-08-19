# app/services/bookshelf_service.py
import uuid
import asyncio
import os
from typing import Optional, List, Dict, Any
from langchain_ollama.llms import OllamaLLM

from app.schemas.book import Answer, MetaBook
from app.services.pinecone_service import PineconeService
from app.services.database_service import DatabaseService
from app.services.document_processing_service import DocumentProcessingService
from app.schemas.job import JobInfo
from app.core.config import Settings
from app.utils.book_processor import BookProcessor
from app.core.logging import LoggerMixin
from app.core.exceptions import BookshelfError, LLMServiceError


class BookshelfService(LoggerMixin):
    def __init__(self, settings: Settings, database: DatabaseService):
        self.settings = settings
        self.database = database
        
        # Create services for each request (stateless)
        self.pinecone = PineconeService(settings=settings)
        self.bp = BookProcessor(settings=settings)  
        self.document_processor = DocumentProcessingService(settings=settings)

        # Initialize LangChain Ollama
        self.llm = OllamaLLM(model=settings.LLM_MODEL, base_url=settings.OLLAMA_API_BASE_URL)
        self.logger.info("BookshelfService initialized", llm_model=settings.LLM_MODEL)

    async def enqueue_file(
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
        await self.database.create_job(book_id, info)
        return book_id

    async def get_job_status(self, book_id: str):
        return await self.database.get_job(book_id)

    async def get_book(self, book_id: str) -> MetaBook:
        book = await self.pinecone.get_book_metadata(book_id)
        return book

    async def get_all_books(self) -> list[str]:
        """Get all books with synchronization between Pinecone and database"""
        # Get indexes from Pinecone and jobs from DB in parallel
        pinecone_indexes, all_jobs = await asyncio.gather(
            self.pinecone.list_indexes(),
            self.database.get_all_jobs()
        )
        
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
        await self._sync_pinecone_and_db(pinecone_book_ids, db_book_ids)
        
        # Return actual list of book_id
        return list(pinecone_book_ids - (db_book_ids - pinecone_book_ids))

    async def _sync_pinecone_and_db(self, pinecone_book_ids: set, db_book_ids: set):
        """Synchronization between Pinecone and database"""
        # Indexes in Pinecone without DB records - create DB records
        orphaned_pinecone = pinecone_book_ids - db_book_ids
        if orphaned_pinecone:
            sync_tasks = []
            for book_id in orphaned_pinecone:
                sync_tasks.append(self._create_db_record_from_pinecone(book_id))
            await asyncio.gather(*sync_tasks, return_exceptions=True)
        
        # DB records without Pinecone indexes - remove from DB
        orphaned_db = db_book_ids - pinecone_book_ids
        if orphaned_db:
            delete_tasks = []
            for book_id in orphaned_db:
                delete_tasks.append(self.database.delete_job(book_id))
            await asyncio.gather(*delete_tasks, return_exceptions=True)

    async def _create_db_record_from_pinecone(self, book_id: str):
        """Create DB record based on Pinecone metadata"""
        try:
            book_metadata = await self.pinecone.get_book_metadata(book_id)
            namespace = f"book_{book_id}"
            index_name = self.pinecone.create_index_name(book_id)
            
            job_info = {
                "filename": book_metadata.get("sample_metadata", {}).get("doc_title", f"book_{book_id}"),
                "path": None,  # File already processed
                "namespace": namespace,
                "index_name": index_name,
                "status": "done",
                "total_chunks": book_metadata.get("vector_count", 0),
                "processed_chunks": book_metadata.get("vector_count", 0),
                "progress": 100
            }
            await self.database.create_job(book_id, job_info)
        except Exception as e:
            print(f"Error creating DB record for book_id {book_id}: {e}")

    async def ask_book_question(self, book_id: str, question: str) -> Answer:
        """Get LLM response based on book context"""
        try:
            # Get embedding and query Pinecone in parallel
            q_emb_list = await self.bp.get_embeddings([question])
            if not q_emb_list:
                raise RuntimeError("get_embeddings returned empty result")
            
            q_emb = q_emb_list[0]
            resp = await self.pinecone.query_top_k(
                book_id,
                vector=q_emb,
                top_k=self.settings.TOP_K,
                include_metadata=True,
                include_values=False,
            )

            # Process search results
            matches = resp.get("matches", []) if isinstance(resp, dict) else []
            context_chunks = self._process_search_matches(matches)

            # Build context and call LLM
            combined_context = self._build_context(context_chunks)
            prompt = self._build_prompt(combined_context, question)
            
            llm_resp_text = await self.llm.ainvoke(prompt)

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
            return Answer(
                text=f"Error processing request: {e}", 
                sources=[], 
                prompt=""
            )

    def _process_search_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process search results from Pinecone"""
        context_chunks = []
        for m in matches:
            mid = m.get("id")
            score = m.get("score") or m.get("distance")
            meta = m.get("metadata") or {}
            text = meta.get("text") or meta.get("content") or meta.get("chunk_text") or ""
            
            context_chunks.append({
                "id": mid, 
                "score": score, 
                "text": text, 
                "metadata": meta
            })
        return context_chunks

    def _build_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Build LLM context from found chunks"""
        parts = []
        for c in context_chunks:
            src = (
                c["metadata"].get("source") or 
                c["metadata"].get("page") or 
                f"chunk:{c['id']}"
            )
            header = f"[source: {src} | id: {c['id']} | score: {c['score']}]"
            parts.append(f"{header}\n{c['text']}")

        combined_context = "\n\n---\n\n".join(parts)
        if len(combined_context) > self.settings.MAX_CONTEXT_CHARS:
            combined_context = (
                combined_context[:self.settings.MAX_CONTEXT_CHARS] + 
                "\n\n...[truncated]..."
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

    async def remove_book(self, book_id: str) -> Dict[str, Any]:
        """Remove book from Pinecone index and delete corresponding DB record"""
        self.logger.info("Removing book", book_id=book_id)
        
        # Check book existence in database
        job = await self.database.get_job(book_id)
        if not job:
            self.logger.warning("Book not found for removal", book_id=book_id)
            return {"success": False, "message": f"Book with id {book_id} not found"}
        
        try:
            # Delete index from Pinecone
            index_name = self.pinecone.create_index_name(book_id)
            await self.pinecone.delete_index(index_name)
            self.logger.info("Pinecone index deleted", book_id=book_id, index_name=index_name)
            
            # Delete record from database
            await self.database.delete_job(book_id)
            self.logger.info("Database record deleted", book_id=book_id)
            
            return {"success": True, "message": f"Book {book_id} successfully removed"}
            
        except Exception as e:
            self.logger.error("Failed to remove book", book_id=book_id, error=str(e))
            return {"success": False, "message": f"Error removing book {book_id}: {str(e)}"}
