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


class BookshelfService:
    def __init__(self, settings: Settings, database: DatabaseService):
        self.pinecone = PineconeService(settings=settings)
        self.bp = BookProcessor(settings=settings)  # Оставляем для совместимости с ask_book_question
        self.document_processor = DocumentProcessingService(settings=settings)
        self.database = database

        self.ollama_base_url = settings.OLLAMA_API_BASE_URL
        self.llm_model = settings.LLM_MODEL
        self.max_context_chars = settings.MAX_CONTEXT_CHARS
        self.top_k = settings.TOP_K

        # Инициализация LangChain Ollama для асинхронной работы
        self.llm = OllamaLLM(model=self.llm_model, base_url=self.ollama_base_url)

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
        indexes = await self.pinecone.list_indexes()
        return indexes

    async def ask_book_question(self, book_id: str, question: str) -> Answer:
        """
        Основной метод: получает эмбеддинг вопроса, делает запрос в Pinecone,
        формирует контекст и вызывает LLM для генерации ответа.
        """
        try:
            # 1) получаем эмбеддинг вопроса
            # Ожидается, что get_embeddings — асинхронная функция, возвращающая List[List[float]]
            q_emb_list = await self.bp.get_embeddings([question])
            if not q_emb_list:
                raise RuntimeError("get_embeddings вернул пустой результат")
            q_emb = q_emb_list[0]
            # 2) запрос в Pinecone (через вынесенный метод)
            # Каждый индекс соответствует одной книге и формируется из book_id.

            resp = await self.pinecone.query_top_k(
                book_id,
                vector=q_emb,
                top_k=self.top_k,
                filter=None,
                include_metadata=True,
                include_values=False,
            )

            matches = resp.get("matches", []) if isinstance(resp, dict) else []

            # 3) нормализуем и собираем чанки
            context_chunks: List[Dict[str, Any]] = []
            for m in matches:
                mid = m.get("id")
                score = m.get("score") or m.get("distance")
                meta = m.get("metadata") or {}
                text = (
                    meta.get("text")
                    or meta.get("content")
                    or meta.get("chunk_text")
                    or ""
                )
                context_chunks.append(
                    {"id": mid, "score": score, "text": text, "metadata": meta}
                )

            # 4) формируем контекст для LLM
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
            if len(combined_context) > self.max_context_chars:
                combined_context = (
                    combined_context[: self.max_context_chars] + "\n\n...[truncated]..."
                )

            # 5) формируем промпт
            prompt = (
                "Ты — помощник. Отвечай используя ТОЛЬКО предоставленные фрагменты и только по содержанию фрагментов, не добавляя ничего лишнего. Если информации недостаточно, "
                "честно скажи об этом.\n\n"
                f"Контекст:\n{combined_context}\n\n"
                f"Вопрос: {question}\n\n"
                "Ответ (коротко):"
            )

            # 6) асинхронный вызов LLM через ainvoke
            try:
                llm_resp_text = await self.llm.ainvoke(prompt)
            except Exception as exc:
                print("LLM async call failed: %s", exc)
                raise

            # 7) собираем ответ
            answer = Answer(
                text=llm_resp_text,
                sources=[
                    {"id": c["id"], "score": c["score"], "metadata": c["metadata"]}
                    for c in context_chunks
                ],
                prompt=prompt,
            )
            return answer

        except Exception as e:
            print("Ошибка в ask_book_question: %s", e)
            return Answer(
                text=f"Ошибка при обработке запроса: {e}", sources=[], prompt=prompt
            )

    async def remove_book(self, book_id: str):
        index_name = self.pinecone.create_index_name(book_id)
        self.pinecone.delete_index(index_name)
        # return self.pinecone.delete_index_safe(index_name)

    async def process_file(self, book_id: str):
        """
        Обработка файла с использованием DocumentProcessingService.
        Значительно упрощенная версия - вся сложная логика вынесена в отдельный сервис.
        """
        job = await self.database.get_job(book_id)
        if not job:
            return

        await self.database.update_job(book_id, {"status": "processing"})
        path = job.get("path")
        filename = job.get("filename")
        namespace = job.get("namespace")

        try:
            # 1. Обрабатываем документ через специализированный сервис
            processing_result = await self.document_processor.process_document(
                book_id, path, filename
            )
            
            chunks_with_embeddings = processing_result["chunks"]
            embedding_dimension = processing_result["embedding_dimension"]
            total_chunks = processing_result["total_chunks"]
            
            await self.database.update_job(book_id, {
                "total_chunks": total_chunks,
                "extra": {"document_metadata": processing_result["document_metadata"]}
            })

            # 2. Убеждаемся, что индекс существует с правильной размерностью
            if embedding_dimension == 0:
                raise RuntimeError("Не удалось определить размерность эмбеддингов")
            
            index_name = self.pinecone.create_index_name(book_id)
            chosen_index = await self.pinecone.ensure_index_for_dimension(
                embedding_dimension, index_name
            )
            await self.database.update_job(book_id, {"index_name": chosen_index})

            # 3. Загружаем данные в Pinecone батчами
            BATCH_SIZE = 64
            processed_count = 0
            
            for i in range(0, len(chunks_with_embeddings), BATCH_SIZE):
                batch_chunks = chunks_with_embeddings[i:i + BATCH_SIZE]
                
                # Подготавливаем данные для upsert
                upsert_data = self.document_processor.prepare_for_pinecone_upsert(
                    batch_chunks
                )
                
                # Загружаем в Pinecone
                self.pinecone.upsert(upsert_data, namespace=namespace)
                
                # Обновляем прогресс
                processed_count += len(batch_chunks)
                await self.database.increment_processed(book_id, n=len(batch_chunks))
                
                # Уведомляем callback (если есть)
                callback_url = job.get("callback_url")
                if callback_url:
                    asyncio.create_task(self._notify_callback(callback_url, book_id))

            await self.database.finish_job(book_id, success=True)
            
        except Exception as e:
            await self.database.finish_job(book_id, success=False, error=str(e))
        finally:
            # Удаляем временный файл
            try:
                os.remove(path)
            except Exception:
                pass

    async def _notify_callback(self, callback_url: str, job_id: str):
        """Best-effort notifier. Non-blocking.
        Keep simple: no retries here; in prod use httpx with retry/backoff.
        """
        import httpx

        payload = await self.database.get_job(job_id)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json=payload, timeout=5.0)
        except Exception:
            # ignore failures for now
            pass
