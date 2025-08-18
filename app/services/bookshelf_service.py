# app/services/bookshelf_service.py
import uuid
import asyncio
import os
from typing import Optional, List, Dict, Any
from langchain_ollama.llms import OllamaLLM


from app.schemas.book import Answer, MetaBook
from app.services.pinecone_service import PineconeService
from app.services.database_service import DatabaseService
from app.schemas.job import JobInfo
from app.core.config import Settings
from app.utils.book_processor import BookProcessor


class BookshelfService:
    def __init__(self, settings: Settings, database: DatabaseService):
        self.pinecone = PineconeService(settings=settings)
        self.bp = BookProcessor(settings=settings)
        self.database = database

        self.ollama_base_url = settings.OLLAMA_API_BASE_URL
        self.llm_model = settings.LLM_MODEL
        self.max_context_chars = settings.MAX_CONTEXT_CHARS
        self.top_k = settings.TOP_K

        # Инициализация LangChain Ollama (sync). Поскольку LangChain-объекты, как правило,
        # синхронные, вызовы LLM будут выполняться в threadpool (asyncio.to_thread).
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
        book = self.pinecone.get_book_metadata(book_id)
        return book

    async def get_all_books(self) -> list[str]:
        indexes = self.pinecone.list_indexes()
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

            # 6) вызов LLM — LangChain Ollama sync объект в threadpool
            def _sync_llm_call(p: str) -> str:
                try:
                    # invoke — рекомендованный интерфейс
                    return str(self.llm.invoke(p))
                except Exception as exc:
                    print("LLM sync call failed: %s", exc)
                    raise

            llm_resp_text = await asyncio.to_thread(_sync_llm_call, prompt)

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
        job = await self.database.get_job(book_id)
        if not job:
            return

        await self.database.update_job(book_id, {"status": "processing"})
        path = job.get("path")
        filename = job.get("filename")

        try:
            with open(path, "rb") as f:
                raw = f.read()
            ext = self.bp.detect_extension(filename, raw)

            # Extract text + structural info
            heading_points = []
            pages = None
            if ext == "docx":
                markdown = self.bp.convert_docx_to_markdown(path)
                full_text, heading_points = self.bp.extract_headings_from_docx(path)
                text_for_chunk = markdown or full_text
            elif ext == "pdf":
                full_text, pages = self.bp.extract_text_from_pdf(path)
                text_for_chunk = full_text
                heading_points = []
            else:
                # odt, txt or fallback
                try:
                    text_for_chunk = raw.decode("utf-8")
                except Exception:
                    text_for_chunk = raw.decode("utf-8", errors="ignore")

            # split into chunks (token-based)
            chunks = self.bp.split_into_token_chunks(text_for_chunk)
            total = len(chunks)
            await self.database.update_job(book_id, {"total_chunks": total})

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
                    metas.append(self.bp.sanitize_metadata(raw_meta))
                    ids.append(f"{book_id}-{c['chunk_index']}")

                # get embeddings (await)
                embeddings = await self.bp.get_embeddings(texts_for_emb)

                # On first batch, ensure index exists and is dense with proper dimension
                if not first_batch_done:
                    emb_dim = len(embeddings[0]) if embeddings else None
                    if emb_dim is None:
                        raise RuntimeError("Received empty embeddings from provider")
                    # This will create or switch to a dense-compatible index if needed
                    index = self.pinecone.create_index_name(book_id)
                    chosen_index = self.pinecone.ensure_index_for_dimension(
                        emb_dim, index
                    )
                    # store chosen index name in job for debugging/inspection
                    await self.database.update_job(book_id, {"index_name": chosen_index})
                    first_batch_done = True

                # prepare upsert tuples
                upsert_items = [
                    (ids[j], embeddings[j], metas[j]) for j in range(len(ids))
                ]

                # perform upsert into the shared index under the job's namespace
                self.pinecone.upsert(upsert_items, namespace=namespace)

                # update progress
                await self.database.increment_processed(book_id, n=len(window))

                # optional: notify callback_url if provided (best-effort, non-blocking)
                cb = job.get("callback_url")
                if cb:
                    # fire-and-forget
                    asyncio.create_task(self._notify_callback(cb, book_id))

            await self.database.finish_job(book_id, success=True)
        except Exception as e:
            await self.database.finish_job(book_id, success=False, error=str(e))
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

        payload = await self.database.get_job(job_id)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json=payload, timeout=5.0)
        except Exception:
            # ignore failures for now
            pass
