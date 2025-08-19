# app/services/background_processor.py
"""
Отдельный сервис для background обработки файлов.
Изолирован от основного BookshelfService для лучшей архитектуры.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List

from app.services.pinecone_service import PineconeService
from app.services.database_service import DatabaseService
from app.services.document_processing_service import DocumentProcessingService
from app.core.config import Settings
from app.core.logging import LoggerMixin
from app.core.exceptions import DocumentProcessingError, PineconeServiceError


class BackgroundProcessor(LoggerMixin):
    """Отдельный класс для фоновой обработки файлов"""
    
    def __init__(self, settings: Settings, database: DatabaseService):
        self.settings = settings
        self.database = database
        self.pinecone = PineconeService(settings=settings)
        self.document_processor = DocumentProcessingService(settings=settings)
        self.logger.info("BackgroundProcessor initialized")

    async def process_file(self, book_id: str) -> None:
        """
        Обработка файла с использованием DocumentProcessingService.
        Изолированный метод для background tasks.
        """
        self.logger.info("Starting file processing", book_id=book_id)
        
        job = await self.database.get_job(book_id)
        if not job:
            self.logger.error("Job not found", book_id=book_id)
            return

        await self.database.update_job(book_id, {"status": "processing"})
        path = job.get("path")
        filename = job.get("filename")
        namespace = job.get("namespace")

        try:
            # 1. Обрабатываем документ через специализированный сервис
            self.logger.info("Processing document", book_id=book_id, filename=filename)
            processing_result = await self.document_processor.process_document(
                book_id, path, filename
            )
            
            chunks_with_embeddings = processing_result["chunks"]
            embedding_dimension = processing_result["embedding_dimension"]
            total_chunks = processing_result["total_chunks"]
            
            self.logger.info("Document processed successfully", 
                           book_id=book_id, total_chunks=total_chunks, 
                           embedding_dimension=embedding_dimension)
            
            await self.database.update_job(book_id, {
                "total_chunks": total_chunks,
                "extra": {"document_metadata": processing_result["document_metadata"]}
            })

            # 2. Убеждаемся, что индекс существует с правильной размерностью
            if embedding_dimension == 0:
                raise DocumentProcessingError("Не удалось определить размерность эмбеддингов")
            
            index_name = self.pinecone.create_index_name(book_id)
            chosen_index = await self.pinecone.ensure_index_for_dimension(
                embedding_dimension, index_name
            )
            await self.database.update_job(book_id, {"index_name": chosen_index})

            # 3. Загружаем данные в Pinecone батчами
            await self._upsert_chunks_to_pinecone(
                chunks_with_embeddings, namespace, book_id, job, chosen_index
            )

            await self.database.finish_job(book_id, success=True)
            self.logger.info("File processing completed successfully", book_id=book_id)
            
        except Exception as e:
            self.logger.error("File processing failed", book_id=book_id, error=str(e))
            await self.database.finish_job(book_id, success=False, error=str(e))
        finally:
            # Удаляем временный файл
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    self.logger.debug("Temporary file removed", path=path)
            except Exception as e:
                self.logger.warning("Failed to remove temporary file", path=path, error=str(e))

    async def _upsert_chunks_to_pinecone(
        self, 
        chunks_with_embeddings: List[Dict[str, Any]], 
        namespace: str, 
        book_id: str, 
        job: Dict[str, Any],
        index_name: str
    ) -> None:
        """Загружает чанки в Pinecone батчами"""
        BATCH_SIZE = 64
        processed_count = 0
        total_chunks = len(chunks_with_embeddings)
        
        self.logger.info("Starting Pinecone upsert", 
                       book_id=book_id, total_chunks=total_chunks, batch_size=BATCH_SIZE)
        
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_chunks = chunks_with_embeddings[i:i + BATCH_SIZE]
            
            try:
                # Подготавливаем данные для upsert
                upsert_data = self.document_processor.prepare_for_pinecone_upsert(
                    batch_chunks
                )
                
                # Загружаем в Pinecone
                await self.pinecone.upsert(upsert_data, index_name=index_name, namespace=namespace)
                
                # Обновляем прогресс
                processed_count += len(batch_chunks)
                await self.database.increment_processed(book_id, n=len(batch_chunks))
                
                self.logger.debug("Batch upserted", 
                                book_id=book_id, batch_num=i//BATCH_SIZE + 1, 
                                processed=processed_count, total=total_chunks)
                
                # Уведомляем callback (если есть)
                callback_url = job.get("callback_url")
                if callback_url:
                    asyncio.create_task(self._notify_callback(callback_url, book_id))
                    
            except Exception as e:
                self.logger.error("Failed to upsert batch", 
                                book_id=book_id, batch_num=i//BATCH_SIZE + 1, error=str(e))
                raise PineconeServiceError(f"Failed to upsert batch: {e}") from e

    async def _notify_callback(self, callback_url: str, job_id: str):
        """Best-effort notifier. Non-blocking."""
        import httpx

        payload = await self.database.get_job(job_id)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json=payload, timeout=5.0)
        except Exception:
            # ignore failures for now
            pass
