# app/services/document_processing_service.py
"""
Централизованный сервис для обработки документов.
Отвечает за:
1. Парсинг и извлечение текста из разных форматов
2. Создание чанков с метаданными
3. Генерацию эмбеддингов
4. Подготовку данных для индексации в Pinecone
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from app.utils.book_processor import BookProcessor
from app.core.config import Settings


class DocumentProcessingService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self.bp = BookProcessor(settings=settings)

    async def process_document(
        self, 
        book_id: str, 
        file_path: str, 
        filename: str
    ) -> Dict[str, Any]:
        """
        Полная обработка документа: извлечение текста, чанкинг, эмбеддинги.
        
        Returns:
            Dict с полями:
            - chunks: List[Dict] - готовые чанки с эмбеддингами и метаданными
            - embedding_dimension: int - размерность векторов
            - total_chunks: int - количество чанков
            - document_metadata: Dict - метаданные документа
        """
        
        # 1. Определяем тип файла и извлекаем текст
        with open(file_path, "rb") as f:
            raw_data = f.read()
        
        file_extension = self.bp.detect_extension(filename, raw_data)
        
        # 2. Извлекаем текст и структурную информацию
        text_content, structural_info = await self._extract_text_and_structure(
            file_path, file_extension, raw_data
        )
        
        # 3. Разбиваем на чанки
        raw_chunks = self.bp.split_into_token_chunks(text_content)
        
        # 4. Обогащаем чанки метаданными и создаем финальный текст для эмбеддингов
        enriched_chunks = self._enrich_chunks_with_metadata(
            raw_chunks, book_id, filename, structural_info
        )
        
        # 5. Генерируем эмбеддинги батчами
        chunks_with_embeddings = await self._generate_embeddings_for_chunks(
            enriched_chunks
        )
        
        # 6. Подготавливаем результат
        result = {
            "chunks": chunks_with_embeddings,
            "embedding_dimension": len(chunks_with_embeddings[0]["embedding"]) if chunks_with_embeddings else 0,
            "total_chunks": len(chunks_with_embeddings),
            "document_metadata": {
                "book_id": book_id,
                "filename": filename,
                "file_extension": file_extension,
                "original_text_length": len(text_content),
                "has_headings": bool(structural_info.get("headings")),
                "has_pages": bool(structural_info.get("pages")),
            }
        }
        
        return result

    async def _extract_text_and_structure(
        self, file_path: str, file_extension: str, raw_data: bytes
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Извлекает текст и структурную информацию в зависимости от типа файла.
        
        Returns:
            Tuple[str, Dict]: (text_content, structural_info)
            structural_info содержит:
            - headings: List[Tuple[int, str]] для docx
            - pages: List[Dict] для pdf
        """
        
        structural_info = {"headings": [], "pages": []}
        
        if file_extension == "docx":
            # Для DOCX получаем и markdown, и структуру заголовков
            markdown_text = self.bp.convert_docx_to_markdown(file_path)
            full_text, heading_points = self.bp.extract_headings_from_docx(file_path)
            
            # Предпочитаем markdown, но если его нет - используем обычный текст
            text_content = markdown_text or full_text
            structural_info["headings"] = heading_points
            
        elif file_extension == "pdf":
            # Для PDF извлекаем текст и информацию о страницах
            full_text, pages_info = self.bp.extract_text_from_pdf(file_path)
            text_content = full_text
            structural_info["pages"] = pages_info
            
        else:
            # Для остальных форматов (txt, odt) - простое чтение
            try:
                text_content = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                text_content = raw_data.decode("utf-8", errors="ignore")
        
        return text_content, structural_info

    def _enrich_chunks_with_metadata(
        self, 
        raw_chunks: List[Dict], 
        book_id: str, 
        filename: str, 
        structural_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Обогащает чанки метаданными: заголовки, страницы, позиция в документе.
        """
        
        enriched_chunks = []
        headings = structural_info.get("headings", [])
        pages = structural_info.get("pages", [])
        
        for chunk in raw_chunks:
            # Базовые метаданные
            metadata = {
                "doc_id": book_id,
                "doc_title": filename,
                "chunk_index": chunk["chunk_index"],
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "actual_tokens": chunk["actual_tokens"],
            }
            
            # Добавляем информацию о заголовках для DOCX
            if headings:
                heading_chain = self._find_relevant_heading(
                    chunk["token_start"], headings
                )
                metadata["heading_chain"] = heading_chain
                
                # Формируем финальный текст с заголовком в начале
                if heading_chain:
                    chunk_text = f"{heading_chain}\n\n{chunk['text']}"
                else:
                    chunk_text = chunk["text"]
            else:
                chunk_text = chunk["text"]
                metadata["heading_chain"] = None
            
            # Добавляем информацию о страницах для PDF
            if pages:
                page_info = self._find_relevant_page(
                    chunk["token_start"], pages
                )
                metadata.update(page_info)
            
            # Санитизируем метаданные для Pinecone
            sanitized_metadata = self.bp.sanitize_metadata(metadata)
            
            enriched_chunks.append({
                "id": f"{book_id}-{chunk['chunk_index']}",
                "text": chunk_text,
                "metadata": sanitized_metadata,
                "original_chunk": chunk,
            })
        
        return enriched_chunks

    def _find_relevant_heading(
        self, token_position: int, headings: List[Tuple[int, str]]
    ) -> Optional[str]:
        """
        Находит наиболее подходящий заголовок для данной позиции в тексте.
        """
        if not headings:
            return None
        
        # Ищем последний заголовок, который идет перед или на текущей позиции
        relevant_heading = None
        for char_start, heading_chain in headings:
            if char_start <= token_position:
                relevant_heading = heading_chain
            else:
                break
        
        return relevant_heading

    def _find_relevant_page(
        self, token_position: int, pages: List[Dict]
    ) -> Dict[str, Any]:
        """
        Находит информацию о странице для данной позиции в тексте.
        """
        page_info = {"page": 1, "page_start": 0, "page_end": 0}
        
        for page_data in pages:
            if page_data["start"] <= token_position <= page_data["end"]:
                page_info = {
                    "page": page_data["page"],
                    "page_start": page_data["start"],
                    "page_end": page_data["end"],
                }
                break
        
        return page_info

    async def _generate_embeddings_for_chunks(
        self, enriched_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Генерирует эмбеддинги для всех чанков батчами.
        """
        if not enriched_chunks:
            return []
        
        # Извлекаем тексты для эмбеддингов
        texts_for_embedding = [chunk["text"] for chunk in enriched_chunks]
        
        # Генерируем эмбеддинги
        embeddings = await self.bp.get_embeddings(texts_for_embedding)
        
        # Добавляем эмбеддинги к чанкам
        chunks_with_embeddings = []
        for i, chunk in enumerate(enriched_chunks):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embeddings[i]
            chunks_with_embeddings.append(chunk_with_embedding)
        
        return chunks_with_embeddings

    def prepare_for_pinecone_upsert(
        self, chunks_with_embeddings: List[Dict[str, Any]]
    ) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """
        Подготавливает данные в формате, необходимом для Pinecone upsert.
        
        Returns:
            List[Tuple[str, List[float], Dict[str, Any]]]: (id, vector, metadata)
        """
        upsert_data = []
        
        for chunk in chunks_with_embeddings:
            # Добавляем текст в метаданные для возможности поиска
            metadata_with_text = chunk["metadata"].copy()
            metadata_with_text["text"] = chunk["text"]
            
            upsert_data.append((
                chunk["id"],
                chunk["embedding"],
                metadata_with_text
            ))
        
        return upsert_data
