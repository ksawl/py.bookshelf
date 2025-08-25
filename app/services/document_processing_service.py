# app/services/document_processing_service.py
"""
Centralized service for document processing.
Responsible for:
1. Parsing and text extraction from different formats
2. Creating chunks with metadata
3. Generating embeddings
4. Preparing data for Pinecone indexing
"""

from typing import List, Dict, Any, Tuple, Optional
from app.utils.book_processor import BookProcessor
from app.core.config import Settings


class DocumentProcessingService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self.bp = BookProcessor(settings=settings)

    def process_document(
        self, book_id: str, file_path: str, filename: str
    ) -> Dict[str, Any]:
        """
        Complete document processing: text extraction, chunking, embeddings.

        Returns:
            Dict with fields:
            - chunks: List[Dict] - ready chunks with embeddings and metadata
            - embedding_dimension: int - vector dimension
            - total_chunks: int - number of chunks
            - document_metadata: Dict - document metadata
        """

        # Determine file type and extract text
        with open(file_path, "rb") as f:
            raw_data = f.read()

        file_extension = self.bp.detect_extension(filename, raw_data)

        # Extract text and structural information (синхронно)
        text_content, structural_info = self._extract_text_and_structure(
            file_path, file_extension, raw_data
        )

        # Split into chunks
        raw_chunks = self.bp.split_into_token_chunks(text_content)

        # Enrich chunks with metadata and create final text for embeddings (синхронно)
        enriched_chunks = self._enrich_chunks_with_metadata(
            raw_chunks, book_id, filename, structural_info
        )

        # Generate embeddings in batches (синхронно)
        chunks_with_embeddings = self._generate_embeddings_for_chunks(
            enriched_chunks
        )

        # Prepare result
        result = {
            "chunks": chunks_with_embeddings,
            "embedding_dimension": len(chunks_with_embeddings[0]["embedding"])
            if chunks_with_embeddings
            else 0,
            "total_chunks": len(chunks_with_embeddings),
            "document_metadata": {
                "book_id": book_id,
                "filename": filename,
                "file_extension": file_extension,
                "original_text_length": len(text_content),
                "has_headings": bool(structural_info.get("headings")),
                "has_pages": bool(structural_info.get("pages")),
            },
        }

        return result

    def _extract_text_and_structure(
        self, file_path: str, file_extension: str, raw_data: bytes
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and structural information depending on file type.

        Returns:
            Tuple[str, Dict]: (text_content, structural_info)
            structural_info contains:
            - headings: List[Tuple[int, str]] for docx
            - pages: List[Dict] for pdf
        """

        structural_info = {"headings": [], "pages": []}

        if file_extension == "docx":
            # For DOCX get both markdown and heading structure
            markdown_text = self.bp.convert_docx_to_markdown(file_path)
            full_text, heading_points = self.bp.extract_headings_from_docx(file_path)

            # Prefer markdown, but if not available - use plain text
            text_content = markdown_text or full_text
            structural_info["headings"] = heading_points

        elif file_extension == "pdf":
            # For PDF extract text and page information
            full_text, pages_info = self.bp.extract_text_from_pdf(file_path)
            text_content = full_text
            structural_info["pages"] = pages_info

        else:
            # For other formats (txt, odt) - simple reading
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
        structural_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Enrich chunks with metadata: headings, pages, position in document.
        """

        enriched_chunks = []
        headings = structural_info.get("headings", [])
        pages = structural_info.get("pages", [])

        for chunk in raw_chunks:
            # Basic metadata
            metadata = {
                "doc_id": book_id,
                "doc_title": filename,
                "chunk_index": chunk["chunk_index"],
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "actual_tokens": chunk["actual_tokens"],
            }

            # Add heading information for DOCX
            if headings:
                heading_chain = self._find_relevant_heading(
                    chunk["token_start"], headings
                )
                metadata["heading_chain"] = heading_chain

                # Form final text with heading at the beginning
                if heading_chain:
                    chunk_text = f"{heading_chain}\n\n{chunk['text']}"
                else:
                    chunk_text = chunk["text"]
            else:
                chunk_text = chunk["text"]
                metadata["heading_chain"] = None

            # Add page information for PDF
            if pages:
                page_info = self._find_relevant_page(chunk["token_start"], pages)
                metadata.update(page_info)

            # Sanitize metadata for Pinecone
            sanitized_metadata = self.bp.sanitize_metadata(metadata)

            enriched_chunks.append(
                {
                    "id": f"{book_id}-{chunk['chunk_index']}",
                    "text": chunk_text,
                    "metadata": sanitized_metadata,
                    "original_chunk": chunk,
                }
            )

        return enriched_chunks

    def _find_relevant_heading(
        self, token_position: int, headings: List[Tuple[int, str]]
    ) -> Optional[str]:
        """
        Find most suitable heading for given position in text.
        """
        if not headings:
            return None

        # Find last heading that comes before or at current position
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
        Find page information for given position in text.
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

    def _generate_embeddings_for_chunks(
        self, enriched_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all chunks in batches.
        """
        if not enriched_chunks:
            return []

        # Extract texts for embeddings
        texts_for_embedding = [chunk["text"] for chunk in enriched_chunks]

        # Generate embeddings synchronously
        embeddings = self.bp.get_embeddings_sync(texts_for_embedding)

        # Add embeddings to chunks
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
        Prepare data in format required for Pinecone upsert.

        Returns:
            List[Tuple[str, List[float], Dict[str, Any]]]: (id, vector, metadata)
        """
        upsert_data = []

        for chunk in chunks_with_embeddings:
            # Add text to metadata for search capability
            metadata_with_text = chunk["metadata"].copy()
            metadata_with_text["text"] = chunk["text"]

            upsert_data.append((chunk["id"], chunk["embedding"], metadata_with_text))

        return upsert_data
