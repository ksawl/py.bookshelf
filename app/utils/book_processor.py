# app/utils/book_processor.py
"""
Utility functions: format detection, conversions, extraction, chunking, embeddings.
Keep low-level details here so controller stays orchestrator-only.
"""

import os
import json
import asyncio
import mammoth
import docx
import fitz
import tiktoken
import numpy as np
from openai import OpenAI
from typing import List, Tuple, Any, Dict, Optional
from sentence_transformers import SentenceTransformer

from app.core.config import Settings


class BookProcessor:
    def __init__(self, settings: Settings):
        self._settings = settings

    def detect_extension(self, filename: str, data: bytes) -> str:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext:
            return ext
        if data.startswith(b"%PDF"):
            return "pdf"
        return "txt"

    def convert_docx_to_markdown(self, path: str) -> str:
        with open(path, "rb") as f:
            res = mammoth.convert_to_markdown(f)
        return res.value

    def extract_headings_from_docx(
        self, path: str
    ) -> Tuple[str, List[Tuple[int, str]]]:
        """Return full text and list of (char_start, heading_chain)."""
        doc = docx.Document(path)
        paragraphs = []
        for para in doc.paragraphs:
            paragraphs.append((para.style.name if para.style else "", para.text))

        full_text = ""
        offsets = []
        pos = 0
        for style, text in paragraphs:
            offsets.append((pos, text or "", style or ""))
            full_text += (text or "") + "\n\n"
            pos = len(full_text)

        heading_points = []
        current = {}
        for char_start, text, style in offsets:
            s = style.lower() if style else ""
            if s.startswith("heading"):
                # e.g., "Heading 1" => level=1
                try:
                    level = int("".join([c for c in s if c.isdigit()]))
                except Exception:
                    level = 1
                current[level] = text.strip()
                # clear deeper levels
                for k in list(current.keys()):
                    if k > level:
                        current.pop(k, None)
                chain = " > ".join(current[i] for i in sorted(current) if current[i])
                heading_points.append((char_start, chain))
        return full_text, heading_points

    def extract_text_from_pdf(self, path: str) -> Tuple[str, List[dict]]:
        doc = fitz.open(path)
        full_text = ""
        pages = []
        pos = 0
        for i in range(len(doc)):
            page = doc.load_page(i)
            txt = page.get_text("text") or ""
            full_text += txt + "\n\n"
            start = pos
            pos = len(full_text)
            pages.append({"page": i + 1, "start": start, "end": pos})
        return full_text, pages

    def split_into_token_chunks(
        self,
        text: str,
        chunk_tokens: Optional[int] = None,
        overlap_pct: Optional[float] = None,
        encoding_name: Optional[str] = None,
    ) -> List[dict]:
        """
        Split text into chunks of ~chunk_size tokens with proper overlap.

        Overlap calculation:
        - overlap_pct = 0.2 (20%) means each chunk overlaps with previous by 20% of chunk_tokens
        - step_size = chunk_tokens * (1 - overlap_pct) = chunk_tokens * 0.8
        - This ensures 20% overlap between consecutive chunks
        """
        if not chunk_tokens:
            chunk_tokens = self._settings.CHUNK_TOKENS

        if not overlap_pct:
            overlap_pct = self._settings.OVERLAP_PCT

        if not encoding_name:
            encoding_name = self._settings.ENCODING_NAME

        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)

        # Calculate step size for proper overlap
        # For 20% overlap: step = chunk_tokens * 0.8
        step_size = int(chunk_tokens * (1 - overlap_pct))

        # Ensure step_size is at least 1 to avoid infinite loop
        step_size = max(1, step_size)

        chunks = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))

            # Skip chunks that are too small (less than 10% of target size)
            if end - start < chunk_tokens * 0.1 and idx > 0:
                break

            slice_tokens = tokens[start:end]
            chunk_text = enc.decode(slice_tokens)

            chunks.append(
                {
                    "chunk_index": idx,
                    "text": chunk_text,
                    "token_start": start,
                    "token_end": end,
                    "actual_tokens": len(slice_tokens),
                }
            )

            idx += 1
            start += step_size

        return chunks

    async def get_embeddings_openai(self, texts: List[str]) -> List[List[float]]:
        if not self._settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        if not texts:
            return []

        _openai_client = OpenAI(api_key=self._settings.OPENAI_API_KEY)

        def _sync_call():
            resp = _openai_client.embeddings.create(
                model=self._settings.EMBEDDING_MODEL, input=texts
            )
            # resp.data - list of objects; each has .embedding vector
            return [item.embedding for item in resp.data]

        # Call in separate thread to not block asyncio loop
        return await asyncio.to_thread(_sync_call)

    def _encode_sync(
        self, texts: List[str], batch_size: int = 64, normalize: bool = True
    ) -> List[List[float]]:
        """
        Синхронный вызов: кодирует список строк в numpy массивы и возвращает list of lists.
        """
        if not texts:
            return []

        # load model once on import
        _model = SentenceTransformer(self._settings.EMBED_MODEL_NAME)
        embeddings = _model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        return embeddings.tolist()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Wrap sync call in separate thread to not block FastAPI loop
        return await asyncio.to_thread(self._encode_sync, texts)

    def sanitize_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Приводит метаданные к типам, которые поддерживает Pinecone.
        - Оставляет str, int, float, bool, List[str].
        - Если значение None -> пропускает поле.
        - Для иных типов -> приводит к str (fallback).
        """
        out: Dict[str, Any] = {}
        for k, v in (meta or {}).items():
            if v is None:
                # Skip None (alternative: out[k] = "")
                continue
            if isinstance(v, (str, bool, int, float)):
                out[k] = v
                continue
            if isinstance(v, list):
                # Allow only string lists
                if all(isinstance(x, str) for x in v):
                    out[k] = v
                else:
                    # Convert elements to strings
                    out[k] = [str(x) for x in v]
                continue
            # Fallback: serialize to string
            out[k] = str(v)
        return out

    def to_primitive(self, obj: Any):
        # Primitives - return as is
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # dict / mapping
        if isinstance(obj, dict):
            return {k: self.to_primitive(v) for k, v in obj.items()}
        # lists/tuples/sets
        if isinstance(obj, (list, tuple, set)):
            return [self.to_primitive(v) for v in obj]

        # pydantic v2
        if hasattr(obj, "model_dump"):
            try:
                return self.to_primitive(obj.model_dump())
            except Exception:
                pass

        # pydantic v1 / dataclasses / openapi models
        if hasattr(obj, "dict"):
            try:
                return self.to_primitive(obj.dict())
            except Exception:
                pass

        # some OpenAPI-generated models
        if hasattr(obj, "to_dict"):
            try:
                return self.to_primitive(obj.to_dict())
            except Exception:
                pass

        # sometimes has json serialization method
        if hasattr(obj, "to_json"):
            try:
                return json.loads(obj.to_json())
            except Exception:
                pass

        # fallback: try vars/__dict__
        try:
            d = vars(obj)
            if isinstance(d, dict) and d:
                return self.to_primitive(d)
        except TypeError:
            pass

        # Last resort - string representation (without falling)
        try:
            return str(obj)
        except Exception:
            return repr(obj)
