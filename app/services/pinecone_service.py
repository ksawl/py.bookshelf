# app/services/pinecone_service.py
"""
Thin wrapper around modern pinecone client to centralize index management and upserts.
This implementation uses the new `Pinecone` class (pinecone>=1.x) and ensures that
we never try to upsert dense vectors into a sparse index.

Public methods:
- ensure_index_for_dimension(dimension) -> returns chosen_index_name
- upsert(vectors, namespace=None)
- query(...)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pinecone import Pinecone, ServerlessSpec
from typing import List, Optional, Tuple, Dict, Any, AsyncGenerator, TYPE_CHECKING

if TYPE_CHECKING:
    from pinecone import Index

from app.schemas.book import MetaBook
from app.core.config import Settings
from app.core.logging import LoggerMixin
from app.core.exceptions import PineconeServiceError


class PineconeService(LoggerMixin):
    def __init__(self, settings: Settings, default_dim: int = 384):
        self._settings = settings
        self.index_name = settings.PINECONE_INDEX
        self.default_dim = default_dim
        self._pc: Optional[Pinecone] = None
        self._index_cache: Dict[str, Index] = {}

        if not settings.PINECONE_API_KEY:
            raise PineconeServiceError("PINECONE_API_KEY not set")

    @property
    def pc(self) -> Pinecone:
        """Lazy initialization of Pinecone client."""
        if self._pc is None:
            self._pc = Pinecone(
                api_key=self._settings.PINECONE_API_KEY,
                environment=self._settings.PINECONE_ENV
            )
            self.logger.info("Pinecone client initialized")
        return self._pc

    def create_index_name(self, book_id: str) -> str:
        return f"book-{book_id}"

    @asynccontextmanager
    async def get_index(self, index_name: str) -> AsyncGenerator[Index, None]:
        """Context manager for getting Pinecone index with proper async handling."""
        if index_name in self._index_cache:
            yield self._index_cache[index_name]
            return

        try:
            self.logger.debug("Creating index connection", index_name=index_name)
            index = await asyncio.to_thread(lambda: self.pc.Index(index_name))
            self._index_cache[index_name] = index
            yield index
        except Exception as e:
            self.logger.error("Failed to get index", index_name=index_name, error=str(e))
            raise PineconeServiceError(f"Failed to get index {index_name}: {e}") from e

    async def _describe_index(self, name: str) -> Dict[str, Any]:
        """Async wrapper for describe_index."""
        try:
            return await asyncio.to_thread(self.pc.describe_index, name)
        except Exception as e:
            self.logger.warning("Failed to describe index", index_name=name, error=str(e))
            return {}

    def _spec_from_env(self) -> ServerlessSpec:
        """
        Попробовать распарсить PINECONE_ENV, либо взять отдельные переменные.
        Ожидаемые форматы PINECONE_ENV: "us-west1-gcp", "us-east1-aws" и т.п.
        """
        env = self._settings.PINECONE_ENV or ""
        cloud = None
        region = None

        # try parse env like "us-west1-gcp" -> region=us-west1, cloud=gcp
        if "-" in env:
            parts = env.split("-")
            # naive: last part is cloud (gcp/aws), rest joined is region
            if parts[-1] in {"gcp", "aws", "azure"}:
                cloud = parts[-1]
                region = "-".join(parts[:-1]) if len(parts) > 1 else None

        # fallback to explicit vars
        cloud = cloud or self._settings.PINECONE_SERVERLESS_CLOUD
        region = region or self._settings.PINECONE_SERVERLESS_REGION

        if not cloud or not region:
            raise RuntimeError(
                "Cannot determine serverless spec for Pinecone. "
                "Set PINECONE_ENV (like 'us-west1-gcp') or "
                "set PINECONE_SERVERLESS_CLOUD and PINECONE_SERVERLESS_REGION."
            )

        # ServerlessSpec expects cloud like 'aws' or 'gcp' and region like 'us-west1'
        return ServerlessSpec(cloud=cloud, region=region)

    async def query_top_k(
        self,
        book_id: str,
        vector: List[float],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> Dict[str, Any]:
        """
        Выполняет запрос в указанном индексе Pinecone и возвращает результат запроса.

        Args:
            book_id: ID книги для формирования имени индекса и namespace
            vector: вектор запроса (List[float])
            top_k: сколько матчей вернуть
            filter: metadata filter (Pinecone filter dict) - опционально
            include_metadata: вернуть metadata
            include_values: вернуть сами векторные значения

        Returns:
            dict: оригинальный ответ от Pinecone Index.query
        """
        if not book_id:
            raise PineconeServiceError("book_id is required for query")
        
        if not vector:
            raise PineconeServiceError("vector is required for query")

        if not top_k:
            top_k = self._settings.TOP_K

        idx_name = self.create_index_name(book_id)
        namespace = f"book_{book_id}"

        try:
            self.logger.debug("Executing Pinecone query", 
                            book_id=book_id, index_name=idx_name, 
                            top_k=top_k, vector_dim=len(vector))

            async with self.get_index(idx_name) as index:
                resp = await asyncio.to_thread(
                    index.query,
                    vector=vector,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter,
                    include_metadata=include_metadata,
                    include_values=include_values,
                )

                # Normalize response to dict
                if hasattr(resp, "to_dict"):
                    result = resp.to_dict()
                elif isinstance(resp, dict):
                    result = resp
                else:
                    result = {"matches": list(getattr(resp, "matches", []))}

                self.logger.debug("Query completed successfully", 
                                book_id=book_id, matches_count=len(result.get("matches", [])))
                return result

        except Exception as e:
            self.logger.error("Pinecone query failed", 
                            book_id=book_id, index_name=idx_name, error=str(e))
            raise PineconeServiceError(f"Query failed for book {book_id}: {e}") from e

    async def ensure_index_for_dimension(
        self, desired_dim: int, target: Optional[str] = None
    ) -> str:
        if not target:
            target = self._settings.PINECONE_INDEX

        def _sync_operations():
            existing = [i["name"] for i in self.pc.list_indexes()]
            
            def _create_index(name, dim):
                # build ServerlessSpec from env (or raise)
                spec = self._spec_from_env()
                # create index with spec
                self.pc.create_index(name=name, dimension=dim, metric="cosine", spec=spec)
            
            return existing, _create_index
        
        existing, _create_index = await asyncio.to_thread(_sync_operations)

        if target in existing:
            info = await self._describe_index(target)
            dim = info.get("dimension") or info.get("index_config", {}).get("dimension")
            if not dim:
                new_name = f"{self.index_name}-dense-{desired_dim}"
                if new_name not in existing:
                    self.logger.info("Creating new index", index_name=new_name, dimension=desired_dim)
                    await asyncio.to_thread(_create_index, new_name, desired_dim)
                target = new_name
            elif int(dim) != int(desired_dim):
                new_name = f"{self.index_name}-dense-{desired_dim}"
                if new_name not in existing:
                    self.logger.info("Creating new index for different dimension", 
                                   index_name=new_name, dimension=desired_dim)
                    await asyncio.to_thread(_create_index, new_name, desired_dim)
                target = new_name
        else:
            # create the requested index with ServerlessSpec
            self.logger.info("Creating new index", index_name=target, dimension=desired_dim)
            await asyncio.to_thread(_create_index, target, desired_dim)

        self.logger.info("Index ensured", index_name=target, dimension=desired_dim)
        return target

    async def upsert(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        index_name: str,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async upsert operation."""
        if not vectors:
            raise PineconeServiceError("No vectors provided for upsert")

        try:
            self.logger.debug("Upserting vectors", 
                            index_name=index_name, count=len(vectors), namespace=namespace)

            async with self.get_index(index_name) as index:
                result = await asyncio.to_thread(
                    index.upsert,
                    vectors=vectors,
                    namespace=namespace
                )
                
                self.logger.debug("Upsert completed", 
                                index_name=index_name, count=len(vectors))
                return result

        except Exception as e:
            self.logger.error("Upsert failed", 
                            index_name=index_name, count=len(vectors), error=str(e))
            raise PineconeServiceError(f"Upsert failed: {e}") from e

    async def delete_index(self, name: str) -> None:
        """Async deletion of Pinecone index."""
        try:
            self.logger.info("Deleting index", index_name=name)
            await asyncio.to_thread(self.pc.delete_index, name)
            
            # Remove from cache if present
            if name in self._index_cache:
                del self._index_cache[name]
                
            self.logger.info("Index deleted successfully", index_name=name)
        except Exception as e:
            self.logger.error("Failed to delete index", index_name=name, error=str(e))
            raise PineconeServiceError(f"Failed to delete index {name}: {e}") from e

    async def list_indexes(self) -> List[str]:
        """List all available Pinecone indexes."""
        try:
            def _sync_list():
                return [i["name"] for i in self.pc.list_indexes()]
            
            indexes = await asyncio.to_thread(_sync_list)
            self.logger.debug("Listed indexes", count=len(indexes))
            return indexes
        except Exception as e:
            self.logger.error("Failed to list indexes", error=str(e))
            raise PineconeServiceError(f"Failed to list indexes: {e}") from e

    async def get_book_metadata(self, book_id: str) -> MetaBook:
        """
        Собирает метаданные по загруженной книге (namespace = f"book_{book_id}").

        Возвращаемая структура (пример):
        {
        "book_id": "...",
        "namespace": "book_...",
        "index_name": "books-shared-index-dense-384",
        "vector_count": 123,
        "sample_metadata": { "doc_id": "...", "doc_title": "...", "chunk_index": 0, "heading_chain": "H1" },
        "raw_stats": {...},   # если удалось получить
        "error": None
        }
        """
        namespace = f"book_{book_id}"
        index_name = self.create_index_name(book_id)
        out: MetaBook = {
            "book_id": book_id,
            "namespace": namespace,
            "index_name": index_name,
            "vector_count": 0,
            "sample_metadata": None,
            "raw_stats": None,
            "error": None,
        }

        # Async setup and stats
        info = await self._describe_index(index_name)
        host = info.get("host") or info.get("server", {}).get("host") or index_name
        if not host:
            raise RuntimeError(
                "Unable to determine index host for chosen index: %s" % index_name
            )

        # 1) Попытка получить статистику индекса (describe_index_stats / describe_index)
        stats = {}
        try:
            def _get_stats():
                # несколько вариантов вызова для совместимости с разными SDK-версиями
                try:
                    return self.pc.describe_index_stats(index_name=index_name)
                except TypeError:
                    # некоторые версии ожидают positional arg
                    try:
                        return self.pc.describe_index_stats(index_name)
                    except Exception:
                        return {}
                except Exception:
                    # fallback: попытка вызвать на уровне index client
                    try:
                        index = self.pc.Index(index_name)
                        return index.describe_index_stats()
                    except Exception:
                        return {}
            
            stats = await asyncio.to_thread(_get_stats)
        except Exception:
            stats = {}

        out["raw_stats"] = stats

        # Извлечём количество векторов для нашего namespace
        try:
            # в разных версиях структура stats разная
            if isinstance(stats, dict):
                # часто stats["namespaces"] -> { namespace: {"vector_count": N, ...}, ...}
                ns_map = stats.get("namespaces") or stats.get("namespaces", {})
                if isinstance(ns_map, dict) and namespace in ns_map:
                    ns_info = ns_map[namespace]
                    out["vector_count"] = int(
                        ns_info.get("vector_count") or ns_info.get("count") or 0
                    )
                else:
                    # некоторые SDK возвращают {"namespace": {...}} for a single ns, or root-level counts
                    # try direct keys
                    if namespace in stats:
                        try:
                            out["vector_count"] = int(
                                stats[namespace].get("vector_count", 0)
                            )
                        except Exception:
                            out["vector_count"] = 0
                    else:
                        # fallback: try to read total vector count for index and hope namespace==index
                        total = (
                            stats.get("total_vector_count")
                            or stats.get("total_count")
                            or stats.get("vector_count")
                        )
                        if total:
                            out["vector_count"] = int(total)
        except Exception:
            out["vector_count"] = out.get("vector_count", 0)

        # 2) Попробуем получить пример метаданных одного элемента из namespace.
        # Для этого нам нужен векторный размер (dimension), чтобы сделать dummy-query (нулевой вектор).
        try:
            info = await self._describe_index(index_name) or {}
            dim = info.get("dimension") or info.get("index_config", {}).get("dimension")
            if dim:
                dim = int(dim)
                # construct zero vector (cosine/dot metrics tolerate zeros for sampling; if index empty, query вернёт пусто)
                zero_vec = [0.0] * dim
                # безопасно запрашиваем top_k=1 в конкретном namespace
                try:
                    # Используем новый async API
                    async with self.get_index(index_name) as index:
                        resp = await asyncio.to_thread(
                            index.query,
                            vector=zero_vec,
                            top_k=1,
                            namespace=namespace,
                            include_metadata=True,
                        )
                    # нормализуем ответ в dict-like
                    matches = None
                    if isinstance(resp, dict):
                        matches = resp.get("matches") or resp.get("results") or []
                    else:
                        # объект-ответ: попробуем атрибуты
                        matches = (
                            getattr(resp, "matches", None)
                            or getattr(resp, "results", None)
                            or []
                        )
                    if matches:
                        first = matches[0]
                        # metadata может быть в first["metadata"] или first.metadata
                        meta = None
                        if isinstance(first, dict):
                            meta = first.get("metadata") or first.get("meta") or {}
                        else:
                            meta = (
                                getattr(first, "metadata", None)
                                or getattr(first, "meta", None)
                                or {}
                            )
                        out["sample_metadata"] = meta or {}
                except Exception:
                    # если query падает (например, index не инициализирован) — просто игнорируем
                    out["sample_metadata"] = None
            else:
                out["sample_metadata"] = None
        except Exception:
            out["sample_metadata"] = None
        # 3) Заполняем index_name в результате (может измениться, если клиент переключал имя)
        out["index_name"] = index_name

        out["raw_stats"] = self._to_primitive(out.get("raw_stats"))
        out["sample_metadata"] = self._to_primitive(out.get("sample_metadata"))

        return out

    def _to_primitive(self, obj: Any, _seen: set = None) -> Any:
        """
        Простая функция для сериализации объектов в примитивные типы.
        Упрощенная версия без зависимости от BookProcessor с защитой от циклических ссылок.
        """
        if _seen is None:
            _seen = set()
        
        # Проверяем циклические ссылки
        obj_id = id(obj)
        if obj_id in _seen:
            return f"<circular reference to {type(obj).__name__}>"
        
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Обработка проблемных типов, которые не сериализуются
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Logger':
            return f"<Logger: {getattr(obj, 'name', 'unknown')}>"
        
        # Обработка других проблемных типов
        if hasattr(obj, '__module__') and obj.__module__ in ['logging', 'threading', 'asyncio']:
            return f"<{type(obj).__name__}>"
        
        # Добавляем объект в множество посещенных
        _seen.add(obj_id)
        
        try:
            if isinstance(obj, dict):
                result = {k: self._to_primitive(v, _seen) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                result = [self._to_primitive(v, _seen) for v in obj]
            elif hasattr(obj, 'to_dict'):
                try:
                    result = self._to_primitive(obj.to_dict(), _seen)
                except Exception:
                    result = str(obj)
            elif hasattr(obj, '__dict__'):
                try:
                    # Безопасный доступ к __dict__ с фильтрацией проблемных атрибутов
                    obj_dict = {}
                    for key, value in obj.__dict__.items():
                        # Пропускаем приватные атрибуты и атрибуты с подчеркиванием
                        if key.startswith('_'):
                            continue
                        # Пропускаем типы, которые точно не сериализуются  
                        if isinstance(value, (type, type(lambda: None))) or (hasattr(value, '__class__') and value.__class__.__name__ == 'Logger'):
                            continue
                        obj_dict[key] = value
                    result = self._to_primitive(obj_dict, _seen)
                except Exception:
                    result = str(obj)
            else:
                result = str(obj)
        except Exception:
            result = str(obj)
        finally:
            # Удаляем объект из множества посещенных при выходе
            _seen.discard(obj_id)
        
        return result
