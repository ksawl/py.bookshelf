"""
Thin wrapper around modern pinecone client to centralize index management and upserts.
This implementation uses the new `Pinecone` class (pinecone>=1.x) and ensures that
we never try to upsert dense vectors into a sparse index.

Public methods:
- ensure_index_for_dimension(dimension) -> returns chosen_index_name
- upsert(vectors, namespace=None)
- query(...)
"""

from typing import List, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec

from app.utils.book_processor import to_primitive
from app.core import config as settings
from app.schemas.book import MetaBook


class PineconeService:
    def __init__(self, default_dim: int = 384):
        self.index_name = settings.PINECONE_INDEX
        api_key = settings.PINECONE_API_KEY
        env = settings.PINECONE_ENV

        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set")

        # create client instance
        self.pc = Pinecone(api_key=api_key, environment=env)
        self.index = None
        self._host = None
        # default dim is a fallback; prefer using the actual embeddings dimension
        self.default_dim = default_dim

    def create_index_name(self, book_id: str):
        return f"book-{book_id}"

    def _describe_index(self, name: str) -> dict:
        try:
            return self.pc.describe_index(name)
        except Exception:
            return {}

    def _spec_from_env(self) -> ServerlessSpec:
        """
        Попробовать распарсить PINECONE_ENV, либо взять отдельные переменные.
        Ожидаемые форматы PINECONE_ENV: "us-west1-gcp", "us-east1-aws" и т.п.
        """
        env = settings.PINECONE_ENV or ""
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
        cloud = cloud or settings.PINECONE_SERVERLESS_CLOUD
        region = region or settings.PINECONE_SERVERLESS_REGION

        if not cloud or not region:
            raise RuntimeError(
                "Cannot determine serverless spec for Pinecone. "
                "Set PINECONE_ENV (like 'us-west1-gcp') or "
                "set PINECONE_SERVERLESS_CLOUD and PINECONE_SERVERLESS_REGION."
            )

        # ServerlessSpec expects cloud like 'aws' or 'gcp' and region like 'us-west1'
        return ServerlessSpec(cloud=cloud, region=region)

    def ensure_index_for_dimension(
        self, desired_dim: int, target: int = settings.PINECONE_INDEX
    ) -> str:
        existing = [i["name"] for i in self.pc.list_indexes()]

        def _create_index(name, dim):
            # build ServerlessSpec from env (or raise)
            spec = self._spec_from_env()
            # create index with spec
            self.pc.create_index(name=name, dimension=dim, metric="cosine", spec=spec)

        if target in existing:
            info = self._describe_index(target)
            dim = info.get("dimension") or info.get("index_config", {}).get("dimension")
            if not dim:
                new_name = f"{self.index_name}-dense-{desired_dim}"
                if new_name not in existing:
                    _create_index(new_name, desired_dim)
                target = new_name
            elif int(dim) != int(desired_dim):
                new_name = f"{self.index_name}-dense-{desired_dim}"
                if new_name not in existing:
                    _create_index(new_name, desired_dim)
                target = new_name
        else:
            # create the requested index with ServerlessSpec
            _create_index(target, desired_dim)

        # describe chosen index and init Index client as before
        info = self._describe_index(target)
        host = info.get("host") or info.get("server", {}).get("host") or target
        if not host:
            raise RuntimeError(
                "Unable to determine index host for chosen index: %s" % target
            )
        self._host = host
        self.index = self.pc.Index(host=host)
        self.index_name = target
        return target

    def upsert(
        self,
        vectors: List[Tuple[str, List[float], dict]],
        namespace: Optional[str] = None,
    ):
        if not self.index:
            raise RuntimeError(
                "Index client not initialized. Call ensure_index_for_dimension() first."
            )
        if namespace:
            return self.index.upsert(vectors=vectors, namespace=namespace)
        return self.index.upsert(vectors=vectors)

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
    ):
        if not self.index:
            raise RuntimeError(
                "Index client not initialized. Call ensure_index_for_dimension() first."
            )
        return self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=include_metadata,
        )

    def delete_index(self, name: str):
        try:
            self.pc.delete_index(name)
        except Exception as e:
            # логируем ошибку
            print("Index delete error:", e)

    def list_indexes(self):
        return [i["name"] for i in self.pc.list_indexes()]

    def get_book_metadata(self, book_id: str) -> MetaBook:
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

        info = self._describe_index(index_name)
        host = info.get("host") or info.get("server", {}).get("host") or index_name
        if not host:
            raise RuntimeError(
                "Unable to determine index host for chosen index: %s" % index_name
            )
        self._host = host
        self.index = self.pc.Index(host=host)
        self.index_name = index_name

        # 1) Попытка получить статистику индекса (describe_index_stats / describe_index)
        stats = {}
        try:
            # несколько вариантов вызова для совместимости с разными SDK-версиями
            try:
                stats = self.pc.describe_index_stats(index_name=index_name)
            except TypeError:
                # некоторые версии ожидают positional arg
                try:
                    stats = self.pc.describe_index_stats(index_name)
                except Exception:
                    stats = {}
            except Exception:
                # fallback: попытка вызвать на уровне index client (если он инициализирован)
                if self.index:
                    try:
                        stats = self.index.describe_index_stats()
                    except Exception:
                        stats = {}
                else:
                    stats = {}
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
            info = self._describe_index(index_name) or {}
            dim = info.get("dimension") or info.get("index_config", {}).get("dimension")
            if dim:
                dim = int(dim)
                # construct zero vector (cosine/dot metrics tolerate zeros for sampling; if index empty, query вернёт пусто)
                zero_vec = [0.0] * dim
                # безопасно запрашиваем top_k=1 в конкретном namespace
                try:
                    # new SDK может возвращать dict или объект; мы обрабатываем оба варианта ниже
                    resp = self.index.query(
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

        out["raw_stats"] = to_primitive(out.get("raw_stats"))
        out["sample_metadata"] = to_primitive(out.get("sample_metadata"))

        return out
