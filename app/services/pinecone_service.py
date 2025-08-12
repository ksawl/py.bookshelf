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

from app.core import config as settings


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

    def create_index(self, book_id: str):
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
        return self.pc.delete_index(name)

    def list_indexes(self):
        return [i["name"] for i in self.pc.list_indexes()]
