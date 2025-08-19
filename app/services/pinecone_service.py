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
        Parse PINECONE_ENV or use separate environment variables.
        Expected PINECONE_ENV formats: "us-west1-gcp", "us-east1-aws" etc.
        """
        env = self._settings.PINECONE_ENV or ""
        cloud = None
        region = None

        # Parse env like "us-west1-gcp" -> region=us-west1, cloud=gcp
        if "-" in env:
            parts = env.split("-")
            # Last part is cloud (gcp/aws), rest joined is region
            if parts[-1] in {"gcp", "aws", "azure"}:
                cloud = parts[-1]
                region = "-".join(parts[:-1]) if len(parts) > 1 else None

        # Fallback to explicit variables
        cloud = cloud or self._settings.PINECONE_SERVERLESS_CLOUD
        region = region or self._settings.PINECONE_SERVERLESS_REGION

        if not cloud or not region:
            raise RuntimeError(
                "Cannot determine serverless spec for Pinecone. "
                "Set PINECONE_ENV (like 'us-west1-gcp') or "
                "set PINECONE_SERVERLESS_CLOUD and PINECONE_SERVERLESS_REGION."
            )

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
        Execute query in specified Pinecone index and return query results.

        Args:
            book_id: Book ID for forming index name and namespace
            vector: Query vector (List[float])
            top_k: Number of matches to return
            filter: Metadata filter (Pinecone filter dict) - optional
            include_metadata: Return metadata
            include_values: Return vector values

        Returns:
            dict: Original response from Pinecone Index.query
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
            # Create the requested index with ServerlessSpec
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
        Collect metadata for uploaded book (namespace = f"book_{book_id}").

        Returns MetaBook with:
        - book_id, namespace, index_name
        - vector_count: number of vectors in namespace
        - sample_metadata: example metadata from one vector
        - raw_stats: raw Pinecone index stats (if available)
        - error: error message if any
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

        # Get index info and stats
        info = await self._describe_index(index_name)
        host = info.get("host") or info.get("server", {}).get("host") or index_name
        if not host:
            raise RuntimeError(
                "Unable to determine index host for chosen index: %s" % index_name
            )

        # Get index statistics with multiple compatibility approaches
        stats = {}
        try:
            def _get_stats():
                # Try different call variants for SDK version compatibility
                try:
                    return self.pc.describe_index_stats(index_name=index_name)
                except TypeError:
                    # Some versions expect positional arg
                    try:
                        return self.pc.describe_index_stats(index_name)
                    except Exception:
                        return {}
                except Exception:
                    # Fallback: try calling at index client level
                    try:
                        index = self.pc.Index(index_name)
                        return index.describe_index_stats()
                    except Exception:
                        return {}
            
            stats = await asyncio.to_thread(_get_stats)
        except Exception:
            stats = {}

        out["raw_stats"] = stats

        # Extract vector count for our namespace
        try:
            # Stats structure varies between SDK versions
            if isinstance(stats, dict):
                # Common format: stats["namespaces"] -> { namespace: {"vector_count": N, ...}, ...}
                ns_map = stats.get("namespaces") or stats.get("namespaces", {})
                if isinstance(ns_map, dict) and namespace in ns_map:
                    ns_info = ns_map[namespace]
                    out["vector_count"] = int(
                        ns_info.get("vector_count") or ns_info.get("count") or 0
                    )
                else:
                    # Some SDKs return {"namespace": {...}} for single ns, or root-level counts
                    if namespace in stats:
                        try:
                            out["vector_count"] = int(
                                stats[namespace].get("vector_count", 0)
                            )
                        except Exception:
                            out["vector_count"] = 0
                    else:
                        # Fallback: try to read total vector count
                        total = (
                            stats.get("total_vector_count")
                            or stats.get("total_count")
                            or stats.get("vector_count")
                        )
                        if total:
                            out["vector_count"] = int(total)
        except Exception:
            out["vector_count"] = out.get("vector_count", 0)

        # Try to get sample metadata from one element in namespace
        # Need vector dimension to make dummy query (zero vector)
        try:
            info = await self._describe_index(index_name) or {}
            dim = info.get("dimension") or info.get("index_config", {}).get("dimension")
            if dim:
                dim = int(dim)
                # Construct zero vector (cosine/dot metrics tolerate zeros for sampling)
                zero_vec = [0.0] * dim
                # Safely query top_k=1 in specific namespace
                try:
                    async with self.get_index(index_name) as index:
                        resp = await asyncio.to_thread(
                            index.query,
                            vector=zero_vec,
                            top_k=1,
                            namespace=namespace,
                            include_metadata=True,
                        )
                    # Normalize response to dict-like
                    matches = None
                    if isinstance(resp, dict):
                        matches = resp.get("matches") or resp.get("results") or []
                    else:
                        # Object response: try attributes
                        matches = (
                            getattr(resp, "matches", None)
                            or getattr(resp, "results", None)
                            or []
                        )
                    if matches:
                        first = matches[0]
                        # Metadata can be in first["metadata"] or first.metadata
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
                    # If query fails (e.g., index not initialized) - just ignore
                    out["sample_metadata"] = None
            else:
                out["sample_metadata"] = None
        except Exception:
            out["sample_metadata"] = None
        # Fill index_name in result (may have changed if client switched name)
        out["index_name"] = index_name

        out["raw_stats"] = self._to_primitive(out.get("raw_stats"))
        out["sample_metadata"] = self._to_primitive(out.get("sample_metadata"))

        return out

    def _to_primitive(self, obj: Any, _seen: set = None) -> Any:
        """
        Simple function for serializing objects to primitive types.
        Simplified version without BookProcessor dependency with circular reference protection.
        """
        if _seen is None:
            _seen = set()
        
        # Check for circular references
        obj_id = id(obj)
        if obj_id in _seen:
            return f"<circular reference to {type(obj).__name__}>"
        
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle problematic types that don't serialize
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Logger':
            return f"<Logger: {getattr(obj, 'name', 'unknown')}>"
        
        # Handle other problematic types
        if hasattr(obj, '__module__') and obj.__module__ in ['logging', 'threading', 'asyncio']:
            return f"<{type(obj).__name__}>"
        
        # Add object to visited set
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
                    # Safe access to __dict__ with problematic attribute filtering
                    obj_dict = {}
                    for key, value in obj.__dict__.items():
                        # Skip private attributes and underscore attributes
                        if key.startswith('_'):
                            continue
                        # Skip types that definitely don't serialize
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
            # Remove object from visited set on exit
            _seen.discard(obj_id)
        
        return result
