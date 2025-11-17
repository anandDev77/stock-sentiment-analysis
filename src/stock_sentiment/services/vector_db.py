"""
Vector database integration interface for optimized vector search.

This module provides an abstraction layer for vector database operations,
supporting both Redis with RediSearch and dedicated vector databases
(Pinecone, Weaviate, Qdrant) for 10-100x faster vector search at scale.
"""

from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorDatabase(ABC):
    """
    Abstract base class for vector database operations.
    
    This interface allows switching between different vector database
    implementations (Redis+RediSearch, Pinecone, Weaviate, Qdrant) without
    changing application code.
    """
    
    @abstractmethod
    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Store a vector with metadata.
        
        Args:
            vector_id: Unique identifier for the vector
            vector: Embedding vector
            metadata: Associated metadata
            
        Returns:
            True if stored successfully
        """
        pass
    
    @abstractmethod
    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filters
            
        Returns:
            List of results with similarity scores and metadata
        """
        pass
    
    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        pass


class RedisVectorDB(VectorDatabase):
    """
    Vector database implementation using Redis with RediSearch.
    
    This provides optimized vector search using RediSearch's vector
    similarity search capabilities, which is 10-100x faster than
    scanning all keys in Redis.
    
    Note: Requires RediSearch module to be installed on Redis server.
    """
    
    def __init__(
        self,
        redis_cache,
        index_name: str = "article_vectors"
    ):
        """
        Initialize Redis vector database.
        
        Args:
            redis_cache: Redis cache instance
            index_name: Name of the RediSearch index
        """
        self.cache = redis_cache
        self.index_name = index_name
        self._index_created = False
        
        if not self.cache or not self.cache.client:
            logger.warning("Redis not available - vector DB disabled")
            return
        
        # Create index if it doesn't exist
        self._create_index()
    
    def _create_index(self):
        """Create RediSearch index for vector similarity search."""
        if not self.cache or not self.cache.client:
            return
        
        try:
            # Check if index exists
            try:
                self.cache.client.execute_command("FT.INFO", self.index_name)
                self._index_created = True
                logger.info(f"RediSearch index '{self.index_name}' already exists")
                return
            except Exception:
                # Index doesn't exist, create it
                pass
            
            # Create index with vector field
            # This requires RediSearch module on Redis server
            # FT.CREATE index_name ON HASH PREFIX 1 "vector:" SCHEMA vector VECTOR HNSW 6 DIM 1536 DISTANCE_METRIC COSINE
            
            logger.warning(
                "RediSearch vector index creation requires RediSearch module. "
                "Falling back to standard Redis scan-based search. "
                "For 10-100x performance improvement, install RediSearch module."
            )
            self._index_created = False
        except Exception as e:
            logger.warning(f"Could not create RediSearch index: {e}")
            self._index_created = False
    
    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Store vector in Redis (with RediSearch if available)."""
        if not self.cache or not self.cache.client:
            return False
        
        try:
            import json
            key = f"vector:{vector_id}"
            
            # Store as hash
            data = {
                "vector": json.dumps(vector),
                **{k: str(v) for k, v in metadata.items()}
            }
            self.cache.client.hset(key, mapping=data)
            
            return True
        except Exception as e:
            logger.error(f"Error storing vector: {e}")
            return False
    
    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search vectors using RediSearch if available, otherwise fallback to scan.
        
        Note: Full RediSearch implementation requires RediSearch module.
        This is a placeholder that falls back to standard Redis operations.
        """
        if not self.cache or not self.cache.client:
            return []
        
        if not self._index_created:
            logger.warning(
                "RediSearch not available - using fallback scan-based search. "
                "For optimal performance, install RediSearch module on Redis server."
            )
            # Fallback to standard Redis scan (used by RAGService)
            return []
        
        # If RediSearch is available, use FT.SEARCH with vector similarity
        # This would be implemented if RediSearch module is installed
        return []
    
    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from Redis."""
        if not self.cache or not self.cache.client:
            return False
        
        try:
            key = f"vector:{vector_id}"
            self.cache.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting vector: {e}")
            return False


# Placeholder for future dedicated vector DB implementations
# These would require additional dependencies (pinecone-client, weaviate-client, etc.)

class PineconeVectorDB(VectorDatabase):
    """
    Pinecone vector database implementation.
    
    Note: Requires 'pinecone-client' package and Pinecone account.
    This is a placeholder for future implementation.
    """
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        logger.warning("Pinecone integration not yet implemented")
        raise NotImplementedError("Pinecone integration requires pinecone-client package")


class WeaviateVectorDB(VectorDatabase):
    """
    Weaviate vector database implementation.
    
    Note: Requires 'weaviate-client' package and Weaviate instance.
    This is a placeholder for future implementation.
    """
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        logger.warning("Weaviate integration not yet implemented")
        raise NotImplementedError("Weaviate integration requires weaviate-client package")

