"""
Vector database integration interface for optimized vector search.

This module provides an abstraction layer for vector database operations,
supporting Azure AI Search for 10-100x faster vector search at scale.

The VectorDatabase abstract base class allows for extensibility if other
vector databases need to be integrated in the future.
"""

from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime

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
        filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional OData filter string (for Azure AI Search) or Dict (for others)
            
        Returns:
            List of results with similarity scores and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the vector database is available and ready to use."""
        pass
    
    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        pass


class AzureAISearchVectorDB(VectorDatabase):
    """
    Azure AI Search vector database implementation.
    
    This provides native vector search with indexing, filtering, and hybrid search
    capabilities. 10-100x faster than Redis SCAN-based search for large datasets.
    
    Features:
    - Native vector indexing (HNSW algorithm)
    - OData filter support (date ranges, sources, etc.)
    - Hybrid search (vector + keyword)
    - Built-in relevance scoring
    """
    
    def __init__(
        self,
        settings: Settings,
        redis_cache: Optional[Any] = None
    ):
        """
        Initialize Azure AI Search vector database.
        
        Args:
            settings: Application settings instance
            redis_cache: Optional Redis cache (for query embedding caching)
        """
        self.settings = settings
        self.redis_cache = redis_cache
        self._client = None
        self._index_client = None
        self._index_created = False
        
        if not settings.is_azure_ai_search_available():
            logger.warning("Azure AI Search not configured - vector DB disabled")
            return
        
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient
            
            search_config = settings.azure_ai_search
            credential = AzureKeyCredential(search_config.api_key)
            
            self._client = SearchClient(
                endpoint=search_config.endpoint,
                index_name=search_config.index_name,
                credential=credential
            )
            
            self._index_client = SearchIndexClient(
                endpoint=search_config.endpoint,
                credential=credential
            )
            
            # Ensure index exists
            self._ensure_index_exists()
            
        except ImportError:
            logger.error(
                "azure-search-documents package not installed. "
                "Install it with: pip install azure-search-documents>=11.4.0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure AI Search: {e}")
    
    def _ensure_index_exists(self) -> bool:
        """Create index if it doesn't exist."""
        if not self._index_client:
            return False
        
        try:
            from azure.search.documents.indexes.models import (
                SearchIndex,
                SimpleField,
                SearchableField,
                VectorSearch,
                HnswAlgorithmConfiguration,
                VectorSearchAlgorithmKind,
                VectorSearchAlgorithmMetric,
                VectorSearchProfile,
                VectorField,
                SearchFieldDataType
            )
            
            search_config = self.settings.azure_ai_search
            index_name = search_config.index_name
            
            # Check if index exists
            try:
                existing_index = self._index_client.get_index(index_name)
                logger.info(f"Azure AI Search index '{index_name}' already exists")
                self._index_created = True
                return True
            except Exception:
                # Index doesn't exist, create it
                pass
            
            # Define index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String, retrievable=True),
                VectorField(
                    name="contentVector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    dimensions=search_config.vector_dimension,
                    vector_search_profile="default-vector-profile"
                ),
                SimpleField(name="symbol", type=SearchFieldDataType.String, filterable=True, facetable=True, retrievable=True),
                SearchableField(name="title", type=SearchFieldDataType.String, retrievable=True),
                SearchableField(name="summary", type=SearchFieldDataType.String, retrievable=True),
                SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True, retrievable=True),
                SimpleField(name="url", type=SearchFieldDataType.String, retrievable=True),
                SimpleField(name="timestamp", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True, retrievable=True),
                SimpleField(name="article_id", type=SearchFieldDataType.String, retrievable=True)
            ]
            
            # Vector search configuration
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-algorithm",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": VectorSearchAlgorithmMetric.COSINE
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm="default-algorithm"
                    )
                ]
            )
            
            # Create index
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            self._index_client.create_index(index)
            logger.info(f"Created Azure AI Search index '{index_name}'")
            self._index_created = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Azure AI Search index: {e}")
            self._index_created = False
            return False
    
    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """Store a vector with metadata in Azure AI Search."""
        if not self._client or not self._index_created:
            return False
        
        try:
            # Prepare document
            document = {
                "id": vector_id,
                "content": f"{metadata.get('title', '')} {metadata.get('summary', '')}",
                "contentVector": vector,
                "symbol": metadata.get("symbol", ""),
                "title": metadata.get("title", ""),
                "summary": metadata.get("summary", ""),
                "source": metadata.get("source", ""),
                "url": metadata.get("url", ""),
                "article_id": metadata.get("article_id", vector_id)
            }
            
            # Parse timestamp if provided
            timestamp = metadata.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        from dateutil import parser
                        timestamp = parser.parse(timestamp)
                    except Exception:
                        timestamp = None
                if timestamp:
                    document["timestamp"] = timestamp
            
            # Upload document
            self._client.upload_documents(documents=[document])
            return True
            
        except Exception as e:
            logger.error(f"Error storing vector in Azure AI Search: {e}")
            return False
    
    def batch_store_vectors(
        self,
        vectors: List[Dict[str, Any]]
    ) -> int:
        """
        Store multiple vectors in batch.
        
        Args:
            vectors: List of dicts with keys: vector_id, vector, metadata
            
        Returns:
            Number of successfully stored vectors
        """
        if not self._client or not self._index_created:
            return 0
        
        if not vectors:
            return 0
        
        try:
            documents = []
            for item in vectors:
                vector_id = item.get("vector_id", "")
                vector = item.get("vector", [])
                metadata = item.get("metadata", {})
                
                document = {
                    "id": vector_id,
                    "content": f"{metadata.get('title', '')} {metadata.get('summary', '')}",
                    "contentVector": vector,
                    "symbol": metadata.get("symbol", ""),
                    "title": metadata.get("title", ""),
                    "summary": metadata.get("summary", ""),
                    "source": metadata.get("source", ""),
                    "url": metadata.get("url", ""),
                    "article_id": metadata.get("article_id", vector_id)
                }
                
                timestamp = metadata.get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        try:
                            from dateutil import parser
                            timestamp = parser.parse(timestamp)
                        except Exception:
                            timestamp = None
                    if timestamp:
                        document["timestamp"] = timestamp
                
                documents.append(document)
            
            # Upload in batch
            result = self._client.upload_documents(documents=documents)
            
            # Count successful uploads
            success_count = sum(1 for r in result if r.succeeded)
            return success_count
            
        except Exception as e:
            logger.error(f"Error batch storing vectors in Azure AI Search: {e}")
            return 0
    
    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using Azure AI Search.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional OData filter string (e.g., "symbol eq 'AAPL' and timestamp ge 2024-12-10T00:00:00Z")
            
        Returns:
            List of results with similarity scores and metadata
        """
        if not self._client or not self._index_created:
            return []
        
        try:
            from azure.search.documents.models import VectorizedQuery
            
            # Create vectorized query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="contentVector"
            )
            
            # Build search options
            search_options = {
                "vector_queries": [vector_query],
                "top": top_k,
                "select": ["id", "symbol", "title", "summary", "source", "url", "timestamp", "article_id"]
            }
            
            if filter:
                search_options["filter"] = filter
            
            # Perform search
            results = self._client.search(
                search_text="",  # Empty for pure vector search, or use for hybrid search
                **search_options
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "article_id": result.get("article_id", result.get("id", "")),
                    "symbol": result.get("symbol", ""),
                    "title": result.get("title", ""),
                    "summary": result.get("summary", ""),
                    "source": result.get("source", ""),
                    "url": result.get("url", ""),
                    "timestamp": result.get("timestamp", ""),
                    "similarity": result.get("@search.score", 0.0)  # Azure AI Search relevance score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vectors in Azure AI Search: {e}")
            return []
    
    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (vector + keyword) in Azure AI Search.
        
        Args:
            query_text: Keyword search query
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional OData filter string
            
        Returns:
            List of results with similarity scores and metadata
        """
        if not self._client or not self._index_created:
            return []
        
        try:
            from azure.search.documents.models import VectorizedQuery
            
            # Create vectorized query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k * 2,  # Get more candidates for RRF
                fields="contentVector"
            )
            
            # Build search options
            search_options = {
                "vector_queries": [vector_query],
                "search_text": query_text,  # Keyword search
                "top": top_k,
                "select": ["id", "symbol", "title", "summary", "source", "url", "timestamp", "article_id"]
            }
            
            if filter:
                search_options["filter"] = filter
            
            # Perform hybrid search (Azure AI Search handles RRF internally)
            results = self._client.search(**search_options)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "article_id": result.get("article_id", result.get("id", "")),
                    "symbol": result.get("symbol", ""),
                    "title": result.get("title", ""),
                    "summary": result.get("summary", ""),
                    "source": result.get("source", ""),
                    "url": result.get("url", ""),
                    "timestamp": result.get("timestamp", ""),
                    "similarity": result.get("@search.score", 0.0),
                    "rrf_score": result.get("@search.reranker_score", result.get("@search.score", 0.0))
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search in Azure AI Search: {e}")
            return []
    
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID from Azure AI Search."""
        if not self._client or not self._index_created:
            return False
        
        try:
            self._client.delete_documents(documents=[{"id": vector_id}])
            return True
        except Exception as e:
            logger.error(f"Error deleting vector from Azure AI Search: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Azure AI Search is available and ready to use."""
        return (
            self._client is not None
            and self._index_client is not None
            and self._index_created
        )

