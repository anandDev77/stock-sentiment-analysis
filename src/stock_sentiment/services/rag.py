"""
Retrieval Augmented Generation (RAG) service for enhanced sentiment analysis.

This module provides RAG functionality using Azure OpenAI embeddings to retrieve
relevant context from stored articles for improved sentiment analysis accuracy.
"""

import json
import hashlib
from typing import List, Dict, Optional
from openai import AzureOpenAI
import numpy as np

from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger
from .cache import RedisCache

logger = get_logger(__name__)


class RAGService:
    """
    Retrieval Augmented Generation service for enhanced sentiment analysis.
    
    This service uses Azure OpenAI embeddings to:
    - Store article embeddings in Redis
    - Retrieve relevant articles using cosine similarity
    - Provide context for sentiment analysis
    
    Attributes:
        client: Azure OpenAI client for embeddings
        cache: Redis cache instance
        settings: Application settings
        embeddings_enabled: Whether embeddings are working
        
    Example:
        >>> settings = get_settings()
        >>> cache = RedisCache(settings=settings)
        >>> rag = RAGService(settings=settings, redis_cache=cache)
        >>> rag.store_article(article_dict, "AAPL")
        >>> context = rag.retrieve_relevant_context("Apple earnings", "AAPL")
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_cache: Optional[RedisCache] = None
    ):
        """
        Initialize RAG service with Azure OpenAI and Redis cache.
        
        Args:
            settings: Application settings (uses global if not provided)
            redis_cache: Redis cache instance for storing embeddings
            
        Raises:
            ValueError: If Azure OpenAI configuration is invalid
        """
        self.settings = settings or get_settings()
        self.cache = redis_cache
        self.embeddings_enabled = False
        
        # Initialize Azure OpenAI for embeddings
        azure_config = self.settings.azure_openai
        self.client = AzureOpenAI(
            azure_endpoint=azure_config.endpoint,
            api_key=azure_config.api_key,
            api_version=azure_config.api_version
        )
        
        # Get embedding deployment name
        self.embedding_deployment = azure_config.embedding_deployment
        
        # Test if embeddings work
        self._test_embeddings()
    
    def _test_embeddings(self) -> None:
        """
        Test if embedding model is available and working.
        
        Sets embeddings_enabled flag based on test result.
        """
        if not self.embedding_deployment:
            logger.warning("No embedding deployment configured - RAG disabled")
            self.embeddings_enabled = False
            return
        
        try:
            test_embedding = self.get_embedding("test")
            if test_embedding:
                self.embeddings_enabled = True
                logger.info("RAG embeddings enabled and working")
            else:
                logger.warning("RAG embeddings disabled - embedding model not available")
                self.embeddings_enabled = False
        except Exception as e:
            logger.warning(f"RAG embeddings disabled - {e}")
            self.embeddings_enabled = False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding vector for text using Azure OpenAI.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding")
            return None
        
        if not self.embedding_deployment:
            logger.warning("No embedding deployment configured")
            return None
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_deployment,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding using {self.embedding_deployment}")
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def store_article(self, article: Dict, symbol: str) -> bool:
        """
        Store article with embedding for RAG retrieval.
        
        Args:
            article: Article dictionary with title, summary, url, etc.
            symbol: Stock ticker symbol
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.embeddings_enabled:
            logger.debug("Skipping article storage - embeddings not enabled")
            return False
        
        if not self.cache or not self.cache.client:
            logger.warning("Redis cache not available for article storage")
            return False
        
        try:
            # Create article ID
            article_id = hashlib.md5(
                f"{symbol}:{article.get('title', '')}:{article.get('url', '')}".encode()
            ).hexdigest()
            
            # Get embedding for article text
            article_text = f"{article.get('title', '')} {article.get('summary', '')}"
            if not article_text.strip():
                logger.warning("Empty article text, skipping storage")
                return False
            
            embedding = self.get_embedding(article_text)
            
            if embedding:
                # Prepare metadata
                metadata = {
                    'article_id': article_id,
                    'symbol': symbol,
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'timestamp': str(article.get('timestamp', ''))
                }
                
                # Store with symbol prefix for easier retrieval
                embedding_key = f"embedding:{symbol}:{article_id}"
                metadata_key = f"article:{symbol}:{article_id}"
                
                # Store in Redis with 7 days TTL
                ttl = 86400 * 7
                self.cache.client.setex(
                    embedding_key,
                    ttl,
                    json.dumps(embedding)
                )
                self.cache.client.setex(
                    metadata_key,
                    ttl,
                    json.dumps(metadata, default=str)
                )
                
                logger.debug(f"Stored article {article_id[:8]}... for {symbol}")
                return True
            else:
                logger.warning("Could not get embedding for article, skipping storage")
                return False
        except Exception as e:
            logger.error(f"Error storing article: {e}")
            return False
    
    def retrieve_relevant_context(
        self,
        query: str,
        symbol: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant articles for RAG context using cosine similarity.
        
        Args:
            query: Query text to find similar articles for
            symbol: Stock ticker symbol
            top_k: Number of top articles to return (default: from settings)
            
        Returns:
            List of article metadata dictionaries with similarity scores
        """
        if not self.embeddings_enabled:
            logger.debug("Skipping context retrieval - embeddings not enabled")
            return []
        
        if not self.cache or not self.cache.client:
            logger.warning("Redis cache not available for context retrieval")
            return []
        
        if top_k is None:
            top_k = self.settings.app.rag_top_k
        
        try:
            # Get embedding for query
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                logger.warning("Could not get query embedding")
                return []
            
            # Get all article embeddings for this symbol
            pattern = f"embedding:{symbol}:*"
            keys = self.cache.client.keys(pattern)
            
            if not keys:
                logger.debug(f"No stored articles found for {symbol}")
                return []
            
            logger.debug(f"Searching {len(keys)} articles for {symbol}")
            
            # Calculate similarities
            similarities = []
            for key in keys:
                article_id = key.split(":")[-1]
                embedding_data = self.cache.client.get(key)
                if embedding_data:
                    try:
                        embedding = json.loads(embedding_data)
                        similarity = self._cosine_similarity(query_embedding, embedding)
                        similarities.append((similarity, article_id))
                    except Exception as e:
                        logger.warning(f"Error processing embedding: {e}")
                        continue
            
            # Sort by similarity and get top_k
            similarities.sort(reverse=True, key=lambda x: x[0])
            results = []
            
            for similarity, article_id in similarities[:top_k]:
                metadata_key = f"article:{symbol}:{article_id}"
                metadata_data = self.cache.client.get(metadata_key)
                if metadata_data:
                    try:
                        metadata = json.loads(metadata_data)
                        metadata['similarity'] = similarity
                        results.append(metadata)
                    except Exception as e:
                        logger.warning(f"Error loading metadata: {e}")
                        continue
            
            if results:
                top_similarity = results[0].get('similarity', 0)
                logger.info(
                    f"Retrieved {len(results)} relevant articles "
                    f"(top similarity: {top_similarity:.3f})"
                )
            else:
                logger.debug("No similar articles found")
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Use numpy if available (faster)
            vec1_array = np.array(vec1)
            vec2_array = np.array(vec2)
            dot_product = np.dot(vec1_array, vec2_array)
            norm1 = np.linalg.norm(vec1_array)
            norm2 = np.linalg.norm(vec2_array)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            # Fallback without numpy
            try:
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = sum(a * a for a in vec1) ** 0.5
                norm2 = sum(b * b for b in vec2) ** 0.5
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return dot_product / (norm1 * norm2)
            except Exception:
                logger.warning("Error calculating cosine similarity")
                return 0.0

