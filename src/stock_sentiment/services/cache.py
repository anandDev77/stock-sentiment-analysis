"""
Redis cache service for caching API responses and data.

This module provides a Redis-based caching layer to reduce API calls
and improve application performance. It handles caching of:
- Sentiment analysis results
- Stock price data
- News articles
- Article embeddings for RAG
"""

import json
import hashlib
from typing import Optional, Any, Dict, List
import redis
from redis import Redis

from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    """
    Redis cache utility for caching API responses and data.
    
    This class provides a high-level interface for Redis caching operations,
    with automatic serialization/deserialization and key generation.
    
    Attributes:
        client: Redis client instance (None if connection failed)
        settings: Application settings instance
        
    Example:
        >>> cache = RedisCache()
        >>> cache.set("key", {"data": "value"}, ttl=3600)
        >>> value = cache.get("key")
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Redis connection.
        
        Args:
            settings: Optional settings instance (uses global if not provided)
            
        Note:
            If Redis connection fails, the client will be None and caching
            will be disabled gracefully without raising exceptions.
        """
        self.settings = settings or get_settings()
        self.client: Optional[Redis] = None
        
        if not self.settings.is_redis_available():
            logger.warning("Redis configuration not available - caching disabled")
            return
        
        try:
            redis_config = self.settings.redis
            self.client = redis.Redis(
                host=redis_config.host,
                port=redis_config.port,
                password=redis_config.password,
                ssl=redis_config.ssl,
                ssl_cert_reqs=None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"Redis connected to {redis_config.host}:{redis_config.port}")
        except Exception as e:
            logger.error(f"Could not connect to Redis: {e}")
            self.client = None
    
    def _generate_key(self, prefix: str, *args) -> str:
        """
        Generate a cache key from prefix and arguments.
        
        Args:
            prefix: Key prefix (e.g., "sentiment", "stock")
            *args: Additional arguments to include in key
            
        Returns:
            MD5 hash of the key string
        """
        key_string = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/error
        """
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache hit for key: {key[:20]}...")
                return json.loads(value)
            logger.debug(f"Cache miss for key: {key[:20]}...")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding cached value for key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            serialized = json.dumps(value, default=str)
            result = self.client.setex(key, ttl, serialized)
            if result:
                logger.debug(f"Cached value for key: {key[:20]}... (TTL: {ttl}s)")
            return bool(result)
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing value for cache: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.client:
            return False
        
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def cache_sentiment(
        self, 
        text: str, 
        sentiment: Dict[str, float], 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache sentiment analysis result.
        
        Args:
            text: Original text that was analyzed
            sentiment: Sentiment scores dictionary
            ttl: Time to live in seconds (default: from settings)
            
        Returns:
            True if cached successfully
        """
        if ttl is None:
            ttl = self.settings.app.cache_ttl_sentiment
        
        key = self._generate_key("sentiment", text)
        return self.set(key, sentiment, ttl)
    
    def get_cached_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """
        Get cached sentiment analysis result.
        
        Args:
            text: Original text that was analyzed
            
        Returns:
            Cached sentiment scores or None
        """
        key = self._generate_key("sentiment", text)
        return self.get(key)
    
    def cache_stock_data(
        self, 
        symbol: str, 
        data: Dict, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache stock price data.
        
        Args:
            symbol: Stock ticker symbol
            data: Stock data dictionary
            ttl: Time to live in seconds (default: from settings)
            
        Returns:
            True if cached successfully
        """
        if ttl is None:
            ttl = self.settings.app.cache_ttl_stock
        
        key = self._generate_key("stock", symbol)
        return self.set(key, data, ttl)
    
    def get_cached_stock_data(self, symbol: str) -> Optional[Dict]:
        """
        Get cached stock price data.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Cached stock data or None
        """
        key = self._generate_key("stock", symbol)
        return self.get(key)
    
    def cache_news(
        self, 
        symbol: str, 
        news: List[Dict], 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache news articles.
        
        Args:
            symbol: Stock ticker symbol
            news: List of news article dictionaries
            ttl: Time to live in seconds (default: from settings)
            
        Returns:
            True if cached successfully
        """
        if ttl is None:
            ttl = self.settings.app.cache_ttl_news
        
        key = self._generate_key("news", symbol)
        return self.set(key, news, ttl)
    
    def get_cached_news(self, symbol: str) -> Optional[List[Dict]]:
        """
        Get cached news articles.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Cached news articles or None
        """
        key = self._generate_key("news", symbol)
        return self.get(key)
    
    def store_article_embedding(
        self, 
        article_id: str, 
        embedding: List[float], 
        metadata: Dict,
        ttl: int = 604800  # 7 days
    ) -> bool:
        """
        Store article embedding for RAG retrieval.
        
        Args:
            article_id: Unique article identifier
            embedding: Vector embedding
            metadata: Article metadata dictionary
            ttl: Time to live in seconds (default: 7 days)
            
        Returns:
            True if stored successfully
        """
        if not self.client:
            return False
        
        try:
            embedding_key = f"embedding:{article_id}"
            metadata_key = f"article:{article_id}"
            
            self.client.setex(embedding_key, ttl, json.dumps(embedding))
            self.client.setex(metadata_key, ttl, json.dumps(metadata, default=str))
            logger.debug(f"Stored embedding for article: {article_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False
    
    def get_article_embedding(self, article_id: str) -> Optional[tuple]:
        """
        Get article embedding and metadata.
        
        Args:
            article_id: Unique article identifier
            
        Returns:
            Tuple of (embedding, metadata) or None
        """
        if not self.client:
            return None
        
        try:
            embedding_key = f"embedding:{article_id}"
            metadata_key = f"article:{article_id}"
            
            embedding_data = self.client.get(embedding_key)
            metadata_data = self.client.get(metadata_key)
            
            if embedding_data and metadata_data:
                embedding = json.loads(embedding_data)
                metadata = json.loads(metadata_data)
                return embedding, metadata
            return None
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

