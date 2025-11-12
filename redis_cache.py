import os
import redis
import json
import hashlib
from typing import Optional, Any, Dict, List
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

class RedisCache:
    """Redis cache utility for caching API responses and data."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.host = os.getenv("REDIS_HOST")
        self.port = int(os.getenv("REDIS_PORT", 6380))
        self.password = os.getenv("REDIS_PASSWORD")
        self.ssl = os.getenv("REDIS_SSL", "true").lower() == "true"
        
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                ssl=self.ssl,
                ssl_cert_reqs=None,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            print(f"✅ REDIS: Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ REDIS: Could not connect to Redis: {e}")
            self.client = None
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generate a cache key from prefix and arguments."""
        key_string = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            return None
        try:
            value = self.client.get(key)
            if value:
                print(f"✅ REDIS: Cache hit for key: {key[:20]}...")
                return json.loads(value)
            print(f"❌ REDIS: Cache miss for key: {key[:20]}...")
            return None
        except Exception as e:
            print(f"❌ REDIS: Error getting from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (default 1 hour)."""
        if not self.client:
            return False
        try:
            serialized = json.dumps(value, default=str)
            result = self.client.setex(key, ttl, serialized)
            if result:
                print(f"✅ REDIS: Cached value for key: {key[:20]}... (TTL: {ttl}s)")
            return result
        except Exception as e:
            print(f"❌ REDIS: Error setting cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.client:
            return False
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Error deleting from Redis cache: {e}")
            return False
    
    def cache_sentiment(self, text: str, sentiment: Dict[str, float], ttl: int = 86400) -> bool:
        """Cache sentiment analysis result (default 24 hours)."""
        key = self._generate_key("sentiment", text)
        return self.set(key, sentiment, ttl)
    
    def get_cached_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """Get cached sentiment analysis result."""
        key = self._generate_key("sentiment", text)
        return self.get(key)
    
    def cache_stock_data(self, symbol: str, data: Dict, ttl: int = 300) -> bool:
        """Cache stock price data (default 5 minutes)."""
        key = self._generate_key("stock", symbol)
        return self.set(key, data, ttl)
    
    def get_cached_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get cached stock price data."""
        key = self._generate_key("stock", symbol)
        return self.get(key)
    
    def cache_news(self, symbol: str, news: List[Dict], ttl: int = 1800) -> bool:
        """Cache news articles (default 30 minutes)."""
        key = self._generate_key("news", symbol)
        return self.set(key, news, ttl)
    
    def get_cached_news(self, symbol: str) -> Optional[List[Dict]]:
        """Get cached news articles."""
        key = self._generate_key("news", symbol)
        return self.get(key)
    
    def store_article_embedding(self, article_id: str, embedding: List[float], metadata: Dict) -> bool:
        """Store article embedding for RAG."""
        if not self.client:
            return False
        try:
            # Store embedding as JSON
            embedding_key = f"embedding:{article_id}"
            metadata_key = f"article:{article_id}"
            
            self.client.setex(embedding_key, 86400 * 7, json.dumps(embedding))  # 7 days
            self.client.setex(metadata_key, 86400 * 7, json.dumps(metadata, default=str))  # 7 days
            return True
        except Exception as e:
            print(f"Error storing embedding: {e}")
            return False
    
    def get_article_embedding(self, article_id: str) -> Optional[tuple]:
        """Get article embedding and metadata."""
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
            print(f"Error getting embedding: {e}")
            return None
    
    def search_similar_articles(self, symbol: str, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """Search for similar articles using cosine similarity (simplified)."""
        if not self.client:
            return []
        
        try:
            # Get all article IDs for this symbol
            pattern = f"article:{symbol}:*"
            keys = self.client.keys(pattern)
            
            similarities = []
            for key in keys:
                article_id = key.split(":")[-1]
                embedding_data = self.client.get(f"embedding:{article_id}")
                if embedding_data:
                    embedding = json.loads(embedding_data)
                    # Simple cosine similarity
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    similarities.append((similarity, article_id))
            
            # Sort by similarity and return top_k
            similarities.sort(reverse=True, key=lambda x: x[0])
            results = []
            
            for similarity, article_id in similarities[:top_k]:
                metadata_data = self.client.get(f"article:{article_id}")
                if metadata_data:
                    metadata = json.loads(metadata_data)
                    metadata['similarity'] = similarity
                    results.append(metadata)
            
            return results
        except Exception as e:
            print(f"Error searching similar articles: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)
        except:
            return 0.0

