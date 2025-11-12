import os
import json
from openai import AzureOpenAI
from typing import List, Dict, Optional
from dotenv import load_dotenv
from redis_cache import RedisCache
import hashlib

load_dotenv()

class RAGService:
    """Retrieval Augmented Generation service for enhanced sentiment analysis."""
    
    def __init__(self, redis_cache: RedisCache):
        """Initialize RAG service with Azure OpenAI and Redis cache."""
        self.cache = redis_cache
        self.embeddings_enabled = False  # Track if embeddings are working
        
        # Initialize Azure OpenAI for embeddings
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        # Embedding model (usually different from chat model)
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be set in .env file")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # Test if embeddings work
        self._test_embeddings()
    
    def _test_embeddings(self):
        """Test if embedding model is available."""
        try:
            test_embedding = self.get_embedding("test")
            if test_embedding:
                self.embeddings_enabled = True
                print("✅ RAG: Embeddings enabled and working")
            else:
                print("⚠️ RAG: Embeddings disabled - embedding model not available")
        except Exception as e:
            print(f"⚠️ RAG: Embeddings disabled - {e}")
            self.embeddings_enabled = False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Azure OpenAI."""
        if not text or len(text.strip()) == 0:
            return None
            
        try:
            # Try embedding deployment first if specified
            if self.embedding_deployment:
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_deployment,
                        input=text
                    )
                    print(f"✅ RAG: Got embedding using {self.embedding_deployment}")
                    return response.data[0].embedding
                except Exception as e:
                    print(f"⚠️ RAG: Embedding model {self.embedding_deployment} failed: {e}")
            
            # Fallback: Don't try chat model for embeddings (it usually doesn't support it)
            # Instead, return None and disable RAG
            print(f"⚠️ RAG: No embedding model available, RAG disabled")
            return None
            
        except Exception as e:
            print(f"❌ RAG: Error getting embedding: {e}")
            return None
    
    def store_article(self, article: Dict, symbol: str) -> bool:
        """Store article with embedding for RAG retrieval."""
        if not self.embeddings_enabled:
            print(f"⚠️ RAG: Skipping article storage - embeddings not enabled")
            return False
            
        try:
            # Create article ID
            article_id = hashlib.md5(
                f"{symbol}:{article.get('title', '')}:{article.get('url', '')}".encode()
            ).hexdigest()
            
            # Get embedding for article text
            article_text = f"{article.get('title', '')} {article.get('summary', '')}"
            if not article_text.strip():
                return False
                
            embedding = self.get_embedding(article_text)
            
            if embedding:
                # Store in Redis with symbol prefix
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
                
                if self.cache.client:
                    self.cache.client.setex(
                        embedding_key, 
                        86400 * 7, 
                        json.dumps(embedding)
                    )
                    self.cache.client.setex(
                        metadata_key,
                        86400 * 7,
                        json.dumps(metadata, default=str)
                    )
                    print(f"✅ RAG: Stored article {article_id[:8]}... for {symbol}")
                    return True
            else:
                print(f"⚠️ RAG: Could not get embedding for article, skipping storage")
            return False
        except Exception as e:
            print(f"❌ RAG: Error storing article: {e}")
            return False
    
    def retrieve_relevant_context(self, query: str, symbol: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant articles for RAG context."""
        if not self.embeddings_enabled:
            print(f"⚠️ RAG: Skipping context retrieval - embeddings not enabled")
            return []
            
        try:
            # Get embedding for query
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                print(f"⚠️ RAG: Could not get query embedding")
                return []
            
            # Search for similar articles
            if not self.cache.client:
                print(f"⚠️ RAG: Redis not available for context retrieval")
                return []
            
            # Get all article embeddings for this symbol
            pattern = f"embedding:{symbol}:*"
            keys = self.cache.client.keys(pattern)
            
            if not keys:
                print(f"⚠️ RAG: No stored articles found for {symbol}")
                return []
            
            print(f"🔍 RAG: Searching {len(keys)} articles for {symbol}")
            
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
                        print(f"⚠️ RAG: Error processing embedding: {e}")
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
                    except:
                        continue
            
            if results:
                print(f"✅ RAG: Retrieved {len(results)} relevant articles (similarity: {results[0].get('similarity', 0):.3f})")
            else:
                print(f"⚠️ RAG: No similar articles found")
            
            return results
        except Exception as e:
            print(f"❌ RAG: Error retrieving context: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except:
            # Fallback without numpy
            try:
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = sum(a * a for a in vec1) ** 0.5
                norm2 = sum(b * b for b in vec2) ** 0.5
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot_product / (norm1 * norm2)
            except:
                return 0.0
