"""
Sentiment analysis service using Azure OpenAI.

This module provides AI-powered sentiment analysis with support for:
- Azure OpenAI GPT-4 for sentiment analysis
- RAG (Retrieval Augmented Generation) for context-aware analysis
- Redis caching for performance optimization
- TextBlob fallback for error handling
"""

import json
import re
from typing import Dict, List, Optional
from openai import AzureOpenAI
from textblob import TextBlob

from ..config.settings import Settings, get_settings
from ..models.sentiment import SentimentScores, SentimentResult
from ..utils.logger import get_logger
from .cache import RedisCache
from .rag import RAGService

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    AI-powered sentiment analyzer using Azure OpenAI.
    
    This class provides sentiment analysis capabilities with:
    - Azure OpenAI GPT-4 for high-quality sentiment analysis
    - RAG integration for context-aware analysis
    - Redis caching to reduce API calls
    - TextBlob fallback for reliability
    
    Attributes:
        client: Azure OpenAI client instance
        settings: Application settings
        cache: Redis cache instance (optional)
        rag_service: RAG service instance (optional)
        cache_hits: Counter for cache hits
        cache_misses: Counter for cache misses
        rag_uses: Counter for RAG usage
        
    Example:
        >>> settings = get_settings()
        >>> cache = RedisCache(settings=settings)
        >>> analyzer = SentimentAnalyzer(settings=settings, redis_cache=cache)
        >>> result = analyzer.analyze_sentiment("Apple stock is rising!", symbol="AAPL")
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_cache: Optional[RedisCache] = None,
        rag_service: Optional[RAGService] = None
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            settings: Application settings (uses global if not provided)
            redis_cache: Redis cache instance for caching results
            rag_service: RAG service for context retrieval
            
        Raises:
            ValueError: If Azure OpenAI configuration is invalid
        """
        self.settings = settings or get_settings()
        self.cache = redis_cache
        self.rag_service = rag_service
        
        # Initialize Azure OpenAI client
        azure_config = self.settings.azure_openai
        self.client = AzureOpenAI(
            azure_endpoint=azure_config.endpoint,
            api_key=azure_config.api_key,
            api_version=azure_config.api_version
        )
        self.deployment_name = azure_config.deployment_name
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.rag_uses = 0
        
        logger.info(
            f"SentimentAnalyzer initialized with deployment: {self.deployment_name}"
        )
    
    def analyze_sentiment(
        self,
        text: str,
        symbol: Optional[str] = None,
        context: Optional[List[Dict]] = None
    ) -> Dict[str, float]:
        """
        Analyze sentiment of given text using Azure OpenAI with optional RAG context.
        
        Args:
            text: Text to analyze
            symbol: Optional stock symbol for RAG context retrieval
            context: Optional additional context items
            
        Returns:
            Dictionary with sentiment scores: {'positive': float, 'negative': float, 'neutral': float}
            
        Note:
            Scores are normalized to sum to approximately 1.0
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for sentiment analysis")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get_cached_sentiment(text)
            if cached_result:
                self.cache_hits += 1
                logger.debug(f"Cache hit for sentiment analysis (text length: {len(text)})")
                return cached_result
            else:
                self.cache_misses += 1
                logger.debug(f"Cache miss for sentiment analysis (text length: {len(text)})")
        
        # Retrieve relevant context using RAG if available
        rag_context = ""
        rag_used = False
        if self.rag_service and symbol:
            relevant_articles = self.rag_service.retrieve_relevant_context(
                query=text,
                symbol=symbol,
                top_k=self.settings.app.rag_top_k
            )
            if relevant_articles:
                rag_used = True
                self.rag_uses += 1
                rag_context = "\n\nRelevant context from recent news:\n"
                for i, article in enumerate(relevant_articles, 1):
                    title = article.get('title', '')
                    summary = article.get('summary', '')[:200]
                    rag_context += f"{i}. {title}: {summary}\n"
                logger.info(f"Using {len(relevant_articles)} relevant articles for RAG context")
            else:
                logger.debug("No relevant RAG context found")
        
        # Use provided context if available
        if context:
            context_text = "\n\nAdditional context:\n"
            for i, item in enumerate(context[:3], 1):
                title = item.get('title', item.get('text', ''))
                context_text += f"{i}. {title}\n"
            rag_context += context_text
        
        # Build prompt
        prompt = f"""Analyze the sentiment of the following text about stocks/finance.
Return ONLY a valid JSON object with scores (0-1) for: positive, negative, neutral.
The scores should sum to approximately 1.0.
{rag_context}
Text: "{text}"

Return only the JSON object, no other text:
"""
        
        try:
            logger.info(f"Sending request to Azure OpenAI (RAG: {'Yes' if rag_used else 'No'})")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial sentiment analyzer. "
                            "Always respond with valid JSON only. "
                            "Consider the context provided when analyzing sentiment."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                sentiment_scores = SentimentScores.from_dict(result)
                
                # Cache the result
                if self.cache:
                    cached = self.cache.cache_sentiment(
                        text,
                        sentiment_scores.to_dict(),
                        ttl=self.settings.app.cache_ttl_sentiment
                    )
                    if cached:
                        logger.debug("Stored sentiment result in cache")
                
                logger.info(
                    f"Sentiment analyzed - Positive: {sentiment_scores.positive:.2f}, "
                    f"Negative: {sentiment_scores.negative:.2f}, "
                    f"Neutral: {sentiment_scores.neutral:.2f}"
                )
                return sentiment_scores.to_dict()
            else:
                logger.warning("Could not parse JSON from Azure OpenAI response, using TextBlob fallback")
                return self._textblob_fallback(text)
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, using TextBlob fallback")
            return self._textblob_fallback(text)
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}, using TextBlob fallback")
            return self._textblob_fallback(text)
    
    def _textblob_fallback(self, text: str) -> Dict[str, float]:
        """
        Fallback sentiment analysis using TextBlob.
        
        This method is used when Azure OpenAI fails or returns invalid responses.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return {
                    'positive': min(polarity, 1.0),
                    'negative': 0.0,
                    'neutral': max(0.0, 1 - polarity)
                }
            elif polarity < -0.1:
                return {
                    'positive': 0.0,
                    'negative': min(abs(polarity), 1.0),
                    'neutral': max(0.0, 1 - abs(polarity))
                }
            else:
                return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        except Exception as e:
            logger.error(f"TextBlob fallback error: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about cache and RAG usage.
        
        Returns:
            Dictionary with statistics:
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            - rag_uses: Number of times RAG was used
            - total_requests: Total number of requests
        """
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'rag_uses': self.rag_uses,
            'total_requests': self.cache_hits + self.cache_misses
        }
    
    def batch_analyze(
        self,
        texts: List[str],
        symbol: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            symbol: Optional stock symbol for RAG context
            
        Returns:
            List of sentiment score dictionaries
        """
        return [self.analyze_sentiment(text, symbol) for text in texts]

