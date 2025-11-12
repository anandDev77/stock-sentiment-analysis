import os
from openai import AzureOpenAI
import json
from typing import Dict, List, Optional
import re
from textblob import TextBlob
from dotenv import load_dotenv
from redis_cache import RedisCache
from rag_service import RAGService

# Load environment variables
load_dotenv()

class SentimentAnalyzer:
    def __init__(self, redis_cache: Optional[RedisCache] = None, rag_service: Optional[RAGService] = None):
        """Initialize the sentiment analyzer with Azure OpenAI, Redis cache, and RAG."""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be set in .env file")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # Initialize cache and RAG service
        self.cache = redis_cache
        self.rag_service = rag_service
        
        # Debug counters
        self.cache_hits = 0
        self.cache_misses = 0
        self.rag_uses = 0

    def analyze_sentiment(self, text: str, symbol: Optional[str] = None, context: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Analyze sentiment of given text using Azure OpenAI with RAG context.
        Returns sentiment scores for positive, negative, and neutral.
        """
        # Check cache first
        if self.cache:
            cached_result = self.cache.get_cached_sentiment(text)
            if cached_result:
                self.cache_hits += 1
                print(f"✅ CACHE HIT: Using cached sentiment for text (length: {len(text)})")
                return cached_result
            else:
                self.cache_misses += 1
                print(f"❌ CACHE MISS: Analyzing sentiment for text (length: {len(text)})")
        
        # Retrieve relevant context using RAG if available
        rag_context = ""
        rag_used = False
        if self.rag_service and symbol:
            relevant_articles = self.rag_service.retrieve_relevant_context(text, symbol, top_k=3)
            if relevant_articles:
                rag_used = True
                self.rag_uses += 1
                rag_context = "\n\nRelevant context from recent news:\n"
                for i, article in enumerate(relevant_articles, 1):
                    rag_context += f"{i}. {article.get('title', '')}: {article.get('summary', '')[:200]}\n"
                print(f"✅ RAG: Using {len(relevant_articles)} relevant articles for context")
            else:
                print(f"⚠️ RAG: No relevant context found")
        
        # Use provided context if available
        if context:
            context_text = "\n\nAdditional context:\n"
            for i, item in enumerate(context[:3], 1):
                context_text += f"{i}. {item.get('title', item.get('text', ''))}\n"
            rag_context += context_text
        
        prompt = f"""Analyze the sentiment of the following text about stocks/finance.
Return ONLY a valid JSON object with scores (0-1) for: positive, negative, neutral.
The scores should sum to approximately 1.0.
{rag_context}
Text: "{text}"

Return only the JSON object, no other text:
"""

        try:
            print(f"🔄 API CALL: Sending request to Azure OpenAI (RAG: {'Yes' if rag_used else 'No'})")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyzer. Always respond with valid JSON only. Consider the context provided when analyzing sentiment."},
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
                positive = float(result.get('positive', 0))
                negative = float(result.get('negative', 0))
                neutral = float(result.get('neutral', 0))
                
                # Normalize to ensure they sum to 1
                total = positive + negative + neutral
                if total > 0:
                    positive = positive / total
                    negative = negative / total
                    neutral = neutral / total
                else:
                    # Fallback if all are 0
                    neutral = 1.0
                
                sentiment_result = {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral
                }
                
                # Cache the result
                if self.cache:
                    cached = self.cache.cache_sentiment(text, sentiment_result)
                    if cached:
                        print(f"✅ CACHE: Stored sentiment result in cache")
                
                print(f"✅ API RESPONSE: Sentiment analyzed (P:{positive:.2f}, N:{negative:.2f}, Neu:{neutral:.2f})")
                return sentiment_result
            else:
                print(f"⚠️ FALLBACK: Could not parse JSON, using TextBlob")
                # Fallback to TextBlob for parsing errors
                return self._textblob_fallback(text)

        except Exception as e:
            print(f"❌ ERROR: Azure OpenAI error: {e}, using TextBlob fallback")
            # Fallback to TextBlob
            return self._textblob_fallback(text)
    
    def _textblob_fallback(self, text: str) -> Dict[str, float]:
        """Fallback sentiment analysis using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            return {'positive': min(polarity, 1.0), 'negative': 0, 'neutral': max(0, 1-polarity)}
        elif polarity < -0.1:
            return {'positive': 0, 'negative': min(abs(polarity), 1.0), 'neutral': max(0, 1-abs(polarity))}
        else:
            return {'positive': 0, 'negative': 0, 'neutral': 1}
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about cache and RAG usage."""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'rag_uses': self.rag_uses,
            'total_requests': self.cache_hits + self.cache_misses
        }

    def batch_analyze(self, texts: List[str], symbol: Optional[str] = None) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze_sentiment(text, symbol) for text in texts]
