"""
Service layer for the Stock Sentiment Analysis application.

This module contains the core business logic services:
- SentimentAnalyzer: AI-powered sentiment analysis
- StockDataCollector: Stock data and news collection
- RAGService: Retrieval Augmented Generation for context
- RedisCache: Caching layer for performance optimization
- MultiTierCache: Multi-tier caching (L1: Memory, L2: Redis, L3: Disk)
- CrossEncoderReranker: Re-ranking for improved precision
- MessageQueue: Async job processing using Redis Streams
- ABTestingFramework: A/B testing for prompt optimization
"""

from .cache import RedisCache, CacheStats
from .collector import StockDataCollector
from .sentiment import SentimentAnalyzer
from .rag import RAGService
from .cost_tracker import CostTracker
from .multi_tier_cache import MultiTierCache
from .reranker import CrossEncoderReranker
from .message_queue import MessageQueue
from .ab_testing import ABTestingFramework, Variant
from .vector_db import VectorDatabase, RedisVectorDB

__all__ = [
    "RedisCache",
    "CacheStats",
    "StockDataCollector",
    "SentimentAnalyzer",
    "RAGService",
    "CostTracker",
    "MultiTierCache",
    "CrossEncoderReranker",
    "MessageQueue",
    "ABTestingFramework",
    "Variant",
    "VectorDatabase",
    "RedisVectorDB",
]

