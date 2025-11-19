"""
Orchestrator service for sentiment analysis.

This module provides the core sentiment analysis logic without Streamlit dependencies,
making it reusable for both the Streamlit dashboard and REST API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from ..config.settings import Settings
from ..services.collector import StockDataCollector
from ..services.sentiment import SentimentAnalyzer
from ..services.rag import RAGService
from ..services.cache import RedisCache
from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_aggregated_sentiment(
    symbol: str,
    collector: StockDataCollector,
    analyzer: SentimentAnalyzer,
    rag_service: Optional[RAGService] = None,
    redis_cache: Optional[RedisCache] = None,
    settings: Optional[Settings] = None,
    data_source_filters: Optional[Dict[str, bool]] = None,
    return_detailed: bool = False
) -> Dict[str, Any]:
    """
    Get aggregated sentiment analysis for a stock symbol.
    
    This function orchestrates the complete sentiment analysis pipeline:
    1. Collect stock data and news from multiple sources
    2. Store articles in RAG for context retrieval
    3. Analyze sentiment for all articles with RAG context
    4. Aggregate sentiment scores
    
    Args:
        symbol: Stock symbol to analyze (e.g., "AAPL")
        collector: StockDataCollector instance
        analyzer: SentimentAnalyzer instance
        rag_service: RAGService instance (optional)
        redis_cache: RedisCache instance (optional)
        settings: Application settings (optional)
        data_source_filters: Dictionary of data source enable/disable flags
            Example: {"yfinance": True, "alpha_vantage": False, "finnhub": True, "reddit": False}
        return_detailed: If True, also return raw data and individual sentiment scores
            (for dashboard use). Default: False (for API use).
    
    Returns:
        Dictionary containing:
            - symbol: Stock symbol
            - positive: Aggregated positive sentiment score (0.0 to 1.0)
            - negative: Aggregated negative sentiment score (0.0 to 1.0)
            - neutral: Aggregated neutral sentiment score (0.0 to 1.0)
            - net_sentiment: Net sentiment (positive - negative, -1.0 to 1.0)
            - dominant_sentiment: "positive", "negative", or "neutral"
            - sources_analyzed: Number of articles analyzed
            - timestamp: ISO format timestamp
            - operation_summary: Dictionary with operation statistics
            - data: (if return_detailed=True) Raw stock data and news articles
            - news_sentiments: (if return_detailed=True) Individual sentiment scores for news articles
            - social_sentiments: (if return_detailed=True) Individual sentiment scores for social media posts
    """
    if not settings:
        from ..config.settings import get_settings
        settings = get_settings()
    
    # Initialize operation summary for logging
    operation_summary = {
        'redis_used': False,
        'stock_cached': False,
        'news_cached': False,
        'sentiment_cache_hits': 0,
        'sentiment_cache_misses': 0,
        'rag_used': False,
        'rag_queries': 0,
        'rag_articles_found': 0,
        'articles_stored': 0
    }
    
    logger.info(f"=== Starting sentiment analysis for {symbol} ===")
    logger.info(f"Sentiment cache enabled: {settings.app.cache_sentiment_enabled}")
    logger.info(f"Sentiment cache TTL: {settings.app.cache_ttl_sentiment}s ({settings.app.cache_ttl_sentiment/3600:.1f} hours)")
    
    try:
        # Step 1: Collect stock data and news
        logger.info(f"[{symbol}] Step 1: Collecting stock data and news...")
        
        if redis_cache:
            redis_cache.last_tier_used = None
            logger.info(f"[{symbol}] Checking Redis for stock data...")
        
        # Collect data with source filters
        data = collector.collect_all_data(symbol, data_source_filters=data_source_filters)
        
        logger.info(f"[{symbol}] Collected {len(data.get('news', []))} news articles")
        
        # Track cache status for stock data
        if redis_cache:
            redis_cache.last_tier_used = None
            cached_stock = redis_cache.get_cached_stock_data(symbol)
            if redis_cache.last_tier_used == "Redis":
                operation_summary['redis_used'] = True
                operation_summary['stock_cached'] = True
                logger.info(f"[{symbol}] ✅ Stock data retrieved from Redis cache")
            else:
                operation_summary['redis_used'] = True  # Redis was checked
                operation_summary['stock_cached'] = False
                logger.info(f"[{symbol}] 🔄 Stock data fetched fresh (cache miss)")
        
        # Track cache status for news data
        if redis_cache:
            redis_cache.last_tier_used = None
            cached_news = redis_cache.get_cached_news(symbol)
            if redis_cache.last_tier_used == "Redis":
                operation_summary['news_cached'] = True
                logger.info(f"[{symbol}] ✅ News data retrieved from Redis cache")
            else:
                operation_summary['news_cached'] = False
                logger.info(f"[{symbol}] 🔄 News data fetched fresh (cache miss)")
        
        # Show data source breakdown
        if data.get('news'):
            source_breakdown = {}
            for article in data['news']:
                source = article.get('source', 'Unknown')
                if 'Alpha Vantage' in source:
                    source_breakdown['Alpha Vantage'] = source_breakdown.get('Alpha Vantage', 0) + 1
                elif 'Finnhub' in source:
                    source_breakdown['Finnhub'] = source_breakdown.get('Finnhub', 0) + 1
                elif 'Reddit' in source or 'r/' in source:
                    source_breakdown['Reddit'] = source_breakdown.get('Reddit', 0) + 1
                else:
                    source_breakdown['Yahoo Finance'] = source_breakdown.get('Yahoo Finance', 0) + 1
            
            if source_breakdown:
                breakdown_text = " | ".join([f"{k}: {v}" for k, v in source_breakdown.items()])
                logger.info(f"API: Data source breakdown - {breakdown_text}")
        
        # Step 2: Store articles in RAG
        logger.info(f"[{symbol}] Step 2: Storing articles in RAG...")
        
        if rag_service and data['news']:
            article_count = len(data['news'])
            logger.info(f"[{symbol}] Storing {article_count} articles in RAG...")
            rag_service.store_articles_batch(data['news'], symbol)
            operation_summary['articles_stored'] += article_count
            logger.info(f"[{symbol}] ✅ Stored {article_count} articles in RAG")
        
        # Also store Reddit posts in RAG if available
        if rag_service and data.get('social_media'):
            reddit_count = len(data['social_media'])
            logger.info(f"[{symbol}] Storing {reddit_count} Reddit posts in RAG...")
            reddit_articles = []
            for post in data['social_media']:
                reddit_articles.append({
                    'title': post.get('title', ''),
                    'summary': post.get('summary', ''),
                    'source': post.get('source', 'Reddit'),
                    'url': post.get('url', ''),
                    'timestamp': post.get('timestamp', datetime.now())
                })
            rag_service.store_articles_batch(reddit_articles, symbol)
            operation_summary['articles_stored'] += reddit_count
            logger.info(f"[{symbol}] ✅ Stored {reddit_count} Reddit posts in RAG")
        
        # Step 3: Analyze sentiment
        logger.info(f"[{symbol}] Step 3: Analyzing sentiment...")
        
        news_texts = [
            article.get('summary', article.get('title', ''))
            for article in data['news']
        ]
        
        logger.info(f"[{symbol}] Analyzing sentiment for {len(news_texts)} articles...")
        
        # Track sentiment cache status
        if redis_cache and settings.app.cache_sentiment_enabled:
            logger.info(f"[{symbol}] Checking sentiment cache for {len(news_texts)} articles...")
            for text in news_texts:
                if text:
                    redis_cache.last_tier_used = None
                    cached_sentiment = redis_cache.get_cached_sentiment(text)
                    if redis_cache.last_tier_used == "Redis":
                        operation_summary['sentiment_cache_hits'] += 1
                    else:
                        operation_summary['sentiment_cache_misses'] += 1
            logger.info(f"[{symbol}] Sentiment cache: {operation_summary['sentiment_cache_hits']} hits, {operation_summary['sentiment_cache_misses']} misses")
        else:
            if not settings.app.cache_sentiment_enabled:
                logger.info(f"[{symbol}] Sentiment cache is disabled - all analyses will use RAG")
            operation_summary['sentiment_cache_misses'] = len(news_texts)
        
        # Track RAG usage before analysis
        initial_rag_uses = getattr(analyzer, 'rag_uses', 0) if hasattr(analyzer, 'rag_uses') else 0
        initial_rag_attempts = getattr(analyzer, 'rag_attempts', 0) if hasattr(analyzer, 'rag_attempts') else 0
        
        # Batch analyze with parallel processing
        worker_count = settings.app.analysis_parallel_workers or settings.app.sentiment_max_workers
        worker_timeout = settings.app.analysis_worker_timeout
        logger.info(
            f"[{symbol}] Starting batch sentiment analysis "
            f"(workers={worker_count}, timeout={worker_timeout}s)..."
        )
        news_sentiments = analyzer.batch_analyze(
            texts=news_texts,
            symbol=symbol,
            max_workers=worker_count,
            worker_timeout=worker_timeout
        )
        
        # Track RAG usage after analysis
        final_rag_uses = getattr(analyzer, 'rag_uses', 0) if hasattr(analyzer, 'rag_uses') else 0
        final_rag_attempts = getattr(analyzer, 'rag_attempts', 0) if hasattr(analyzer, 'rag_attempts') else 0
        
        rag_queries_made = final_rag_attempts - initial_rag_attempts
        rag_successful = final_rag_uses - initial_rag_uses
        
        if rag_queries_made > 0:
            operation_summary['rag_used'] = True
            operation_summary['rag_queries'] = rag_queries_made
            operation_summary['rag_articles_found'] = rag_successful
            logger.info(f"[{symbol}] ✅ RAG was used: {rag_queries_made} queries made, {rag_successful} successful")
        else:
            logger.info(f"[{symbol}] ℹ️ RAG was not used (sentiment was cached or cache enabled)")
        
        # Handle empty texts
        for i, text in enumerate(news_texts):
            if not text:
                news_sentiments[i] = {'positive': 0, 'negative': 0, 'neutral': 1}
        
        # Analyze social media posts separately
        social_texts = [post.get('text', '') for post in data.get('social_media', [])]
        social_sentiments = []
        if social_texts:
            logger.info(f"[{symbol}] Analyzing sentiment for {len(social_texts)} social media posts...")
            social_sentiments = analyzer.batch_analyze(
                texts=social_texts,
                symbol=symbol,
                max_workers=worker_count,
                worker_timeout=worker_timeout
            )
            # Handle empty texts
            for i, text in enumerate(social_texts):
                if not text:
                    social_sentiments[i] = {'positive': 0, 'negative': 0, 'neutral': 1}
            logger.info(f"[{symbol}] Analyzed {len(social_sentiments)} social media posts")
        
        # Combine for aggregation (but keep separate for detailed return)
        all_sentiments = news_sentiments + social_sentiments
        
        # Step 4: Aggregate sentiment scores
        logger.info(f"[{symbol}] Step 4: Aggregating sentiment scores...")
        
        if not all_sentiments:
            logger.warning(f"[{symbol}] No sentiment results to aggregate")
            result = {
                'symbol': symbol,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'net_sentiment': 0.0,
                'dominant_sentiment': 'neutral',
                'sources_analyzed': 0,
                'timestamp': datetime.now().isoformat(),
                'operation_summary': operation_summary
            }
            if return_detailed:
                result['data'] = data
                result['news_sentiments'] = news_sentiments
                result['social_sentiments'] = social_sentiments
            return result
        
        # Calculate aggregated scores from all sentiments
        total_positive = sum(s.get('positive', 0) for s in all_sentiments)
        total_negative = sum(s.get('negative', 0) for s in all_sentiments)
        total_neutral = sum(s.get('neutral', 0) for s in all_sentiments)
        count = len(all_sentiments)
        
        avg_positive = total_positive / count if count > 0 else 0.0
        avg_negative = total_negative / count if count > 0 else 0.0
        avg_neutral = total_neutral / count if count > 0 else 0.0
        
        # Normalize to ensure they sum to 1.0
        total = avg_positive + avg_negative + avg_neutral
        if total > 0:
            avg_positive /= total
            avg_negative /= total
            avg_neutral /= total
        
        net_sentiment = avg_positive - avg_negative
        
        # Determine dominant sentiment
        if avg_positive > avg_negative and avg_positive > avg_neutral:
            dominant_sentiment = 'positive'
        elif avg_negative > avg_positive and avg_negative > avg_neutral:
            dominant_sentiment = 'negative'
        else:
            dominant_sentiment = 'neutral'
        
        # Log final summary
        logger.info(f"[{symbol}] === Operation Summary ===")
        logger.info(f"[{symbol}] Redis used: {operation_summary['redis_used']}")
        logger.info(f"[{symbol}]   - Stock cached: {operation_summary['stock_cached']}")
        logger.info(f"[{symbol}]   - News cached: {operation_summary['news_cached']}")
        logger.info(f"[{symbol}]   - Sentiment: {operation_summary['sentiment_cache_hits']} hits, {operation_summary['sentiment_cache_misses']} misses")
        logger.info(f"[{symbol}] RAG used: {operation_summary['rag_used']}")
        if operation_summary['rag_used']:
            logger.info(f"[{symbol}]   - RAG queries: {operation_summary['rag_queries']}")
            logger.info(f"[{symbol}]   - Articles found: {operation_summary['rag_articles_found']}")
        logger.info(f"[{symbol}] Articles stored in RAG: {operation_summary['articles_stored']}")
        logger.info(f"[{symbol}] === End Summary ===")
        
        # Base result with aggregated sentiment
        result = {
            'symbol': symbol,
            'positive': round(avg_positive, 4),
            'negative': round(avg_negative, 4),
            'neutral': round(avg_neutral, 4),
            'net_sentiment': round(net_sentiment, 4),
            'dominant_sentiment': dominant_sentiment,
            'sources_analyzed': count,
            'timestamp': datetime.now().isoformat(),
            'operation_summary': operation_summary
        }
        
        # Add detailed data if requested (for dashboard)
        if return_detailed:
            result['data'] = data
            result['news_sentiments'] = news_sentiments
            result['social_sentiments'] = social_sentiments
        
        return result
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis for {symbol}: {e}", exc_info=True)
        raise

