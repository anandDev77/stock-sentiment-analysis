"""
Data loading and processing logic.
"""

import streamlit as st
import time
from datetime import datetime
from typing import Optional, Any

from ..utils.logger import get_logger
from ..utils.ui_helpers import show_toast, get_error_recovery_ui

logger = get_logger(__name__)


def load_stock_data(
    symbol: str,
    collector,
    analyzer,
    rag_service: Optional[Any],
    redis_cache: Optional[Any],
    settings
) -> bool:
    """
    Load and process stock data with sentiment analysis.
    
    Args:
        symbol: Stock symbol to analyze
        collector: StockDataCollector instance
        analyzer: SentimentAnalyzer instance
        rag_service: RAGService instance (optional)
        redis_cache: RedisCache instance (optional)
        settings: Application settings
        
    Returns:
        True if successful, False otherwise
    """
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
    
    logger.info(f"=== Starting data load for {symbol} ===")
    logger.info(f"Sentiment cache enabled: {settings.app.cache_sentiment_enabled}")
    logger.info(f"Sentiment cache TTL: {settings.app.cache_ttl_sentiment}s ({settings.app.cache_ttl_sentiment/3600:.1f} hours)")
    
    # Multi-step progress bar for better UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetching stock data (20%)
        status_text.text("📊 Fetching stock data...")
        progress_bar.progress(0.2)
        
        logger.info(f"[{symbol}] Step 1: Fetching stock data...")
        
        if redis_cache:
            redis_cache.last_tier_used = None  # Reset before call
            logger.info(f"[{symbol}] Checking Redis for stock data...")
        
        # Get data source filters from UI
        data_source_filters = None
        if 'search_filters' in st.session_state and 'data_sources' in st.session_state.search_filters:
            data_source_filters = st.session_state.search_filters.get('data_sources')
            enabled_sources = [k for k, v in data_source_filters.items() if v]
            logger.info(f"[{symbol}] Data source filters applied - Enabled: {', '.join(enabled_sources)}")
        
        # Step 2: Collecting news articles (40%)
        status_text.text("📰 Collecting news articles from multiple sources...")
        progress_bar.progress(0.4)
        
        logger.info(f"[{symbol}] Step 2: Collecting news articles from multiple sources...")
        
        # Collect data with source filters using collect_all_data
        data = collector.collect_all_data(symbol, data_source_filters=data_source_filters)
        
        logger.info(f"[{symbol}] Collected {len(data.get('news', []))} news articles")
        
        # Clear any previous errors
        if symbol in st.session_state.data_errors:
            del st.session_state.data_errors[symbol]
        
    except Exception as e:
        # Error handling with retry
        progress_bar.empty()
        status_text.empty()
        
        error_msg = f"Failed to fetch data: {str(e)}"
        logger.error(f"Data collection error: {e}")
        
        st.session_state.data_errors[symbol] = error_msg
        
        if get_error_recovery_ui(error_msg, retry_key=f"retry_data_{symbol}"):
            st.rerun()
        
        st.stop()
        return False
    
    try:
        # Step 3: Storing in RAG (60%)
        status_text.text("💾 Storing articles in RAG for context retrieval...")
        progress_bar.progress(0.6)
        
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
        
        st.session_state.data = data
        st.session_state.symbol = symbol
        
        # Reset pagination when new data is loaded
        st.session_state.article_page = 1
        st.session_state.show_all_articles = False
        
        # Show data source breakdown
        if data.get('news'):
            source_breakdown = {}
            for article in data['news']:
                source = article.get('source', 'Unknown')
                # Categorize by data source
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
                logger.info(f"App: Data source breakdown - {breakdown_text}")
        
        # Store articles in RAG using batch processing
        if rag_service and data['news']:
            article_count = len(data['news'])
            logger.info(f"[{symbol}] Step 3: Storing {article_count} articles in RAG...")
            rag_service.store_articles_batch(data['news'], symbol)
            operation_summary['articles_stored'] += article_count
            logger.info(f"[{symbol}] ✅ Stored {article_count} articles in RAG")
        
        # Also store Reddit posts in RAG if available
        if rag_service and data.get('social_media'):
            reddit_count = len(data['social_media'])
            logger.info(f"[{symbol}] Storing {reddit_count} Reddit posts in RAG...")
            # Convert Reddit posts to article format for RAG
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
        
        # Analyze sentiment with RAG context using parallel processing
        news_texts = [
            article.get('summary', article.get('title', ''))
            for article in data['news']
        ]
        
        logger.info(f"[{symbol}] Step 4: Analyzing sentiment for {len(news_texts)} articles...")
        
        # Track sentiment cache status during batch processing
        if redis_cache and settings.app.cache_sentiment_enabled:
            logger.info(f"[{symbol}] Checking sentiment cache for {len(news_texts)} articles...")
            for i, text in enumerate(news_texts):
                if text:
                    # Reset before each check to track individual cache status
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
        
        # Set RAG filters from UI if available
        if 'search_filters' in st.session_state:
            filters = st.session_state.search_filters
            exclude_sources = ["Unknown"] if filters.get("exclude_unknown", True) else None
            
            # Log filter application
            filter_log = []
            if filters.get("date_range"):
                start, end = filters.get("date_range", (None, None))
                filter_log.append(f"date_range={start.strftime('%Y-%m-%d') if start else 'any'} to {end.strftime('%Y-%m-%d') if end else 'any'}")
            if filters.get("days_back"):
                filter_log.append(f"days_back={filters.get('days_back')}")
            if filters.get("sources"):
                filter_log.append(f"sources={', '.join(filters.get('sources', []))}")
            if exclude_sources:
                filter_log.append(f"exclude_sources={', '.join(exclude_sources)}")
            
            if filter_log:
                logger.info(f"App: Applying RAG filters from UI - {', '.join(filter_log)}")
            
            analyzer.set_rag_filters(
                date_range=filters.get("date_range"),
                sources=filters.get("sources"),
                exclude_sources=exclude_sources,
                days_back=filters.get("days_back")
            )
        
        # Step 4: Analyzing sentiment (80%)
        status_text.text("🤖 Analyzing sentiment with AI...")
        progress_bar.progress(0.8)
        
        # Track RAG usage before analysis
        initial_rag_uses = getattr(analyzer, 'rag_uses', 0) if hasattr(analyzer, 'rag_uses') else 0
        initial_rag_attempts = getattr(analyzer, 'rag_attempts', 0) if hasattr(analyzer, 'rag_attempts') else 0
        
        # Batch analyze with parallel processing
        logger.info(f"[{symbol}] Starting batch sentiment analysis (max_workers=5)...")
        news_sentiments = analyzer.batch_analyze(
            texts=news_texts,
            symbol=symbol,
            max_workers=5  # Process up to 5 articles concurrently
        )
        
        # Track RAG usage after analysis
        final_rag_uses = getattr(analyzer, 'rag_uses', 0) if hasattr(analyzer, 'rag_uses') else 0
        final_rag_attempts = getattr(analyzer, 'rag_attempts', 0) if hasattr(analyzer, 'rag_attempts') else 0
        
        rag_queries_made = final_rag_attempts - initial_rag_attempts
        rag_successful = final_rag_uses - initial_rag_uses
        
        if rag_queries_made > 0:
            operation_summary['rag_used'] = True
            operation_summary['rag_queries'] = rag_queries_made
            operation_summary['rag_articles_found'] = rag_successful  # Approximate
            logger.info(f"[{symbol}] ✅ RAG was used: {rag_queries_made} queries made, {rag_successful} successful")
        else:
            logger.info(f"[{symbol}] ℹ️ RAG was not used (sentiment was cached or cache enabled)")
        
        # Handle empty texts (fallback to neutral)
        for i, text in enumerate(news_texts):
            if not text:
                news_sentiments[i] = {'positive': 0, 'negative': 0, 'neutral': 1}

        # Analyze social media posts in parallel
        social_texts = [post.get('text', '') for post in data['social_media']]
        social_sentiments = analyzer.batch_analyze(
            texts=social_texts,
            symbol=symbol,
            max_workers=5
        )
        
        # Handle empty texts (fallback to neutral)
        for i, text in enumerate(social_texts):
            if not text:
                social_sentiments[i] = {'positive': 0, 'negative': 0, 'neutral': 1}
        
        # Step 5: Complete (100%)
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(1.0)
        
        st.session_state.news_sentiments = news_sentiments
        st.session_state.social_sentiments = social_sentiments
        
        # Store operation summary
        st.session_state.operation_summary = operation_summary
        
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
        
        # Show success toast
        time.sleep(0.3)  # Brief pause to show completion
        progress_bar.empty()
        status_text.empty()
        
        show_toast(f"✅ Successfully analyzed {symbol}!", "success")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        
        error_msg = f"Failed to analyze sentiment: {str(e)}"
        logger.error(f"Sentiment analysis error: {e}")
        
        if get_error_recovery_ui(error_msg, retry_key=f"retry_sentiment_{symbol}"):
            st.rerun()
        
        # Show partial data if available
        if data and data.get('news'):
            st.warning("⚠️ Some data was collected but sentiment analysis failed. Showing available data.")
        else:
            st.stop()
        return False
    
    st.session_state.load_data = False
    st.session_state.title_shown = False  # Show title again after loading
    # Force rerun to update UI with new cache status
    st.rerun()
    return True

