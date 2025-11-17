"""
Data loading and processing logic.
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

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
    # Initialize cache status for current request
    cache_status = {
        'stock': {'hit': False, 'miss': False},
        'news': {'hit': False, 'miss': False},
        'sentiment': {'hits': 0, 'misses': 0}
    }
    # Update session state immediately so UI can show it
    st.session_state.cache_status = cache_status
    
    # Multi-step progress bar for better UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetching stock data (20%)
        status_text.text("📊 Fetching stock data...")
        progress_bar.progress(0.2)
        
        if redis_cache:
            redis_cache.last_tier_used = None  # Reset before call
        
        # Get data source filters from UI
        data_source_filters = None
        if 'search_filters' in st.session_state and 'data_sources' in st.session_state.search_filters:
            data_source_filters = st.session_state.search_filters.get('data_sources')
            enabled_sources = [k for k, v in data_source_filters.items() if v]
            logger.info(f"App: Data source filters applied - Enabled: {', '.join(enabled_sources)}")
        
        # Step 2: Collecting news articles (40%)
        status_text.text("📰 Collecting news articles from multiple sources...")
        progress_bar.progress(0.4)
        
        # Collect data with source filters using collect_all_data
        data = collector.collect_all_data(symbol, data_source_filters=data_source_filters)
        
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
                cache_status['stock']['hit'] = True
                cache_status['stock']['miss'] = False
            else:
                cache_status['stock']['hit'] = False
                cache_status['stock']['miss'] = True
        
        # Track cache status for news data
        if redis_cache:
            redis_cache.last_tier_used = None
            cached_news = redis_cache.get_cached_news(symbol)
            if redis_cache.last_tier_used == "Redis":
                cache_status['news']['hit'] = True
                cache_status['news']['miss'] = False
            else:
                cache_status['news']['hit'] = False
                cache_status['news']['miss'] = True
        
        # Update cache status
        st.session_state.cache_status = cache_status
        
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
            logger.info(f"App: Storing {len(data['news'])} articles in RAG from multiple sources")
            rag_service.store_articles_batch(data['news'], symbol)
        
        # Also store Reddit posts in RAG if available
        if rag_service and data.get('social_media'):
            logger.info(f"App: Storing {len(data['social_media'])} Reddit posts in RAG")
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
        
        # Analyze sentiment with RAG context using parallel processing
        news_texts = [
            article.get('summary', article.get('title', ''))
            for article in data['news']
        ]
        
        # Track sentiment cache status during batch processing
        if redis_cache:
            for text in news_texts:
                if text:
                    # Reset before each check to track individual cache status
                    redis_cache.last_tier_used = None
                    cached_sentiment = redis_cache.get_cached_sentiment(text)
                    if redis_cache.last_tier_used == "Redis":
                        cache_status['sentiment']['hits'] += 1
                    else:
                        cache_status['sentiment']['misses'] += 1
            # Update cache status after sentiment tracking
            st.session_state.cache_status = cache_status
        
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
        
        # Batch analyze with parallel processing
        news_sentiments = analyzer.batch_analyze(
            texts=news_texts,
            symbol=symbol,
            max_workers=5  # Process up to 5 articles concurrently
        )
        
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
        
        # Final update of cache status
        st.session_state.cache_status = cache_status
        
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

