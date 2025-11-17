"""
Stock Sentiment Analysis Dashboard - Streamlit Application

This is the main Streamlit application for the Stock Sentiment Analysis dashboard.
It provides an interactive interface for analyzing stock sentiment using AI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from stock_sentiment.config.settings import get_settings
from stock_sentiment.services.sentiment import SentimentAnalyzer
from stock_sentiment.services.collector import StockDataCollector
from stock_sentiment.services.cache import RedisCache
from stock_sentiment.services.rag import RAGService
from stock_sentiment.services.cost_tracker import CostTracker
from stock_sentiment.utils.logger import get_logger, setup_logger

# Initialize root logger at app startup
setup_logger("stock_sentiment", level="INFO")
logger = get_logger(__name__)
logger.info("Stock Sentiment Dashboard starting up")

# Page configuration with custom theme
st.set_page_config(
    page_title="Stock Sentiment Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI/UX
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    /* Subheader styling */
    h2 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 600;
    }
    
    /* Metric cards enhancement */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem 0;
    }
    
    /* Card-like containers */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #1f77b4;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #7f8c8d;
    }
    
    /* Chart containers */
    .plotly-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize settings
try:
    settings = get_settings()
except ValueError as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# Initialize Redis cache and RAG service
@st.cache_resource
def get_redis_cache(_settings):
    """Get Redis cache instance."""
    try:
        if _settings.is_redis_available():
            cache = RedisCache(settings=_settings)
            if cache.client:
                return cache
        return None
    except Exception as e:
        logger.warning(f"Redis cache not available: {e}")
        return None

@st.cache_resource
def get_rag_service(_settings, _cache):
    """Get RAG service instance."""
    if _cache and _settings.is_rag_available():
        try:
            return RAGService(settings=_settings, redis_cache=_cache)
        except Exception as e:
            logger.warning(f"RAG service not available: {e}")
            return None
    return None

@st.cache_resource
def get_collector(_settings, _cache):
    """Get stock data collector instance."""
    return StockDataCollector(settings=_settings, redis_cache=_cache)

@st.cache_resource
def get_cost_tracker(_settings, _cache):
    """Get or create cost tracker instance."""
    if _cache and _cache.client:
        return CostTracker(cache=_cache, settings=_settings)
    return None

@st.cache_resource
def get_analyzer(_settings, _cache, _rag_service, _cost_tracker):
    """Get sentiment analyzer instance."""
    try:
        return SentimentAnalyzer(
            settings=_settings,
            redis_cache=_cache,
            rag_service=_rag_service,
            cost_tracker=_cost_tracker
        )
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI: {e}")
        return None

# Initialize components
redis_cache = get_redis_cache(settings)
rag_service = get_rag_service(settings, redis_cache)
collector = get_collector(settings, redis_cache)
cost_tracker = get_cost_tracker(settings, redis_cache)
analyzer = get_analyzer(settings, redis_cache, rag_service, cost_tracker)

if analyzer is None:
    st.error("Failed to initialize sentiment analyzer. Please check your configuration.")
    st.stop()

# Initialize session state
if 'load_data' not in st.session_state:
    st.session_state.load_data = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'news_sentiments' not in st.session_state:
    st.session_state.news_sentiments = []
if 'social_sentiments' not in st.session_state:
    st.session_state.social_sentiments = []
if 'symbol' not in st.session_state:
    st.session_state.symbol = "AAPL"
if 'title_shown' not in st.session_state:
    st.session_state.title_shown = False

# Main header - only show once
if not st.session_state.title_shown:
    st.title("📈 Stock Sentiment Dashboard")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 2rem;'>
            <h3 style='color: white; margin: 0;'>🤖 AI-Powered Financial Intelligence</h3>
            <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
                Analyze stock sentiment using Azure OpenAI with RAG and Redis caching for enhanced accuracy
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.session_state.title_shown = True

# Sidebar for input - Enhanced UI
with st.sidebar:
    # Logo/Header section
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0; border-bottom: 2px solid #e0e0e0; margin-bottom: 1.5rem;'>
            <h2 style='color: #1f77b4; margin: 0;'>⚙️ Configuration</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Stock symbol input with better styling
    symbol = st.text_input(
        "📊 Stock Symbol",
        value=st.session_state.symbol,
        key="stock_symbol",
        help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
    ).upper()
    
    # Status indicators with better visual design
    st.markdown("### 🔌 System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        # Check Redis connection with actual ping test
        redis_connected = False
        redis_error = None
        if redis_cache:
            if redis_cache.client:
                try:
                    redis_cache.client.ping()
                    redis_connected = True
                except Exception as e:
                    redis_error = str(e)
                    logger.warning(f"Redis ping failed: {e}")
            else:
                redis_error = "Redis client not initialized"
        else:
            redis_error = "Redis cache not available"
        
        if redis_connected:
            st.markdown(
                """
                <div style='background: #d4edda; color: #155724; padding: 0.75rem; 
                            border-radius: 8px; text-align: center; font-weight: 600;'>
                    ✅ Redis
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='background: #f8d7da; color: #721c24; padding: 0.75rem; 
                            border-radius: 8px; text-align: center; font-weight: 600;'
                            title='{redis_error or "Redis not configured"}'>
                    ⚠️ Redis
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with status_col2:
        # Check RAG service with detailed status
        rag_enabled = False
        rag_error = None
        if rag_service:
            try:
                rag_enabled = rag_service.embeddings_enabled
                if not rag_enabled:
                    rag_error = "Embeddings not enabled (check embedding deployment)"
            except Exception as e:
                rag_error = str(e)
                logger.warning(f"RAG check failed: {e}")
        else:
            rag_error = "RAG service not initialized"
        
        if rag_enabled:
            st.markdown(
                """
                <div style='background: #d4edda; color: #155724; padding: 0.75rem; 
                            border-radius: 8px; text-align: center; font-weight: 600;'>
                    ✅ RAG
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='background: #fff3cd; color: #856404; padding: 0.75rem; 
                            border-radius: 8px; text-align: center; font-weight: 600;'
                            title='{rag_error or "RAG not configured"}'>
                    ⚠️ RAG
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced load button
    if st.button("🚀 Load Data", type="primary"):
        st.session_state.load_data = True
        st.session_state.symbol = symbol
        st.session_state.title_shown = False  # Reset to show title again after load
    
    st.markdown("---")
    
    # Cache status indicator with detailed info
    if 'cache_status' in st.session_state and st.session_state.cache_status:
        cache_status = st.session_state.cache_status
        st.markdown("### 🔄 Cache Status (Last Request)")
        
        cache_col1, cache_col2 = st.columns(2)
        with cache_col1:
            # Stock data cache status
            if cache_status['stock']['hit']:
                st.success("✅ Stock Data: **CACHED**")
            elif cache_status['stock']['miss']:
                st.info("🔄 Stock Data: **FRESH** (from API)")
            else:
                st.warning("⚠️ Stock Data: Unknown status")
            
            # News cache status
            if cache_status['news']['hit']:
                st.success("✅ News: **CACHED**")
            elif cache_status['news']['miss']:
                st.info("🔄 News: **FRESH** (from API)")
            else:
                st.warning("⚠️ News: Unknown status")
        
        with cache_col2:
            total_sentiment = cache_status['sentiment']['hits'] + cache_status['sentiment']['misses']
            if total_sentiment > 0:
                hit_rate = (cache_status['sentiment']['hits'] / total_sentiment) * 100
                st.metric(
                    "Sentiment Cache",
                    f"{hit_rate:.0f}%",
                    f"{cache_status['sentiment']['hits']}/{total_sentiment} hits"
                )
            else:
                st.info("No sentiment analysis yet")
            
            # Show last cache tier used (for all cache operations)
            if redis_cache and redis_cache.last_tier_used:
                tier_display = redis_cache.last_tier_used
                if tier_display == "Redis":
                    tier_emoji = "🔴"
                    tier_desc = "Redis Cache"
                    tier_color = "success"
                elif tier_display == "MISS":
                    tier_emoji = "⚪"
                    tier_desc = "Cache Miss"
                    tier_color = "info"
                else:
                    tier_emoji = "🔴"
                    tier_desc = tier_display
                    tier_color = "success"
                
                if tier_color == "success":
                    st.success(f"{tier_emoji} Last Cache: **{tier_desc}**")
                else:
                    st.info(f"{tier_emoji} Last Cache: **{tier_desc}**")
    
    st.markdown("---")
    
    # Connection details for troubleshooting
    with st.expander("🔍 Connection Details", expanded=False):
        st.markdown("### Redis Connection")
        if redis_cache:
            if redis_cache.client:
                try:
                    redis_cache.client.ping()
                    st.success("✅ Redis: Connected and responding")
                    # Show Redis info
                    try:
                        info = redis_cache.client.info('server')
                        st.code(f"Redis Version: {info.get('redis_version', 'Unknown')}")
                    except:
                        pass
                except Exception as e:
                    st.error(f"❌ Redis: Connection failed - {e}")
            else:
                st.warning("⚠️ Redis: Client not initialized")
                if settings.is_redis_available():
                    st.info("Redis config exists but connection failed. Check your .env file.")
                else:
                    st.info("Redis not configured in .env file")
        else:
            st.warning("⚠️ Redis: Cache instance not created")
        
        st.markdown("### RAG Service")
        if rag_service:
            st.success(f"✅ RAG Service: Initialized")
            st.code(f"Embeddings Enabled: {rag_service.embeddings_enabled}")
            if hasattr(rag_service, 'embedding_deployment'):
                st.code(f"Embedding Model: {rag_service.embedding_deployment or 'Not configured'}")
        else:
            st.warning("⚠️ RAG Service: Not initialized")
            if settings.is_rag_available():
                st.info("RAG config exists but service failed to initialize. Check embedding deployment.")
            else:
                st.info("RAG not configured (missing embedding deployment in .env)")
        
        st.markdown("### Configuration Check")
        st.code(f"Redis Available: {settings.is_redis_available()}")
        st.code(f"RAG Available: {settings.is_rag_available()}")
        st.code(f"Azure OpenAI Available: {settings.is_azure_openai_available()}")
    
    st.markdown("---")
    
    # Performance stats with better design
    st.markdown("### 📊 Performance Metrics")
    
    # Get cache stats from Redis (persistent across reloads)
    cache_stats = {'cache_hits': 0, 'cache_misses': 0, 'cache_sets': 0}
    if redis_cache and redis_cache.client:
        try:
            # Test connection first
            redis_cache.client.ping()
            cache_stats = redis_cache.get_cache_stats()
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            cache_stats = {'cache_hits': 0, 'cache_misses': 0, 'cache_sets': 0}
    
    # Get analyzer stats (sentiment analysis specific)
    analyzer_stats = {'rag_uses': 0, 'rag_attempts': 0, 'total_requests': 0}
    if analyzer:
        try:
            # Check if analyzer has get_stats method
            if hasattr(analyzer, 'get_stats'):
                analyzer_stats = analyzer.get_stats()
            else:
                # Fallback to direct attribute access
                analyzer_stats = {
                    'rag_uses': getattr(analyzer, 'rag_uses', 0),
                    'rag_attempts': getattr(analyzer, 'rag_attempts', 0),
                    'cache_hits': getattr(analyzer, 'cache_hits', 0),
                    'cache_misses': getattr(analyzer, 'cache_misses', 0)
                }
        except Exception as e:
            logger.warning(f"Error getting analyzer stats: {e}")
            analyzer_stats = {'rag_uses': 0, 'rag_attempts': 0, 'total_requests': 0}
    
    # Cache metrics from Redis (persistent)
    cache_col1, cache_col2 = st.columns(2)
    with cache_col1:
        st.metric(
            "Cache Hits",
            cache_stats.get('cache_hits', 0),
            delta=None,
            delta_color="normal",
            help="Total cache hits (persisted in Redis)"
        )
    with cache_col2:
        st.metric(
            "Cache Misses",
            cache_stats.get('cache_misses', 0),
            delta=None,
            delta_color="normal",
            help="Total cache misses (persisted in Redis)"
        )
    
    # Calculate hit rate
    total_cache_ops = cache_stats.get('cache_hits', 0) + cache_stats.get('cache_misses', 0)
    if total_cache_ops > 0:
        hit_rate = (cache_stats.get('cache_hits', 0) / total_cache_ops) * 100
        st.metric(
            "Cache Hit Rate",
            f"{hit_rate:.1f}%",
            delta=f"{cache_stats.get('cache_hits', 0)}/{total_cache_ops}",
            delta_color="normal",
            help="Percentage of cache hits vs total cache operations"
        )
    
    # RAG uses (from analyzer)
    rag_uses = analyzer_stats.get('rag_uses', 0)
    rag_attempts = analyzer_stats.get('rag_attempts', 0)
    rag_success_rate = (rag_uses / rag_attempts * 100) if rag_attempts > 0 else 0.0
    
    st.metric(
        "RAG Uses",
        rag_uses,
        delta=f"{rag_attempts} attempts ({rag_success_rate:.1f}% success)" if rag_attempts > 0 else "No attempts",
        delta_color="normal",
        help=f"RAG successfully used {rag_uses} times out of {rag_attempts} attempts. "
             f"RAG is used when relevant articles are found for context. "
             f"If this is 0, it may mean: 1) No articles stored yet, 2) Articles don't match queries, "
             f"3) Similarity threshold too high, or 4) RAG service not initialized."
    )
    
    # Cost tracking (if available)
    if cost_tracker:
        try:
            cost_summary = cost_tracker.get_cost_summary(days=7)
            st.markdown("### 💰 Cost Tracking (Last 7 Days)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Cost",
                    f"${cost_summary.get('total_cost', 0):.4f}",
                    help="Total API costs in USD"
                )
            with col2:
                st.metric(
                    "Avg Daily",
                    f"${cost_summary.get('average_daily_cost', 0):.4f}",
                    help="Average daily cost"
                )
            with col3:
                st.metric(
                    "API Calls",
                    f"{cost_summary.get('total_calls', 0):,}",
                    help="Total API calls made"
                )
        except Exception as e:
            logger.warning(f"Error displaying cost tracking: {e}")
    
    # Cache management buttons
    cache_col1, cache_col2 = st.columns(2)
    with cache_col1:
        if st.button("🔄 Reset Cache Stats", use_container_width=True, help="Reset cache statistics counters (hits/misses)"):
            if redis_cache:
                redis_cache.reset_cache_stats()
                st.success("Cache statistics reset!")
                st.rerun()
    
    with cache_col2:
        # Initialize confirmation state
        if 'confirm_clear_cache' not in st.session_state:
            st.session_state.confirm_clear_cache = False
        
        if st.session_state.confirm_clear_cache:
            # Show confirmation buttons
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("✅ Confirm", use_container_width=True, type="primary"):
                    if redis_cache and redis_cache.client:
                        if redis_cache.clear_all_cache():
                            st.success("All cache data cleared!")
                            st.session_state.confirm_clear_cache = False
                            # Clear session state cache status
                            if 'cache_status' in st.session_state:
                                del st.session_state.cache_status
                            st.rerun()
                        else:
                            st.error("Failed to clear cache. Check logs for details.")
                            st.session_state.confirm_clear_cache = False
                    else:
                        st.warning("Redis cache not available")
                        st.session_state.confirm_clear_cache = False
            with confirm_col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_clear_cache = False
                    st.rerun()
        else:
            if st.button("🗑️ Clear All Cache", use_container_width=True, help="Clear all cached data from Redis (stock, news, sentiment, RAG)"):
                if redis_cache and redis_cache.client:
                    st.session_state.confirm_clear_cache = True
                    st.warning("⚠️ This will delete ALL cached data. Please confirm.")
                    st.rerun()
                else:
                    st.warning("Redis cache not available")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem; color: #7f8c8d; font-size: 0.85rem;'>
            Powered by Azure OpenAI<br>
            with RAG & Redis Caching
        </div>
        """,
        unsafe_allow_html=True
    )

# Load data if button clicked
if st.session_state.load_data and symbol:
    # Initialize cache status for current request
    cache_status = {
        'stock': {'hit': False, 'miss': False},
        'news': {'hit': False, 'miss': False},
        'sentiment': {'hits': 0, 'misses': 0}
    }
    # Update session state immediately so UI can show it
    st.session_state.cache_status = cache_status
    
    with st.spinner("🔄 Collecting data and analyzing sentiment..."):
        # Collect stock data and track cache status in real-time
        if redis_cache:
            redis_cache.last_tier_used = None  # Reset before call
        stock_data = collector.get_stock_price(symbol)
        if redis_cache:
            if redis_cache.last_tier_used == "Redis":
                cache_status['stock']['hit'] = True
            elif redis_cache.last_tier_used == "MISS" or redis_cache.last_tier_used is None:
                cache_status['stock']['miss'] = True
        # Update immediately after stock collection
        st.session_state.cache_status = cache_status
        
        # Collect news data and track cache status in real-time
        if redis_cache:
            redis_cache.last_tier_used = None  # Reset before call
        news_data = collector.get_news_headlines(symbol)
        if redis_cache:
            if redis_cache.last_tier_used == "Redis":
                cache_status['news']['hit'] = True
            elif redis_cache.last_tier_used == "MISS" or redis_cache.last_tier_used is None:
                cache_status['news']['miss'] = True
        # Update immediately after news collection
        st.session_state.cache_status = cache_status
        
        # Combine data
        data = {
            'price_data': stock_data,
            'news': news_data,
            'social_media': collector.get_reddit_sentiment_data(symbol)
        }
        
        st.session_state.data = data
        st.session_state.symbol = symbol
        
        # Store articles in RAG using batch processing (industry best practice)
        # Batch processing is 10-100x faster than individual API calls
        if rag_service and data['news']:
            rag_service.store_articles_batch(data['news'], symbol, batch_size=100)
        
        # Analyze sentiment with RAG context using parallel processing (industry best practice)
        # Parallel processing provides 5-10x performance improvement
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
        
        st.session_state.news_sentiments = news_sentiments
        st.session_state.social_sentiments = social_sentiments
        
        # Final update of cache status (already updated incrementally above)
        st.session_state.cache_status = cache_status
        
        st.session_state.load_data = False
        st.session_state.title_shown = False  # Show title again after loading
        # Force rerun to update UI with new cache status
        st.rerun()

# Create tabs with better styling
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Price Analysis",
    "📰 News & Sentiment",
    "🔧 Technical Analysis",
    "🤖 AI Insights"
])

data = st.session_state.data
news_sentiments = st.session_state.news_sentiments
social_sentiments = st.session_state.social_sentiments

if data is None:
    # Enhanced empty state
    st.markdown(
        """
        <div class='empty-state'>
            <h2 style='color: #7f8c8d; margin-bottom: 1rem;'>👆 Get Started</h2>
            <p style='font-size: 1.1rem; color: #95a5a6;'>
                Enter a stock symbol in the sidebar and click <strong>'Load Data'</strong> to begin analysis
            </p>
            <div style='margin-top: 2rem; padding: 2rem; background: #f8f9fa; border-radius: 10px;'>
                <h4 style='color: #34495e;'>💡 Popular Symbols</h4>
                <div style='display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 1rem;'>
                    <span style='padding: 0.5rem 1rem; background: white; border-radius: 6px; font-weight: 600;'>AAPL</span>
                    <span style='padding: 0.5rem 1rem; background: white; border-radius: 6px; font-weight: 600;'>MSFT</span>
                    <span style='padding: 0.5rem 1rem; background: white; border-radius: 6px; font-weight: 600;'>GOOGL</span>
                    <span style='padding: 0.5rem 1rem; background: white; border-radius: 6px; font-weight: 600;'>TSLA</span>
                    <span style='padding: 0.5rem 1rem; background: white; border-radius: 6px; font-weight: 600;'>AMZN</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Aggregate sentiment scores
    def aggregate_sentiments(sentiments):
        """Aggregate sentiment scores from multiple analyses."""
        if not sentiments:
            return {'positive': 0, 'negative': 0, 'neutral': 1}
        df = pd.DataFrame(sentiments)
        return df.mean().to_dict()

    news_agg = aggregate_sentiments(news_sentiments)
    social_agg = aggregate_sentiments(social_sentiments)
    
    current_symbol = st.session_state.get('symbol', symbol)
    price_data = data.get('price_data', {})
    company_name = price_data.get('company_name', current_symbol)

    # Tab 1: Overview - Enhanced design
    with tab1:
        # Hero section with key metrics
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h2 style='color: white; margin: 0;'>{company_name}</h2>
                        <p style='margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;'>{current_symbol}</p>
                    </div>
                    <div style='text-align: right;'>
                        <h1 style='color: white; margin: 0; border: none;'>${price_data.get('price', 0):.2f}</h1>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Key metrics in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            market_cap = price_data.get('market_cap', 0)
            if market_cap > 0:
                st.metric(
                    "Market Cap",
                    f"${market_cap/1e9:.2f}B",
                    help="Total market capitalization"
                )
            else:
                st.metric("Market Cap", "N/A")
        
        with col2:
            net_sentiment = news_agg['positive'] - news_agg['negative']
            delta_color = "normal" if abs(net_sentiment) < 0.1 else ("inverse" if net_sentiment < 0 else "normal")
            st.metric(
                "Net Sentiment",
                f"{net_sentiment:+.2%}",
                delta=f"{'Positive' if net_sentiment > 0 else 'Negative' if net_sentiment < 0 else 'Neutral'}",
                delta_color=delta_color,
                help="Overall sentiment score from news analysis"
            )
        
        with col3:
            st.metric(
                "Positive",
                f"{news_agg['positive']:.1%}",
                help="Positive sentiment percentage"
            )
        
        with col4:
            st.metric(
                "Negative",
                f"{news_agg['negative']:.1%}",
                help="Negative sentiment percentage"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sentiment visualization with enhanced design
        st.subheader("📊 Sentiment Breakdown")
        news_sentiment_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Score': [news_agg['positive'], news_agg['negative'], news_agg['neutral']]
        })
        
        fig_news = px.bar(
            news_sentiment_df,
            x='Sentiment',
            y='Score',
            labels={'Score': 'Score', 'Sentiment': 'Sentiment Type'},
            color='Sentiment',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#95a5a6'
            },
            text='Score'
        )
        fig_news.update_traces(
            texttemplate='%{text:.1%}',
            textposition='outside',
            marker_line_color='white',
            marker_line_width=2
        )
        fig_news.update_layout(
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            yaxis=dict(tickformat='.0%', range=[0, 1])
        )
        st.plotly_chart(fig_news, width='stretch', key="overview_sentiment_chart")
        
        # Quick insights
        st.markdown("---")
        st.subheader("💡 Quick Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            if news_agg['positive'] > 0.5:
                st.success(f"✅ **Strong Positive Sentiment** ({news_agg['positive']:.1%})")
            elif news_agg['negative'] > 0.5:
                st.error(f"⚠️ **Strong Negative Sentiment** ({news_agg['negative']:.1%})")
            else:
                st.info("⚖️ **Balanced Sentiment** - Mixed market perception")
        
        with insights_col2:
            if net_sentiment > 0.2:
                st.success("📈 **Bullish Outlook** - Positive market sentiment")
            elif net_sentiment < -0.2:
                st.warning("📉 **Bearish Outlook** - Negative market sentiment")
            else:
                st.info("📊 **Neutral Outlook** - Balanced market sentiment")
        
        # Social media note
        if not data.get('social_media'):
            st.info(
                "ℹ️ **Note:** Social media sentiment data requires API integration. "
                "Currently using news articles for sentiment analysis."
            )

    # Tab 2: Price Analysis - Enhanced
    with tab2:
        st.header(f"📈 Price Analysis - {current_symbol}")
        
        try:
            ticker = yf.Ticker(current_symbol)
            period = st.selectbox(
                "📅 Time Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=5,
                key="price_period"
            )
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # Enhanced metrics display
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
                    delta_color = "normal" if change >= 0 else "inverse"
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)",
                        delta_color=delta_color
                    )
                
                with metrics_col2:
                    high = hist['High'].max()
                    st.metric("52W High", f"${high:.2f}")
                
                with metrics_col3:
                    low = hist['Low'].min()
                    st.metric("52W Low", f"${low:.2f}")
                
                with metrics_col4:
                    volume = hist['Volume'].iloc[-1]
                    st.metric("Volume", f"{volume:,.0f}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Enhanced price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=3),
                    fill='tonexty',
                    fillcolor='rgba(31, 119, 180, 0.1)'
                ))
                fig.update_layout(
                    title=dict(
                        text=f"{current_symbol} Price Chart ({period})",
                        font=dict(size=20, color='#2c3e50')
                    ),
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch', key="price_chart")
                
                # Volume chart with better styling
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=hist.index,
                    y=hist['Volume'],
                    name='Volume',
                    marker_color='rgba(31, 119, 180, 0.6)',
                    marker_line_color='rgba(31, 119, 180, 0.8)',
                    marker_line_width=1
                ))
                fig_vol.update_layout(
                    title=dict(
                        text="Trading Volume",
                        font=dict(size=18, color='#2c3e50')
                    ),
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig_vol, width='stretch', key="volume_chart")
            else:
                st.warning("⚠️ No price data available for this symbol.")
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            st.error(f"❌ Error fetching price data: {e}")

    # Tab 3: News & Sentiment - Enhanced
    with tab3:
        st.header(f"📰 News & Sentiment Analysis")
        
        # Sentiment summary cards
        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
        
        with sentiment_col1:
            st.markdown(
                f"""
                <div style='background: #d4edda; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #155724; margin: 0;'>{news_agg['positive']:.1%}</h3>
                    <p style='color: #155724; margin: 0.5rem 0 0 0; font-weight: 600;'>Positive</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with sentiment_col2:
            st.markdown(
                f"""
                <div style='background: #f8d7da; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #721c24; margin: 0;'>{news_agg['negative']:.1%}</h3>
                    <p style='color: #721c24; margin: 0.5rem 0 0 0; font-weight: 600;'>Negative</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with sentiment_col3:
            st.markdown(
                f"""
                <div style='background: #e2e3e5; padding: 1.5rem; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #383d41; margin: 0;'>{news_agg['neutral']:.1%}</h3>
                    <p style='color: #383d41; margin: 0.5rem 0 0 0; font-weight: 600;'>Neutral</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sentiment over time
        if data.get('news'):
            st.subheader("📅 Sentiment Trend Over Time")
            news_df = pd.DataFrame(data['news'])
            news_df['sentiment'] = [s['positive'] - s['negative'] for s in news_sentiments]
            news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            news_df = news_df.sort_values('timestamp')
            
            fig_news_time = px.line(
                news_df,
                x='timestamp',
                y='sentiment',
                title='News Sentiment Over Time',
                markers=True,
                color_discrete_sequence=['#1f77b4']
            )
            fig_news_time.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Neutral"
            )
            fig_news_time.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_news_time, width='stretch', key="sentiment_trend")
        
        # News articles with enhanced design
        if data.get('news'):
            st.subheader("📰 Recent News Articles")
            st.markdown(f"*Showing {len(data['news'][:10])} of {len(data['news'])} articles*")
            
            for i, article in enumerate(data['news'][:10]):
                sentiment = news_sentiments[i] if i < len(news_sentiments) else {
                    'positive': 0, 'negative': 0, 'neutral': 1
                }
                
                # Determine sentiment badge
                if sentiment['positive'] > sentiment['negative'] and sentiment['positive'] > sentiment['neutral']:
                    badge_color = "#2ecc71"
                    badge_text = "🟢 Positive"
                elif sentiment['negative'] > sentiment['positive'] and sentiment['negative'] > sentiment['neutral']:
                    badge_color = "#e74c3c"
                    badge_text = "🔴 Negative"
                else:
                    badge_color = "#95a5a6"
                    badge_text = "⚪ Neutral"
                
                title = article.get('title', 'No title available')
                source = article.get('source', 'News Source')
                if source in ['Unknown', 'Unknown Source', '']:
                    source = 'News Source'
                
                # Enhanced article card
                with st.expander(
                    f"{badge_text} | {title[:60]}{'...' if len(title) > 60 else ''} | {source}",
                    expanded=False
                ):
                    # Sentiment metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Positive", f"{sentiment['positive']:.1%}")
                    with metric_col2:
                        st.metric("Negative", f"{sentiment['negative']:.1%}")
                    with metric_col3:
                        st.metric("Neutral", f"{sentiment['neutral']:.1%}")
                    
                    st.divider()
                    
                    # Article content
                    if title and title != 'No title available':
                        st.markdown(f"**Title:** {title}")
                    
                    summary = article.get('summary', '')
                    if summary:
                        st.markdown(f"**Summary:** {summary}")
                    
                    url = article.get('url', '')
                    if url:
                        st.markdown(f"🔗 [Read full article]({url})", unsafe_allow_html=True)
                    else:
                        st.info("No article link available")

    # Tab 4: Technical Analysis - Enhanced
    with tab4:
        st.header(f"🔧 Technical Analysis - {current_symbol}")
        
        try:
            ticker = yf.Ticker(current_symbol)
            period = st.selectbox(
                "📅 Analysis Period",
                ["1mo", "3mo", "6mo", "1y", "2y"],
                index=2,
                key="tech_period"
            )
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # Calculate technical indicators
                hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
                hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()
                
                # RSI calculation
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                hist['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
                hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
                
                # Enhanced price chart with moving averages
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='#ff9800', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='#e74c3c', width=2, dash='dash')
                ))
                fig.update_layout(
                    title=dict(
                        text=f"{current_symbol} Price with Moving Averages",
                        font=dict(size=20, color='#2c3e50')
                    ),
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, width='stretch', key="tech_price_ma")
                
                # RSI and MACD side by side
                tech_col1, tech_col2 = st.columns(2)
                
                with tech_col1:
                    st.subheader("📊 Relative Strength Index (RSI)")
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#9b59b6', width=2)
                    ))
                    fig_rsi.add_hline(
                        y=70,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Overbought (70)"
                    )
                    fig_rsi.add_hline(
                        y=30,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Oversold (30)"
                    )
                    fig_rsi.update_layout(
                        height=350,
                        yaxis=dict(range=[0, 100]),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False
                    )
                    st.plotly_chart(fig_rsi, width='stretch', key="rsi_chart")
                    
                    current_rsi = hist['RSI'].iloc[-1]
                    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Normal"
                    st.metric("Current RSI", f"{current_rsi:.2f}", rsi_status)
                
                with tech_col2:
                    st.subheader("📈 MACD Indicator")
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='#3498db', width=2)
                    ))
                    fig_macd.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='#e74c3c', width=2)
                    ))
                    fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_macd.update_layout(
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_macd, width='stretch', key="macd_chart")
                    
                    current_macd = hist['MACD'].iloc[-1]
                    current_signal = hist['Signal'].iloc[-1]
                    macd_trend = "Bullish" if current_macd > current_signal else "Bearish"
                    st.metric("MACD", f"{current_macd:.2f}", f"Signal: {current_signal:.2f} ({macd_trend})")
            else:
                st.warning("⚠️ No data available for technical analysis.")
        except Exception as e:
            logger.error(f"Error performing technical analysis: {e}")
            st.error(f"❌ Error performing technical analysis: {e}")

    # Tab 5: AI Insights - Enhanced
    with tab5:
        st.header(f"🤖 AI-Powered Insights")
        
        # Overall sentiment analysis with enhanced design
        net_sentiment = news_agg['positive'] - news_agg['negative']
        sentiment_label = (
            'Positive' if net_sentiment > 0.1
            else 'Negative' if net_sentiment < -0.1
            else 'Neutral'
        )
        perception = (
            'generally positive' if net_sentiment > 0.1
            else 'generally negative' if net_sentiment < -0.1
            else 'relatively neutral'
        )
        
        # Insight card
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;'>
                <h3 style='color: white; margin: 0 0 1rem 0;'>📊 Overall Sentiment Analysis</h3>
                <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 2rem; font-weight: 700;'>{news_agg['positive']:.1%}</div>
                        <div style='opacity: 0.9;'>Positive</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 2rem; font-weight: 700;'>{news_agg['negative']:.1%}</div>
                        <div style='opacity: 0.9;'>Negative</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 2rem; font-weight: 700;'>{news_agg['neutral']:.1%}</div>
                        <div style='opacity: 0.9;'>Neutral</div>
                    </div>
                </div>
                <div style='margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.3);'>
                    <p style='margin: 0; font-size: 1.1rem;'>
                        <strong>Overall Sentiment:</strong> {sentiment_label} ({net_sentiment:+.2%})
                    </p>
                    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
                        Market perception of {current_symbol} is {perception}.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Key insights with better design
        st.subheader("🔍 Key Insights")
        
        insights = []
        if news_agg['positive'] > 0.5:
            insights.append(("✅", f"News sentiment is predominantly positive ({news_agg['positive']:.1%})", "success"))
        elif news_agg['negative'] > 0.5:
            insights.append(("⚠️", f"News sentiment is predominantly negative ({news_agg['negative']:.1%})", "warning"))
        
        if net_sentiment > 0.2:
            insights.append(("📈", "Strong positive sentiment overall - Bullish outlook", "success"))
        elif net_sentiment < -0.2:
            insights.append(("📉", "Strong negative sentiment overall - Bearish outlook", "error"))
        else:
            insights.append(("⚖️", "Sentiment is relatively balanced", "info"))
        
        for icon, text, alert_type in insights:
            if alert_type == "success":
                st.success(f"{icon} {text}")
            elif alert_type == "warning":
                st.warning(f"{icon} {text}")
            elif alert_type == "error":
                st.error(f"{icon} {text}")
            else:
                st.info(f"{icon} {text}")
        
        # Sentiment breakdown chart
        st.subheader("📈 Detailed Sentiment Breakdown")
        news_sentiment_df = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Score': [news_agg['positive'], news_agg['negative'], news_agg['neutral']]
        })
        
        fig_comparison = px.bar(
            news_sentiment_df,
            x='Sentiment',
            y='Score',
            labels={'Score': 'Score', 'Sentiment': 'Sentiment Type'},
            color='Sentiment',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Negative': '#e74c3c',
                'Neutral': '#95a5a6'
            },
            text='Score'
        )
        fig_comparison.update_traces(
            texttemplate='%{text:.1%}',
            textposition='outside',
            marker_line_color='white',
            marker_line_width=2
        )
        fig_comparison.update_layout(
            title="News Sentiment Breakdown",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickformat='.0%', range=[0, 1])
        )
        st.plotly_chart(fig_comparison, width='stretch', key="ai_sentiment_breakdown")
