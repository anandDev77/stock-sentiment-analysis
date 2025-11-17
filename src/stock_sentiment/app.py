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
import time
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
from stock_sentiment.utils.ui_helpers import (
    show_toast, filter_articles, get_error_recovery_ui,
    generate_comparison_insights
)

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
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []
if 'data_errors' not in st.session_state:
    st.session_state.data_errors = {}
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False
if 'comparison_stocks' not in st.session_state:
    st.session_state.comparison_stocks = []

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
    
    # Initialize search filters if not exists
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {
            "date_range": None,
            "sources": None,
            "exclude_unknown": True,
            "days_back": None,
            "data_sources": {
                "yfinance": True,  # Always enabled (primary source)
                "alpha_vantage": True,  # Default to enabled if configured
                "finnhub": True,  # Default to enabled if configured
                "reddit": False  # Default to disabled
            }
        }
    
    # Initialize data source filters based on settings
    settings = get_settings()
    if 'data_sources' not in st.session_state.search_filters:
        st.session_state.search_filters["data_sources"] = {
            "yfinance": True,  # Always enabled
            "alpha_vantage": settings.data_sources.alpha_vantage_enabled,
            "finnhub": settings.data_sources.finnhub_enabled,
            "reddit": settings.data_sources.reddit_enabled
        }
    
    # Search Filters Section
    with st.expander("🔍 Search Filters", expanded=False):
        # Date Range Filter
        st.subheader("📅 Date Range")
        use_date_filter = st.checkbox("Filter by date", value=False, key="use_date_filter")
        
        date_range = None
        days_back = None
        date_option = None
        
        if use_date_filter:
            date_option = st.radio(
                "Date range",
                ["Last 3 days", "Last 7 days", "Last 30 days", "Custom range"],
                horizontal=False,
                key="date_option"
            )
            
            if date_option == "Custom range":
                from datetime import datetime, timedelta
                start_date = st.date_input(
                    "Start date",
                    value=datetime.now().date() - timedelta(days=7),
                    key="start_date"
                )
                end_date = st.date_input(
                    "End date",
                    value=datetime.now().date(),
                    key="end_date"
                )
                if start_date and end_date:
                    date_range = (
                        datetime.combine(start_date, datetime.min.time()),
                        datetime.combine(end_date, datetime.max.time())
                    )
            else:
                days_map = {"Last 3 days": 3, "Last 7 days": 7, "Last 30 days": 30}
                days_back = days_map[date_option]
        
        # Source Filter
        st.subheader("📰 News Sources")
        use_source_filter = st.checkbox("Filter by source", value=False, key="use_source_filter")
        
        selected_sources = None
        exclude_unknown = st.checkbox("Exclude 'Unknown' sources", value=True, key="exclude_unknown")
        
        if use_source_filter:
            source_options = ["Trusted Only", "Custom"]
            source_choice = st.radio(
                "Source filter",
                source_options,
                horizontal=False,
                key="source_choice"
            )
            
            if source_choice == "Trusted Only":
                selected_sources = ["Reuters", "Bloomberg", "CNBC", "Wall Street Journal"]
            elif source_choice == "Custom":
                # Get available sources from previous data if available
                available_sources = ["Reuters", "Bloomberg", "CNBC", "Wall Street Journal", "Yahoo Finance", "MarketWatch"]
                if st.session_state.data and 'news' in st.session_state.data:
                    news_sources = set()
                    for article in st.session_state.data.get('news', []):
                        source = article.get('source', '')
                        if source and source != 'Unknown':
                            news_sources.add(source)
                    if news_sources:
                        available_sources = sorted(list(news_sources))
                
                selected_sources = st.multiselect(
                    "Select sources",
                    available_sources,
                    default=available_sources[:3] if available_sources else [],
                    key="selected_sources"
                )
        
        # Data Source Filter
        st.subheader("📡 Data Sources")
        st.markdown("Enable/disable data sources to see how sentiment varies by source type")
        
        data_source_filters = st.session_state.search_filters.get("data_sources", {
            "yfinance": True,
            "alpha_vantage": settings.data_sources.alpha_vantage_enabled,
            "finnhub": settings.data_sources.finnhub_enabled,
            "reddit": settings.data_sources.reddit_enabled
        })
        
        # Show available sources based on configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # yfinance is always enabled (primary source)
            st.checkbox(
                "Yahoo Finance (yfinance)",
                value=True,
                disabled=True,
                help="Primary data source - always enabled"
            )
            
            alpha_vantage_available = settings.data_sources.alpha_vantage_enabled and settings.data_sources.alpha_vantage_api_key
            data_source_filters["alpha_vantage"] = st.checkbox(
                "Alpha Vantage",
                value=data_source_filters.get("alpha_vantage", alpha_vantage_available),
                disabled=not alpha_vantage_available,
                help="Financial news API (500 calls/day free tier)" if alpha_vantage_available else "Not configured - set DATA_SOURCE_ALPHA_VANTAGE_API_KEY in .env",
                key="filter_alpha_vantage"
            )
        
        with col2:
            finnhub_available = settings.data_sources.finnhub_enabled and settings.data_sources.finnhub_api_key
            data_source_filters["finnhub"] = st.checkbox(
                "Finnhub",
                value=data_source_filters.get("finnhub", finnhub_available),
                disabled=not finnhub_available,
                help="Company news API (60 calls/minute free tier)" if finnhub_available else "Not configured - set DATA_SOURCE_FINNHUB_API_KEY in .env",
                key="filter_finnhub"
            )
            
            reddit_available = settings.data_sources.reddit_enabled and settings.data_sources.reddit_client_id
            data_source_filters["reddit"] = st.checkbox(
                "Reddit",
                value=data_source_filters.get("reddit", reddit_available),
                disabled=not reddit_available,
                help="Social media sentiment from Reddit posts" if reddit_available else "Not configured - set DATA_SOURCE_REDDIT_CLIENT_ID in .env",
                key="filter_reddit"
            )
        
        # Store filters in session state
        st.session_state.search_filters = {
            "date_range": date_range if use_date_filter else None,
            "sources": selected_sources if use_source_filter else None,
            "exclude_unknown": exclude_unknown,
            "days_back": days_back if use_date_filter and date_option and date_option != "Custom range" else None,
            "data_sources": data_source_filters
        }
    
    # Recent searches
    if st.session_state.recent_searches:
        st.markdown("### 🔍 Recent Searches")
        for sym in st.session_state.recent_searches[-5:]:  # Show last 5
            if st.button(sym, key=f"recent_{sym}", width='stretch'):
                st.session_state.symbol = sym
                st.session_state.load_data = True
                st.rerun()
        st.markdown("---")
    
    # Enhanced load button
    if st.button("🚀 Load Data", type="primary", width='stretch'):
        st.session_state.load_data = True
        st.session_state.symbol = symbol
        st.session_state.title_shown = False  # Reset to show title again after load
        
        # Add to recent searches
        if symbol and symbol not in st.session_state.recent_searches:
            st.session_state.recent_searches.append(symbol)
            # Keep only last 10
            if len(st.session_state.recent_searches) > 10:
                st.session_state.recent_searches = st.session_state.recent_searches[-10:]
    
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
        if st.button("🔄 Reset Cache Stats", width='stretch', help="Reset cache statistics counters (hits/misses)"):
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
                if st.button("✅ Confirm", width='stretch', type="primary"):
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
                if st.button("❌ Cancel", width='stretch'):
                    st.session_state.confirm_clear_cache = False
                    st.rerun()
        else:
            if st.button("🗑️ Clear All Cache", width='stretch', help="Clear all cached data from Redis (stock, news, sentiment, RAG)"):
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
        # This method handles all sources and respects the filters
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
        
        # Store articles in RAG using batch processing (industry best practice)
        # Batch processing is 10-100x faster than individual API calls
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
        
        # Final update of cache status (already updated incrementally above)
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
    
    st.session_state.load_data = False
    st.session_state.title_shown = False  # Show title again after loading
    # Force rerun to update UI with new cache status
    st.rerun()

# Create tabs with better styling - including comparison tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Price Analysis",
    "📰 News & Sentiment",
    "🔧 Technical Analysis",
    "🤖 AI Insights",
    "📊 Comparison"
])

data = st.session_state.data
news_sentiments = st.session_state.news_sentiments
social_sentiments = st.session_state.social_sentiments

# Show active data sources and breakdown if data is available
if data is not None:
    
    # Show active data sources indicator
    if 'search_filters' in st.session_state and 'data_sources' in st.session_state.search_filters:
        active_sources = [k.replace('_', ' ').title() for k, v in st.session_state.search_filters['data_sources'].items() if v]
        if active_sources:
            st.info(f"📡 **Active Data Sources:** {', '.join(active_sources)}")
    
    # Show data source breakdown if available
    if data.get('news'):
        source_counts = {}
        for article in data['news']:
            source = article.get('source', 'Unknown')
            if 'Alpha Vantage' in source:
                source_counts['Alpha Vantage'] = source_counts.get('Alpha Vantage', 0) + 1
            elif 'Finnhub' in source:
                source_counts['Finnhub'] = source_counts.get('Finnhub', 0) + 1
            elif 'Reddit' in source or 'r/' in source:
                source_counts['Reddit'] = source_counts.get('Reddit', 0) + 1
            else:
                source_counts['Yahoo Finance'] = source_counts.get('Yahoo Finance', 0) + 1
        
        if len(source_counts) > 1:  # Only show if multiple sources
            st.markdown("#### 📊 Articles by Source")
            breakdown_cols = st.columns(len(source_counts))
            for idx, (source, count) in enumerate(source_counts.items()):
                with breakdown_cols[idx]:
                    st.metric(f"{source}", count)
            st.markdown("---")

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
            yaxis=dict(tickformat='.0%', range=[0, 1]),
            clickmode='event+select'  # Enable click events for drill-down
        )
        
        # Enhanced hover template
        fig_news.update_traces(
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1%}<extra></extra>"
        )
        
        st.plotly_chart(fig_news, width='stretch', key="overview_sentiment_chart", on_select="rerun")
        
        # Show drill-down info if chart is clicked
        if fig_news.data and hasattr(st.session_state, 'selected_chart_data'):
            selected_data = st.session_state.get('selected_chart_data')
            if selected_data:
                st.info(f"Selected: {selected_data}")
        
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
            
            # Normalize timestamps to handle both timezone-aware and naive datetimes
            def normalize_timestamp(ts):
                """Normalize timestamp to naive datetime for pandas compatibility."""
                if isinstance(ts, datetime):
                    if ts.tzinfo is not None:
                        # Convert timezone-aware to UTC, then make naive
                        from datetime import timezone as tz
                        return ts.astimezone(tz.utc).replace(tzinfo=None)
                    return ts
                return ts
            
            # Normalize all timestamps before pandas conversion
            news_df['timestamp'] = news_df['timestamp'].apply(normalize_timestamp)
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
            
            # Search and filter section
            search_col1, search_col2, search_col3 = st.columns([3, 1, 1])
            
            with search_col1:
                search_query = st.text_input(
                    "🔍 Search articles...",
                    key="article_search",
                    placeholder="Search by title, source, or content",
                    help="Filter articles by keywords"
                )
            
            with search_col2:
                sentiment_filter = st.selectbox(
                    "Sentiment",
                    ["All", "Positive", "Negative", "Neutral"],
                    key="sentiment_filter",
                    help="Filter by sentiment"
                )
            
            with search_col3:
                sort_option = st.selectbox(
                    "Sort by",
                    ["Date (Newest)", "Date (Oldest)", "Sentiment (Positive)", "Sentiment (Negative)", "Source"],
                    key="sort_option"
                )
            
            # Get unique sources for filter
            unique_sources = list(set([article.get('source', 'Unknown') for article in data['news']]))
            source_filter = st.multiselect(
                "Filter by Source",
                options=unique_sources,
                key="source_filter",
                help="Select sources to display"
            )
            
            # Apply filters
            filtered_articles = filter_articles(
                data['news'],
                search_query=search_query if search_query else None,
                sentiment_filter=sentiment_filter.lower() if sentiment_filter != "All" else None,
                source_filter=source_filter if source_filter else None,
                sentiments=news_sentiments
            )
            
            # Sort articles
            if sort_option == "Date (Newest)":
                filtered_articles.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
            elif sort_option == "Date (Oldest)":
                filtered_articles.sort(key=lambda x: x.get('timestamp', datetime.min))
            elif sort_option == "Sentiment (Positive)":
                # Sort by positive sentiment score
                filtered_indices = [data['news'].index(a) for a in filtered_articles if a in data['news']]
                filtered_articles.sort(
                    key=lambda x: news_sentiments[data['news'].index(x)]['positive'] if x in data['news'] else 0,
                    reverse=True
                )
            elif sort_option == "Sentiment (Negative)":
                filtered_articles.sort(
                    key=lambda x: news_sentiments[data['news'].index(x)]['negative'] if x in data['news'] else 0,
                    reverse=True
                )
            elif sort_option == "Source":
                filtered_articles.sort(key=lambda x: x.get('source', ''))
            
            # Update news_sentiments to match filtered articles
            filtered_sentiments = []
            for article in filtered_articles:
                if article in data['news']:
                    idx = data['news'].index(article)
                    filtered_sentiments.append(news_sentiments[idx] if idx < len(news_sentiments) else {'positive': 0, 'negative': 0, 'neutral': 1})
                else:
                    filtered_sentiments.append({'positive': 0, 'negative': 0, 'neutral': 1})
            
            # Pagination controls
            total_articles = len(filtered_articles)
            articles_per_page = settings.app.ui_articles_per_page
            
            # Initialize session state for pagination
            if 'article_page' not in st.session_state:
                st.session_state.article_page = 1
            if 'show_all_articles' not in st.session_state:
                st.session_state.show_all_articles = False
            
            # Reset page if filters changed
            if search_query or sentiment_filter != "All" or source_filter:
                st.session_state.article_page = 1
            
            # Pagination controls
            pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([2, 1, 1, 1])
            
            with pag_col1:
                show_all = st.checkbox(
                    "Show All Articles",
                    value=st.session_state.show_all_articles,
                    key="show_all_articles_checkbox",
                    help="Display all articles at once (may be slow for large lists)"
                )
                st.session_state.show_all_articles = show_all
            
            if not show_all and total_articles > articles_per_page:
                total_pages = (total_articles + articles_per_page - 1) // articles_per_page
                
                with pag_col2:
                    if st.button("◀ Previous", disabled=st.session_state.article_page <= 1):
                        st.session_state.article_page = max(1, st.session_state.article_page - 1)
                        st.rerun()
                
                with pag_col3:
                    st.markdown(f"**Page {st.session_state.article_page} of {total_pages}**")
                
                with pag_col4:
                    if st.button("Next ▶", disabled=st.session_state.article_page >= total_pages):
                        st.session_state.article_page = min(total_pages, st.session_state.article_page + 1)
                        st.rerun()
            
            # Determine which articles to display
            if show_all:
                articles_to_display = filtered_articles
                sentiments_to_display = filtered_sentiments
                start_idx = 0
                end_idx = total_articles
            else:
                start_idx = (st.session_state.article_page - 1) * articles_per_page
                end_idx = min(start_idx + articles_per_page, total_articles)
                articles_to_display = filtered_articles[start_idx:end_idx]
                sentiments_to_display = filtered_sentiments[start_idx:end_idx]
            
            # Display article count
            if search_query or sentiment_filter != "All" or source_filter:
                st.info(f"🔍 Found {total_articles} article(s) matching your filters")
            
            if show_all:
                st.markdown(f"*Showing all {total_articles} articles*")
            else:
                st.markdown(f"*Showing articles {start_idx + 1}-{end_idx} of {total_articles}*")
            
            # Display articles
            for i, article in enumerate(articles_to_display):
                # Use filtered sentiments
                sentiment = sentiments_to_display[i] if i < len(sentiments_to_display) else {
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
    
    # Tab 6: Stock Comparison - Enhanced with AI Insights
    with tab6:
        st.header("📊 Stock Comparison")
        st.markdown("Compare multiple stocks side-by-side with AI-powered insights")
        
        # Initialize comparison state
        if 'comparison_stocks' not in st.session_state:
            st.session_state.comparison_stocks = []
        if 'comparison_data' not in st.session_state:
            st.session_state.comparison_data = {}
        if 'comparison_sentiments' not in st.session_state:
            st.session_state.comparison_sentiments = {}
        if 'comparison_insights' not in st.session_state:
            st.session_state.comparison_insights = None
        
        current_symbol = st.session_state.get('symbol', 'AAPL')
        
        # Stock selector
        col1, col2 = st.columns([3, 1])
        with col1:
            compare_stocks = st.multiselect(
                "Select stocks to compare (2-5 recommended)",
                options=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC'] + 
                        [s for s in st.session_state.recent_searches if s not in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']],
                default=[current_symbol] if current_symbol else [],
                key="compare_stocks_select",
                help="Select 2 or more stocks to compare their sentiment and performance"
            )
        
        with col2:
            if st.button("🔄 Compare", type="primary", width='stretch'):
                if len(compare_stocks) < 2:
                    st.warning("⚠️ Please select at least 2 stocks to compare")
                else:
                    st.session_state.comparison_stocks = compare_stocks
                    st.session_state.comparison_data = {}
                    st.session_state.comparison_sentiments = {}
                    st.session_state.comparison_insights = None
                    
                    # Collect data for all stocks
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_stocks = len(compare_stocks)
                    for idx, sym in enumerate(compare_stocks):
                        try:
                            status_text.text(f"📊 Analyzing {sym} ({idx+1}/{total_stocks})...")
                            progress_bar.progress((idx + 1) / total_stocks)
                            
                            comp_data = collector.collect_all_data(sym)
                            comp_texts = [a.get('summary', a.get('title', '')) for a in comp_data.get('news', [])]
                            
                            if comp_texts:
                                comp_sentiments = analyzer.batch_analyze(comp_texts, symbol=sym)
                                
                                # Store articles in RAG for future context
                                if rag_service and comp_data.get('news'):
                                    logger.info(f"Storing {len(comp_data['news'])} articles in RAG for {sym}")
                                    rag_service.store_articles_batch(comp_data['news'], sym)
                                
                                # Aggregate sentiment
                                if comp_sentiments:
                                    df = pd.DataFrame(comp_sentiments)
                                    st.session_state.comparison_data[sym] = comp_data
                                    st.session_state.comparison_sentiments[sym] = {
                                        'positive': float(df['positive'].mean()),
                                        'negative': float(df['negative'].mean()),
                                        'neutral': float(df['neutral'].mean())
                                    }
                        except Exception as e:
                            logger.error(f"Error comparing {sym}: {e}")
                            st.warning(f"⚠️ Failed to load data for {sym}: {str(e)}")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Generate AI insights
                    if st.session_state.comparison_data and st.session_state.comparison_sentiments:
                        status_text.text("🤖 Generating AI comparison insights...")
                        st.session_state.comparison_insights = generate_comparison_insights(
                            st.session_state.comparison_data,
                            st.session_state.comparison_sentiments,
                            analyzer
                        )
                        status_text.empty()
                        show_toast("✅ Comparison complete!", "success")
                        st.rerun()
        
        # Display comparison if data available
        if st.session_state.get('comparison_data') and st.session_state.get('comparison_sentiments'):
            st.markdown("---")
            
            # AI Insights Section
            if st.session_state.comparison_insights:
                st.subheader("🤖 AI-Powered Comparison Insights")
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;'>
                        <div style='white-space: pre-wrap;'>{st.session_state.comparison_insights}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("---")
            
            # Comparison metrics
            comp_df = pd.DataFrame([
                {
                    'Symbol': sym,
                    'Price': comp_data.get('price_data', {}).get('price', 0),
                    'Market Cap': comp_data.get('price_data', {}).get('market_cap', 0) / 1e9,  # In billions
                    'Positive': sent['positive'],
                    'Negative': sent['negative'],
                    'Neutral': sent['neutral'],
                    'Net Sentiment': sent['positive'] - sent['negative']
                }
                for sym, comp_data in st.session_state.comparison_data.items()
                if sym in st.session_state.comparison_sentiments
                for sent in [st.session_state.comparison_sentiments[sym]]
            ])
            
            if not comp_df.empty:
                # Comparison chart
                st.subheader("📊 Sentiment Comparison Chart")
                fig_comp = px.bar(
                    comp_df,
                    x='Symbol',
                    y=['Positive', 'Negative', 'Neutral'],
                    title="Sentiment Comparison Across Stocks",
                    barmode='group',
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Negative': '#e74c3c',
                        'Neutral': '#95a5a6'
                    }
                )
                fig_comp.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Sentiment Score",
                    xaxis_title="Stock Symbol",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_comp, width='stretch', key="comparison_chart")
                
                # Net Sentiment Comparison
                st.subheader("📈 Net Sentiment Comparison")
                fig_net = px.bar(
                    comp_df,
                    x='Symbol',
                    y='Net Sentiment',
                    title="Net Sentiment (Positive - Negative)",
                    color='Net Sentiment',
                    color_continuous_scale=['#e74c3c', '#95a5a6', '#2ecc71']
                )
                fig_net.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Net Sentiment",
                    xaxis_title="Stock Symbol",
                    showlegend=False
                )
                fig_net.update_traces(
                    texttemplate='%{y:+.2%}',
                    textposition='outside'
                )
                st.plotly_chart(fig_net, width='stretch', key="net_sentiment_chart")
                
                # Comparison table
                st.subheader("📋 Detailed Comparison Table")
                st.dataframe(
                    comp_df.style.format({
                        'Price': '${:.2f}',
                        'Market Cap': '${:.2f}B',
                        'Positive': '{:.2%}',
                        'Negative': '{:.2%}',
                        'Neutral': '{:.2%}',
                        'Net Sentiment': '{:+.2%}'
                    }),
                    width='stretch',
                    height=400
                )
        else:
            st.info("👆 Select stocks above and click 'Compare' to see the comparison analysis")
