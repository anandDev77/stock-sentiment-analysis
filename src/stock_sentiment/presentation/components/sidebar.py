"""
Sidebar component for the Streamlit application.
"""

import streamlit as st
from typing import Optional, Any
from datetime import datetime, timedelta

from ...config.settings import get_settings
from ...utils.logger import get_logger

logger = get_logger(__name__)


def render_sidebar(
    redis_cache: Optional[Any],
    rag_service: Optional[Any],
    analyzer: Optional[Any],
    cost_tracker: Optional[Any],
    settings
) -> str:
    """
    Render the sidebar with all controls and status indicators.
    
    Args:
        redis_cache: RedisCache instance
        rag_service: RAGService instance
        analyzer: SentimentAnalyzer instance
        cost_tracker: CostTracker instance
        settings: Application settings
        
    Returns:
        Selected stock symbol
    """
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
        
        # Stock symbol input
        symbol = st.text_input(
            "📊 Stock Symbol",
            value=st.session_state.symbol,
            key="stock_symbol",
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        # System status indicators
        _render_system_status(redis_cache, rag_service)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Search filters
        _render_search_filters(settings)
        
        # Recent searches
        _render_recent_searches()
        
        # Load button
        if st.button("🚀 Load Data", type="primary", width='stretch'):
            st.session_state.load_data = True
            st.session_state.symbol = symbol
            st.session_state.title_shown = False
            
            # Add to recent searches
            if symbol and symbol not in st.session_state.recent_searches:
                st.session_state.recent_searches.append(symbol)
                # Keep only last 10
                if len(st.session_state.recent_searches) > 10:
                    st.session_state.recent_searches = st.session_state.recent_searches[-10:]
        
        st.markdown("---")
        
        # Cache status
        _render_cache_status(redis_cache)
        
        st.markdown("---")
        
        # Connection details
        _render_connection_details(redis_cache, rag_service, settings)
        
        st.markdown("---")
        
        # Performance metrics
        _render_performance_metrics(redis_cache, analyzer, cost_tracker)
        
        # Cache management
        _render_cache_management(redis_cache)
        
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
    
    return symbol


def _render_system_status(redis_cache: Optional[Any], rag_service: Optional[Any]):
    """Render system status indicators."""
    st.markdown("### 🔌 System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        # Check Redis connection
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
        # Check RAG service
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


def _render_search_filters(settings):
    """Render search filters section."""
    # Initialize search filters if not exists
    if 'search_filters' not in st.session_state:
        st.session_state.search_filters = {
            "date_range": None,
            "sources": None,
            "exclude_unknown": True,
            "days_back": None,
            "data_sources": {
                "yfinance": True,
                "alpha_vantage": settings.data_sources.alpha_vantage_enabled,
                "finnhub": settings.data_sources.finnhub_enabled,
                "reddit": settings.data_sources.reddit_enabled
            }
        }
    
    # Initialize data source filters based on settings
    if 'data_sources' not in st.session_state.search_filters:
        st.session_state.search_filters["data_sources"] = {
            "yfinance": True,
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


def _render_recent_searches():
    """Render recent searches section."""
    if st.session_state.recent_searches:
        st.markdown("### 🔍 Recent Searches")
        for sym in st.session_state.recent_searches[-5:]:  # Show last 5
            if st.button(sym, key=f"recent_{sym}", width='stretch'):
                st.session_state.symbol = sym
                st.session_state.load_data = True
                st.rerun()
        st.markdown("---")


def _render_cache_status(redis_cache: Optional[Any]):
    """Render cache status indicators."""
    if 'cache_status' in st.session_state and st.session_state.cache_status:
        cache_status = st.session_state.cache_status
        st.markdown("### 🔄 Cache Status (Last Request)")
        
        cache_col1, cache_col2 = st.columns(2)
        with cache_col1:
            if cache_status['stock']['hit']:
                st.success("✅ Stock Data: **CACHED**")
            elif cache_status['stock']['miss']:
                st.info("🔄 Stock Data: **FRESH** (from API)")
            else:
                st.warning("⚠️ Stock Data: Unknown status")
            
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


def _render_connection_details(redis_cache: Optional[Any], rag_service: Optional[Any], settings):
    """Render connection details for troubleshooting."""
    with st.expander("🔍 Connection Details", expanded=False):
        st.markdown("### Redis Connection")
        if redis_cache:
            if redis_cache.client:
                try:
                    redis_cache.client.ping()
                    st.success("✅ Redis: Connected and responding")
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


def _render_performance_metrics(redis_cache: Optional[Any], analyzer: Optional[Any], cost_tracker: Optional[Any]):
    """Render performance metrics section."""
    st.markdown("### 📊 Performance Metrics")
    
    # Get cache stats from Redis
    cache_stats = {'cache_hits': 0, 'cache_misses': 0, 'cache_sets': 0}
    if redis_cache and redis_cache.client:
        try:
            redis_cache.client.ping()
            cache_stats = redis_cache.get_cache_stats()
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
    
    # Get analyzer stats
    analyzer_stats = {'rag_uses': 0, 'rag_attempts': 0, 'total_requests': 0}
    if analyzer:
        try:
            if hasattr(analyzer, 'get_stats'):
                analyzer_stats = analyzer.get_stats()
            else:
                analyzer_stats = {
                    'rag_uses': getattr(analyzer, 'rag_uses', 0),
                    'rag_attempts': getattr(analyzer, 'rag_attempts', 0),
                    'cache_hits': getattr(analyzer, 'cache_hits', 0),
                    'cache_misses': getattr(analyzer, 'cache_misses', 0)
                }
        except Exception as e:
            logger.warning(f"Error getting analyzer stats: {e}")
    
    # Cache metrics
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
    
    # RAG uses
    rag_uses = analyzer_stats.get('rag_uses', 0)
    rag_attempts = analyzer_stats.get('rag_attempts', 0)
    rag_success_rate = (rag_uses / rag_attempts * 100) if rag_attempts > 0 else 0.0
    
    st.metric(
        "RAG Uses",
        rag_uses,
        delta=f"{rag_attempts} attempts ({rag_success_rate:.1f}% success)" if rag_attempts > 0 else "No attempts",
        delta_color="normal",
        help=f"RAG successfully used {rag_uses} times out of {rag_attempts} attempts."
    )
    
    # Cost tracking
    if cost_tracker:
        try:
            cost_summary = cost_tracker.get_cost_summary(days=7)
            st.markdown("### 💰 Cost Tracking (Last 7 Days)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cost", f"${cost_summary.get('total_cost', 0):.4f}", help="Total API costs in USD")
            with col2:
                st.metric("Avg Daily", f"${cost_summary.get('average_daily_cost', 0):.4f}", help="Average daily cost")
            with col3:
                st.metric("API Calls", f"{cost_summary.get('total_calls', 0):,}", help="Total API calls made")
        except Exception as e:
            logger.warning(f"Error displaying cost tracking: {e}")


def _render_cache_management(redis_cache: Optional[Any]):
    """Render cache management buttons."""
    cache_col1, cache_col2 = st.columns(2)
    with cache_col1:
        if st.button("🔄 Reset Cache Stats", width='stretch', help="Reset cache statistics counters (hits/misses)"):
            if redis_cache:
                redis_cache.reset_cache_stats()
                st.success("Cache statistics reset!")
                st.rerun()
    
    with cache_col2:
        if 'confirm_clear_cache' not in st.session_state:
            st.session_state.confirm_clear_cache = False
        
        if st.session_state.confirm_clear_cache:
            confirm_col1, confirm_col2 = st.columns(2)
            with confirm_col1:
                if st.button("✅ Confirm", width='stretch', type="primary"):
                    if redis_cache and redis_cache.client:
                        if redis_cache.clear_all_cache():
                            st.success("All cache data cleared!")
                            st.session_state.confirm_clear_cache = False
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
            if st.button("🗑️ Clear All Cache", width='stretch', help="Clear all cached data from Redis"):
                if redis_cache and redis_cache.client:
                    st.session_state.confirm_clear_cache = True
                    st.warning("⚠️ This will delete ALL cached data. Please confirm.")
                    st.rerun()
                else:
                    st.warning("Redis cache not available")

