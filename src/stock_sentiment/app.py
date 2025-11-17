"""
Stock Sentiment Analysis Dashboard - Streamlit Application

This is the main Streamlit application for the Stock Sentiment Analysis dashboard.
It provides an interactive interface for analyzing stock sentiment using AI.
"""

import streamlit as st

# Add src directory to Python path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from stock_sentiment.presentation.styles import apply_custom_styles
from stock_sentiment.presentation.initialization import (
    initialize_settings,
    initialize_services,
    initialize_session_state,
    setup_app
)
from stock_sentiment.presentation.components.sidebar import render_sidebar
from stock_sentiment.presentation.components.empty_state import render_empty_state
from stock_sentiment.presentation.data_loader import load_stock_data
from stock_sentiment.presentation.tabs import (
    render_overview_tab,
    render_price_analysis_tab,
    render_news_sentiment_tab,
    render_technical_analysis_tab,
    render_ai_insights_tab,
    render_comparison_tab
)

# Setup application
setup_app()
apply_custom_styles()

# Initialize settings and services
settings = initialize_settings()
redis_cache, rag_service, collector, cost_tracker, analyzer = initialize_services(settings)
initialize_session_state()

# Render sidebar and get selected symbol
symbol = render_sidebar(redis_cache, rag_service, analyzer, cost_tracker, settings)

# Load data if button clicked
if st.session_state.load_data and symbol:
    load_stock_data(symbol, collector, analyzer, rag_service, redis_cache, settings)

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

# Show active data sources and breakdown if data is available
data = st.session_state.data
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

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Price Analysis",
    "📰 News & Sentiment",
    "🔧 Technical Analysis",
    "🤖 AI Insights",
    "📊 Comparison"
])

# Get data from session state
news_sentiments = st.session_state.news_sentiments
social_sentiments = st.session_state.social_sentiments
current_symbol = st.session_state.get('symbol', symbol)

# Render tabs based on data availability
if data is None:
    render_empty_state()
else:
    # Render each tab
    with tab1:
        render_overview_tab(data, news_sentiments, social_sentiments, current_symbol)
    
    with tab2:
        render_price_analysis_tab(current_symbol)
    
    with tab3:
        render_news_sentiment_tab(data, news_sentiments, current_symbol)
    
    with tab4:
        render_technical_analysis_tab(current_symbol)
    
    with tab5:
        render_ai_insights_tab(news_sentiments, current_symbol)
    
    with tab6:
        render_comparison_tab(collector, analyzer, rag_service)
