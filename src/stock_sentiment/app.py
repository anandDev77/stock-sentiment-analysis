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
from stock_sentiment.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Sentiment Dashboard",
    page_icon="📈",
    layout="wide"
)

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
def get_analyzer(_settings, _cache, _rag_service):
    """Get sentiment analyzer instance."""
    try:
        return SentimentAnalyzer(
            settings=_settings,
            redis_cache=_cache,
            rag_service=_rag_service
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
analyzer = get_analyzer(settings, redis_cache, rag_service)

if analyzer is None:
    st.error("Failed to initialize sentiment analyzer. Please check your configuration.")
    st.stop()

# Streamlit app layout
st.title("📈 Stock Sentiment Dashboard")
st.markdown("Analyze stock sentiment using AI-powered sentiment analysis with Azure OpenAI")

# Sidebar for input
with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input(
        "Enter Stock Symbol (e.g., AAPL):",
        value="AAPL",
        key="stock_symbol"
    ).upper()
    
    st.info(
        "ℹ️ Using Azure OpenAI for sentiment analysis with RAG and Redis caching. "
        "Configure in .env file."
    )
    
    # Show cache status
    if redis_cache and redis_cache.client:
        st.success("✅ Redis cache connected")
    else:
        st.warning("⚠️ Redis cache not available - caching disabled")
    
    # Show RAG status
    if rag_service:
        if rag_service.embeddings_enabled:
            st.success("✅ RAG enabled (embeddings working)")
        else:
            st.warning("⚠️ RAG disabled (embedding model not available)")
    else:
        st.warning("⚠️ RAG service not available")
    
    if st.button("🔍 Load Data", type="primary", use_container_width=True):
        st.session_state.load_data = True
        st.session_state.symbol = symbol

# Add debug stats to sidebar after analyzer is initialized
with st.sidebar:
    st.divider()
    try:
        stats = analyzer.get_stats()
        st.subheader("📊 Performance Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cache Hits", stats.get('cache_hits', 0))
            st.metric("RAG Uses", stats.get('rag_uses', 0))
        with col2:
            st.metric("Cache Misses", stats.get('cache_misses', 0))
            total = stats.get('total_requests', 0)
            if total > 0:
                hit_rate = (stats.get('cache_hits', 0) / total) * 100
                st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
    except Exception:
        pass

# Initialize session state
if 'load_data' not in st.session_state:
    st.session_state.load_data = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'news_sentiments' not in st.session_state:
    st.session_state.news_sentiments = []
if 'social_sentiments' not in st.session_state:
    st.session_state.social_sentiments = []

# Load data if button clicked or symbol changed
if st.session_state.load_data and symbol:
    with st.spinner("Collecting data and analyzing sentiment..."):
        # Collect data
        data = collector.collect_all_data(symbol)
        st.session_state.data = data
        st.session_state.symbol = symbol
        
        # Store articles in RAG for future retrieval
        if rag_service and data['news']:
            for article in data['news']:
                rag_service.store_article(article, symbol)
        
        # Analyze sentiment with RAG context
        news_sentiments = []
        for article in data['news']:
            text_to_analyze = article.get('summary', article.get('title', ''))
            if text_to_analyze:
                # Pass symbol for RAG context retrieval
                news_sentiments.append(analyzer.analyze_sentiment(text_to_analyze, symbol=symbol))
            else:
                news_sentiments.append({'positive': 0, 'negative': 0, 'neutral': 1})

        social_sentiments = []
        for post in data['social_media']:
            text_to_analyze = post.get('text', '')
            if text_to_analyze:
                social_sentiments.append(analyzer.analyze_sentiment(text_to_analyze, symbol=symbol))
            else:
                social_sentiments.append({'positive': 0, 'negative': 0, 'neutral': 1})
        
        st.session_state.news_sentiments = news_sentiments
        st.session_state.social_sentiments = social_sentiments
        st.session_state.load_data = False

# Create tabs
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
    st.info("👆 Enter a stock symbol and click 'Load Data' to get started")
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

    # Tab 1: Overview
    with tab1:
        st.header(f"📊 Overview - {current_symbol}")
        
        # Stock info and metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Price", f"${data['price_data']['price']:.2f}")
        
        with col2:
            market_cap = data['price_data']['market_cap']
            if market_cap > 0:
                st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
            else:
                st.metric("Market Cap", "N/A")

        with col3:
            # Calculate net sentiment from news only
            net_sentiment = news_agg['positive'] - news_agg['negative']
            st.metric("Net Sentiment", f"{net_sentiment:.2%}")

        with col4:
            company_name = data['price_data'].get('company_name', current_symbol)
            st.metric("Company", company_name)
        
        st.divider()
        
        # Quick sentiment overview
        st.subheader("News Sentiment Summary")
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
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        )
        fig_news.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_news, width='stretch', key="overview_news_sentiment")
        
        # Social media data not available
        if not data['social_media']:
            st.info(
                "ℹ️ Social media sentiment data is not available. We only use free APIs. "
                "To add social media data, integrate with Reddit API (PRAW) or Twitter API."
            )

    # Tab 2: Price Analysis
    with tab2:
        st.header(f"📈 Price Analysis - {current_symbol}")
        
        try:
            ticker = yf.Ticker(current_symbol)
            period = st.selectbox(
                "Time Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=5,
                key="price_period"
            )
            hist = ticker.history(period=period)
            
            if not hist.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{change:+.2f} ({change_pct:+.2f}%)"
                    )
                
                with col2:
                    high = hist['High'].max()
                    st.metric("52W High", f"${high:.2f}")
                
                with col3:
                    low = hist['Low'].min()
                    st.metric("52W Low", f"${low:.2f}")
                
                with col4:
                    volume = hist['Volume'].iloc[-1]
                    st.metric("Volume", f"{volume:,.0f}")
                
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(
                    title=f"{current_symbol} Price Chart ({period})",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, width='stretch', key="price_chart")
                
                # Volume chart
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=hist.index,
                    y=hist['Volume'],
                    name='Volume',
                    marker_color='rgba(31, 119, 180, 0.5)'
                ))
                fig_vol.update_layout(
                    title="Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=300
                )
                st.plotly_chart(fig_vol, width='stretch', key="volume_chart")
            else:
                st.warning("No price data available for this symbol.")
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            st.error(f"Error fetching price data: {e}")

    # Tab 3: News & Sentiment
    with tab3:
        st.header(f"📰 News & Sentiment - {current_symbol}")
        
        # Sentiment breakdown
        st.subheader("News Sentiment")
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
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        )
        fig_news.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_news, width='stretch', key="news_sentiment_breakdown")
        st.caption(
            f"Positive: {news_agg['positive']:.2%} | "
            f"Negative: {news_agg['negative']:.2%} | "
            f"Neutral: {news_agg['neutral']:.2%}"
        )
        
        # Social media sentiment not available
        if not data['social_media']:
            st.info(
                "ℹ️ Social media sentiment data is not available. We only use free APIs. "
                "To add social media data, integrate with Reddit API (PRAW) or Twitter API."
            )

        # Sentiment over time
        st.subheader("📅 Sentiment Over Time")

        if data['news']:
            news_df = pd.DataFrame(data['news'])
            news_df['sentiment'] = [s['positive'] - s['negative'] for s in news_sentiments]
            news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            news_df = news_df.sort_values('timestamp')

            fig_news_time = px.line(
                news_df,
                x='timestamp',
                y='sentiment',
                title='News Sentiment Over Time',
                markers=True
            )
            fig_news_time.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_news_time.update_layout(height=400)
            st.plotly_chart(fig_news_time, width='stretch', key="news_sentiment_time")

        # Detailed news articles
        if data['news']:
            st.subheader("📰 Recent News Articles")
            for i, article in enumerate(data['news'][:10]):
                # Determine sentiment label
                sentiment = news_sentiments[i] if i < len(news_sentiments) else {
                    'positive': 0, 'negative': 0, 'neutral': 1
                }
                
                if sentiment['positive'] > sentiment['negative'] and sentiment['positive'] > sentiment['neutral']:
                    sentiment_label = "🟢 Positive"
                elif sentiment['negative'] > sentiment['positive'] and sentiment['negative'] > sentiment['neutral']:
                    sentiment_label = "🔴 Negative"
                else:
                    sentiment_label = "⚪ Neutral"
                
                # Get article data
                title = article.get('title', 'No title available')
                source = article.get('source', 'Unknown Source')
                if source == 'Unknown' or not source or source == 'Unknown Source':
                    source = 'News Source'
                
                # Truncate title for expander header if too long
                display_title = title
                if len(display_title) > 60:
                    display_title = display_title[:60] + "..."
                
                with st.expander(f"{sentiment_label} | {display_title} | {source}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive", f"{sentiment['positive']:.1%}")
                    with col2:
                        st.metric("Negative", f"{sentiment['negative']:.1%}")
                    with col3:
                        st.metric("Neutral", f"{sentiment['neutral']:.1%}")
                    
                    st.divider()
                    
                    # Show full title
                    if title and title != 'No title available':
                        st.write(f"**Title:** {title}")
                    
                    # Show summary if available
                    summary = article.get('summary', '')
                    if summary:
                        st.write(f"**Summary:** {summary}")
                    elif not title or title == 'No title available':
                        st.write("No summary or title available.")
                    
                    # Show link if URL is available
                    url = article.get('url', '')
                    if url:
                        st.markdown(f"🔗 [Read full article]({url})")
                    else:
                        st.info("No article link available")

    # Tab 4: Technical Analysis
    with tab4:
        st.header(f"🔧 Technical Analysis - {current_symbol}")
        
        try:
            ticker = yf.Ticker(current_symbol)
            period = st.selectbox(
                "Time Period",
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
                
                # Price chart with moving averages
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1, dash='dash')
                ))
                fig.update_layout(
                    title=f"{current_symbol} Price with Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400
                )
                st.plotly_chart(fig, width='stretch', key="tech_price_ma")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
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
                        title="Relative Strength Index (RSI)",
                        xaxis_title="Date",
                        yaxis_title="RSI",
                        height=300,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_rsi, width='stretch', key="rsi_chart")
                    
                    current_rsi = hist['RSI'].iloc[-1]
                    st.metric("Current RSI", f"{current_rsi:.2f}")
                
                with col2:
                    # MACD Chart
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ))
                    fig_macd.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    ))
                    fig_macd.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_macd.update_layout(
                        title="MACD (Moving Average Convergence Divergence)",
                        xaxis_title="Date",
                        yaxis_title="MACD",
                        height=300
                    )
                    st.plotly_chart(fig_macd, width='stretch', key="macd_chart")
                    
                    current_macd = hist['MACD'].iloc[-1]
                    current_signal = hist['Signal'].iloc[-1]
                    st.metric("MACD", f"{current_macd:.2f}", f"Signal: {current_signal:.2f}")
                
            else:
                st.warning("No data available for technical analysis.")
        except Exception as e:
            logger.error(f"Error performing technical analysis: {e}")
            st.error(f"Error performing technical analysis: {e}")

    # Tab 5: AI Insights
    with tab5:
        st.header(f"🤖 AI Insights - {current_symbol}")
        
        # Overall AI-generated insights
        st.subheader("📊 Overall Sentiment Analysis")
        
        # Use only news data
        overall_positive = news_agg['positive']
        overall_negative = news_agg['negative']
        overall_neutral = news_agg['neutral']
        net_sentiment = overall_positive - overall_negative
        
        # Generate AI insight summary
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
        
        insight_text = f"""
        Based on the sentiment analysis of news articles for {current_symbol}:
        
        - **Overall Sentiment**: {sentiment_label}
        - **Net Sentiment Score**: {net_sentiment:.2%}
        - **Positive Sentiment**: {overall_positive:.2%}
        - **Negative Sentiment**: {overall_negative:.2%}
        - **Neutral Sentiment**: {overall_neutral:.2%}
        
        The sentiment analysis indicates that the market perception of {current_symbol} is {perception}.
        """
        
        st.markdown(insight_text)
        
        # Sentiment visualization
        st.subheader("📈 News Sentiment Breakdown")
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
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
        )
        fig_comparison.update_layout(
            title="News Sentiment Breakdown",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_comparison, width='stretch', key="sentiment_comparison")
        
        # Note about social media
        if not data['social_media']:
            st.info(
                "ℹ️ Social media sentiment data is not available. We only use free APIs. "
                "To add social media data, integrate with Reddit API (PRAW) or Twitter API."
            )
        
        # Key insights
        st.subheader("🔍 Key Insights")
        
        insights = []
        if news_agg['positive'] > 0.5:
            insights.append(f"✅ News sentiment is predominantly positive ({news_agg['positive']:.2%})")
        elif news_agg['negative'] > 0.5:
            insights.append(f"⚠️ News sentiment is predominantly negative ({news_agg['negative']:.2%})")
        
        if net_sentiment > 0.2:
            insights.append("📈 Strong positive sentiment overall")
        elif net_sentiment < -0.2:
            insights.append("📉 Strong negative sentiment overall")
        
        if insights:
            for insight in insights:
                st.write(insight)
        else:
            st.info("Sentiment is relatively balanced.")

