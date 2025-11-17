"""
Price analysis tab component.
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from ...utils.logger import get_logger

logger = get_logger(__name__)


def render_price_analysis_tab(current_symbol):
    """Render the price analysis tab."""
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
            
            # Volume chart
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
                title=dict(text="Trading Volume", font=dict(size=18, color='#2c3e50')),
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

