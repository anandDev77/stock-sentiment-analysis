import requests
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import time
from datetime import datetime, timedelta
from redis_cache import RedisCache

class StockDataCollector:
    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = redis_cache

    def get_stock_price(self, symbol: str) -> Dict:
        """Fetch current stock price and basic info with caching."""
        # Check cache first
        if self.cache:
            cached_data = self.cache.get_cached_stock_data(symbol)
            if cached_data:
                print(f"✅ CACHE: Using cached stock data for {symbol}")
                return cached_data
            print(f"❌ CACHE: Fetching fresh stock data for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                result = {
                    'symbol': symbol,
                    'price': current_price,
                    'company_name': info.get('longName', symbol),
                    'market_cap': info.get('marketCap', 0),
                    'timestamp': datetime.now()
                }
                
                # Cache the result (5 minutes TTL)
                if self.cache:
                    self.cache.cache_stock_data(symbol, result, ttl=300)
                
                return result
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")

        return {'symbol': symbol, 'price': 0, 'company_name': symbol, 'market_cap': 0}

    def get_news_headlines(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Fetch recent news headlines for a stock with caching."""
        # Check cache first
        if self.cache:
            cached_news = self.cache.get_cached_news(symbol)
            if cached_news:
                print(f"✅ CACHE: Using cached news for {symbol} ({len(cached_news)} articles)")
                return cached_news
            print(f"❌ CACHE: Fetching fresh news for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            # DEBUG: Print the raw response structure
            print(f"\n=== DEBUG: News API Response for {symbol} ===")
            print(f"Number of articles: {len(news) if news else 0}")
            if news and len(news) > 0:
                print(f"\nFirst article keys: {list(news[0].keys())}")
                print(f"\nFirst article full data:")
                import json
                print(json.dumps(news[0], indent=2, default=str))
            print("=" * 50 + "\n")
            
            if not news:
                return []

            headlines = []
            for i, article in enumerate(news[:limit]):
                # The new yfinance API structure has data nested in 'content'
                content = article.get('content', {})
                
                # Extract title from content
                title = (content.get('title') or 
                        article.get('title') or 
                        article.get('headline') or 
                        'No title available')
                
                # Extract summary from content
                summary = (content.get('summary') or 
                          content.get('description') or 
                          article.get('summary') or 
                          article.get('description') or 
                          '')
                
                # Extract publisher/provider
                provider = article.get('provider', {})
                publisher = (provider.get('displayName') if isinstance(provider, dict) else None) or \
                           (article.get('publisher') if isinstance(article.get('publisher'), str) else None) or \
                           article.get('source') or \
                           'News Source'
                
                # Extract URL - try canonicalUrl from content first, then clickThroughUrl
                url = ''
                canonical_url = content.get('canonicalUrl', {})
                if isinstance(canonical_url, dict):
                    url = canonical_url.get('url', '')
                
                if not url:
                    click_through = content.get('clickThroughUrl', {})
                    if isinstance(click_through, dict):
                        url = click_through.get('url', '')
                
                # Try at article level as fallback
                if not url:
                    canonical_url = article.get('canonicalUrl', {})
                    if isinstance(canonical_url, dict):
                        url = canonical_url.get('url', '')
                
                if not url:
                    click_through = article.get('clickThroughUrl', {})
                    if isinstance(click_through, dict):
                        url = click_through.get('url', '')
                
                # Handle timestamp - try pubDate from content, then displayTime
                timestamp_str = content.get('pubDate') or content.get('displayTime')
                if timestamp_str:
                    try:
                        # Parse ISO format datetime string (e.g., '2025-11-12T14:35:10Z')
                        from dateutil import parser
                        timestamp = parser.parse(timestamp_str)
                    except:
                        try:
                            # Try parsing as timestamp number
                            if isinstance(timestamp_str, (int, float)):
                                timestamp = datetime.fromtimestamp(timestamp_str)
                            else:
                                # Try parsing ISO format manually
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except:
                            timestamp = datetime.now()
                else:
                    # Fallback to providerPublishTime or current time
                    timestamp_val = article.get('providerPublishTime') or article.get('publishTime')
                    if timestamp_val:
                        try:
                            if isinstance(timestamp_val, (int, float)):
                                timestamp = datetime.fromtimestamp(timestamp_val)
                            else:
                                timestamp = datetime.now()
                        except:
                            timestamp = datetime.now()
                    else:
                        timestamp = datetime.now()
                
                headline_data = {
                    'title': title,
                    'summary': summary,
                    'source': publisher,
                    'timestamp': timestamp,
                    'url': url
                }
                
                # DEBUG: Print what we extracted
                if i == 0:
                    print(f"\n=== DEBUG: Extracted data from first article ===")
                    print(f"Title: {title}")
                    print(f"Summary: {summary[:100] if summary else 'None'}...")
                    print(f"Publisher: {publisher}")
                    print(f"URL: {url}")
                    print("=" * 50 + "\n")
                
                headlines.append(headline_data)

            # Cache the results (30 minutes TTL)
            if self.cache:
                self.cache.cache_news(symbol, headlines, ttl=1800)

            return headlines

        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_reddit_sentiment_data(self, symbol: str) -> List[Dict]:
        """
        Get social media sentiment data.
        Currently returns empty list as we only use free APIs.
        To add real social media data, integrate with:
        - Reddit API (PRAW) - requires API credentials
        - Twitter API - requires API credentials and may have costs
        - Other free alternatives if available
        """
        # Return empty list - no fake data
        # In the future, integrate with real APIs here
        return []

    def collect_all_data(self, symbol: str) -> Dict:
        """Collect all available data for a stock symbol."""
        return {
            'price_data': self.get_stock_price(symbol),
            'news': self.get_news_headlines(symbol),
            'social_media': self.get_reddit_sentiment_data(symbol)
        }
