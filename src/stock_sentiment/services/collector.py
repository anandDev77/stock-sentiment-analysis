"""
Stock data collection service.

This module provides functionality to collect stock market data including:
- Stock prices and company information
- News articles and headlines
"""

from typing import List, Dict, Optional
from datetime import datetime
import yfinance as yf
from dateutil import parser

from ..config.settings import Settings, get_settings
from ..utils.logger import get_logger
from ..utils.validators import validate_stock_symbol, sanitize_text
from .cache import RedisCache

logger = get_logger(__name__)


class StockDataCollector:
    """
    Service for collecting stock market data from free APIs.
    
    This class fetches stock data using yfinance and caches results
    in Redis for performance optimization.
    
    Attributes:
        cache: Redis cache instance (optional)
        settings: Application settings
        headers: HTTP headers for API requests
        
    Example:
        >>> settings = get_settings()
        >>> cache = RedisCache(settings=settings)
        >>> collector = StockDataCollector(settings=settings, redis_cache=cache)
        >>> data = collector.get_stock_price("AAPL")
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_cache: Optional[RedisCache] = None
    ):
        """
        Initialize the stock data collector.
        
        Args:
            settings: Application settings (uses global if not provided)
            redis_cache: Redis cache instance for caching
        """
        self.settings = settings or get_settings()
        self.cache = redis_cache
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info("StockDataCollector initialized")
    
    def get_stock_price(self, symbol: str) -> Dict:
        """
        Fetch current stock price and basic company information.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            
        Returns:
            Dictionary with stock data:
            - symbol: Stock ticker
            - price: Current stock price
            - company_name: Company name
            - market_cap: Market capitalization
            - timestamp: Data collection timestamp
        """
        if not validate_stock_symbol(symbol):
            logger.warning(f"Invalid stock symbol: {symbol}")
            return {
                'symbol': symbol,
                'price': 0.0,
                'company_name': symbol,
                'market_cap': 0,
                'timestamp': datetime.now()
            }
        
        # Check cache first
        if self.cache:
            cached_data = self.cache.get_cached_stock_data(symbol)
            if cached_data:
                # Convert timestamp string back to datetime if needed
                if isinstance(cached_data.get('timestamp'), str):
                    try:
                        cached_data['timestamp'] = datetime.fromisoformat(
                            cached_data['timestamp']
                        )
                    except ValueError:
                        cached_data['timestamp'] = datetime.now()
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                result = {
                    'symbol': symbol,
                    'price': current_price,
                    'company_name': info.get('longName', symbol),
                    'market_cap': info.get('marketCap', 0) or 0,
                    'timestamp': datetime.now()
                }
                
                # Cache the result
                if self.cache:
                    self.cache.cache_stock_data(
                        symbol,
                        result,
                        ttl=self.settings.app.cache_ttl_stock
                    )
                
                return result
        except Exception as e:
            logger.error(f"Error fetching stock price for {symbol}: {e}")
        
        # Return default on error
        return {
            'symbol': symbol,
            'price': 0.0,
            'company_name': symbol,
            'market_cap': 0,
            'timestamp': datetime.now()
        }
    
    def get_news_headlines(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Fetch recent news headlines for a stock.
        
        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles to return
            
        Returns:
            List of news article dictionaries with:
            - title: Article title
            - summary: Article summary
            - source: News source/publisher
            - url: Article URL
            - timestamp: Publication timestamp
        """
        if not validate_stock_symbol(symbol):
            logger.warning(f"Invalid stock symbol: {symbol}")
            return []
        
        # Check cache first
        if self.cache:
            cached_news = self.cache.get_cached_news(symbol)
            if cached_news:
                # Convert timestamp strings back to datetime
                for article in cached_news:
                    if isinstance(article.get('timestamp'), str):
                        try:
                            article['timestamp'] = datetime.fromisoformat(
                                article['timestamp']
                            )
                        except ValueError:
                            article['timestamp'] = datetime.now()
                return cached_news
        
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.info(f"No news found for {symbol}")
                return []
            
            headlines = []
            for i, article in enumerate(news[:limit]):
                try:
                    # Extract data from nested structure
                    content = article.get('content', {})
                    
                    # Extract title
                    title = (
                        content.get('title') or
                        article.get('title') or
                        article.get('headline') or
                        'No title available'
                    )
                    title = sanitize_text(title)
                    
                    # Extract summary
                    summary = (
                        content.get('summary') or
                        content.get('description') or
                        article.get('summary') or
                        article.get('description') or
                        ''
                    )
                    summary = sanitize_text(summary)
                    
                    # Extract publisher
                    provider = article.get('provider', {})
                    publisher = (
                        provider.get('displayName') if isinstance(provider, dict) else None
                    ) or (
                        article.get('publisher') if isinstance(article.get('publisher'), str) else None
                    ) or article.get('source') or 'News Source'
                    
                    # Extract URL
                    url = self._extract_url(content, article)
                    
                    # Extract timestamp
                    timestamp = self._extract_timestamp(content, article)
                    
                    headline_data = {
                        'title': title,
                        'summary': summary,
                        'source': publisher,
                        'timestamp': timestamp,
                        'url': url
                    }
                    
                    headlines.append(headline_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing article {i} for {symbol}: {e}")
                    continue
            
            # Cache the results
            if self.cache:
                self.cache.cache_news(
                    symbol,
                    headlines,
                    ttl=self.settings.app.cache_ttl_news
                )
            
            return headlines
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def _extract_url(self, content: Dict, article: Dict) -> str:
        """
        Extract article URL from nested structure.
        
        Args:
            content: Content dictionary from article
            article: Full article dictionary
            
        Returns:
            Article URL or empty string
        """
        # Try canonicalUrl from content first
        canonical_url = content.get('canonicalUrl', {})
        if isinstance(canonical_url, dict):
            url = canonical_url.get('url', '')
            if url:
                return url
        
        # Try clickThroughUrl from content
        click_through = content.get('clickThroughUrl', {})
        if isinstance(click_through, dict):
            url = click_through.get('url', '')
            if url:
                return url
        
        # Try at article level
        canonical_url = article.get('canonicalUrl', {})
        if isinstance(canonical_url, dict):
            url = canonical_url.get('url', '')
            if url:
                return url
        
        click_through = article.get('clickThroughUrl', {})
        if isinstance(click_through, dict):
            url = click_through.get('url', '')
            if url:
                return url
        
        return ''
    
    def _extract_timestamp(self, content: Dict, article: Dict) -> datetime:
        """
        Extract publication timestamp from nested structure.
        
        Args:
            content: Content dictionary from article
            article: Full article dictionary
            
        Returns:
            Datetime object or current time as fallback
        """
        # Try pubDate or displayTime from content
        timestamp_str = content.get('pubDate') or content.get('displayTime')
        if timestamp_str:
            try:
                return parser.parse(str(timestamp_str))
            except (ValueError, TypeError):
                pass
        
        # Try providerPublishTime or publishTime from article
        timestamp_val = article.get('providerPublishTime') or article.get('publishTime')
        if timestamp_val:
            try:
                if isinstance(timestamp_val, (int, float)):
                    return datetime.fromtimestamp(timestamp_val)
                else:
                    return parser.parse(str(timestamp_val))
            except (ValueError, TypeError):
                pass
        
        # Fallback to current time
        return datetime.now()
    
    def get_reddit_sentiment_data(self, symbol: str) -> List[Dict]:
        """
        Get social media sentiment data.
        
        Currently returns empty list as we only use free APIs.
        To add real social media data, integrate with:
        - Reddit API (PRAW) - requires API credentials
        - Twitter API - requires API credentials and may have costs
        - Other free alternatives if available
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Empty list (placeholder for future integration)
        """
        return []
    
    def collect_all_data(self, symbol: str) -> Dict:
        """
        Collect all available data for a stock symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with:
            - price_data: Stock price and company info
            - news: List of news articles
            - social_media: List of social media posts (currently empty)
        """
        return {
            'price_data': self.get_stock_price(symbol),
            'news': self.get_news_headlines(symbol),
            'social_media': self.get_reddit_sentiment_data(symbol)
        }

