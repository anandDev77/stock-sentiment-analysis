"""
Stock data models.

This module defines the data structures for stock information,
news articles, and social media posts.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class StockData:
    """
    Stock price and company information.
    
    Attributes:
        symbol: Stock ticker symbol (e.g., "AAPL")
        price: Current stock price
        company_name: Full company name
        market_cap: Market capitalization in USD
        timestamp: When this data was collected
    """
    symbol: str
    price: float
    company_name: str
    market_cap: int
    timestamp: datetime
    
    def __post_init__(self):
        """Validate stock data."""
        if self.price < 0:
            raise ValueError("Stock price cannot be negative")
        if self.market_cap < 0:
            raise ValueError("Market cap cannot be negative")
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "symbol": self.symbol,
            "price": self.price,
            "company_name": self.company_name,
            "market_cap": self.market_cap,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class NewsArticle:
    """
    News article information.
    
    Attributes:
        title: Article title
        summary: Article summary or description
        source: Publisher or news source
        url: Link to the full article
        timestamp: Publication timestamp
    """
    title: str
    summary: str
    source: str
    url: str
    timestamp: datetime
    
    def __post_init__(self):
        """Validate and clean article data."""
        if not self.title:
            self.title = "No title available"
        if not self.source:
            self.source = "News Source"
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "timestamp": self.timestamp.isoformat()
        }
    
    @property
    def text_for_analysis(self) -> str:
        """
        Get text content for sentiment analysis.
        
        Returns:
            Combined title and summary for analysis
        """
        return f"{self.title} {self.summary}".strip()


@dataclass
class SocialMediaPost:
    """
    Social media post information.
    
    Attributes:
        text: Post content
        platform: Platform name (e.g., "Reddit", "Twitter")
        author: Post author username
        subreddit: Subreddit name (if Reddit)
        url: Link to the post
        timestamp: Post timestamp
    """
    text: str
    platform: str
    author: str
    source: str  # Alias for platform for compatibility
    subreddit: Optional[str] = None
    url: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize source from platform."""
        if not self.source:
            self.source = self.platform
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "platform": self.platform,
            "source": self.source,
            "author": self.author,
            "subreddit": self.subreddit,
            "url": self.url or "",
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

