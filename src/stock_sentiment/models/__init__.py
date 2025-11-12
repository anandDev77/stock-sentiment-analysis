"""
Data models for the Stock Sentiment Analysis application.

This module defines the data structures used throughout the application
for type safety and better code organization.
"""

from .sentiment import SentimentResult, SentimentScores
from .stock import StockData, NewsArticle, SocialMediaPost

__all__ = [
    "SentimentResult",
    "SentimentScores",
    "StockData",
    "NewsArticle",
    "SocialMediaPost",
]

