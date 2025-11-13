# Stock Sentiment Analysis Dashboard - Technical Architecture Documentation

**Version:** 1.0  
**Last Updated:** November 2024  
**Author:** Technical Architecture Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [High-Level Architecture](#high-level-architecture)
4. [Core Components](#core-components)
5. [Data Flow Architecture](#data-flow-architecture)
6. [RAG (Retrieval Augmented Generation) Flow](#rag-retrieval-augmented-generation-flow)
7. [Caching Strategy](#caching-strategy)
8. [Data Models](#data-models)
9. [Key Attributes and Their Meanings](#key-attributes-and-their-meanings)
10. [Technology Stack](#technology-stack)
11. [Performance Considerations](#performance-considerations)
12. [Security Architecture](#security-architecture)

---

## Executive Summary

### What This Application Does (Layman's Terms)

Imagine you want to know how people feel about a company's stock (like Apple or Microsoft). This application:

1. **Collects Information**: Gathers recent news articles and stock price data from the internet
2. **Understands Context**: Uses artificial intelligence to read and understand the news articles
3. **Analyzes Sentiment**: Determines if the news is positive, negative, or neutral about the stock
4. **Provides Insights**: Shows you charts, graphs, and summaries to help you make investment decisions

Think of it as a smart assistant that reads hundreds of news articles about a stock and tells you whether the overall sentiment is good or bad, helping you understand market trends without reading everything yourself.

### Technical Overview

This is a **web-based application** built using:
- **Streamlit** (Python web framework) for the user interface
- **Azure OpenAI GPT-4** for intelligent sentiment analysis
- **Redis** for high-speed data caching
- **RAG (Retrieval Augmented Generation)** for context-aware analysis
- **yfinance** for real-time stock market data

The application follows a **microservices-inspired architecture** with clear separation of concerns, making it maintainable, scalable, and testable.

---

## System Overview

### Purpose

The Stock Sentiment Analysis Dashboard provides real-time sentiment analysis of financial news and social media content related to stock symbols. It combines:

- **Real-time data collection** from financial APIs
- **AI-powered sentiment analysis** using Large Language Models (LLMs)
- **Context-aware analysis** through RAG technology
- **Intelligent caching** to optimize performance and reduce costs
- **Interactive visualization** for data exploration

### Key Capabilities

1. **Stock Data Collection**: Fetches current stock prices, company information, and historical data
2. **News Aggregation**: Collects relevant news articles from multiple sources
3. **Sentiment Analysis**: Analyzes text to determine positive, negative, or neutral sentiment
4. **Context Enhancement**: Uses RAG to provide relevant context from stored articles
5. **Data Visualization**: Creates interactive charts and graphs
6. **Performance Optimization**: Caches results to minimize API calls and improve response times

---

## High-Level Architecture

### System Architecture Diagram

![Diagram 1](diagrams/diagram_1.png)


### Component Interaction Flow

![Diagram 2](diagrams/diagram_2.png)



---

## Core Components

### 1. Main Application (`app.py`)

**Purpose**: The entry point and orchestration layer of the application.

**Responsibilities**:
- Manages the Streamlit user interface
- Coordinates between different services
- Handles user interactions and session state
- Renders visualizations and charts

**Key Functions**:
- Initializes all service components
- Manages application state
- Coordinates data collection and analysis
- Displays results in multiple tabs

### 2. Data Collector Service (`StockDataCollector`)

**Purpose**: Fetches stock market data and news articles from external APIs.

**How It Works**:
1. Receives a stock symbol (e.g., "AAPL" for Apple)
2. Checks Redis cache first to avoid unnecessary API calls
3. If not cached, fetches data from yfinance API
4. Stores the fetched data in Redis for future use
5. Returns structured data to the application

**Key Methods**:
- `get_stock_price(symbol)`: Fetches current stock price and company info
- `get_news_headlines(symbol)`: Retrieves recent news articles
- `collect_all_data(symbol)`: Orchestrates collection of all data types

**Technical Details**:
- Uses `yfinance` library to access Yahoo Finance data
- Implements caching with configurable TTL (Time To Live)
- Handles API errors gracefully with fallback mechanisms

### 3. Sentiment Analyzer Service (`SentimentAnalyzer`)

**Purpose**: Analyzes text to determine sentiment (positive, negative, or neutral) using AI.

**How It Works**:
1. Receives text to analyze (e.g., a news article headline)
2. Checks if sentiment for this text was previously analyzed (cached)
3. If not cached, uses RAG to find relevant context from other articles
4. Sends the text and context to Azure OpenAI GPT-4
5. Receives sentiment scores (positive, negative, neutral)
6. Caches the result for future use
7. Returns sentiment scores

**Key Methods**:
- `analyze_sentiment(text, symbol)`: Main analysis method
- `batch_analyze(texts, symbol)`: Analyzes multiple texts efficiently

**Technical Details**:
- Uses Azure OpenAI GPT-4 for high-quality sentiment analysis
- Implements RAG (Retrieval Augmented Generation) for context-aware analysis
- Falls back to TextBlob library if Azure OpenAI fails
- Returns normalized scores that sum to 1.0

### 4. RAG Service (`RAGService`)

**Purpose**: Provides context-aware analysis by retrieving relevant articles from a knowledge base.

**What is RAG?** (Layman's Terms)

Imagine you're reading a news article about Apple. Instead of analyzing it in isolation, RAG:
1. Looks through all previously stored articles about Apple
2. Finds the 3 most similar/relevant articles
3. Provides that context to the AI, so it understands the full picture
4. The AI can then make a more informed decision about sentiment

**How It Works**:
1. **Storage Phase**: When a news article is collected:
   - Converts the article text into a mathematical representation (embedding vector)
   - Stores the embedding and article metadata in Redis

2. **Retrieval Phase**: When analyzing sentiment:
   - Converts the query text into an embedding
   - Searches all stored embeddings for similar articles
   - Calculates similarity scores using cosine similarity
   - Returns the top K most relevant articles

**Key Methods**:
- `store_article(article, symbol)`: Stores article with embedding
- `retrieve_relevant_context(query, symbol)`: Finds similar articles
- `get_embedding(text)`: Converts text to embedding vector

**Technical Details**:
- Uses Azure OpenAI embedding models (e.g., text-embedding-ada-002)
- Implements cosine similarity for semantic search
- Uses Redis SCAN (not KEYS) for production-safe key iteration
- Applies similarity threshold filtering (default: 0.3)

### 5. Redis Cache Service (`RedisCache`)

**Purpose**: Stores frequently accessed data to improve performance and reduce API costs.

**What Gets Cached**:
1. **Stock Data**: Current prices, company info (TTL: 1 hour)
2. **News Articles**: Recent news headlines (TTL: 2 hours)
3. **Sentiment Results**: Analyzed sentiment scores (TTL: 24 hours)
4. **Article Embeddings**: Vector representations for RAG (TTL: 7 days)
5. **Query Embeddings**: Cached query embeddings (TTL: 24 hours)
6. **Cache Statistics**: Hit/miss counters (persistent)

**Key Methods**:
- `get(key)`: Retrieve cached value
- `set(key, value, ttl)`: Store value with expiration
- `get_cached_stock_data(symbol)`: Get cached stock data
- `get_cached_news(symbol)`: Get cached news articles
- `cache_sentiment(text, scores, ttl)`: Cache sentiment result

**Technical Details**:
- Uses MD5 hashing for consistent key generation
- Implements normalized key generation (uppercase, trimmed)
- Tracks cache statistics in Redis for monitoring
- Uses JSON serialization for complex data structures

---

## Data Flow Architecture

### Complete Request Flow

![Diagram 3](diagrams/diagram_3.png)


### RAG Retrieval Flow (Detailed)

![Diagram 4](diagrams/diagram_4.png)



---

## RAG (Retrieval Augmented Generation) Flow

### Conceptual Understanding

**What is RAG?**

RAG stands for **Retrieval Augmented Generation**. It's a technique that enhances AI responses by providing relevant context from a knowledge base.

**Simple Analogy**: 
Imagine you're taking an exam. Without RAG, you can only use what you memorized. With RAG, you can look up relevant information from your notes before answering, making your answers more accurate and informed.

### RAG Architecture in This Application

![Diagram 5](diagrams/diagram_5.png)



### RAG Process Steps

1. **Query Embedding Generation**
   - **Input**: Text to analyze (e.g., "Apple announces new iPhone")
   - **Process**: Convert text to numerical vector using embedding model
   - **Output**: 1536-dimensional vector representing semantic meaning
   - **Caching**: Query embeddings are cached for 24 hours

2. **Article Embedding Retrieval**
   - **Process**: Use Redis SCAN to find all article embeddings for the symbol
   - **Pattern**: `embedding:{symbol}:*`
   - **Safety**: Uses SCAN instead of KEYS to avoid blocking Redis

3. **Similarity Calculation**
   - **Method**: Cosine similarity between query and article embeddings
   - **Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`
   - **Range**: 0.0 (no similarity) to 1.0 (identical meaning)
   - **Implementation**: Uses NumPy for efficient vector operations

4. **Threshold Filtering**
   - **Purpose**: Remove low-quality matches
   - **Default Threshold**: 0.3 (30% similarity)
   - **Rationale**: Prevents irrelevant articles from polluting context

5. **Top-K Selection**
   - **Default K**: 3 articles
   - **Selection**: Highest similarity scores above threshold
   - **Result**: Most relevant articles for context

6. **Context Building**
   - **Format**: Structured markdown with article metadata
   - **Includes**: Title, source, summary, similarity score
   - **Purpose**: Provides clear context to the LLM

7. **Prompt Augmentation**
   - **Enhancement**: Adds RAG context to the sentiment analysis prompt
   - **Benefit**: LLM can consider recent news when analyzing sentiment
   - **Result**: More accurate and context-aware sentiment analysis

---

## Caching Strategy

### Cache Hierarchy

![Diagram 6](diagrams/diagram_6.png)



### Cache TTL Strategy

| Data Type | TTL | Rationale |
|-----------|-----|------------|
| Stock Data | 1 hour | Stock prices change frequently but not second-by-second |
| News Articles | 2 hours | News updates regularly but not constantly |
| Sentiment Results | 24 hours | Sentiment for same text doesn't change |
| Article Embeddings | 7 days | Embeddings are stable, articles don't change |
| Query Embeddings | 24 hours | Same queries benefit from caching |

### Cache Key Structure

```
Pattern: {prefix}:{normalized_args_hash}

Examples:
- stock:AAPL → MD5 hash
- news:AAPL → MD5 hash
- sentiment:{text_hash} → MD5 hash
- embedding:{symbol}:{article_id} → Direct key
- query_embedding:{text_hash} → MD5 hash
```

**Key Generation**:
- Normalizes inputs (uppercase, trimmed)
- Uses MD5 hashing for consistency
- Ensures same inputs generate same keys across app reloads

---

## Data Models

### SentimentScores

**Purpose**: Represents sentiment analysis results as numerical scores.

**Attributes**:

| Attribute | Type | Range | Description |
|-----------|------|-------|-------------|
| `positive` | float | 0.0 - 1.0 | Probability that sentiment is positive |
| `negative` | float | 0.0 - 1.0 | Probability that sentiment is negative |
| `neutral` | float | 0.0 - 1.0 | Probability that sentiment is neutral |

**Properties**:
- Scores are normalized to sum to 1.0
- `net_sentiment`: Calculated as `positive - negative` (-1.0 to 1.0)
- `dominant_sentiment`: Returns "positive", "negative", or "neutral"

**Example**:
```python
scores = SentimentScores(
    positive=0.65,  # 65% positive
    negative=0.20,  # 20% negative
    neutral=0.15    # 15% neutral
)
# Net sentiment: 0.45 (moderately positive)
```

### StockData

**Purpose**: Represents stock price and company information.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `symbol` | str | Stock ticker symbol (e.g., "AAPL") |
| `price` | float | Current stock price in USD |
| `company_name` | str | Full company name |
| `market_cap` | int | Market capitalization in USD |
| `timestamp` | datetime | When data was collected |

### NewsArticle

**Purpose**: Represents a news article about a stock.

**Attributes**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `title` | str | Article headline |
| `summary` | str | Article summary or description |
| `source` | str | Publisher name |
| `url` | str | Link to full article |
| `timestamp` | datetime | Publication date |

**Property**:
- `text_for_analysis`: Combines title and summary for sentiment analysis

---

## Key Attributes and Their Meanings

### Application Settings (`AppSettings`)

#### Logging Configuration

**`log_level`** (str, default: "INFO")
- **Purpose**: Controls verbosity of application logs
- **Values**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Usage**: Set to DEBUG for troubleshooting, INFO for production
- **Example**: `APP_LOG_LEVEL=DEBUG` in `.env`

**`debug`** (bool, default: false)
- **Purpose**: Enables additional debugging features
- **Impact**: May include more detailed error messages and stack traces

#### Cache TTL Settings

**`cache_ttl_sentiment`** (int, default: 86400 seconds = 24 hours)
- **Purpose**: How long sentiment analysis results are cached
- **Rationale**: Sentiment for the same text doesn't change, so long TTL reduces API calls
- **Trade-off**: Longer TTL = fewer API calls but potentially stale data

**`cache_ttl_stock`** (int, default: 3600 seconds = 1 hour)
- **Purpose**: How long stock price data is cached
- **Rationale**: Stock prices change frequently but not second-by-second
- **Trade-off**: Balance between freshness and API rate limits

**`cache_ttl_news`** (int, default: 7200 seconds = 2 hours)
- **Purpose**: How long news articles are cached
- **Rationale**: News updates regularly but not constantly
- **Trade-off**: Fresh news vs. API costs

#### RAG Configuration

**`rag_top_k`** (int, default: 3)
- **Purpose**: Number of similar articles to retrieve for context
- **Impact**: 
  - Higher K = more context but potentially more noise
  - Lower K = less context but more focused
- **Recommendation**: 3-5 articles typically provides good balance

**`rag_similarity_threshold`** (float, default: 0.3)
- **Purpose**: Minimum similarity score (0.0-1.0) for article inclusion
- **Meaning**: 
  - 0.0 = no similarity required (includes everything)
  - 1.0 = perfect match required (very restrictive)
  - 0.3 = moderate similarity (filters out irrelevant articles)
- **Impact**: Higher threshold = higher quality but fewer articles

### Azure OpenAI Settings

**`endpoint`** (str, required)
- **Purpose**: Azure OpenAI service endpoint URL
- **Format**: `https://{resource-name}.openai.azure.com`
- **Security**: Contains service location, should be kept private

**`api_key`** (str, required)
- **Purpose**: Authentication key for Azure OpenAI API
- **Security**: Highly sensitive, never commit to version control
- **Rotation**: Should be rotated regularly

**`deployment_name`** (str, default: "gpt-4")
- **Purpose**: Name of the GPT-4 deployment in Azure
- **Note**: Must match the deployment name in Azure Portal

**`embedding_deployment`** (str, optional)
- **Purpose**: Name of the embedding model deployment
- **Required for**: RAG functionality
- **Common values**: "text-embedding-ada-002", "text-embedding-3-small"

### Redis Settings

**`host`** (str, required)
- **Purpose**: Redis server hostname or IP address
- **Format**: `{name}.redis.cache.windows.net` for Azure Cache

**`port`** (int, default: 6380)
- **Purpose**: Redis server port
- **Note**: 6380 is standard for SSL-enabled Redis

**`password`** (str, required)
- **Purpose**: Redis authentication password
- **Security**: Sensitive, should be kept private

**`ssl`** (bool, default: true)
- **Purpose**: Enable SSL/TLS encryption for Redis connection
- **Security**: Required for production, especially Azure Cache

### Sentiment Analysis Attributes

**`temperature`** (float, default: 0.2)
- **Purpose**: Controls randomness in AI responses
- **Range**: 0.0 (deterministic) to 2.0 (very creative)
- **For Sentiment**: Lower values (0.1-0.3) provide more consistent results

**`max_tokens`** (int, default: 200)
- **Purpose**: Maximum length of AI response
- **For Sentiment**: 200 tokens sufficient for JSON response
- **Cost Impact**: More tokens = higher API costs

**`response_format`** (dict, optional)
- **Purpose**: Forces structured JSON output
- **Value**: `{"type": "json_object"}`
- **Benefit**: More reliable JSON parsing, fewer errors

### RAG Service Attributes

**`embedding_dimension`** (int, implicit: 1536)
- **Purpose**: Size of embedding vectors
- **For text-embedding-ada-002**: 1536 dimensions
- **Storage**: Each embedding is ~6KB in Redis

**`similarity_metric`** (str, implicit: "cosine")
- **Purpose**: Method for comparing embeddings
- **Cosine Similarity**: Measures angle between vectors (0-1 scale)
- **Alternative**: Euclidean distance (not used here)

---

## Technology Stack

### Frontend
- **Streamlit**: Python web framework for rapid UI development
- **Plotly**: Interactive data visualization library
- **Pandas**: Data manipulation and analysis

### Backend Services
- **Python 3.8+**: Programming language
- **Azure OpenAI**: Large Language Model service (GPT-4)
- **Redis**: In-memory data store for caching
- **yfinance**: Financial data API wrapper

### Data Processing
- **NumPy**: Numerical computing for vector operations
- **TextBlob**: Fallback sentiment analysis library
- **Pydantic**: Data validation and settings management

### Infrastructure
- **Azure Cloud**: Hosting for OpenAI and Redis
- **Docker** (optional): Containerization support
- **Git**: Version control

---

## Performance Considerations

### Optimization Strategies

1. **Multi-Level Caching**
   - Session state (fastest, per-user)
   - Redis cache (fast, shared across users)
   - API calls (slowest, only when needed)

2. **Batch Processing**
   - Process multiple articles in single operations
   - Reduce API round trips

3. **Async Operations** (Future Enhancement)
   - Parallel API calls where possible
   - Non-blocking I/O operations

4. **Embedding Caching**
   - Cache query embeddings to avoid regeneration
   - Store article embeddings for 7 days

### Performance Metrics

| Operation | Without Cache | With Cache | Improvement |
|-----------|---------------|------------|-------------|
| Stock Data Fetch | ~500ms | ~5ms | 100x faster |
| News Fetch | ~800ms | ~5ms | 160x faster |
| Sentiment Analysis | ~2000ms | ~5ms | 400x faster |
| RAG Retrieval | ~300ms | ~50ms | 6x faster |

### Scalability Considerations

- **Horizontal Scaling**: Stateless design allows multiple instances
- **Redis Clustering**: Can use Redis Cluster for high availability
- **API Rate Limits**: Caching reduces load on external APIs
- **Cost Optimization**: Caching significantly reduces Azure OpenAI API costs

---

## Security Architecture

### Security Measures

1. **Environment Variables**
   - Sensitive data stored in `.env` file (not committed)
   - API keys never hardcoded
   - `.env.example` provided as template

2. **Azure Security**
   - API keys managed through Azure Key Vault (recommended)
   - SSL/TLS for all external connections
   - Network isolation for Redis

3. **Data Privacy**
   - No user data stored permanently
   - Session state cleared on app restart
   - Cache TTLs ensure data expiration

4. **Input Validation**
   - Stock symbol validation
   - Text sanitization
   - SQL injection prevention (not applicable, no SQL)

### Best Practices

- **Key Rotation**: Rotate API keys regularly
- **Least Privilege**: Use minimal required permissions
- **Monitoring**: Log security events
- **Updates**: Keep dependencies updated

---

## Conclusion

This application demonstrates a production-ready architecture combining:

- **Modern AI**: Azure OpenAI GPT-4 for intelligent analysis
- **Advanced Techniques**: RAG for context-aware processing
- **Performance**: Redis caching for optimization
- **User Experience**: Interactive Streamlit dashboard
- **Best Practices**: Clean architecture, proper error handling, comprehensive logging

The modular design allows for easy extension and maintenance, making it suitable for both learning and production use.

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Maintained By**: Development Team

