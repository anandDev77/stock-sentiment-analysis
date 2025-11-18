# Stock Sentiment Analysis Dashboard - Technical Architecture Documentation V2

**Version:** 2.0  
**Last Updated:** December 2024  
**Author:** Technical Architecture Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [High-Level Architecture](#high-level-architecture)
4. [Modular Application Structure](#modular-application-structure)
5. [Core Components](#core-components)
6. [Data Flow Architecture](#data-flow-architecture)
7. [RAG (Retrieval Augmented Generation) Flow](#rag-retrieval-augmented-generation-flow)
8. [Vector Database Architecture](#vector-database-architecture)
9. [Data Sources Integration](#data-sources-integration)
10. [Caching Strategy](#caching-strategy)
11. [Mathematical Models & Algorithms](#mathematical-models--algorithms)
12. [Data Models](#data-models)
13. [Key Attributes and Their Meanings](#key-attributes-and-their-meanings)
14. [Technology Stack](#technology-stack)
15. [Performance Considerations](#performance-considerations)
16. [Security Architecture](#security-architecture)

---

## Executive Summary

### What This Application Does (Layman's Terms)

Imagine you want to know how people feel about a company's stock (like Apple or Microsoft). This application:

1. **Collects Information**: Gathers recent news articles and stock price data from multiple sources (Yahoo Finance, Alpha Vantage, Finnhub, Reddit)
2. **Stores Knowledge**: Uses Azure AI Search to store and index articles for fast retrieval
3. **Understands Context**: Uses artificial intelligence to read and understand the news articles with relevant context from similar articles
4. **Analyzes Sentiment**: Determines if the news is positive, negative, or neutral about the stock using Azure OpenAI GPT-4
5. **Provides Insights**: Shows you charts, graphs, and summaries to help you make investment decisions

Think of it as a smart assistant that reads hundreds of news articles about a stock, remembers them in a searchable knowledge base, and tells you whether the overall sentiment is good or bad, helping you understand market trends without reading everything yourself.

### Technical Overview

This is a **web-based application** built using:
- **Streamlit** (Python web framework) for the user interface
- **Azure OpenAI GPT-4** for intelligent sentiment analysis
- **Azure AI Search** for high-performance vector search (10-100x faster than Redis SCAN)
- **Redis** for high-speed data caching
- **RAG (Retrieval Augmented Generation)** with hybrid search for context-aware analysis
- **Multiple Data Sources**: yfinance, Alpha Vantage, Finnhub, Reddit for comprehensive news coverage

The application follows a **modular, layered architecture** with clear separation of concerns:
- **Presentation Layer**: UI components, tabs, styling
- **Application Layer**: Data loading, orchestration
- **Service Layer**: Business logic (collector, sentiment, RAG, cache, vector DB)
- **Infrastructure Layer**: External APIs, databases

This architecture makes it maintainable, scalable, testable, and suitable for production deployment.

---

## System Overview

### Purpose

The Stock Sentiment Analysis Dashboard provides real-time sentiment analysis of financial news and social media content related to stock symbols. It combines:

- **Multi-source data collection** from financial APIs (yfinance, Alpha Vantage, Finnhub, Reddit)
- **AI-powered sentiment analysis** using Large Language Models (LLMs)
- **Context-aware analysis** through RAG technology with hybrid search
- **High-performance vector search** using Azure AI Search
- **Intelligent caching** to optimize performance and reduce costs
- **Interactive visualization** for data exploration

### Key Capabilities

1. **Multi-Source Stock Data Collection**: Fetches current stock prices, company information, and historical data from yfinance
2. **News Aggregation**: Collects relevant news articles from multiple sources (Yahoo Finance, Alpha Vantage, Finnhub, Reddit)
3. **Vector Search**: Stores and retrieves articles using Azure AI Search for 10-100x faster search
4. **Hybrid Search**: Combines semantic (vector) and keyword search using Reciprocal Rank Fusion (RRF)
5. **Sentiment Analysis**: Analyzes text to determine positive, negative, or neutral sentiment with RAG context
6. **Temporal Decay**: Boosts recent articles in search results (financial news is time-sensitive)
7. **Data Visualization**: Creates interactive charts and graphs
8. **Performance Optimization**: Multi-level caching to minimize API calls and improve response times
9. **Configurable Caching**: Enable/disable sentiment caching to test RAG functionality

---

## High-Level Architecture

### System Architecture Diagram

![Architecture Diagram](diagrams/architecture.png)

### Component Interaction Flow

![Component Flow](diagrams/components.png)

### Data Flow Diagram

![Data Flow](diagrams/dataflow.png)

---

## Modular Application Structure

### V2 Architecture Overview

The application has been refactored from a monolithic `app.py` into a modular, layered architecture:

```
src/stock_sentiment/
├── app.py                          # Thin orchestrator (entry point)
├── presentation/                   # Presentation Layer
│   ├── styles.py                  # Custom CSS styling
│   ├── initialization.py          # App setup, service initialization
│   ├── data_loader.py             # Data loading orchestration
│   ├── components/                # Reusable UI components
│   │   ├── sidebar.py             # Sidebar with filters, settings
│   │   └── empty_state.py         # Empty state component
│   └── tabs/                      # Tab modules
│       ├── overview_tab.py        # Overview dashboard
│       ├── price_analysis_tab.py  # Price charts and analysis
│       ├── news_sentiment_tab.py  # News and sentiment display
│       ├── technical_analysis_tab.py  # Technical indicators
│       ├── ai_insights_tab.py     # AI-generated insights
│       └── comparison_tab.py      # Multi-stock comparison
├── services/                      # Service Layer (Business Logic)
│   ├── collector.py               # Multi-source data collection
│   ├── sentiment.py               # AI sentiment analysis
│   ├── rag.py                     # RAG service with hybrid search
│   ├── cache.py                   # Redis caching
│   └── vector_db.py               # Azure AI Search integration
├── config/                        # Configuration
│   └── settings.py                # Pydantic settings management
├── models/                        # Data Models
│   ├── sentiment.py              # SentimentScores model
│   └── stock.py                   # StockData model
└── utils/                         # Utilities
    ├── logger.py                  # Logging configuration
    ├── retry.py                   # Retry logic
    ├── circuit_breaker.py         # Circuit breaker pattern
    └── preprocessing.py           # Text preprocessing
```

### Presentation Layer

#### 1. Main Application (`app.py`)

**Purpose**: Thin orchestrator that coordinates the application flow.

**Responsibilities**:
- Imports and initializes presentation layer modules
- Coordinates between different components
- Manages tab rendering

**Key Flow**:
```python
1. Setup application (styles, initialization)
2. Initialize settings and services
3. Render sidebar (filters, settings)
4. Load data if requested
5. Render tabs based on data availability
```

#### 2. Initialization Module (`presentation/initialization.py`)

**Purpose**: Centralized initialization of all application components.

**Key Functions**:
- `initialize_settings()`: Load and validate application settings
- `initialize_services()`: Create service instances (Redis, RAG, Collector, Analyzer)
- `initialize_session_state()`: Initialize Streamlit session state
- `setup_app()`: Configure Streamlit page settings

**Service Initialization**:
- Uses `@st.cache_resource` for singleton services
- Handles graceful degradation if services unavailable
- Provides fallback mechanisms

#### 3. Data Loader (`presentation/data_loader.py`)

**Purpose**: Orchestrates the complete data loading and processing pipeline.

**Key Functions**:
- `load_stock_data()`: Main orchestration function

**Process Flow**:
1. **Step 1**: Fetch stock data (with Redis cache check)
2. **Step 2**: Collect news from multiple sources (with source filters)
3. **Step 3**: Store articles in RAG (Azure AI Search)
4. **Step 4**: Analyze sentiment for all articles (with RAG context)
5. **Step 5**: Store results in session state

**Features**:
- Progress tracking with progress bars
- Comprehensive logging for demos
- Operation summary tracking (Redis usage, RAG usage, cache hits/misses)
- Error handling with user-friendly messages

#### 4. Sidebar Component (`presentation/components/sidebar.py`)

**Purpose**: Provides user controls and system information.

**Sections**:
- **System Status**: Redis and RAG service availability
- **Search Filters**: 
  - Stock symbol input
  - Data source toggles (yfinance, Alpha Vantage, Finnhub, Reddit)
  - RAG filter controls (exclude sources)
- **Sentiment Cache Controls**:
  - Enable/disable sentiment caching
  - TTL slider (0.1 to 168 hours)
  - Allows testing RAG by disabling cache
- **Operation Summary**: 
  - Redis usage (stock/news/sentiment cache hits)
  - RAG usage (queries made, articles found)
  - Articles stored in RAG
  - Summary of last operation

#### 5. Tab Modules (`presentation/tabs/`)

Each tab is a self-contained module with a `render_*_tab()` function:

- **Overview Tab**: Dashboard with key metrics, sentiment distribution
- **Price Analysis Tab**: Stock price charts, historical data
- **News & Sentiment Tab**: Article list with sentiment scores
- **Technical Analysis Tab**: Technical indicators, moving averages
- **AI Insights Tab**: AI-generated insights and recommendations
- **Comparison Tab**: Multi-stock comparison functionality

---

## Core Components

### 1. Data Collector Service (`StockDataCollector`)

**Purpose**: Fetches stock market data and news articles from multiple external APIs.

**Data Sources**:
1. **yfinance** (Primary, always enabled)
   - Stock prices, company info
   - News headlines
   - Historical data

2. **Alpha Vantage** (Optional)
   - Company news
   - Free tier: 500 calls/day
   - API key required

3. **Finnhub** (Optional)
   - Company news
   - Free tier: 60 calls/minute
   - API key required

4. **Reddit** (Optional)
   - Social media sentiment
   - Uses PRAW (Python Reddit API Wrapper)
   - Requires Reddit app registration

**How It Works**:
1. Receives a stock symbol (e.g., "AAPL" for Apple)
2. Checks Redis cache first to avoid unnecessary API calls
3. If not cached, fetches data from enabled sources in parallel
4. Deduplicates articles across sources
5. Stores the fetched data in Redis for future use
6. Returns structured data to the application

**Key Methods**:
- `get_stock_price(symbol)`: Fetches current stock price and company info
- `get_news_headlines(symbol)`: Retrieves news from yfinance
- `get_alpha_vantage_news(symbol)`: Retrieves news from Alpha Vantage
- `get_finnhub_news(symbol)`: Retrieves news from Finnhub
- `get_reddit_sentiment_data(symbol)`: Retrieves Reddit posts
- `collect_all_data(symbol, data_source_filters)`: Orchestrates collection from all enabled sources

**Technical Details**:
- Uses `yfinance` library for Yahoo Finance data
- Implements caching with configurable TTL (Time To Live)
- Handles API errors gracefully with fallback mechanisms
- Supports source filtering (enable/disable individual sources)
- Deduplicates articles by title and URL

### 2. Sentiment Analyzer Service (`SentimentAnalyzer`)

**Purpose**: Analyzes text to determine sentiment (positive, negative, or neutral) using AI with RAG context.

**How It Works**:
1. Receives text to analyze (e.g., a news article headline)
2. Checks if sentiment for this text was previously analyzed (cached)
3. If not cached and RAG available:
   - Retrieves relevant context from stored articles using hybrid search
   - Formats context with article metadata
4. Sends the text and context to Azure OpenAI GPT-4
5. Receives sentiment scores (positive, negative, neutral) as JSON
6. Caches the result for future use (if caching enabled)
7. Returns sentiment scores

**Key Methods**:
- `analyze_sentiment(text, symbol)`: Main analysis method with RAG context
- `batch_analyze(texts, symbol, max_workers)`: Analyzes multiple texts efficiently in parallel

**Technical Details**:
- Uses Azure OpenAI GPT-4 for high-quality sentiment analysis
- Implements RAG (Retrieval Augmented Generation) for context-aware analysis
- Falls back to TextBlob library if Azure OpenAI fails
- Returns normalized scores that sum to 1.0
- Supports configurable sentiment caching (can be disabled for RAG testing)
- Uses few-shot learning with examples in the prompt
- Implements retry logic with exponential backoff
- Circuit breaker pattern for resilience

**Prompt Engineering**:
- System prompt with role definition
- Few-shot examples (positive, negative, neutral)
- Structured JSON output format
- Context injection from RAG results

### 3. RAG Service (`RAGService`)

**Purpose**: Provides context-aware analysis by retrieving relevant articles from a knowledge base using hybrid search.

**What is RAG?** (Layman's Terms)

Imagine you're reading a news article about Apple. Instead of analyzing it in isolation, RAG:
1. Looks through all previously stored articles about Apple in Azure AI Search
2. Finds the most similar/relevant articles using hybrid search (semantic + keyword)
3. Combines results using Reciprocal Rank Fusion (RRF)
4. Applies temporal decay to boost recent articles
5. Provides that context to the AI, so it understands the full picture
6. The AI can then make a more informed decision about sentiment

**How It Works**:

**Storage Phase**: When a news article is collected:
1. Preprocesses article text (cleaning, normalization)
2. Converts the article text into a mathematical representation (embedding vector) using Azure OpenAI
3. Stores the embedding and article metadata in Azure AI Search
4. Also marks as stored in Redis for duplicate checking

**Retrieval Phase**: When analyzing sentiment:
1. Converts the query text into an embedding
2. Performs hybrid search:
   - **Semantic search**: Vector similarity search in Azure AI Search
   - **Keyword search**: Full-text search in Azure AI Search
3. Combines results using Reciprocal Rank Fusion (RRF)
4. Applies temporal decay to boost recent articles
5. Filters by similarity threshold
6. Returns the top K most relevant articles

**Key Methods**:
- `store_articles_batch(articles, symbol, batch_size)`: Stores multiple articles with batch embedding generation
- `retrieve_relevant_context(query, symbol, top_k, use_hybrid)`: Finds similar articles using hybrid search
- `get_embedding(text)`: Converts text to embedding vector
- `get_embeddings_batch(texts, batch_size)`: Batch embedding generation (industry best practice)

**Technical Details**:
- Uses Azure OpenAI embedding models (e.g., text-embedding-ada-002, 1536 dimensions)
- Uses Azure AI Search for vector storage and search (10-100x faster than Redis SCAN)
- Implements hybrid search (semantic + keyword) with RRF
- Applies similarity threshold filtering (default: 0.01 for RRF scores)
- Supports OData filters (symbol, date range, sources)
- Batch embedding generation (reduces API calls from N to N/batch_size)
- Temporal decay for recency boosting
- Falls back to Redis SCAN if Azure AI Search unavailable

### 4. Vector Database Service (`AzureAISearchVectorDB`)

**Purpose**: Provides high-performance vector search using Azure AI Search.

**Why Azure AI Search?**
- **10-100x faster** than Redis SCAN-based search for large datasets
- **Native vector indexing** using HNSW (Hierarchical Navigable Small World) algorithm
- **Hybrid search** (vector + keyword) with built-in RRF
- **OData filter support** for complex queries (date ranges, sources, etc.)
- **Built-in relevance scoring** and reranking
- **Scalable** to millions of documents

**Features**:
- Automatic index creation if not exists
- Batch document upload
- Vector search with configurable top_k
- Hybrid search (vector + keyword) with RRF
- OData filter support
- Error handling and logging

**Key Methods**:
- `store_vector(vector_id, vector, metadata)`: Store single vector
- `batch_store_vectors(vectors)`: Batch store vectors
- `search_vectors(query_vector, top_k, filter)`: Pure vector search
- `hybrid_search(query_text, query_vector, top_k, filter)`: Hybrid search
- `is_available()`: Check if service is available
- `delete_vector(vector_id)`: Delete vector by ID

**Index Schema**:
- `id`: Unique document ID (symbol:article_id)
- `contentVector`: Embedding vector (1536 dimensions)
- `content`: Searchable text (title + summary)
- `symbol`: Stock symbol (filterable)
- `title`: Article title
- `summary`: Article summary
- `source`: Article source (filterable)
- `url`: Article URL
- `timestamp`: Publication date (filterable, sortable)
- `article_id`: Original article ID

### 5. Redis Cache Service (`RedisCache`)

**Purpose**: Stores frequently accessed data to improve performance and reduce API costs.

**What Gets Cached**:
1. **Stock Data**: Current prices, company info (TTL: 1 hour)
2. **News Articles**: Recent news headlines (TTL: 2 hours)
3. **Sentiment Results**: Analyzed sentiment scores (TTL: 24 hours, configurable, can be disabled)
4. **Article Embeddings**: Vector representations for RAG (TTL: 7 days)
5. **Query Embeddings**: Cached query embeddings (TTL: 24 hours)
6. **Cache Statistics**: Hit/miss counters (persistent)

**Key Methods**:
- `get(key)`: Retrieve cached value
- `set(key, value, ttl)`: Store value with expiration
- `get_cached_stock_data(symbol)`: Get cached stock data
- `get_cached_news(symbol)`: Get cached news articles
- `get_cached_sentiment(text)`: Get cached sentiment result (respects cache_sentiment_enabled setting)
- `cache_sentiment(text, scores, ttl)`: Cache sentiment result (respects cache_sentiment_enabled setting)

**Technical Details**:
- Uses MD5 hashing for consistent key generation
- Implements normalized key generation (uppercase, trimmed)
- Tracks cache statistics in Redis for monitoring
- Uses JSON serialization for complex data structures
- Supports configurable sentiment caching (can be disabled)

---

## Data Flow Architecture

### Complete Request Flow

![Data Flow Diagram](diagrams/dataflow.png)

### Detailed Request Flow

1. **User Input**: User enters stock symbol and clicks "Load Data"
2. **Sidebar Processing**: Sidebar captures filters (data sources, RAG filters)
3. **Data Loading**:
   - Check Redis for cached stock data
   - If not cached, fetch from yfinance
   - Collect news from enabled sources (yfinance, Alpha Vantage, Finnhub, Reddit)
   - Deduplicate articles
4. **RAG Storage**:
   - Generate embeddings for articles (batch processing)
   - Store in Azure AI Search
   - Mark as stored in Redis
5. **Sentiment Analysis**:
   - For each article:
     - Check sentiment cache (if enabled)
     - If not cached, retrieve RAG context (hybrid search)
     - Send to Azure OpenAI GPT-4 with context
     - Cache result (if caching enabled)
6. **Result Storage**: Store in session state for UI display
7. **UI Rendering**: Display results in tabs

---

## RAG (Retrieval Augmented Generation) Flow

### Conceptual Understanding

**What is RAG?**

RAG stands for **Retrieval Augmented Generation**. It's a technique that enhances AI responses by providing relevant context from a knowledge base.

**Simple Analogy**: 
Imagine you're taking an exam. Without RAG, you can only use what you memorized. With RAG, you can look up relevant information from your notes before answering, making your answers more accurate and informed.

### RAG Architecture in This Application

![RAG Flow Diagram](diagrams/rag_flow.png)

### RAG Process Steps

#### 1. Query Embedding Generation
- **Input**: Text to analyze (e.g., "Apple announces new iPhone")
- **Process**: Convert text to numerical vector using Azure OpenAI embedding model
- **Output**: 1536-dimensional vector representing semantic meaning
- **Caching**: Query embeddings are cached for 24 hours in Redis

#### 2. Hybrid Search
- **Semantic Search**: Vector similarity search in Azure AI Search
  - Uses HNSW algorithm for fast approximate nearest neighbor search
  - Returns articles with highest cosine similarity
- **Keyword Search**: Full-text search in Azure AI Search
  - Searches in `content` field (title + summary)
  - Returns articles with matching keywords
- **Combination**: Uses Reciprocal Rank Fusion (RRF) to combine results

#### 3. Reciprocal Rank Fusion (RRF)
- **Purpose**: Combine semantic and keyword search results without score normalization
- **Formula**: `RRF_score = Σ(1 / (k + rank))` for each search result
  - `k = 60` (standard RRF constant)
  - `rank` = position in search results (1, 2, 3, ...)
- **Benefit**: Works with different score scales (no normalization needed)
- **Result**: Articles appearing high in both searches rank highest

#### 4. Temporal Decay
- **Purpose**: Boost recent articles (financial news is time-sensitive)
- **Formula**: `decay = 1.0 / (1 + age_days / decay_days)`
  - `age_days` = days since article publication
  - `decay_days` = 7 (configurable)
- **Boost**: `boosted_score = current_score * (1 + decay * 0.2)`
- **Result**: Recent articles get up to 20% score boost

#### 5. Threshold Filtering
- **Purpose**: Remove low-quality matches
- **Default Threshold**: 0.01 (for RRF scores, which are typically 0.01-0.15)
- **Rationale**: Prevents irrelevant articles from polluting context
- **Auto-adjustment**: If threshold too high, automatically lowers it

#### 6. Top-K Selection
- **Default K**: 3 articles
- **Selection**: Highest RRF scores above threshold
- **Result**: Most relevant articles for context

#### 7. Context Building
- **Format**: Structured markdown with article metadata
- **Includes**: Title, source, summary, similarity score
- **Purpose**: Provides clear context to the LLM

#### 8. Prompt Augmentation
- **Enhancement**: Adds RAG context to the sentiment analysis prompt
- **Benefit**: LLM can consider recent news when analyzing sentiment
- **Result**: More accurate and context-aware sentiment analysis

---

## Vector Database Architecture

### Azure AI Search Integration

**Why Azure AI Search?**
- **Performance**: 10-100x faster than Redis SCAN for large datasets
- **Scalability**: Handles millions of documents
- **Features**: Native vector indexing, hybrid search, filtering
- **Cost**: Pay-per-use, scales with usage

### Index Structure

```
Index: stock-articles
├── id (Edm.String, key)
├── contentVector (Collection(Edm.Single), 1536 dimensions)
├── content (Edm.String, searchable)
├── symbol (Edm.String, filterable)
├── title (Edm.String, searchable)
├── summary (Edm.String, searchable)
├── source (Edm.String, filterable)
├── url (Edm.String)
├── timestamp (Edm.DateTimeOffset, filterable, sortable)
└── article_id (Edm.String)
```

### Search Modes

1. **Vector Search**: Pure semantic search using cosine similarity
2. **Hybrid Search**: Combines vector and keyword search with RRF
3. **Filtered Search**: OData filters for symbol, date range, sources

### OData Filter Examples

```
# Filter by symbol
symbol eq 'AAPL'

# Filter by symbol and exclude source
symbol eq 'AAPL' and source ne 'Unknown'

# Filter by date range
symbol eq 'AAPL' and timestamp ge 2024-01-01T00:00:00Z and timestamp le 2024-12-31T23:59:59Z

# Complex filter
symbol eq 'AAPL' and (source eq 'Yahoo Finance' or source eq 'Finnhub') and timestamp ge 2024-11-01T00:00:00Z
```

---

## Data Sources Integration

### Supported Data Sources

#### 1. Yahoo Finance (yfinance) - Primary Source
- **Status**: Always enabled
- **Data**: Stock prices, company info, news headlines
- **Library**: `yfinance`
- **Rate Limits**: None (public API)
- **Caching**: 1 hour for stock data, 2 hours for news

#### 2. Alpha Vantage
- **Status**: Optional (requires API key)
- **Data**: Company news
- **API**: REST API
- **Rate Limits**: 500 calls/day (free tier)
- **Configuration**: `DATA_SOURCE_ALPHA_VANTAGE_API_KEY`, `DATA_SOURCE_ALPHA_VANTAGE_ENABLED`

#### 3. Finnhub
- **Status**: Optional (requires API key)
- **Data**: Company news
- **API**: REST API
- **Rate Limits**: 60 calls/minute (free tier)
- **Configuration**: `DATA_SOURCE_FINNHUB_API_KEY`, `DATA_SOURCE_FINNHUB_ENABLED`

#### 4. Reddit
- **Status**: Optional (requires app registration)
- **Data**: Social media posts from relevant subreddits
- **Library**: PRAW (Python Reddit API Wrapper)
- **Rate Limits**: 60 requests/minute (Reddit API limit)
- **Configuration**: `DATA_SOURCE_REDDIT_CLIENT_ID`, `DATA_SOURCE_REDDIT_CLIENT_SECRET`, `DATA_SOURCE_REDDIT_ENABLED`

### Data Collection Flow

1. **Primary Source**: Always fetch from yfinance
2. **Additional Sources**: Fetch from enabled sources in parallel
3. **Deduplication**: Remove duplicate articles by title and URL
4. **Source Tracking**: Track which source each article came from
5. **Caching**: Cache results in Redis with appropriate TTL

### Source Filtering

Users can enable/disable individual sources via UI:
- Toggle switches in sidebar
- Filters applied during data collection
- Logged for transparency

---

## Caching Strategy

### Cache Hierarchy

![Caching Diagram](diagrams/caching.png)

### Cache TTL Strategy

| Data Type | TTL | Rationale | Configurable |
|-----------|-----|-----------|--------------|
| Stock Data | 1 hour | Stock prices change frequently but not second-by-second | Yes |
| News Articles | 2 hours | News updates regularly but not constantly | Yes |
| Sentiment Results | 24 hours | Sentiment for same text doesn't change | Yes (can be disabled) |
| Article Embeddings | 7 days | Embeddings are stable, articles don't change | Yes |
| Query Embeddings | 24 hours | Same queries benefit from caching | Yes |

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

### Sentiment Cache Control

**Feature**: Configurable sentiment caching for RAG testing

**Options**:
- **Enable**: Cache sentiment results (default, reduces API calls)
- **Disable**: Force RAG usage for every analysis (useful for demos)

**TTL Control**: Slider from 0.1 to 168 hours (7 days)

**Use Case**: 
- Disable cache to see RAG in action
- Adjust TTL to test cache expiration behavior
- Monitor RAG usage in operation summary

---

## Mathematical Models & Algorithms

### 1. Cosine Similarity

**Purpose**: Measure semantic similarity between two embedding vectors.

**Formula**:
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` = dot product of vectors A and B
- `||A||` = Euclidean norm (magnitude) of vector A
- `||B||` = Euclidean norm (magnitude) of vector B
- `θ` = angle between vectors

**Range**: 0.0 (orthogonal, no similarity) to 1.0 (identical, perfect similarity)

**Implementation**:
```python
import numpy as np

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
```

**Use Case**: Semantic search in RAG (finding similar articles)

### 2. Reciprocal Rank Fusion (RRF)

**Purpose**: Combine multiple ranked lists without score normalization.

**Formula**:
```
RRF_score(d) = Σ(1 / (k + rank_i(d)))
```

Where:
- `d` = document/article
- `k` = RRF constant (typically 60)
- `rank_i(d)` = rank of document d in search result i
- Sum is over all search results containing document d

**Example**:
- Document appears at rank 1 in semantic search: `1 / (60 + 1) = 0.0164`
- Document appears at rank 2 in keyword search: `1 / (60 + 2) = 0.0161`
- Combined RRF score: `0.0164 + 0.0161 = 0.0325`

**Why RRF?**
- Works with different score scales (no normalization needed)
- Documents appearing high in multiple searches rank highest
- Proven method in information retrieval

**Implementation**:
```python
def reciprocal_rank_fusion(semantic_results, keyword_results, k=60):
    rrf_scores = {}
    
    # Add semantic search results
    for rank, article in enumerate(semantic_results, 1):
        article_id = article.get('article_id', '')
        if article_id:
            rrf_scores[article_id] = rrf_scores.get(article_id, 0) + (1 / (k + rank))
    
    # Add keyword search results
    for rank, article in enumerate(keyword_results, 1):
        article_id = article.get('article_id', '')
        if article_id:
            rrf_scores[article_id] = rrf_scores.get(article_id, 0) + (1 / (k + rank))
    
    # Sort by RRF score
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

**Use Case**: Hybrid search in RAG (combining semantic and keyword results)

### 3. Temporal Decay

**Purpose**: Boost recent articles in search results (financial news is time-sensitive).

**Formula**:
```
decay = 1.0 / (1 + age_days / decay_days)
boosted_score = current_score * (1 + decay * boost_factor)
```

Where:
- `age_days` = days since article publication
- `decay_days` = decay half-life (default: 7 days)
- `boost_factor` = 0.2 (20% maximum boost)

**Examples**:
- Article from today (age=0): `decay = 1.0 / (1 + 0/7) = 1.0`, boost = 20%
- Article from 7 days ago (age=7): `decay = 1.0 / (1 + 7/7) = 0.5`, boost = 10%
- Article from 30 days ago (age=30): `decay = 1.0 / (1 + 30/7) ≈ 0.19`, boost ≈ 3.8%

**Implementation**:
```python
def apply_temporal_decay(results, decay_days=7, boost_factor=0.2):
    from datetime import datetime
    now = datetime.now()
    
    for result in results:
        timestamp = result.get('timestamp', '')
        if not timestamp:
            continue
        
        article_time = parse_timestamp(timestamp)
        age_days = (now - article_time).days
        
        decay = max(0.1, 1.0 / (1 + age_days / decay_days))
        current_score = result.get('rrf_score', result.get('similarity', 0))
        boosted_score = current_score * (1 + decay * boost_factor)
        result['rrf_score'] = boosted_score
    
    return sorted(results, key=lambda x: x.get('rrf_score', 0), reverse=True)
```

**Use Case**: RAG retrieval (boosting recent articles)

### 4. Sentiment Score Normalization

**Purpose**: Ensure sentiment scores sum to 1.0 (probability distribution).

**Formula**:
```
normalized_score = raw_score / sum(all_scores)
```

**Example**:
- Raw scores: positive=0.7, negative=0.2, neutral=0.15
- Sum: 1.05
- Normalized: positive=0.667, negative=0.190, neutral=0.143

**Net Sentiment**:
```
net_sentiment = positive - negative
```

**Range**: -1.0 (completely negative) to +1.0 (completely positive)

**Dominant Sentiment**:
```
dominant = argmax(positive, negative, neutral)
```

**Use Case**: Sentiment analysis output formatting

### 5. Batch Embedding Generation

**Purpose**: Reduce API calls by processing multiple texts in one request.

**Efficiency Gain**:
- **One-by-one**: N articles = N API calls
- **Batch**: N articles = ⌈N / batch_size⌉ API calls
- **Improvement**: Up to batch_size× reduction in API calls

**Example**:
- 100 articles, batch_size=100: 100 calls → 1 call (100× improvement)
- 100 articles, batch_size=50: 100 calls → 2 calls (50× improvement)

**Implementation**:
```python
def get_embeddings_batch(texts, batch_size=100):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=embedding_deployment,
            input=batch
        )
        results.extend([item.embedding for item in response.data])
    return results
```

**Use Case**: RAG article storage (efficient embedding generation)

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
| `source` | str | Publisher name (Yahoo Finance, Alpha Vantage, Finnhub, Reddit) |
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
- **Configurable**: Can be adjusted via UI slider (0.1 to 168 hours)

**`cache_sentiment_enabled`** (bool, default: True)
- **Purpose**: Enable/disable sentiment caching
- **Use Case**: Disable to force RAG usage for every analysis (useful for demos)
- **Configurable**: Can be toggled via UI checkbox

**`cache_ttl_stock`** (int, default: 3600 seconds = 1 hour)
- **Purpose**: How long stock price data is cached
- **Rationale**: Stock prices change frequently but not second-by-second
- **Trade-off**: Balance between freshness and API rate limits

**`cache_ttl_news`** (int, default: 7200 seconds = 2 hours)
- **Purpose**: How long news articles are cached
- **Rationale**: News updates regularly but not constantly
- **Trade-off**: Fresh news vs. API costs

**`cache_ttl_rag_articles`** (int, default: 604800 seconds = 7 days)
- **Purpose**: How long RAG article embeddings are cached
- **Rationale**: Embeddings are stable, articles don't change

#### RAG Configuration

**`rag_top_k`** (int, default: 3)
- **Purpose**: Number of similar articles to retrieve for context
- **Impact**: 
  - Higher K = more context but potentially more noise
  - Lower K = less context but more focused
- **Recommendation**: 3-5 articles typically provides good balance

**`rag_similarity_threshold`** (float, default: 0.01)
- **Purpose**: Minimum similarity score for article inclusion
- **Note**: For RRF scores (typically 0.01-0.15), use 0.01-0.03. For cosine similarity (0.0-1.0), use 0.3-0.7.
- **Impact**: Higher threshold = higher quality but fewer articles
- **Auto-adjustment**: Automatically lowers if too restrictive

**`rag_batch_size`** (int, default: 100)
- **Purpose**: Number of articles to process per batch for embedding generation
- **Max**: 2048 (Azure OpenAI limit)
- **Impact**: Larger batches = fewer API calls but more memory usage
- **Recommendation**: 100-200 for optimal balance

**`rag_temporal_decay_days`** (int, default: 7)
- **Purpose**: Half-life for temporal decay calculation
- **Formula**: `decay = 1.0 / (1 + age_days / decay_days)`
- **Impact**: Lower value = steeper decay (recent articles more important)

**`rag_similarity_auto_adjust_multiplier`** (float, default: 0.8)
- **Purpose**: Multiplier for auto-adjusting similarity threshold when too high
- **Use Case**: If threshold filters all results, automatically lower it by this multiplier

### Azure AI Search Settings

**`endpoint`** (str, required)
- **Purpose**: Azure AI Search service endpoint URL
- **Format**: `https://{service-name}.search.windows.net`
- **Security**: Contains service location, should be kept private

**`api_key`** (str, required)
- **Purpose**: Authentication key for Azure AI Search API
- **Security**: Highly sensitive, never commit to version control
- **Rotation**: Should be rotated regularly

**`index_name`** (str, default: "stock-articles")
- **Purpose**: Name of the search index
- **Note**: Index is created automatically if it doesn't exist

**`vector_dimension`** (int, default: 1536)
- **Purpose**: Dimension of embedding vectors
- **Note**: Must match embedding model dimension (text-embedding-ada-002 = 1536)

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

---

## Technology Stack

### Frontend
- **Streamlit**: Python web framework for rapid UI development
- **Plotly**: Interactive data visualization library
- **Pandas**: Data manipulation and analysis

### Backend Services
- **Python 3.8+**: Programming language
- **Azure OpenAI**: Large Language Model service (GPT-4)
- **Azure AI Search**: Vector database and search service
- **Redis**: In-memory data store for caching
- **yfinance**: Financial data API wrapper

### Data Sources
- **yfinance**: Yahoo Finance (primary source, always enabled)
- **Alpha Vantage**: Company news API (optional)
- **Finnhub**: Financial data API (optional)
- **Reddit (PRAW)**: Social media sentiment (optional)

### Data Processing
- **NumPy**: Numerical computing for vector operations
- **TextBlob**: Fallback sentiment analysis library
- **Pydantic**: Data validation and settings management
- **dateutil**: Date parsing and manipulation

### Infrastructure
- **Azure Cloud**: Hosting for OpenAI, AI Search, and Redis
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
   - Batch embedding generation (reduces API calls from N to N/batch_size)
   - Parallel sentiment analysis (multi-threading)
   - Batch document upload to Azure AI Search

3. **Vector Search Optimization**
   - Azure AI Search (10-100x faster than Redis SCAN)
   - HNSW algorithm for fast approximate nearest neighbor search
   - Hybrid search with built-in RRF

4. **Embedding Caching**
   - Cache query embeddings to avoid regeneration
   - Store article embeddings for 7 days

### Performance Metrics

| Operation | Without Cache | With Cache | Improvement |
|-----------|---------------|------------|-------------|
| Stock Data Fetch | ~500ms | ~5ms | 100× faster |
| News Fetch | ~800ms | ~5ms | 160× faster |
| Sentiment Analysis | ~2000ms | ~5ms | 400× faster |
| RAG Retrieval (Redis SCAN) | ~300ms | ~50ms | 6× faster |
| RAG Retrieval (Azure AI Search) | ~50ms | ~50ms | 10-100× faster than Redis SCAN |

### Scalability Considerations

- **Horizontal Scaling**: Stateless design allows multiple instances
- **Azure AI Search**: Scales to millions of documents
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
   - Network isolation for Redis and AI Search

3. **Data Privacy**
   - No user data stored permanently
   - Session state cleared on app restart
   - Cache TTLs ensure data expiration

4. **Input Validation**
   - Stock symbol validation
   - Text sanitization
   - OData filter injection prevention (proper escaping)

### Best Practices

- **Key Rotation**: Rotate API keys regularly
- **Least Privilege**: Use minimal required permissions
- **Monitoring**: Log security events
- **Updates**: Keep dependencies updated

---

## Conclusion

This application demonstrates a production-ready architecture combining:

- **Modern AI**: Azure OpenAI GPT-4 for intelligent analysis
- **Advanced Techniques**: RAG with hybrid search for context-aware processing
- **High Performance**: Azure AI Search for 10-100× faster vector search
- **Multi-Source Data**: Comprehensive news coverage from multiple APIs
- **Performance**: Multi-level caching for optimization
- **User Experience**: Interactive Streamlit dashboard with modular architecture
- **Best Practices**: Clean architecture, proper error handling, comprehensive logging

The modular, layered design allows for easy extension and maintenance, making it suitable for both learning and production use. The architecture supports:

- **Scalability**: Can handle millions of articles with Azure AI Search
- **Flexibility**: Easy to add new data sources or features
- **Maintainability**: Clear separation of concerns
- **Testability**: Modular components are easy to test
- **Demo-Ready**: Comprehensive logging and operation summaries for demonstrations

---

**Document Version**: 2.0  
**Last Updated**: December 2024  
**Maintained By**: Development Team

