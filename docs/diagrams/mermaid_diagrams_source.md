# Mermaid Diagrams Source Code

This file contains all Mermaid diagram source code extracted from index.md.
This is kept for reference and future modifications.

**Total Diagrams:** 29

---

## Diagram 1

**Location:** Line 176 in index.md

**Context:** ## High-Level Architecture ### Complete System Overview

```mermaid
flowchart TB
    subgraph "User Interface"
        USER[User<br/>Stock Symbol Input]
        DASHBOARD[Streamlit Dashboard<br/>6 Tabs: Overview, Price, News,<br/>Technical, AI Insights, Comparison]
    end
    
    subgraph "Frontend Layer"
        API_CLIENT[API Client<br/>HTTP Requests<br/>Error Handling]
    end
    
    subgraph "API Layer (FastAPI)"
        API_SERVER[FastAPI Server<br/>Port 8000<br/>CORS Enabled]
        ROUTES[API Routes<br/>/sentiment, /price,<br/>/comparison, /system, /cache]
        VALIDATION[Request Validation<br/>Pydantic Models]
    end
    
    subgraph "Service Layer"
        ORCH[Orchestrator<br/>Pipeline Coordination]
        COLLECTOR[Data Collector<br/>Multi-source Collection]
        SENTIMENT[Sentiment Analyzer<br/>GPT-4 + RAG]
        RAG[RAG Service<br/>Hybrid Search]
    end
    
    subgraph "Storage Layer"
        REDIS[Redis Cache<br/>L2 Cache<br/>Stock, News, Sentiment, Embeddings]
        VECTOR_DB[Azure AI Search<br/>Vector Database<br/>HNSW Index]
    end
    
    subgraph "External APIs"
        YFINANCE[yfinance<br/>Stock + News]
        ALPHA[Alpha Vantage<br/>News API]
        FINN[Finnhub<br/>News API]
        REDDIT_API[Reddit<br/>Social Media]
        OPENAI[Azure OpenAI<br/>GPT-4 + Embeddings]
    end
    
    USER --> DASHBOARD
    DASHBOARD --> API_CLIENT
    API_CLIENT -->|HTTP| API_SERVER
    API_SERVER --> ROUTES
    ROUTES --> VALIDATION
    VALIDATION --> ORCH
    
    ORCH --> COLLECTOR
    ORCH --> SENTIMENT
    ORCH --> RAG
    
    COLLECTOR --> YFINANCE
    COLLECTOR --> ALPHA
    COLLECTOR --> FINN
    COLLECTOR --> REDDIT_API
    COLLECTOR --> REDIS
    
    SENTIMENT --> RAG
    SENTIMENT --> OPENAI
    SENTIMENT --> REDIS
    
    RAG --> VECTOR_DB
    RAG --> REDIS
    RAG --> OPENAI
    
    REDIS -.Cache Hit.-> COLLECTOR
    REDIS -.Cache Hit.-> SENTIMENT
    REDIS -.Cache Hit.-> RAG
    
    VECTOR_DB -.Vector Search.-> RAG
    
    style USER fill:#e1f5ff
    style DASHBOARD fill:#e1f5ff
    style API_SERVER fill:#fff4e1
    style ORCH fill:#ffe1f5
    style REDIS fill:#ffe1f5
    style VECTOR_DB fill:#e8f5e9
    style OPENAI fill:#fff9c4
```

---

## Diagram 2

**Location:** Line 264 in index.md

**Context:** ### System Architecture Overview

```mermaid
flowchart TB
    subgraph "Frontend Layer"
        STREAMLIT[Streamlit Dashboard<br/>User Interface]
        API_CLIENT[API Client<br/>HTTP Requests]
    end
    
    subgraph "API Layer"
        FASTAPI[FastAPI Server<br/>REST API Endpoints]
        ROUTES[API Routes<br/>sentiment, price, comparison, system, cache]
        MODELS[Response Models<br/>Pydantic Models]
    end
    
    subgraph "Service Layer"
        ORCHESTRATOR[Orchestrator<br/>Core Business Logic]
        COLLECTOR[Data Collector<br/>Multi-source Data Collection]
        SENTIMENT[Sentiment Analyzer<br/>AI Analysis with RAG]
        RAG[RAG Service<br/>Context Retrieval]
        CACHE[Redis Cache<br/>Caching Layer]
        VECTOR_DB[Azure AI Search<br/>Vector Database]
    end
    
    subgraph "Infrastructure Layer"
        AZURE_OPENAI[Azure OpenAI<br/>GPT-4 & Embeddings]
        REDIS[Redis Cache<br/>In-Memory Storage]
        AZURE_SEARCH[Azure AI Search<br/>Vector Search]
        YFINANCE[yfinance API<br/>Stock Data]
        ALPHA_VANTAGE[Alpha Vantage API<br/>News]
        FINNHUB[Finnhub API<br/>News]
        REDDIT_API[Reddit API<br/>Social Media]
    end
    
    STREAMLIT --> API_CLIENT
    API_CLIENT --> FASTAPI
    FASTAPI --> ROUTES
    ROUTES --> MODELS
    ROUTES --> ORCHESTRATOR
    ORCHESTRATOR --> COLLECTOR
    ORCHESTRATOR --> SENTIMENT
    ORCHESTRATOR --> RAG
    SENTIMENT --> RAG
    RAG --> VECTOR_DB
    RAG --> CACHE
    COLLECTOR --> CACHE
    SENTIMENT --> CACHE
    COLLECTOR --> YFINANCE
    COLLECTOR --> ALPHA_VANTAGE
    COLLECTOR --> FINNHUB
    COLLECTOR --> REDDIT_API
    SENTIMENT --> AZURE_OPENAI
    RAG --> AZURE_OPENAI
    CACHE --> REDIS
    VECTOR_DB --> AZURE_SEARCH
    
    style STREAMLIT fill:#e1f5ff
    style FASTAPI fill:#fff4e1
    style ORCHESTRATOR fill:#ffe1f5
    style RAG fill:#e8f5e9
    style VECTOR_DB fill:#e8f5e9
    style AZURE_OPENAI fill:#fff9c4
```

---

## Diagram 3

**Location:** Line 330 in index.md

**Context:** ### Component Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as Streamlit Frontend
    participant APIClient as API Client
    participant FastAPI as FastAPI Server
    participant Orchestrator as Orchestrator Service
    participant Collector as Data Collector
    participant RAG as RAG Service
    participant Sentiment as Sentiment Analyzer
    participant Cache as Redis Cache
    participant VectorDB as Azure AI Search
    
    User->>Streamlit: Enter stock symbol & click "Load Data"
    Streamlit->>APIClient: get_sentiment(symbol, detailed=True)
    APIClient->>FastAPI: GET /sentiment/{symbol}?detailed=true
    FastAPI->>Orchestrator: get_aggregated_sentiment(symbol)
    
    Orchestrator->>Cache: Check cached stock data
    alt Cache Hit
        Cache-->>Orchestrator: Return cached data
    else Cache Miss
        Orchestrator->>Collector: collect_all_data(symbol)
        Collector->>YFinance: Fetch stock data & news
        Collector->>AlphaVantage: Fetch news (if enabled)
        Collector->>Finnhub: Fetch news (if enabled)
        Collector->>Reddit: Fetch posts (if enabled)
        Collector-->>Orchestrator: Return collected data
        Orchestrator->>Cache: Store data in cache
    end
    
    Note over Orchestrator: STEP 2: Store in RAG
    Orchestrator->>RAG: store_articles_batch(articles, symbol)
    
    alt RAG Service Available
        Note over RAG: Check Embeddings Enabled
        alt Embeddings Enabled
            RAG->>RAG: Preprocess articles<br/>(clean, expand abbreviations)
            RAG->>Redis: Check duplicate markers<br/>(per article)
            RAG->>VectorDB: batch_check_documents_exist<br/>(if Azure AI Search available)
            alt Articles Already Exist
                VectorDB-->>RAG: Return existing count
                RAG->>Redis: Mark as stored
                RAG-->>Orchestrator: Return existing count
            else New Articles to Store
                RAG->>Redis: Check cached embeddings
                alt Embeddings Cached
                    Redis-->>RAG: Return cached embeddings
                else Embeddings Not Cached
                    RAG->>AzureOpenAI: get_embeddings_batch<br/>(batch_size=100, single API call)
                    alt Embedding Generation Success
                        AzureOpenAI-->>RAG: Return embeddings (1536 dims)
                        RAG->>Redis: Cache embeddings (7 days TTL)
                    else Embedding Generation Failed
                        RAG->>RAG: Log error, skip failed articles
                    end
                end
                
                alt Azure AI Search Available
                    RAG->>VectorDB: batch_store_vectors<br/>(vectors + metadata)
                    alt Storage Success
                        VectorDB->>VectorDB: Index with HNSW algorithm
                        VectorDB-->>RAG: Return stored count
                        RAG->>Redis: Set duplicate markers (7 days TTL)
                        RAG-->>Orchestrator: Return stored count
                    else Storage Failed
                        RAG->>RAG: Log error, return partial count
                        RAG-->>Orchestrator: Return stored count (partial)
                    end
                else Azure AI Search Unavailable
                    RAG->>Redis: Store in Redis SCAN format<br/>(fallback storage)
                    Redis-->>RAG: Storage confirmed
                    RAG-->>Orchestrator: Return stored count
                end
            end
        else Embeddings Disabled
            RAG-->>Orchestrator: Return 0 (embeddings not enabled)
        end
    else RAG Service Unavailable
        Note over Orchestrator: RAG skipped, continue without context
    end
    
    loop For each article
        Orchestrator->>Cache: Check sentiment cache
        alt Cache Hit
            Cache-->>Orchestrator: Return cached sentiment
        else Cache Miss
            Orchestrator->>RAG: retrieve_relevant_context(article, symbol)
            RAG->>VectorDB: Hybrid search (semantic + keyword)
            VectorDB-->>RAG: Return similar articles
            RAG-->>Orchestrator: Return context
            Orchestrator->>Sentiment: analyze_sentiment(article, context)
            Sentiment->>AzureOpenAI: GPT-4 API call with context
            AzureOpenAI-->>Sentiment: Return sentiment scores
            Sentiment-->>Orchestrator: Return sentiment
            Orchestrator->>Cache: Store sentiment in cache
        end
    end
    
    Orchestrator-->>FastAPI: Return aggregated sentiment + data
    FastAPI-->>APIClient: Return JSON response
    APIClient-->>Streamlit: Return data
    Streamlit->>User: Display results in dashboard
```

---

## Diagram 4

**Location:** Line 394 in index.md

**Context:** ### API-Driven Architecture Flow

```mermaid
flowchart LR
    subgraph "Frontend (Streamlit)"
        UI[User Interface]
        CLIENT[API Client]
    end
    
    subgraph "API Layer (FastAPI)"
        ENDPOINTS[API Endpoints]
        VALIDATION[Request Validation]
        ORCH[Orchestration]
    end
    
    subgraph "Service Layer"
        SERVICES[Business Logic Services]
    end
    
    subgraph "Infrastructure"
        EXTERNAL[External APIs & Databases]
    end
    
    UI -->|User Action| CLIENT
    CLIENT -->|HTTP Request| ENDPOINTS
    ENDPOINTS -->|Validate| VALIDATION
    VALIDATION -->|Route| ORCH
    ORCH -->|Call| SERVICES
    SERVICES -->|Fetch/Store| EXTERNAL
    EXTERNAL -->|Data| SERVICES
    SERVICES -->|Results| ORCH
    ORCH -->|Response| ENDPOINTS
    ENDPOINTS -->|JSON| CLIENT
    CLIENT -->|Update UI| UI
    
    style UI fill:#e1f5ff
    style ENDPOINTS fill:#fff4e1
    style SERVICES fill:#ffe1f5
    style EXTERNAL fill:#e8f5e9
```

---

## Diagram 5

**Location:** Line 437 in index.md

**Context:** ### Deployment Architecture

```mermaid
flowchart TB
    subgraph "User Browser"
        BROWSER[Web Browser]
    end
    
    subgraph "Frontend Service"
        STREAMLIT_SVC[Streamlit Service<br/>Port 8501]
    end
    
    subgraph "API Service"
        API_SVC[FastAPI Service<br/>Port 8000]
    end
    
    subgraph "Azure Cloud Services"
        AZURE_OPENAI_SVC[Azure OpenAI<br/>GPT-4 & Embeddings]
        AZURE_SEARCH_SVC[Azure AI Search<br/>Vector Database]
        REDIS_SVC[Azure Cache for Redis<br/>Caching]
    end
    
    subgraph "External APIs"
        YFINANCE_API[yfinance<br/>Public API]
        ALPHA_VANTAGE_API[Alpha Vantage<br/>REST API]
        FINNHUB_API[Finnhub<br/>REST API]
        REDDIT_API[Reddit<br/>REST API]
    end
    
    BROWSER -->|HTTP| STREAMLIT_SVC
    STREAMLIT_SVC -->|HTTP| API_SVC
    API_SVC -->|API Calls| AZURE_OPENAI_SVC
    API_SVC -->|Vector Search| AZURE_SEARCH_SVC
    API_SVC -->|Cache Operations| REDIS_SVC
    API_SVC -->|Data Collection| YFINANCE_API
    API_SVC -->|Data Collection| ALPHA_VANTAGE_API
    API_SVC -->|Data Collection| FINNHUB_API
    API_SVC -->|Data Collection| REDDIT_API
    
    style STREAMLIT_SVC fill:#e1f5ff
    style API_SVC fill:#fff4e1
    style AZURE_OPENAI_SVC fill:#fff9c4
    style AZURE_SEARCH_SVC fill:#e8f5e9
    style REDIS_SVC fill:#ffebee
```

---

## Diagram 6

**Location:** Line 543 in index.md

**Context:** ### Component Relationship Diagram

```mermaid
graph TB
    subgraph "Presentation Layer"
        APP[app.py<br/>Entry Point]
        INIT[initialization.py<br/>Service Setup]
        LOADER[data_loader.py<br/>Data Loading]
        CLIENT[api_client.py<br/>HTTP Client]
        SIDEBAR[sidebar.py<br/>UI Controls]
        TABS[tabs/*.py<br/>Tab Modules]
    end
    
    subgraph "API Layer"
        API_MAIN[api/main.py<br/>FastAPI App]
        API_DEPS[api/dependencies.py<br/>DI Container]
        API_ROUTES[api/routes/*.py<br/>Endpoints]
        API_MODELS[api/models/response.py<br/>Response Models]
    end
    
    subgraph "Service Layer"
        ORCH[services/orchestrator.py<br/>Orchestration]
        COLL[services/collector.py<br/>Data Collection]
        SENT[services/sentiment.py<br/>Sentiment Analysis]
        RAG[services/rag.py<br/>RAG Service]
        CACHE[services/cache.py<br/>Redis Cache]
        VDB[services/vector_db.py<br/>Vector DB]
    end
    
    subgraph "Infrastructure"
        CONFIG[config/settings.py<br/>Configuration]
        MODELS[models/*.py<br/>Data Models]
        UTILS[utils/*.py<br/>Utilities]
    end
    
    APP --> INIT
    APP --> LOADER
    APP --> SIDEBAR
    APP --> TABS
    LOADER --> CLIENT
    CLIENT --> API_MAIN
    SIDEBAR --> CLIENT
    
    API_MAIN --> API_ROUTES
    API_ROUTES --> API_DEPS
    API_ROUTES --> API_MODELS
    API_DEPS --> ORCH
    
    ORCH --> COLL
    ORCH --> SENT
    ORCH --> RAG
    SENT --> RAG
    SENT --> CACHE
    RAG --> VDB
    RAG --> CACHE
    COLL --> CACHE
    
    ORCH --> CONFIG
    COLL --> CONFIG
    SENT --> CONFIG
    RAG --> CONFIG
    CACHE --> CONFIG
    VDB --> CONFIG
    
    SENT --> MODELS
    COLL --> MODELS
    API_MODELS --> MODELS
    
    SENT --> UTILS
    RAG --> UTILS
    COLL --> UTILS
    
    style APP fill:#e1f5ff
    style API_MAIN fill:#fff4e1
    style ORCH fill:#ffe1f5
    style CONFIG fill:#e8f5e9
```

---

## Diagram 7

**Location:** Line 2975 in index.md

**Context:** #### User Request Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as Streamlit Frontend<br/>(app.py)
    participant APIClient as API Client<br/>(api_client.py)
    participant FastAPI as FastAPI Server<br/>(api/main.py)
    participant SentimentRoute as Sentiment Route<br/>(routes/sentiment.py)
    participant Orchestrator as Orchestrator<br/>(services/orchestrator.py)
    participant Collector as Data Collector<br/>(services/collector.py)
    participant RAG as RAG Service<br/>(services/rag.py)
    participant Sentiment as Sentiment Analyzer<br/>(services/sentiment.py)
    participant Cache as Redis Cache<br/>(services/cache.py)
    
    User->>Streamlit: 1. Enter symbol "AAPL"<br/>2. Click "Load Data"
    Streamlit->>APIClient: get_sentiment(symbol="AAPL", detailed=True)
    APIClient->>FastAPI: HTTP GET /sentiment/AAPL?detailed=true
    FastAPI->>SentimentRoute: Route to sentiment endpoint
    SentimentRoute->>Orchestrator: get_aggregated_sentiment(symbol, ...)
    
    Note over Orchestrator: STEP 1: Collect Data
    Orchestrator->>Cache: Check cached stock data
    alt Cache Hit
        Cache-->>Orchestrator: Return cached data
    else Cache Miss
        Orchestrator->>Collector: collect_all_data(symbol, data_source_filters)
            Note over Collector: Apply Source Filters
            Collector->>Collector: Validate symbol format
            alt Symbol Valid
                Collector->>YFinance: Fetch stock & news<br/>(always enabled)
                alt YFinance Success
                    YFinance-->>Collector: Return stock + news data
                else YFinance Error
                    Collector->>Collector: Log error, continue with other sources
                end
                
                alt Alpha Vantage Enabled
                    Collector->>AlphaVantage: Fetch news<br/>(if API key configured)
                    alt Alpha Vantage Success
                        AlphaVantage-->>Collector: Return news articles
                    else Alpha Vantage Error
                        Collector->>Collector: Log warning, skip source
                    end
                end
                
                alt Finnhub Enabled
                    Collector->>Finnhub: Fetch news<br/>(if API key configured)
                    alt Finnhub Success
                        Finnhub-->>Collector: Return news articles
                    else Finnhub Error
                        Collector->>Collector: Log warning, skip source
                    end
                end
                
                alt Reddit Enabled
                    Collector->>Reddit: Fetch posts<br/>(if credentials configured)
                    alt Reddit Success
                        Reddit-->>Collector: Return social posts
                    else Reddit Error
                        Collector->>Collector: Log warning, skip source
                    end
                end
                
                Note over Collector: Post-Processing
                Collector->>Collector: Deduplicate articles<br/>(by URL & title similarity)
                Collector->>Collector: Normalize article format<br/>(standard structure)
                Collector-->>Orchestrator: Return collected data
                Orchestrator->>Cache: Store data in cache<br/>(stock: 1h TTL, news: 2h TTL)
            else Symbol Invalid
                Collector-->>Orchestrator: Return error/default data
            end
        end
    
    Note over Orchestrator: STEP 2: Store in RAG
    Orchestrator->>RAG: store_articles_batch(articles, symbol)
    RAG->>AzureAISearch: Check existing articles
    RAG->>AzureOpenAI: Generate embeddings (batch)
    RAG->>AzureAISearch: Store articles with embeddings
    RAG-->>Orchestrator: Articles stored
    
    Note over Orchestrator: STEP 3: Analyze Sentiment
    loop For each article
        Orchestrator->>Cache: Check sentiment cache
        alt Cache Hit
            Cache-->>Orchestrator: Return cached sentiment
        else Cache Miss
            Orchestrator->>RAG: retrieve_relevant_context(article, symbol)
            RAG->>AzureAISearch: Hybrid search
            AzureAISearch-->>RAG: Return similar articles
            RAG-->>Orchestrator: Return context
            Orchestrator->>Sentiment: analyze_sentiment(article, context)
            Sentiment->>AzureOpenAI: GPT-4 API call with context
            AzureOpenAI-->>Sentiment: Return sentiment scores
            Sentiment-->>Orchestrator: Return sentiment
            Orchestrator->>Cache: Store sentiment in cache
        end
    end
    
    Orchestrator->>Orchestrator: Aggregate sentiment scores
    Orchestrator-->>SentimentRoute: Return aggregated result
    SentimentRoute-->>FastAPI: Return JSON response
    FastAPI-->>APIClient: HTTP 200 OK with JSON
    APIClient-->>Streamlit: Return data dictionary
    Streamlit->>Streamlit: Update session state
    Streamlit->>User: Display results in dashboard tabs
```

---

## Diagram 8

**Location:** Line 3079 in index.md

**Context:** #### Data Collection Flow Diagram

```mermaid
flowchart TD
    START([User clicks Load Data<br/>with symbol AAPL]) --> CHECK_CACHE{Check Redis Cache<br/>for stock & news}
    
    CHECK_CACHE -->|Cache Hit| RETURN_CACHED[Return Cached Data<br/>TTL: 1h stock, 2h news]
    CHECK_CACHE -->|Cache Miss| COLLECT[Start Data Collection]
    
    COLLECT --> VALIDATE[Validate Symbol<br/>Check format & length]
    
    VALIDATE -->|Invalid| ERROR_RETURN[Return Error/Default<br/>price: 0.0, empty news]
    VALIDATE -->|Valid| FILTER[Apply Source Filters<br/>yfinance: ✅<br/>Alpha Vantage: ✅<br/>Finnhub: ❌<br/>Reddit: ❌]
    
    FILTER --> YFINANCE[yfinance API<br/>Stock price + News<br/>Always enabled]
    FILTER --> ALPHA_VANTAGE{Alpha Vantage<br/>Enabled?}
    FILTER --> FINNHUB{Finnhub<br/>Enabled?}
    FILTER --> REDDIT{Reddit<br/>Enabled?}
    
    ALPHA_VANTAGE -->|Yes| ALPHA_FETCH[Alpha Vantage API<br/>Company News]
    ALPHA_VANTAGE -->|No| SKIP_ALPHA[Skip Alpha Vantage]
    FINNHUB -->|Yes| FINN_FETCH[Finnhub API<br/>Company News]
    FINNHUB -->|No| SKIP_FINN[Skip Finnhub]
    REDDIT -->|Yes| REDDIT_FETCH[Reddit API<br/>Social Posts]
    REDDIT -->|No| SKIP_REDDIT[Skip Reddit]
    
    YFINANCE --> PARALLEL[Parallel Collection<br/>All enabled sources]
    ALPHA_FETCH --> PARALLEL
    FINN_FETCH --> PARALLEL
    REDDIT_FETCH --> PARALLEL
    
    PARALLEL --> ERROR_HANDLE[Error Handling<br/>Log errors, continue with<br/>available sources]
    
    ERROR_HANDLE --> DEDUPE[Deduplicate Articles<br/>By URL & Title Similarity<br/>85% threshold]
    
    DEDUPE --> NORMALIZE[Normalize Article Format<br/>title, summary, source, url, timestamp<br/>Validate all fields present]
    
    NORMALIZE --> STORE_CACHE[Store in Redis Cache<br/>stock: 1h TTL<br/>news: 2h TTL]
    
    STORE_CACHE --> RETURN[Return Data Dictionary<br/>price_data, news]
    ERROR_RETURN --> RETURN
    
    RETURN_CACHED --> RETURN
    
    RETURN --> END([Data Ready for Processing])
    
    style START fill:#e1f5ff
    style CHECK_CACHE fill:#fff4e1
    style YFINANCE fill:#e8f5e9
    style ALPHA_VANTAGE fill:#e8f5e9
    style DEDUPE fill:#ffe1f5
    style STORE_CACHE fill:#fff9c4
    style END fill:#e1f5ff
```

---

## Diagram 9

**Location:** Line 3156 in index.md

**Context:** #### RAG Storage Flow Diagram

```mermaid
sequenceDiagram
    participant Orchestrator as Orchestrator
    participant RAG as RAG Service
    participant Redis as Redis Cache
    participant VectorDB as Azure AI Search
    participant OpenAI as Azure OpenAI<br/>(Embeddings)
    
    Orchestrator->>RAG: store_articles_batch(articles, symbol)
    
    Note over RAG: STEP 1: Validate & Prepare
    RAG->>RAG: Check embeddings enabled<br/>Check Redis available
    alt Embeddings Disabled or Redis Unavailable
        RAG-->>Orchestrator: Return 0 (cannot store)
    else Service Available
        Note over RAG: STEP 2: Prepare Articles
        RAG->>RAG: Preprocess articles<br/>(clean text, expand abbreviations,<br/>filter financial text)
        RAG->>RAG: Create article IDs<br/>(MD5 hash of symbol:title:url)
        RAG->>RAG: Filter empty texts<br/>Skip articles with no content
        
        Note over RAG: STEP 3: Check Duplicates (per article)
        loop For each article
            RAG->>Redis: Check duplicate marker<br/>(article_hash:SYMBOL:ID)<br/>EXISTS operation
            alt Already in Redis
                Redis-->>RAG: Duplicate found (TTL active)
                Note over RAG: Skip this article, increment count
            else Not in Redis
                Note over RAG: Article needs processing<br/>Add to processing list
            end
        end
        
        alt All Articles Duplicates
            RAG-->>Orchestrator: Return duplicate count<br/>(all already stored)
        else Some New Articles
            Note over RAG: STEP 4: Check Azure AI Search (batch)
            alt Azure AI Search Available
                RAG->>VectorDB: batch_check_documents_exist(vector_ids)<br/>Check before embedding generation
                VectorDB-->>RAG: Return existing document IDs
                
                alt All articles already in Azure AI Search
                    RAG->>Redis: Mark as stored<br/>(for duplicate checking, 7 days TTL)
                    RAG-->>Orchestrator: Return existing count<br/>(no embeddings needed)
                else Some new articles
                    Note over RAG: STEP 5: Generate Embeddings (batch)
                    RAG->>Redis: Check cached embeddings<br/>(per article, 7 days TTL)
                    alt Some embeddings cached
                        Redis-->>RAG: Return cached embeddings<br/>(skip generation for these)
                    end
                    alt Some embeddings not cached
                        RAG->>OpenAI: get_embeddings_batch(texts, batch_size=100)
                        Note over OpenAI: Single API call for<br/>all articles (batch processing)<br/>Up to 2048 inputs per call
                        alt Embedding Success
                            OpenAI-->>RAG: Return embeddings<br/>(1536 dims each)
                            RAG->>Redis: Cache embeddings<br/>(7 days TTL, per article)
                        else Embedding Failed
                            RAG->>RAG: Log error<br/>Skip failed articles<br/>Continue with successful ones
                        end
                    end
                    
                    Note over RAG: STEP 6: Store in Azure AI Search
                    alt Has Valid Embeddings
                        RAG->>VectorDB: batch_store_vectors<br/>(vectors + metadata)<br/>Replace colon in IDs
                        alt Storage Success
                            VectorDB->>VectorDB: Index with HNSW algorithm<br/>O(N log N) build time
                            VectorDB-->>RAG: Return stored count
                            
                            Note over RAG: STEP 7: Mark in Redis
                            RAG->>Redis: Set duplicate markers<br/>(article_hash:SYMBOL:ID, 7 days TTL)
                            
                            RAG-->>Orchestrator: Return total stored count<br/>(existing + newly stored)
                        else Storage Failed
                            RAG->>RAG: Log error<br/>Return partial count
                            RAG-->>Orchestrator: Return stored count (partial)
                        end
                    else No Valid Embeddings
                        RAG-->>Orchestrator: Return 0 (all embeddings failed)
                    end
                end
            else Azure AI Search Unavailable
                Note over RAG: Fallback to Redis Storage
                RAG->>Redis: Store embeddings + metadata<br/>(Redis SCAN format)
                Redis-->>RAG: Storage confirmed
                RAG->>Redis: Set duplicate markers<br/>(7 days TTL)
                RAG-->>Orchestrator: Return stored count<br/>(Redis fallback)
            end
        end
    end
```

---

## Diagram 10

**Location:** Line 3251 in index.md

**Context:** #### Sentiment Analysis Flow Diagram

```mermaid
sequenceDiagram
    participant Orchestrator as Orchestrator
    participant Sentiment as Sentiment Analyzer
    participant Cache as Redis Cache
    participant RAG as RAG Service
    participant VectorDB as Azure AI Search
    participant OpenAI as Azure OpenAI GPT-4
    
    Orchestrator->>Sentiment: analyze_sentiment(text, symbol)
    
    Note over Sentiment: STEP 1: Preprocess Text
    Sentiment->>Sentiment: preprocess_text(text)<br/>(remove HTML, expand abbreviations)
    
    Note over Sentiment: STEP 2: Check Cache
    Sentiment->>Cache: get_cached_sentiment(text)
    alt Cache Hit (if enabled)
        Cache-->>Sentiment: Return cached scores
        Sentiment-->>Orchestrator: Return cached result<br/>(skip RAG & LLM)
    else Cache Miss
        Note over Sentiment: STEP 3: Retrieve RAG Context
        Sentiment->>RAG: retrieve_relevant_context(text, symbol)
        
        Note over RAG: RAG Retrieval Process
        RAG->>RAG: Generate query embedding
        RAG->>VectorDB: Hybrid search<br/>(semantic + keyword)
        VectorDB-->>RAG: Return top K similar articles
        RAG->>RAG: Apply temporal decay
        RAG->>RAG: Apply RRF combination
        RAG-->>Sentiment: Return context articles
        
        Note over Sentiment: STEP 4: Build Prompt
        Sentiment->>Sentiment: Format RAG context<br/>Build prompt with context
        
            Note over Sentiment: STEP 5: Call LLM (with Circuit Breaker)
        Sentiment->>Sentiment: Check Circuit Breaker State
        alt Circuit Breaker OPEN
            Sentiment->>Sentiment: Use TextBlob Fallback<br/>(prevent cascading failures)
            Sentiment-->>Orchestrator: Return TextBlob sentiment
        else Circuit Breaker CLOSED
            Sentiment->>Sentiment: Retry with Exponential Backoff<br/>(max_attempts, initial_delay)
            Sentiment->>OpenAI: GPT-4 API call<br/>(text + RAG context + few-shot examples)
            alt API Success
                OpenAI-->>Sentiment: Return JSON response<br/>{"positive": 0.85, "negative": 0.10, "neutral": 0.05}
                
                Note over Sentiment: STEP 6: Parse & Normalize
                Sentiment->>Sentiment: Parse JSON response
                alt JSON Parse Success
                    Sentiment->>Sentiment: Normalize scores (sum to 1.0)
                    Sentiment->>Sentiment: Validate score ranges<br/>(0.0 ≤ score ≤ 1.0)
                    
                    Note over Sentiment: STEP 7: Cache Result
                    Sentiment->>Cache: cache_sentiment(text, scores, TTL)
                    
                    Sentiment-->>Orchestrator: Return sentiment scores
                else JSON Parse Failed
                    Sentiment->>Sentiment: Extract JSON from markdown/text<br/>(regex fallback)
                    alt Extraction Success
                        Sentiment->>Sentiment: Normalize & validate
                        Sentiment->>Cache: cache_sentiment(text, scores, TTL)
                        Sentiment-->>Orchestrator: Return sentiment scores
                    else Extraction Failed
                        Sentiment->>Sentiment: Use TextBlob Fallback
                        Sentiment-->>Orchestrator: Return TextBlob sentiment
                    end
                end
            else API Error (Retry Exhausted)
                Sentiment->>Sentiment: Circuit Breaker Trip<br/>(increment failure count)
                Sentiment->>Sentiment: Use TextBlob Fallback
                Sentiment-->>Orchestrator: Return TextBlob sentiment
            end
        end
    end
```

---

## Diagram 11

**Location:** Line 3302 in index.md

**Context:** Diagram 11

```mermaid
flowchart TD
    START([Batch Analyze:<br/>30 articles]) --> POOL[ThreadPoolExecutor<br/>max_workers=5]
    
    POOL --> WORKER1[Worker 1:<br/>Article 1-6]
    POOL --> WORKER2[Worker 2:<br/>Article 7-12]
    POOL --> WORKER3[Worker 3:<br/>Article 13-18]
    POOL --> WORKER4[Worker 4:<br/>Article 19-24]
    POOL --> WORKER5[Worker 5:<br/>Article 25-30]
    
    WORKER1 --> ANALYZE1["Analyze Sentiment<br/>1. Preprocess text<br/>2. Check cache<br/>3. RAG retrieval<br/>4. LLM call with retry<br/>5. Parse and normalize<br/>6. Cache result"]
    WORKER2 --> ANALYZE2["Analyze Sentiment<br/>1. Preprocess text<br/>2. Check cache<br/>3. RAG retrieval<br/>4. LLM call with retry<br/>5. Parse and normalize<br/>6. Cache result"]
    WORKER3 --> ANALYZE3["Analyze Sentiment<br/>1. Preprocess text<br/>2. Check cache<br/>3. RAG retrieval<br/>4. LLM call with retry<br/>5. Parse and normalize<br/>6. Cache result"]
    WORKER4 --> ANALYZE4["Analyze Sentiment<br/>1. Preprocess text<br/>2. Check cache<br/>3. RAG retrieval<br/>4. LLM call with retry<br/>5. Parse and normalize<br/>6. Cache result"]
    WORKER5 --> ANALYZE5["Analyze Sentiment<br/>1. Preprocess text<br/>2. Check cache<br/>3. RAG retrieval<br/>4. LLM call with retry<br/>5. Parse and normalize<br/>6. Cache result"]
    
    ANALYZE1 --> ERROR_HANDLE1{Error?}
    ANALYZE2 --> ERROR_HANDLE2{Error?}
    ANALYZE3 --> ERROR_HANDLE3{Error?}
    ANALYZE4 --> ERROR_HANDLE4{Error?}
    ANALYZE5 --> ERROR_HANDLE5{Error?}
    
    ERROR_HANDLE1 -->|Success| COLLECT[Collect Results]
    ERROR_HANDLE1 -->|Error| FALLBACK1["TextBlob Fallback<br/>Return neutral sentiment"]
    ERROR_HANDLE2 -->|Success| COLLECT
    ERROR_HANDLE2 -->|Error| FALLBACK2[TextBlob Fallback]
    ERROR_HANDLE3 -->|Success| COLLECT
    ERROR_HANDLE3 -->|Error| FALLBACK3[TextBlob Fallback]
    ERROR_HANDLE4 -->|Success| COLLECT
    ERROR_HANDLE4 -->|Error| FALLBACK4[TextBlob Fallback]
    ERROR_HANDLE5 -->|Success| COLLECT
    ERROR_HANDLE5 -->|Error| FALLBACK5[TextBlob Fallback]
    
    FALLBACK1 --> COLLECT
    FALLBACK2 --> COLLECT
    FALLBACK3 --> COLLECT
    FALLBACK4 --> COLLECT
    FALLBACK5 --> COLLECT
    
    COLLECT --> AGGREGATE[Aggregate Sentiment Scores<br/>Average positive, negative, neutral]
    AGGREGATE --> END([Return Aggregated Results])
    
    style START fill:#e1f5ff
    style POOL fill:#fff4e1
    style ANALYZE1 fill:#ffe1f5
    style ANALYZE2 fill:#ffe1f5
    style ANALYZE3 fill:#ffe1f5
    style ANALYZE4 fill:#ffe1f5
    style ANALYZE5 fill:#ffe1f5
    style END fill:#e1f5ff
```

---

## Diagram 12

**Location:** Line 3375 in index.md

**Context:** #### RAG Retrieval Flow Diagram

```mermaid
flowchart TD
    START([Query: Apple earnings report<br/>Symbol: AAPL]) --> PREPROCESS[Preprocess Query<br/>Clean & normalize]
    
    PREPROCESS --> VALIDATE_QUERY{Query Valid?<br/>Non-empty, length check}
    
    VALIDATE_QUERY -->|Invalid| RETURN_EMPTY([Return Empty Context])
    VALIDATE_QUERY -->|Valid| EXPAND{Expand Query?}
    
    EXPAND -->|Yes| EXPANDED[Expanded Query:<br/>Apple earnings report<br/>profits revenue results financial]
    EXPAND -->|No| ORIGINAL[Original Query]
    
    EXPANDED --> CHECK_EMBED_CACHE{Query Embedding<br/>Cached?}
    ORIGINAL --> CHECK_EMBED_CACHE
    
    CHECK_EMBED_CACHE -->|Cached| GET_CACHED[Get Cached Embedding<br/>24h TTL]
    CHECK_EMBED_CACHE -->|Not Cached| EMBED[Generate Query Embedding<br/>Azure OpenAI<br/>1536 dimensions]
    
    EMBED -->|Success| CACHE_EMBED[Cache Embedding<br/>24h TTL]
    EMBED -->|Failed| FALLBACK_KEYWORD[Fallback to Keyword-Only<br/>Search]
    
    GET_CACHED --> HYBRID[Hybrid Search]
    CACHE_EMBED --> HYBRID
    FALLBACK_KEYWORD --> KEYWORD_ONLY[Keyword Search Only<br/>BM25 Algorithm]
    
    HYBRID --> CHECK_VECTOR_DB{Azure AI Search<br/>Available?}
    
    CHECK_VECTOR_DB -->|Yes| SEMANTIC[Semantic Search<br/>Vector Similarity<br/>Cosine Similarity]
    CHECK_VECTOR_DB -->|Yes| KEYWORD[Keyword Search<br/>Full-Text Search<br/>BM25 Algorithm]
    CHECK_VECTOR_DB -->|No| REDIS_FALLBACK[Redis SCAN Fallback<br/>Calculate cosine similarity<br/>in-memory]
    
    SEMANTIC --> FILTER1[Apply OData Filters<br/>symbol eq 'AAPL'<br/>date_range, sources]
    KEYWORD --> FILTER2[Apply OData Filters<br/>symbol eq 'AAPL'<br/>date_range, sources]
    
    FILTER1 --> RESULTS1[Semantic Results<br/>Ranked by similarity<br/>0.0 - 1.0]
    FILTER2 --> RESULTS2[Keyword Results<br/>Ranked by BM25 score<br/>0 - 100]
    
    REDIS_FALLBACK --> REDIS_RESULTS[Redis Results<br/>Sorted by similarity]
    
    RESULTS1 --> RRF[Reciprocal Rank Fusion<br/>RRF score calculation<br/>k=60 constant]
    RESULTS2 --> RRF
    REDIS_RESULTS --> TEMPORAL
    
    RRF --> COMBINED[Combined Results<br/>Articles appearing in both<br/>rank highest]
    
    COMBINED --> TEMPORAL[Apply Temporal Decay<br/>Boost recent articles<br/>decay formula applied]
    
    TEMPORAL --> THRESHOLD{Filter by<br/>Similarity Threshold}
    
    THRESHOLD -->|Too Restrictive<br/>No results| AUTO_ADJUST[Auto-Adjust Threshold<br/>Lower by 20%<br/>Retry search]
    AUTO_ADJUST --> THRESHOLD
    
    THRESHOLD -->|Pass| CHECK_RESULTS{Results<br/>Found?}
    
    CHECK_RESULTS -->|No Results| LOG_WARNING[Log Warning<br/>Indexing delay or<br/>filter too restrictive]
    CHECK_RESULTS -->|Results Found| TOP_K[Select Top K Articles<br/>Default: 3<br/>Up to top_k]
    
    LOG_WARNING --> RETURN_EMPTY_FINAL([Return Empty Context<br/>Proceed without RAG])
    TOP_K --> FORMAT[Format Context<br/>Title, Summary, Source,<br/>Relevance Score, Timestamp]
    
    FORMAT --> VALIDATE_CONTEXT{Context<br/>Valid?}
    
    VALIDATE_CONTEXT -->|Invalid| RETURN_EMPTY_FINAL
    VALIDATE_CONTEXT -->|Valid| END([Return Context Articles<br/>for LLM Prompt])
    
    style START fill:#e1f5ff
    style EMBED fill:#fff9c4
    style SEMANTIC fill:#e8f5e9
    style KEYWORD fill:#e8f5e9
    style RRF fill:#ffe1f5
    style TEMPORAL fill:#fff4e1
    style END fill:#e1f5ff
```

---

## Diagram 13

**Location:** Line 3426 in index.md

**Context:** Diagram 13

```mermaid
flowchart LR
    subgraph "Semantic Search Results"
        S1[Article A: rank 1<br/>similarity: 0.92]
        S2[Article B: rank 2<br/>similarity: 0.88]
        S3[Article C: rank 3<br/>similarity: 0.85]
    end
    
    subgraph "Keyword Search Results"
        K1[Article D: rank 1<br/>score: 95]
        K2[Article A: rank 2<br/>score: 88]
        K3[Article E: rank 3<br/>score: 82]
    end
    
    subgraph "RRF Calculation"
        RRF1[Article A:<br/>RRF calculation<br/>Score: 0.0325<br/>Appears in both lists]
        RRF2[Article B:<br/>RRF calculation<br/>Score: 0.0161]
        RRF3[Article D:<br/>RRF calculation<br/>Score: 0.0164]
    end
    
    subgraph "Final Ranking"
        F1[1. Article A: 0.0325<br/>Appears in both!]
        F2[2. Article D: 0.0164]
        F3[3. Article B: 0.0161]
    end
    
    S1 --> RRF1
    K2 --> RRF1
    S2 --> RRF2
    K1 --> RRF3
    
    RRF1 --> F1
    RRF2 --> F3
    RRF3 --> F2
    
    style RRF1 fill:#ffe1f5
    style F1 fill:#e8f5e9
```

---

## Diagram 14

**Location:** Line 3548 in index.md

**Context:** #### Cosine Similarity Visualization

```mermaid
graph TB
    subgraph "Vector Space (2D Example)"
        A[Article A Vector<br/>0.5, 0.3]
        B[Article B Vector<br/>0.4, 0.5]
        O[Origin 0, 0]
    end
    
    subgraph "Calculation"
        DOT[Dot Product<br/>A · B = 0.35]
        MAG_A[Magnitude A<br/>norm A = 0.583]
        MAG_B[Magnitude B<br/>norm B = 0.640]
        COS[Cosine Similarity<br/>cos θ = 0.35 / 0.373<br/>= 0.938]
    end
    
    subgraph "Interpretation"
        HIGH[High Similarity<br/>0.938 ≈ 94%<br/>Very Similar Content]
    end
    
    A --> DOT
    B --> DOT
    A --> MAG_A
    B --> MAG_B
    MAG_A --> COS
    MAG_B --> COS
    DOT --> COS
    COS --> HIGH
    
    style A fill:#e8f5e9
    style B fill:#e8f5e9
    style COS fill:#fff9c4
    style HIGH fill:#ffe1f5
```

---

## Diagram 15

**Location:** Line 3685 in index.md

**Context:** #### RRF Calculation Flow Diagram

```mermaid
flowchart TD
    START([Two Ranked Lists:<br/>Semantic + Keyword]) --> SEMANTIC[Semantic Search Results<br/>Rank 1: Article A<br/>Rank 2: Article B<br/>Rank 3: Article C]
    
    START --> KEYWORD[Keyword Search Results<br/>Rank 1: Article D<br/>Rank 2: Article A<br/>Rank 3: Article E]
    
    SEMANTIC --> RRF1[Calculate RRF for Semantic<br/>Article A: 1/61 = 0.0164<br/>Article B: 1/62 = 0.0161<br/>Article C: 1/63 = 0.0159]
    
    KEYWORD --> RRF2[Calculate RRF for Keyword<br/>Article D: 1/61 = 0.0164<br/>Article A: 1/62 = 0.0161<br/>Article E: 1/63 = 0.0159]
    
    RRF1 --> COMBINE[Combine RRF Scores<br/>Sum scores for each article]
    RRF2 --> COMBINE
    
    COMBINE --> CALC[Article A: 0.0164 + 0.0161 = 0.0325<br/>Article B: 0.0161 + 0 = 0.0161<br/>Article C: 0.0159 + 0 = 0.0159<br/>Article D: 0 + 0.0164 = 0.0164<br/>Article E: 0 + 0.0159 = 0.0159]
    
    CALC --> SORT[Sort by RRF Score<br/>Descending Order]
    
    SORT --> RESULT[Final Ranking:<br/>1. Article A: 0.0325<br/>2. Article D: 0.0164<br/>3. Article B: 0.0161<br/>4. Article E: 0.0159<br/>5. Article C: 0.0159]
    
    style START fill:#e1f5ff
    style COMBINE fill:#fff9c4
    style RESULT fill:#e8f5e9
```

---

## Diagram 16

**Location:** Line 3799 in index.md

**Context:** #### Temporal Decay Curve Visualization

```mermaid
graph LR
    subgraph "Decay Function"
        FORMULA[decay = 1.0 / 1 + age/7]
    end
    
    subgraph "Examples"
        DAY0[Day 0: decay = 1.0<br/>Boost: 20%]
        DAY3[Day 3: decay = 0.70<br/>Boost: 14%]
        DAY7[Day 7: decay = 0.50<br/>Boost: 10%]
        DAY14[Day 14: decay = 0.33<br/>Boost: 6.6%]
        DAY30[Day 30: decay = 0.19<br/>Boost: 3.8%]
        DAY60[Day 60: decay = 0.10<br/>Boost: 2.0%]
    end
    
    subgraph "Visual Curve"
        CURVE[Decay Value vs Age<br/>1.0 ┤<br/>0.8 ┤<br/>0.6 ┤    ╲<br/>0.4 ┤      ╲<br/>0.2 ┤        ╲___<br/>0.0 └───────────────<br/>    0  7  14  21  28<br/>    Days Old]
    end
    
    FORMULA --> DAY0
    FORMULA --> DAY7
    FORMULA --> DAY30
    
    DAY0 --> CURVE
    DAY7 --> CURVE
    DAY30 --> CURVE
    
    style FORMULA fill:#fff9c4
    style DAY0 fill:#e8f5e9
    style DAY7 fill:#fff4e1
    style DAY30 fill:#ffe1f5
```

---

## Diagram 17

**Location:** Line 3965 in index.md

**Context:** #### Normalization Process Diagram

```mermaid
flowchart TD
    START([LLM Returns Raw Scores]) --> CHECK{Sum = 1.0?}
    
    CHECK -->|Yes| VALID[Scores Already Normalized<br/>No action needed]
    CHECK -->|No| CALC[Calculate Total<br/>total = Σ scores]
    
    CALC --> DIVIDE[Divide Each Score by Total<br/>normalized_i = score_i / total]
    
    DIVIDE --> VERIFY[Verify Sum = 1.0<br/>Σ normalized = 1.0]
    
    VERIFY --> CLAMP[Clamp to Valid Range<br/>0.0 ≤ score ≤ 1.0]
    
    CLAMP --> EDGE{Total = 0?}
    
    EDGE -->|Yes| FALLBACK[Fallback to Neutral<br/>positive = 0.0<br/>negative = 0.0<br/>neutral = 1.0]
    EDGE -->|No| FINAL[Final Normalized Scores<br/>Sum = 1.0]
    
    VALID --> FINAL
    FALLBACK --> FINAL
    
    FINAL --> END([Return Normalized Scores])
    
    style START fill:#e1f5ff
    style CALC fill:#fff9c4
    style FINAL fill:#e8f5e9
    style END fill:#e1f5ff
```

---

## Diagram 18

**Location:** Line 4091 in index.md

**Context:** #### Batch Processing Flow Diagram

```mermaid
flowchart TD
    START([250 Articles to Embed]) --> SPLIT[Split into Batches<br/>Batch 1: 100 articles<br/>Batch 2: 100 articles<br/>Batch 3: 50 articles]
    
    SPLIT --> BATCH1[Batch 1 API Call<br/>100 articles → 1 call<br/>Latency: 200ms]
    SPLIT --> BATCH2[Batch 2 API Call<br/>100 articles → 1 call<br/>Latency: 200ms]
    SPLIT --> BATCH3[Batch 3 API Call<br/>50 articles → 1 call<br/>Latency: 150ms]
    
    BATCH1 --> PARALLEL[Parallel Processing<br/>All batches processed<br/>concurrently if possible]
    BATCH2 --> PARALLEL
    BATCH3 --> PARALLEL
    
    PARALLEL --> COLLECT[Collect All Embeddings<br/>250 embeddings total]
    
    COLLECT --> CACHE[Cache Embeddings in Redis<br/>TTL: 7 days]
    
    CACHE --> END([250 Embeddings Ready<br/>Total Time: 0.6s<br/>Total Cost: $0.0003])
    
    subgraph "Comparison: Individual Calls"
        INDIV[250 Individual Calls<br/>Latency: 12.5s<br/>Cost: $0.025]
    end
    
    style START fill:#e1f5ff
    style PARALLEL fill:#fff9c4
    style END fill:#e8f5e9
    style INDIV fill:#ffe1f5
```

---

## Diagram 19

**Location:** Line 4194 in index.md

**Context:** #### HNSW Graph Structure Visualization

```mermaid
graph TB
    subgraph "Layer 2 (Sparse - Entry Point)"
        L2A[Node A]
        L2B[Node B]
        L2C[Node C]
        L2A --- L2B
        L2B --- L2C
    end
    
    subgraph "Layer 1 (Medium Density)"
        L1A[Node A]
        L1B[Node B]
        L1C[Node C]
        L1D[Node D]
        L1E[Node E]
        L1A --- L1B
        L1B --- L1C
        L1C --- L1D
        L1D --- L1E
        L1A --- L1D
    end
    
    subgraph "Layer 0 (Dense - All Nodes)"
        L0A[Node A]
        L0B[Node B]
        L0C[Node C]
        L0D[Node D]
        L0E[Node E]
        L0F[Node F]
        L0G[Node G]
        L0H[Node H]
        L0A --- L0B
        L0B --- L0C
        L0C --- L0D
        L0D --- L0E
        L0E --- L0F
        L0F --- L0G
        L0G --- L0H
        L0A --- L0D
        L0B --- L0E
        L0C --- L0F
        L0D --- L0G
    end
    
    L2A -.-> L1A
    L2B -.-> L1B
    L2C -.-> L1C
    
    L1A -.-> L0A
    L1B -.-> L0B
    L1C -.-> L0C
    L1D -.-> L0D
    L1E -.-> L0E
    
    style L2A fill:#fff9c4
    style L1A fill:#e8f5e9
    style L0A fill:#ffe1f5
```

---

## Diagram 20

**Location:** Line 4256 in index.md

**Context:** Diagram 20

```mermaid
sequenceDiagram
    participant Query as Query Vector
    participant L2 as Layer 2<br/>(Sparse)
    participant L1 as Layer 1<br/>(Medium)
    participant L0 as Layer 0<br/>(Dense)
    participant Result as Top K Results
    
    Query->>L2: 1. Start at Layer 2<br/>Find entry point (Node A)
    L2->>L2: 2. Navigate to nearest<br/>neighbor (Node B)
    L2->>L1: 3. Move down to Layer 1<br/>at Node B
    L1->>L1: 4. Navigate to nearest<br/>neighbor (Node D)
    L1->>L0: 5. Move down to Layer 0<br/>at Node D
    L0->>L0: 6. Search local neighborhood<br/>Find top K nearest
    L0->>Result: 7. Return top K results<br/>(e.g., Nodes D, E, F)
    
    Note over Query,Result: Total comparisons: ~20<br/>vs. 1000 for brute force
```

---

## Diagram 21

**Location:** Line 4313 in index.md

**Context:** ### 6.1.1 Data Source Integration Architecture

```mermaid
flowchart TB
    subgraph "User Request"
        USER[User Request<br/>Symbol: AAPL]
    end
    
    subgraph "Data Collector Service"
        COLLECTOR[StockDataCollector<br/>collect_all_data]
        FILTER[Source Filtering<br/>Apply user filters]
        DEDUPE[Deduplication<br/>Remove duplicates]
        NORMALIZE[Data Normalization<br/>Standard format]
    end
    
    subgraph "Data Sources"
        YFINANCE[yfinance API<br/>✅ Always Enabled<br/>Stock + News]
        ALPHA[Alpha Vantage API<br/>⚠️ Optional<br/>Requires API Key<br/>Company News]
        FINN[Finnhub API<br/>⚠️ Optional<br/>Requires API Key<br/>Financial News]
        REDDIT[Reddit API<br/>⚠️ Optional<br/>Requires Credentials<br/>Social Media]
    end
    
    subgraph "Cache Layer"
        REDIS_CACHE[Redis Cache<br/>Check before fetch<br/>Store after fetch]
    end
    
    subgraph "Output"
        RESULT[Combined Results<br/>price_data + news]
    end
    
    USER --> COLLECTOR
    COLLECTOR --> VALIDATE[Validate Symbol<br/>Format check]
    VALIDATE --> FILTER
    FILTER --> REDIS_CACHE
    
    REDIS_CACHE -->|Cache Hit| UPDATE_STATS[Update Cache Stats<br/>Increment hits]
    REDIS_CACHE -->|Cache Miss| YFINANCE
    REDIS_CACHE -->|Cache Miss| ALPHA
    REDIS_CACHE -->|Cache Miss| FINN
    REDIS_CACHE -->|Cache Miss| REDDIT
    
    UPDATE_STATS --> RESULT
    
    YFINANCE --> ERROR_HANDLE[Error Handling<br/>Log per source<br/>Continue with others]
    ALPHA --> ERROR_HANDLE
    FINN --> ERROR_HANDLE
    REDDIT --> ERROR_HANDLE
    
    ERROR_HANDLE --> DEDUPE
    DEDUPE --> NORMALIZE
    NORMALIZE --> VALIDATE_DATA[Validate Data<br/>Check required fields]
    VALIDATE_DATA --> REDIS_CACHE
    REDIS_CACHE -->|Store in Cache| UPDATE_STATS_STORE[Update Cache Stats<br/>Increment sets]
    UPDATE_STATS_STORE --> RESULT
    VALIDATE_DATA --> RESULT
    
    style USER fill:#e1f5ff
    style COLLECTOR fill:#fff4e1
    style YFINANCE fill:#e8f5e9
    style ALPHA fill:#fff9c4
    style FINN fill:#fff9c4
    style REDDIT fill:#fff9c4
    style REDIS_CACHE fill:#ffe1f5
    style RESULT fill:#e8f5e9
```

---

## Diagram 22

**Location:** Line 4532 in index.md

**Context:** Diagram 22

```mermaid
flowchart TD
    START([User Request with Sources]) --> VALIDATE_SYMBOL[Validate Symbol<br/>Format, length check]
    
    VALIDATE_SYMBOL -->|Invalid| ERROR[Return Error<br/>Invalid symbol format]
    VALIDATE_SYMBOL -->|Valid| CHECK_CACHE{Check Redis Cache<br/>for stock & news}
    
    CHECK_CACHE -->|Cache Hit| RETURN[Return Cached Data<br/>TTL: 1h stock, 2h news<br/>Deserialize JSON]
    CHECK_CACHE -->|Cache Miss| PARSE[Parse Source Filters<br/>yfinance, alpha_vantage, etc.<br/>Validate filter format]
    
    PARSE --> CHECK{Source Enabled?<br/>Check API keys}
    
    CHECK -->|yfinance| YFINANCE[yfinance Collection<br/>Always enabled<br/>No API key needed]
    CHECK -->|alpha_vantage| CHECK_ALPHA{Alpha Vantage<br/>API Key?}
    CHECK -->|finnhub| CHECK_FINN{Finnhub<br/>API Key?}
    CHECK -->|reddit| CHECK_REDDIT{Reddit<br/>Credentials?}
    
    CHECK_ALPHA -->|Yes| ALPHA[Alpha Vantage Collection<br/>Company News API]
    CHECK_ALPHA -->|No| SKIP_ALPHA[Skip Alpha Vantage<br/>Log warning]
    CHECK_FINN -->|Yes| FINN[Finnhub Collection<br/>Financial News API]
    CHECK_FINN -->|No| SKIP_FINN[Skip Finnhub<br/>Log warning]
    CHECK_REDDIT -->|Yes| REDDIT[Reddit Collection<br/>Social Media Posts]
    CHECK_REDDIT -->|No| SKIP_REDDIT[Skip Reddit<br/>Log warning]
    
    YFINANCE --> COLLECT[Parallel Collection<br/>All enabled sources<br/>Concurrent execution]
    ALPHA --> COLLECT
    FINN --> COLLECT
    REDDIT --> COLLECT
    
    COLLECT --> ERROR_HANDLE[Error Handling<br/>Log errors per source<br/>Continue with available data]
    
    ERROR_HANDLE --> DEDUPE[Deduplicate Articles<br/>By URL & Title Similarity<br/>85% threshold]
    
    DEDUPE --> NORMALIZE[Normalize Format<br/>Standard structure<br/>Validate all fields]
    
    NORMALIZE --> VALIDATE_DATA{Data<br/>Valid?}
    
    VALIDATE_DATA -->|Invalid| DEFAULT[Return Default Data<br/>Empty news, price: 0.0]
    VALIDATE_DATA -->|Valid| CACHE[Cache in Redis<br/>stock: 1h TTL<br/>news: 2h TTL]
    
    CACHE --> RETURN
    DEFAULT --> RETURN
    ERROR --> RETURN
    
    style START fill:#e1f5ff
    style COLLECT fill:#fff9c4
    style RETURN fill:#e8f5e9
```

---

## Diagram 23

**Location:** Line 4579 in index.md

**Context:** ### 6.8 Data Normalization Flow

```mermaid
flowchart LR
    subgraph "Source-Specific Formats"
        YF_FORMAT[yfinance Format<br/>Nested structure<br/>content.title<br/>content.summary]
        AV_FORMAT[Alpha Vantage Format<br/>REST API JSON<br/>title, description<br/>published_time]
        FINN_FORMAT[Finnhub Format<br/>REST API JSON<br/>headline, summary<br/>datetime]
        REDDIT_FORMAT[Reddit Format<br/>PRAW Object<br/>title, selftext<br/>created_utc]
    end
    
    subgraph "Normalization Process"
        EXTRACT[Extract Fields<br/>title, summary, source, url, timestamp]
        CLEAN[Clean & Sanitize<br/>Remove HTML, normalize whitespace]
        VALIDATE[Validate & Default<br/>Ensure all fields present]
    end
    
    subgraph "Standard Format"
        STANDARD[Standard Article Format<br/>title: str<br/>summary: str<br/>source: str<br/>url: str<br/>timestamp: datetime]
    end
    
    YF_FORMAT --> EXTRACT
    AV_FORMAT --> EXTRACT
    FINN_FORMAT --> EXTRACT
    REDDIT_FORMAT --> EXTRACT
    
    EXTRACT --> CLEAN
    CLEAN --> VALIDATE
    VALIDATE --> STANDARD
    
    style YF_FORMAT fill:#e8f5e9
    style AV_FORMAT fill:#fff9c4
    style FINN_FORMAT fill:#fff9c4
    style REDDIT_FORMAT fill:#fff9c4
    style STANDARD fill:#ffe1f5
```

---

## Diagram 24

**Location:** Line 4645 in index.md

**Context:** ### 7.2 Cache Architecture

```mermaid
flowchart TB
    subgraph "Cache Types"
        STOCK_CACHE[Stock Data Cache<br/>Key: stock:AAPL<br/>TTL: 1 hour<br/>Data: price, market_cap, company_name]
        NEWS_CACHE[News Articles Cache<br/>Key: news:AAPL<br/>TTL: 2 hours<br/>Data: List of articles]
        SENTIMENT_CACHE[Sentiment Cache<br/>Key: sentiment:hash<br/>TTL: 24 hours<br/>Data: positive, negative, neutral]
        EMBEDDING_CACHE[Embedding Cache<br/>Key: embedding:hash<br/>TTL: 7 days<br/>Data: 1536-dim vector]
        DUPLICATE_MARKER[Duplicate Markers<br/>Key: article_hash:SYMBOL:ID<br/>TTL: 7 days<br/>Data: '1' marker]
    end
    
    subgraph "Cache Operations"
        GET[GET Operation<br/>Check if exists]
        SET[SET Operation<br/>Store with TTL]
        EXISTS[EXISTS Check<br/>Fast lookup]
    end
    
    subgraph "Cache Statistics"
        STATS[Cache Stats<br/>Hits, Misses, Sets<br/>Stored in Redis]
    end
    
    STOCK_CACHE --> GET
    NEWS_CACHE --> GET
    SENTIMENT_CACHE --> GET
    EMBEDDING_CACHE --> GET
    DUPLICATE_MARKER --> EXISTS
    
    GET -->|Cache Hit| STATS
    GET -->|Cache Miss| SET
    SET --> STATS
    
    style STOCK_CACHE fill:#e8f5e9
    style NEWS_CACHE fill:#e8f5e9
    style SENTIMENT_CACHE fill:#fff9c4
    style EMBEDDING_CACHE fill:#ffe1f5
    style STATS fill:#fff4e1
```

---

## Diagram 25

**Location:** Line 4753 in index.md

**Context:** ### 7.6 Cache Flow Diagram

```mermaid
flowchart TD
    START([Data Request]) --> CHECK_REDIS{Redis<br/>Available?}
    
    CHECK_REDIS -->|Not Available| FETCH_DIRECT[Fetch from API/Service<br/>Skip caching]
    CHECK_REDIS -->|Available| CHECK{Check Redis Cache<br/>GET operation}
    
    CHECK -->|Cache Hit| UPDATE_STATS_HIT[Update Cache Stats<br/>Increment hits]
    CHECK -->|Cache Miss| FETCH[Fetch from API/Service]
    
    UPDATE_STATS_HIT --> RETURN[Return Cached Data<br/>Deserialize JSON]
    
    FETCH --> VALIDATE_DATA{Data<br/>Valid?}
    FETCH_DIRECT --> VALIDATE_DATA
    
    VALIDATE_DATA -->|Invalid| ERROR_RETURN[Return Error/Default<br/>Log error]
    VALIDATE_DATA -->|Valid| PROCESS[Process Data<br/>Normalize, validate]
    
    PROCESS --> STORE{Store in<br/>Redis?}
    
    STORE -->|Yes| STORE_CACHE[Store in Redis<br/>with TTL<br/>Serialize JSON]
    STORE -->|No| SKIP_STORE[Skip Storage<br/>Redis unavailable]
    
    STORE_CACHE --> UPDATE_STATS_SET[Update Cache Stats<br/>Increment sets]
    SKIP_STORE --> RETURN
    UPDATE_STATS_SET --> RETURN
    ERROR_RETURN --> RETURN
    
    subgraph "Cache Statistics"
        STATS[Track Cache Stats<br/>Hits, Misses, Sets<br/>Stored in Redis]
    end
    
    RETURN --> STATS
    
    style START fill:#e1f5ff
    style CHECK fill:#fff9c4
    style RETURN fill:#e8f5e9
    style STATS fill:#ffe1f5
```

---

## Diagram 26

**Location:** Line 4831 in index.md

**Context:** ### 8.2 Authentication ### 8.2.1 API Request/Response Flow

```mermaid
sequenceDiagram
    participant Client as API Client<br/>(Frontend/External)
    participant FastAPI as FastAPI Server<br/>(api/main.py)
    participant Middleware as Middleware<br/>(CORS, Logging)
    participant Route as API Route<br/>(routes/*.py)
    participant Validator as Request Validator<br/>(Pydantic Models)
    participant Deps as Dependencies<br/>(Service Injection)
    participant Service as Service Layer<br/>(Business Logic)
    participant External as External Services<br/>(APIs, Databases)
    
    Client->>FastAPI: HTTP Request<br/>GET /sentiment/AAPL?detailed=true
    
    Note over FastAPI: Request Processing
    FastAPI->>Middleware: Process Request<br/>(CORS, Logging, Timing)
    alt Health Check / Docs
        Middleware-->>Client: Skip logging, return immediately
    else Regular Request
        Middleware->>Route: Route to Handler<br/>(/sentiment/{symbol})
        Route->>Validator: Validate Parameters<br/>(Path, Query, Body)<br/>Pydantic validation
        
        alt Validation Failed
            Validator-->>Route: Validation Error
            Route->>Middleware: 422 Unprocessable Entity<br/>Error Details
            Middleware->>FastAPI: JSON Error Response
            FastAPI-->>Client: HTTP 422<br/>Validation Error Details
        else Validation Success
            Validator->>Deps: Get Services<br/>(Dependency Injection)<br/>Singleton pattern
            alt Service Initialization Failed
                Deps-->>Route: Service Error
                Route->>Middleware: 500 Internal Server Error
                Middleware->>FastAPI: JSON Error Response
                FastAPI-->>Client: HTTP 500<br/>Service Unavailable
            else Services Available
                Deps->>Service: Call Business Logic<br/>(orchestrator.get_aggregated_sentiment)
                
                Note over Service: Business Logic Execution
                Service->>External: Fetch Data<br/>(APIs, Cache, Vector DB)
                alt External Service Error
                    External-->>Service: Error Response
                    Service->>Service: Handle Error<br/>(Fallback, Retry, or Return Partial)
                    Service-->>Deps: Return Results (with errors)
                else External Service Success
                    External-->>Service: Return Data
                    Service->>Service: Process & Aggregate<br/>(Sentiment scores, normalization)
                    Service-->>Deps: Return Results
                end
                
                Deps-->>Route: Return Data Dictionary
                
                Route->>Validator: Validate Response<br/>(Pydantic Models)
                alt Response Validation Failed
                    Validator-->>Route: Validation Error
                    Route->>Middleware: 500 Internal Server Error<br/>Response Format Error
                    Middleware->>FastAPI: JSON Error Response
                    FastAPI-->>Client: HTTP 500<br/>Response Format Error
                else Response Validation Success
                    Validator->>Middleware: JSON Response<br/>Validated data
                    Middleware->>FastAPI: Add Headers<br/>(CORS, Content-Type)
                    FastAPI-->>Client: HTTP 200 OK<br/>JSON Response with Data
                end
            end
        end
    end
```

---

## Diagram 27

**Location:** Line 5172 in index.md

**Context:** ## Configuration Guide ### 9.1 Configuration Loading Flow

```mermaid
flowchart TD
    START([Application Start]) --> LOAD_ENV[Load .env File<br/>python-dotenv]
    
    LOAD_ENV --> PARSE[Parse Environment Variables<br/>Pydantic BaseSettings]
    
    PARSE --> VALIDATE[Validate Settings<br/>Type checking, format validation]
    
    VALIDATE --> CHECK_REQUIRED{Required<br/>Settings Present?}
    
    CHECK_REQUIRED -->|No| ERROR[Raise ValueError<br/>Show helpful error message]
    CHECK_REQUIRED -->|Yes| CREATE[Create Settings Instance<br/>Singleton Pattern]
    
    CREATE --> NESTED[Initialize Nested Settings<br/>AzureOpenAI, Redis, App, etc.]
    
    NESTED --> VALIDATE_NESTED[Validate Each Nested Setting<br/>URLs, ports, API keys]
    
    VALIDATE_NESTED --> CHECK_OPTIONAL{Optional<br/>Settings Valid?}
    
    CHECK_OPTIONAL -->|Invalid| WARN[Log Warning<br/>Use defaults or disable feature]
    CHECK_OPTIONAL -->|Valid| STORE[Store in Global Singleton<br/>_settings variable]
    
    WARN --> STORE
    STORE --> AVAILABLE[Settings Available<br/>get_settings returns instance]
    
    AVAILABLE --> SERVICES[Services Initialize<br/>Using settings]
    
    style START fill:#e1f5ff
    style VALIDATE fill:#fff9c4
    style ERROR fill:#ffe1f5
    style STORE fill:#e8f5e9
    style AVAILABLE fill:#e8f5e9
```

---

## Diagram 28

**Location:** Line 5208 in index.md

**Context:** Diagram 28

```mermaid
graph TD
    SETTINGS[Settings<br/>Main Container]
    
    SETTINGS --> AZURE_OPENAI[AzureOpenAISettings<br/>endpoint, api_key, deployment_name]
    SETTINGS --> REDIS[RedisSettings<br/>host, port, password, ssl]
    SETTINGS --> AZURE_SEARCH[AzureAISearchSettings<br/>endpoint, api_key, index_name<br/>Optional]
    SETTINGS --> DATA_SOURCES[DataSourceSettings<br/>alpha_vantage, finnhub, reddit<br/>Optional]
    SETTINGS --> APP[AppSettings<br/>cache_ttl, rag_top_k, etc.]
    
    style SETTINGS fill:#e1f5ff
    style AZURE_OPENAI fill:#fff9c4
    style REDIS fill:#fff9c4
    style APP fill:#e8f5e9
```

---

## Diagram 29

**Location:** Line 5466 in index.md

**Context:** ## Troubleshooting & FAQ ### 10.1 Troubleshooting Decision Tree

```mermaid
flowchart TD
    START([Application Issue]) --> CHECK_ERROR{What Error?}
    
    CHECK_ERROR -->|Redis Error| REDIS_ISSUE[Redis Connection Failed]
    CHECK_ERROR -->|OpenAI Error| OPENAI_ISSUE[Azure OpenAI Error]
    CHECK_ERROR -->|No Data| DATA_ISSUE[No Articles Found]
    CHECK_ERROR -->|RAG Error| RAG_ISSUE[RAG Not Finding Articles]
    CHECK_ERROR -->|Timeout| TIMEOUT_ISSUE[API Timeout]
    CHECK_ERROR -->|Other| OTHER_ISSUE[Check Logs]
    
    REDIS_ISSUE --> REDIS_CHECK[Check .env:<br/>REDIS_HOST, REDIS_PASSWORD]
    REDIS_CHECK --> REDIS_TEST[Test Connection:<br/>redis-cli ping]
    REDIS_TEST --> REDIS_FIX{Connection OK?}
    REDIS_FIX -->|No| REDIS_NETWORK[Check Network/Firewall]
    REDIS_FIX -->|Yes| REDIS_WORKING[✅ Working<br/>App continues without cache]
    
    OPENAI_ISSUE --> OPENAI_CHECK[Check .env:<br/>AZURE_OPENAI_API_KEY]
    OPENAI_CHECK --> OPENAI_VERIFY[Verify in Azure Portal:<br/>Deployment, Quota, Region]
    OPENAI_VERIFY --> OPENAI_FIX{Valid?}
    OPENAI_FIX -->|No| OPENAI_UPDATE[Update Configuration]
    OPENAI_FIX -->|Yes| OPENAI_FALLBACK[⚠️ Using TextBlob Fallback]
    
    DATA_ISSUE --> DATA_SYMBOL[Check Symbol Valid<br/>e.g., AAPL not APPL]
    DATA_SYMBOL --> DATA_SOURCES[Check Sources Enabled<br/>yfinance always works]
    DATA_SOURCES --> DATA_API[Check API Status<br/>yfinance may be down]
    DATA_API --> DATA_WORKING{Data Found?}
    DATA_WORKING -->|No| DATA_TRY[Try Different Symbol]
    DATA_WORKING -->|Yes| DATA_SUCCESS[✅ Working]
    
    RAG_ISSUE --> RAG_CONFIG[Check Azure AI Search Config<br/>Optional, has Redis fallback]
    RAG_CONFIG --> RAG_INDEX[Verify Index Exists<br/>Check Azure Portal]
    RAG_INDEX --> RAG_DELAY[Check Indexing Delay<br/>New articles need time]
    RAG_DELAY --> RAG_FILTERS[Check Filters<br/>May be too restrictive]
    RAG_FILTERS --> RAG_WORKING{Articles Found?}
    RAG_WORKING -->|No| RAG_FALLBACK[⚠️ Using Redis SCAN Fallback]
    RAG_WORKING -->|Yes| RAG_SUCCESS[✅ Working]
    
    TIMEOUT_ISSUE --> TIMEOUT_INCREASE[Increase Timeout<br/>APP_API_TIMEOUT=300]
    TIMEOUT_INCREASE --> TIMEOUT_SOURCES[Reduce Data Sources<br/>Fewer API calls]
    TIMEOUT_SOURCES --> TIMEOUT_CACHE[Enable Caching<br/>Faster responses]
    TIMEOUT_CACHE --> TIMEOUT_WORKING{Timeout Fixed?}
    TIMEOUT_WORKING -->|No| TIMEOUT_NETWORK[Check Network Speed]
    TIMEOUT_WORKING -->|Yes| TIMEOUT_SUCCESS[✅ Working]
    
    OTHER_ISSUE --> LOGS[Check Application Logs<br/>Detailed error messages]
    LOGS --> DOCS[Review Documentation<br/>This file]
    
    style START fill:#e1f5ff
    style REDIS_WORKING fill:#e8f5e9
    style OPENAI_FALLBACK fill:#fff9c4
    style DATA_SUCCESS fill:#e8f5e9
    style RAG_SUCCESS fill:#e8f5e9
    style TIMEOUT_SUCCESS fill:#e8f5e9
```

---

