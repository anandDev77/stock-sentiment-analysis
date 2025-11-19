# Mermaid Diagrams Source Code

This file contains all Mermaid diagram source code extracted from INDEX_V3.md.
This is kept for reference and future modifications.

**Total Diagrams:** 29

---

## Diagram 1

**Location:** Line 176 in INDEX_V3.md

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

**Location:** Line 264 in INDEX_V3.md

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

**Location:** Line 330 in INDEX_V3.md

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
    
    Orchestrator->>RAG: store_articles_batch(articles, symbol)
    RAG->>VectorDB: Check existing articles
    RAG->>AzureOpenAI: Generate embeddings (batch)
    RAG->>VectorDB: Store articles with embeddings
    RAG-->>Orchestrator: Articles stored
    
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

**Location:** Line 394 in INDEX_V3.md

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

**Location:** Line 437 in INDEX_V3.md

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

**Location:** Line 543 in INDEX_V3.md

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

**Location:** Line 2975 in INDEX_V3.md

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
        Orchestrator->>Collector: collect_all_data(symbol)
        Collector->>YFinance: Fetch stock & news
        Collector->>AlphaVantage: Fetch news (if enabled)
        Collector->>Finnhub: Fetch news (if enabled)
        Collector->>Reddit: Fetch posts (if enabled)
        Collector-->>Orchestrator: Return collected data
        Orchestrator->>Cache: Store data in cache
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

**Location:** Line 3079 in INDEX_V3.md

**Context:** #### Data Collection Flow Diagram

```mermaid
flowchart TD
    START([User clicks Load Data<br/>with symbol AAPL]) --> CHECK_CACHE{Check Redis Cache<br/>for stock & news}
    
    CHECK_CACHE -->|Cache Hit| RETURN_CACHED[Return Cached Data<br/>TTL: 1h stock, 2h news]
    CHECK_CACHE -->|Cache Miss| COLLECT[Start Data Collection]
    
    COLLECT --> FILTER[Apply Source Filters<br/>yfinance: ✅<br/>Alpha Vantage: ✅<br/>Finnhub: ❌<br/>Reddit: ❌]
    
    FILTER --> YFINANCE[yfinance API<br/>Stock price + News]
    FILTER --> ALPHA_VANTAGE[Alpha Vantage API<br/>Company News]
    FILTER --> FINNHUB[Finnhub API<br/>Company News<br/>SKIPPED]
    FILTER --> REDDIT[Reddit API<br/>Social Posts<br/>SKIPPED]
    
    YFINANCE --> PARALLEL[Parallel Collection]
    ALPHA_VANTAGE --> PARALLEL
    
    PARALLEL --> DEDUPE[Deduplicate Articles<br/>By URL & Title Similarity]
    
    DEDUPE --> NORMALIZE[Normalize Article Format<br/>title, summary, source, url, timestamp]
    
    NORMALIZE --> STORE_CACHE[Store in Redis Cache<br/>stock: 1h TTL<br/>news: 2h TTL]
    
    STORE_CACHE --> RETURN[Return Data Dictionary<br/>price_data, news]
    
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

**Location:** Line 3156 in INDEX_V3.md

**Context:** #### RAG Storage Flow Diagram

```mermaid
sequenceDiagram
    participant Orchestrator as Orchestrator
    participant RAG as RAG Service
    participant Redis as Redis Cache
    participant VectorDB as Azure AI Search
    participant OpenAI as Azure OpenAI<br/>(Embeddings)
    
    Orchestrator->>RAG: store_articles_batch(articles, symbol)
    
    Note over RAG: STEP 1: Prepare Articles
    RAG->>RAG: Preprocess articles<br/>(clean text, expand abbreviations)
    RAG->>RAG: Create article IDs<br/>(MD5 hash of title+url)
    
    Note over RAG: STEP 2: Check Duplicates
    RAG->>Redis: Check duplicate markers<br/>(article_hash:SYMBOL:ID)
    alt Already in Redis
        Redis-->>RAG: Duplicate found
        RAG-->>Orchestrator: Skip (already stored)
    else Not in Redis
        Note over RAG: STEP 3: Check Azure AI Search
        RAG->>VectorDB: batch_check_documents_exist(vector_ids)
        VectorDB-->>RAG: Return existing document IDs
        
        alt Already in Azure AI Search
            RAG->>Redis: Mark as stored (for duplicate checking)
            RAG-->>Orchestrator: Return existing count
        else New Articles
            Note over RAG: STEP 4: Generate Embeddings
            RAG->>Redis: Check cached embeddings
            alt Embedding Cached
                Redis-->>RAG: Return cached embedding
            else Embedding Not Cached
                RAG->>OpenAI: get_embeddings_batch(texts, batch_size=100)
                Note over OpenAI: Single API call for<br/>all articles (batch processing)
                OpenAI-->>RAG: Return embeddings (1536 dims each)
                RAG->>Redis: Cache embeddings (7 days TTL)
            end
            
            Note over RAG: STEP 5: Store in Azure AI Search
            RAG->>VectorDB: batch_store_vectors(vectors_with_metadata)
            VectorDB->>VectorDB: Index with HNSW algorithm
            VectorDB-->>RAG: Return stored count
            
            Note over RAG: STEP 6: Mark in Redis
            RAG->>Redis: Set duplicate markers<br/>(article_hash:SYMBOL:ID, 7 days TTL)
            
            RAG-->>Orchestrator: Return total stored count
        end
    end
```

---

## Diagram 10

**Location:** Line 3251 in INDEX_V3.md

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
        
        Note over Sentiment: STEP 5: Call LLM
        Sentiment->>OpenAI: GPT-4 API call<br/>(text + RAG context + few-shot examples)
        OpenAI-->>Sentiment: Return JSON response<br/>{"positive": 0.85, "negative": 0.10, "neutral": 0.05}
        
        Note over Sentiment: STEP 6: Parse & Normalize
        Sentiment->>Sentiment: Parse JSON response
        Sentiment->>Sentiment: Normalize scores (sum to 1.0)
        
        Note over Sentiment: STEP 7: Cache Result
        Sentiment->>Cache: cache_sentiment(text, scores, TTL)
        
        Sentiment-->>Orchestrator: Return sentiment scores
    end
```

---

## Diagram 11

**Location:** Line 3302 in INDEX_V3.md

**Context:** Diagram 11

```mermaid
flowchart TD
    START([Batch Analyze:<br/>30 articles]) --> POOL[ThreadPoolExecutor<br/>max_workers=5]
    
    POOL --> WORKER1[Worker 1:<br/>Article 1-6]
    POOL --> WORKER2[Worker 2:<br/>Article 7-12]
    POOL --> WORKER3[Worker 3:<br/>Article 13-18]
    POOL --> WORKER4[Worker 4:<br/>Article 19-24]
    POOL --> WORKER5[Worker 5:<br/>Article 25-30]
    
    WORKER1 --> ANALYZE1[Analyze Sentiment<br/>Cache Check → RAG → LLM]
    WORKER2 --> ANALYZE2[Analyze Sentiment<br/>Cache Check → RAG → LLM]
    WORKER3 --> ANALYZE3[Analyze Sentiment<br/>Cache Check → RAG → LLM]
    WORKER4 --> ANALYZE4[Analyze Sentiment<br/>Cache Check → RAG → LLM]
    WORKER5 --> ANALYZE5[Analyze Sentiment<br/>Cache Check → RAG → LLM]
    
    ANALYZE1 --> COLLECT[Collect Results]
    ANALYZE2 --> COLLECT
    ANALYZE3 --> COLLECT
    ANALYZE4 --> COLLECT
    ANALYZE5 --> COLLECT
    
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

**Location:** Line 3375 in INDEX_V3.md

**Context:** #### RAG Retrieval Flow Diagram

```mermaid
flowchart TD
    START([Query: Apple earnings report<br/>Symbol: AAPL]) --> PREPROCESS[Preprocess Query<br/>Clean & normalize]
    
    PREPROCESS --> EXPAND{Expand Query?}
    EXPAND -->|Yes| EXPANDED[Expanded Query:<br/>Apple earnings report<br/>profits revenue results financial]
    EXPAND -->|No| ORIGINAL[Original Query]
    
    EXPANDED --> EMBED[Generate Query Embedding<br/>Azure OpenAI<br/>1536 dimensions]
    ORIGINAL --> EMBED
    
    EMBED --> HYBRID[Hybrid Search]
    
    HYBRID --> SEMANTIC[Semantic Search<br/>Vector Similarity<br/>Cosine Similarity]
    HYBRID --> KEYWORD[Keyword Search<br/>Full-Text Search<br/>BM25 Algorithm]
    
    SEMANTIC --> FILTER1[Apply OData Filters<br/>symbol eq 'AAPL'<br/>date_range, sources]
    KEYWORD --> FILTER2[Apply OData Filters<br/>symbol eq 'AAPL'<br/>date_range, sources]
    
    FILTER1 --> RESULTS1[Semantic Results<br/>Ranked by similarity<br/>0.0 - 1.0]
    FILTER2 --> RESULTS2[Keyword Results<br/>Ranked by BM25 score<br/>0 - 100]
    
    RESULTS1 --> RRF[Reciprocal Rank Fusion<br/>RRF score calculation]
    RESULTS2 --> RRF
    
    RRF --> COMBINED[Combined Results<br/>Articles appearing in both<br/>rank highest]
    
    COMBINED --> TEMPORAL[Apply Temporal Decay<br/>Boost recent articles<br/>decay formula applied]
    
    TEMPORAL --> THRESHOLD{Filter by<br/>Similarity Threshold}
    
    THRESHOLD -->|Too Restrictive| AUTO_ADJUST[Auto-Adjust Threshold<br/>Lower by 20%]
    AUTO_ADJUST --> THRESHOLD
    
    THRESHOLD -->|Pass| TOP_K[Select Top K Articles<br/>Default: 3]
    
    TOP_K --> FORMAT[Format Context<br/>Title, Summary, Source,<br/>Relevance Score]
    
    FORMAT --> END([Return Context Articles<br/>for LLM Prompt])
    
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

**Location:** Line 3426 in INDEX_V3.md

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

**Location:** Line 3548 in INDEX_V3.md

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

**Location:** Line 3685 in INDEX_V3.md

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

**Location:** Line 3799 in INDEX_V3.md

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

**Location:** Line 3965 in INDEX_V3.md

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

**Location:** Line 4091 in INDEX_V3.md

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

**Location:** Line 4194 in INDEX_V3.md

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

**Location:** Line 4256 in INDEX_V3.md

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

**Location:** Line 4313 in INDEX_V3.md

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
    COLLECTOR --> FILTER
    FILTER --> REDIS_CACHE
    
    REDIS_CACHE -->|Cache Miss| YFINANCE
    REDIS_CACHE -->|Cache Miss| ALPHA
    REDIS_CACHE -->|Cache Miss| FINN
    REDIS_CACHE -->|Cache Miss| REDDIT
    
    YFINANCE --> DEDUPE
    ALPHA --> DEDUPE
    FINN --> DEDUPE
    REDDIT --> DEDUPE
    
    DEDUPE --> NORMALIZE
    NORMALIZE --> REDIS_CACHE
    REDIS_CACHE -->|Cache Hit| RESULT
    NORMALIZE --> RESULT
    
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

**Location:** Line 4532 in INDEX_V3.md

**Context:** Diagram 22

```mermaid
flowchart TD
    START([User Request with Sources]) --> PARSE[Parse Source Filters<br/>yfinance, alpha_vantage, etc.]
    
    PARSE --> CHECK{Source Enabled?}
    
    CHECK -->|yfinance| YFINANCE[yfinance Collection<br/>Always enabled]
    CHECK -->|alpha_vantage| ALPHA[Alpha Vantage Collection<br/>Requires API key]
    CHECK -->|finnhub| FINN[Finnhub Collection<br/>Requires API key]
    CHECK -->|reddit| REDDIT[Reddit Collection<br/>Requires credentials]
    
    YFINANCE --> COLLECT[Parallel Collection<br/>All enabled sources]
    ALPHA --> COLLECT
    FINN --> COLLECT
    REDDIT --> COLLECT
    
    COLLECT --> DEDUPE[Deduplicate Articles<br/>By URL & Title]
    
    DEDUPE --> NORMALIZE[Normalize Format<br/>Standard structure]
    
    NORMALIZE --> CACHE[Cache in Redis<br/>2h TTL]
    
    CACHE --> RETURN[Return Combined Results]
    
    style START fill:#e1f5ff
    style COLLECT fill:#fff9c4
    style RETURN fill:#e8f5e9
```

---

## Diagram 23

**Location:** Line 4579 in INDEX_V3.md

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

**Location:** Line 4645 in INDEX_V3.md

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

**Location:** Line 4753 in INDEX_V3.md

**Context:** ### 7.6 Cache Flow Diagram

```mermaid
flowchart TD
    START([Data Request]) --> CHECK{Check Redis Cache}
    
    CHECK -->|Cache Hit| RETURN[Return Cached Data<br/>Update Stats]
    CHECK -->|Cache Miss| FETCH[Fetch from API/Service]
    
    FETCH --> PROCESS[Process Data]
    PROCESS --> STORE[Store in Redis<br/>with TTL]
    
    STORE --> RETURN
    
    subgraph "Cache Statistics"
        STATS[Track Cache Stats<br/>Hits, Misses, Sets]
    end
    
    RETURN --> STATS
    
    style START fill:#e1f5ff
    style CHECK fill:#fff9c4
    style RETURN fill:#e8f5e9
    style STATS fill:#ffe1f5
```

---

## Diagram 26

**Location:** Line 4831 in INDEX_V3.md

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
    FastAPI->>Middleware: Process Request<br/>(CORS, Logging)
    Middleware->>Route: Route to Handler
    Route->>Validator: Validate Parameters<br/>(Path, Query, Body)
    
    alt Validation Failed
        Validator-->>Client: 400 Bad Request<br/>Error Details
    else Validation Success
        Validator->>Deps: Get Services<br/>(Dependency Injection)
        Deps->>Service: Call Business Logic<br/>(orchestrator.get_aggregated_sentiment)
        
        Service->>External: Fetch Data<br/>(APIs, Cache, Vector DB)
        External-->>Service: Return Data
        
        Service->>Service: Process & Aggregate
        Service-->>Deps: Return Results
        Deps-->>Route: Return Data Dictionary
        
        Route->>Validator: Validate Response<br/>(Pydantic Models)
        Validator->>Middleware: JSON Response
        Middleware->>FastAPI: Add Headers
        FastAPI-->>Client: HTTP 200 OK<br/>JSON Response
    end
```

---

## Diagram 27

**Location:** Line 5172 in INDEX_V3.md

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

**Location:** Line 5208 in INDEX_V3.md

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

**Location:** Line 5466 in INDEX_V3.md

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

