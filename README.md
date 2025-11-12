# Stock Sentiment Analysis Dashboard

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional-grade stock sentiment analysis application that leverages Azure OpenAI, Redis caching, and RAG (Retrieval Augmented Generation) to provide real-time sentiment insights for stock market analysis.

## 🚀 Features

- **AI-Powered Sentiment Analysis**: Uses Azure OpenAI GPT-4 for accurate financial sentiment analysis
- **RAG Enhancement**: Retrieval Augmented Generation provides context-aware sentiment analysis
- **Redis Caching**: Reduces API calls and improves performance with intelligent caching
- **Real-Time Stock Data**: Fetches live stock prices and news from free APIs (yfinance)
- **Interactive Dashboard**: Beautiful Streamlit-based web interface with multiple analysis views
- **Comprehensive Analytics**: Price charts, sentiment trends, news analysis, and technical indicators

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Azure Setup](#azure-setup)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.8 or higher
- Azure account with:
  - Azure OpenAI service (with GPT-4 deployment)
  - Azure Cache for Redis
- Azure CLI installed and configured
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-sentiment-analysis.git
cd stock-sentiment-analysis
```

### 2. Create Virtual Environment

```bash
# Using Makefile (recommended)
make venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or manually
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Using Makefile (recommended)
make install          # Production dependencies
make install-dev      # Development dependencies

# Or manually
pip install -r requirements.txt
```

## Configuration

### 1. Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` and fill in your actual values. The `.env.example` file contains detailed comments explaining each variable.

**Required Configuration:**
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Your GPT-4 deployment name

**Optional but Recommended:**
- `REDIS_HOST` - Redis host for caching
- `REDIS_PASSWORD` - Redis password
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` - Required for RAG functionality

**Optional Application Settings:**
- `APP_LOG_LEVEL` - Logging level (default: INFO)
- `APP_RAG_TOP_K` - Number of articles for RAG context (default: 3)
- `APP_RAG_SIMILARITY_THRESHOLD` - Minimum similarity for RAG (default: 0.3)
- Cache TTL settings for performance tuning

See `.env.example` for complete documentation of all available settings.

## Azure Setup

We provide automated scripts to set up Azure infrastructure. See [scripts/README.md](scripts/README.md) for detailed documentation.

### Quick Setup

#### 1. Login to Azure

```bash
az login
az account set --subscription "your-subscription-id"
```

#### 2. Create Resource Group

```bash
az group create --name stock-sentiment-rg --location eastus
```

#### 3. Setup Azure OpenAI (with RAG)

```bash
# Using the setup script
./scripts/setup-azure-openai.sh stock-sentiment-rg --location eastus

# Or using make
make setup-azure RG=stock-sentiment-rg
```

This script will:
- Create Azure OpenAI service
- Deploy GPT-4 model for chat completions
- Deploy text-embedding-ada-002 for RAG
- Output configuration for your `.env` file

#### 4. Setup Azure Redis

```bash
# Using the setup script
./scripts/setup-azure-redis.sh stock-sentiment-rg --location eastus

# Or using make
make setup-redis RG=stock-sentiment-rg
```

#### 5. Setup Everything at Once

```bash
make setup-all RG=stock-sentiment-rg
```

### Manual Setup

If you prefer to set up resources manually:

1. **Azure OpenAI**:
   - Create Cognitive Services resource
   - Deploy GPT-4 model
   - Deploy text-embedding-ada-002 model (for RAG)
   - Get API key and endpoint

2. **Azure Redis**:
   - Create Azure Cache for Redis
   - Get connection details (host, port, password)

## Running the Application

### Start the Dashboard

```bash
# Using streamlit directly (recommended)
streamlit run src/stock_sentiment/app.py

# Or using make
make run
```

**Note**: The application is now located in `src/stock_sentiment/app.py` as part of the refactored structure.

The application will be available at `http://localhost:8501`

### Using the Dashboard

1. **Enter Stock Symbol**: Type a stock ticker (e.g., AAPL, MSFT, GOOGL)
2. **Load Data**: Click "Load Data" to fetch stock information and news
3. **Explore Tabs**:
   - **Overview**: Summary of stock data and overall sentiment
   - **Price Analysis**: Historical price charts and trends
   - **News & Sentiment**: News articles with sentiment analysis
   - **Technical Analysis**: Technical indicators and metrics
   - **AI Insights**: AI-generated insights using RAG

## Project Structure

```
stock-sentiment-analysis/
├── src/
│   └── stock_sentiment/          # Main application package
│       ├── __init__.py
│       ├── app.py                # Streamlit dashboard
│       ├── config/               # Configuration management
│       │   ├── __init__.py
│       │   └── settings.py       # Settings and environment validation
│       ├── models/               # Data models
│       │   ├── __init__.py
│       │   ├── sentiment.py      # Sentiment data models
│       │   └── stock.py           # Stock data models
│       ├── services/             # Business logic services
│       │   ├── __init__.py
│       │   ├── cache.py           # Redis cache service
│       │   ├── collector.py      # Stock data collector
│       │   ├── rag.py             # RAG service
│       │   └── sentiment.py      # Sentiment analyzer
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           ├── logger.py          # Logging configuration
│           └── validators.py      # Input validation
├── scripts/                      # Deployment and setup scripts
│   ├── setup-azure-openai.sh
│   ├── setup-azure-redis.sh
│   ├── add-embedding-model.sh
│   └── README.md
├── tests/                        # Test files
├── docs/                         # Documentation
├── .env.example                  # Example environment file
├── .gitignore                    # Git ignore rules
├── Makefile                      # Common commands
├── pyproject.toml                # Package configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Development

### Makefile Commands

The project includes a comprehensive Makefile with industry-standard commands:

```bash
# Virtual Environment
make venv              # Create virtual environment
make venv-activate     # Show activation command

# Installation
make install           # Install production dependencies
make install-dev       # Install development dependencies

# Running
make run               # Run the Streamlit application

# Testing & Quality
make test              # Run tests with coverage
make lint              # Run linters (flake8, mypy)
make format            # Format code with black
make format-check      # Check formatting without changes

# Cleanup
make clean             # Clean cache and build files
make clean-all         # Clean everything including venv

# Azure Setup
make setup-azure RG=your-resource-group    # Setup Azure OpenAI
make setup-redis RG=your-resource-group   # Setup Azure Redis
make setup-all RG=your-resource-group     # Setup both

# Help
make help              # Show all available commands
```

### Setup Development Environment

```bash
make venv
source venv/bin/activate
make install-dev
```

### Run Tests

```bash
make test
```

### Code Formatting

```bash
make format        # Format code
make format-check  # Check formatting
```

### Linting

```bash
make lint
```

### Clean Build Files

```bash
make clean         # Clean cache and build files
make clean-all     # Clean everything including venv
```

## Architecture

### Components

1. **Sentiment Analyzer**: Uses Azure OpenAI to analyze sentiment with optional RAG context
2. **Data Collector**: Fetches stock data and news from yfinance API
3. **RAG Service**: Manages embeddings and retrieves relevant context for sentiment analysis
4. **Redis Cache**: Caches API responses to reduce costs and improve performance
5. **Streamlit Dashboard**: Interactive web interface for visualization and analysis

### Data Flow

```
User Input (Stock Symbol)
    ↓
Data Collector → Fetch Stock Data & News
    ↓
RAG Service → Store Articles & Generate Embeddings
    ↓
Sentiment Analyzer → Analyze with RAG Context
    ↓
Redis Cache → Store Results
    ↓
Streamlit Dashboard → Display Results
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint | - | Yes |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | - | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Chat model deployment name | `gpt-4` | No |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment | - | No (for RAG) |
| `REDIS_HOST` | Redis host address | - | Yes |
| `REDIS_PORT` | Redis port | `6380` | No |
| `REDIS_PASSWORD` | Redis password | - | Yes |
| `REDIS_SSL` | Enable SSL | `true` | No |
| `APP_LOG_LEVEL` | Logging level | `INFO` | No |
| `APP_CACHE_TTL_SENTIMENT` | Sentiment cache TTL (seconds) | `86400` | No |
| `APP_CACHE_TTL_STOCK` | Stock data cache TTL (seconds) | `300` | No |
| `APP_CACHE_TTL_NEWS` | News cache TTL (seconds) | `1800` | No |

## Troubleshooting

### Redis Connection Issues

- Verify Redis credentials in `.env`
- Check if Redis is accessible from your network
- Ensure SSL settings match your Redis configuration

### Azure OpenAI Errors

- Verify API key and endpoint are correct
- Check if models are deployed in your Azure OpenAI resource
- Ensure you have sufficient quota

### RAG Not Working

- Verify embedding model is deployed: `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- Check deployment name matches your Azure resource
- Run `./scripts/add-embedding-model.sh` to add embedding model

### Import Errors

- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version (3.8+)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 style guide
- Use type hints
- Add docstrings to all functions and classes
- Run `make format` before committing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the dashboard framework
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/) for AI capabilities
- [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- [Plotly](https://plotly.com/) for interactive visualizations

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This application uses free APIs where possible. For production use, consider implementing rate limiting, error handling, and monitoring.
