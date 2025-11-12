# Stock Sentiment Dashboard

A real-time stock sentiment analysis dashboard built with Streamlit and Azure OpenAI. This application analyzes sentiment from news articles to provide insights into stock market sentiment.

## Features

- 📈 Real-time stock price and market cap information
- 📰 News sentiment analysis using AI (Azure OpenAI)
- 📊 Interactive visualizations with Plotly
- ⏱️ Time-series sentiment tracking
- 🔍 Detailed article breakdowns

## Prerequisites

1. **Python 3.8 or higher**
2. **Azure OpenAI Account** - You need an Azure OpenAI resource with:
   - An endpoint URL
   - An API key
   - A deployed model (e.g., gpt-4)

## Installation

1. Clone or download this repository

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your Azure OpenAI configuration:
   ```env
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.api.cognitive.microsoft.com
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
   AZURE_OPENAI_API_VERSION=2023-05-15
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the URL shown in the terminal (usually `http://localhost:8501`)

3. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT) in the sidebar

4. Click "Load Data" to fetch and analyze sentiment

## Project Structure

```
stock-sentiment-analysis2/
├── app.py                 # Main Streamlit dashboard
├── sentiment_analyzer.py # Sentiment analysis using Azure OpenAI
├── data_collector.py      # Stock data and news collection
├── requirements.txt       # Python dependencies
├── .env                  # Azure OpenAI configuration (not in git)
└── README.md             # This file
```

## How It Works

1. **Data Collection**: The app fetches stock data and news articles using `yfinance` (free API)
2. **Sentiment Analysis**: Each text is analyzed using Azure OpenAI to determine positive, negative, and neutral sentiment scores
3. **Visualization**: Results are displayed with interactive charts and detailed breakdowns

## Configuration

The app uses environment variables from the `.env` file for Azure OpenAI configuration:

- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_DEPLOYMENT_NAME`: The name of your deployed model (e.g., "gpt-4")
- `AZURE_OPENAI_API_VERSION`: The API version to use (e.g., "2023-05-15")

## Notes

- The app only uses free APIs for data collection (yfinance for stock prices and news)
- Social media data is not included (would require Reddit/Twitter API integration)
- The app uses TextBlob as a fallback if Azure OpenAI fails to parse the response
- Make sure your `.env` file is properly configured before running the app

## Troubleshooting

- **Azure OpenAI connection error**: Check your `.env` file and verify your endpoint, API key, and deployment name are correct
- **No data for symbol**: Check if the stock symbol is valid and try again
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

## License

This project is open source and available for educational purposes.
