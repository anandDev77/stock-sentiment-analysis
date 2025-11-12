# Stock Sentiment Dashboard

A real-time stock sentiment analysis dashboard built with Streamlit and Ollama. This application analyzes sentiment from news articles and social media posts to provide insights into stock market sentiment.

## Features

- 📈 Real-time stock price and market cap information
- 📰 News sentiment analysis using AI (Ollama)
- 💬 Social media sentiment analysis
- 📊 Interactive visualizations with Plotly
- ⏱️ Time-series sentiment tracking
- 🔍 Detailed article and post breakdowns

## Prerequisites

1. **Python 3.8 or higher**
2. **Ollama** - Download and install from [ollama.ai](https://ollama.ai)
3. **Ollama Model** - Pull a compatible model:
   ```bash
   ollama pull llama2:7b
   ```
   Or use other models like:
   - `llama2:13b`
   - `mistral:7b`
   - `llama3:8b`

## Installation

1. Clone or download this repository

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the URL shown in the terminal (usually `http://localhost:8501`)

3. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT) in the sidebar

4. Select an Ollama model from the dropdown

5. Click "Analyze Sentiment" to see the results

## Project Structure

```
stock-sentiment-analysis2/
├── app.py                 # Main Streamlit dashboard
├── sentiment_analyzer.py  # Sentiment analysis using Ollama
├── data_collector.py      # Stock data and news collection
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How It Works

1. **Data Collection**: The app fetches stock data, news articles, and social media posts using `yfinance` and other APIs
2. **Sentiment Analysis**: Each text is analyzed using Ollama (LLM) to determine positive, negative, and neutral sentiment scores
3. **Visualization**: Results are displayed with interactive charts and detailed breakdowns

## Notes

- The social media data is currently simulated. For production use, integrate with Reddit API (PRAW) or Twitter API
- Make sure Ollama is running and the selected model is available before analyzing
- The app uses TextBlob as a fallback if Ollama fails to parse the response

## Troubleshooting

- **Ollama connection error**: Make sure Ollama is installed and running (`ollama serve`)
- **Model not found**: Pull the model first using `ollama pull <model-name>`
- **No data for symbol**: Check if the stock symbol is valid and try again

## License

This project is open source and available for educational purposes.

