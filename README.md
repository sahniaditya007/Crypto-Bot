# Crypto Bot

A comprehensive cryptocurrency analysis and trading bot that combines sentiment analysis, iceberg order detection, and arbitrage opportunities.

## Project Structure

```
crypto-bot/
├── src/                      # Source code
│   ├── sentiment/           # Sentiment analysis module
│   │   ├── data_collection.py     # Data collection from various sources
│   │   ├── sentiment_analysis.py  # Sentiment analysis implementation
│   │   ├── market_analysis.py     # Market trend analysis
│   │   ├── prediction_generation.py # Price prediction generation
│   │   ├── recommendation_system.py # Investment recommendations
│   │   ├── model_training.py      # Model training utilities
│   │   ├── evaluation_metrics.py  # Model evaluation metrics
│   │   └── main.py               # Main entry point for sentiment analysis
│   │
│   ├── iceberg/             # Iceberg order detection module
│   │   ├── iceberg_detector.py    # Iceberg order detection implementation
│   │   └── run_prediction.py      # Script to run iceberg predictions
│   │
│   ├── arbitrage/           # Arbitrage detection module
│   │   ├── binance_connect.py     # Binance API connection
│   │   ├── arbitrage_checker.py   # Arbitrage opportunity checker
│   │   └── fetch_order_book.py    # Fetch order book data
│
├── config/                   # Configuration files
│   ├── .env.iceberg        # Iceberg module environment variables
│   ├── .env.sentiment      # Sentiment module environment variables
│   ├── .env.arbitrage      # Arbitrage module environment variables
│   ├── .env.prediction     # Prediction module environment variables
│   └── .env.example        # Example environment variables
│
├── data/                    # Data storage directory
│   ├── cryptopanic_news_sentiment.json # Sentiment data from CryptoPanic
│   ├── cryptopanic_news.json           # News data from CryptoPanic
│   ├── historical_data.json            # Historical market data
│   ├── market_data.json                # Market data
│   ├── market_trends.json              # Market trends analysis
│   ├── newsapi_news_sentiment.json     # Sentiment data from NewsAPI
│   ├── newsapi_news.json               # News data from NewsAPI
│   ├── recommendations.json            # Investment recommendations
│   ├── training_history.csv            # Model training history
│   └── iceberg/                        # Iceberg detection data
│
├── models/                  # Trained models storage
│   ├── best_model.keras     # Best trained model for iceberg detection
│   └── sentiment/           # Sentiment analysis models
│
├── logs/                    # Log files
│   ├── crypto_bot.log       # General log file
│   ├── iceberg_detector_*.log # Iceberg detection logs
│
├── cache/                   # Cache files
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-bot.git
cd crypto-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
The project uses multiple environment files for different modules:
- `config/.env.sentiment` for sentiment analysis configuration
- `config/.env.iceberg` for iceberg detection configuration
- `config/.env.example` as a template for setting up your environment variables

Make sure to set up your API keys and configuration in these files.

5. Run the sentiment analysis:
```bash
python -m src.sentiment.main
```

6. Run the iceberg detector:
```bash
python -m src.iceberg.iceberg_detector
```

7. Run the arbitrage checker:
```bash
python -m src.arbitrage.arbitrage_checker
```

## Features

### Sentiment Analysis
- **News Analysis**: Analyze sentiment from news sources like CryptoPanic and NewsAPI.
- **Social Media Sentiment**: Analyze sentiment from Twitter and Reddit.
- **Market Trend Analysis**: Analyze market trends based on price, volume, and sentiment data.
- **Price Predictions**: Generate price predictions using trained models.
- **Investment Recommendations**: Provide actionable investment recommendations based on sentiment and market data.

### Iceberg Order Detection
- **Order Book Analysis**: Analyze order book data to detect hidden iceberg orders.
- **Market Manipulation Detection**: Identify potential market manipulation patterns.

### Arbitrage Detection
- **Real-Time Price Monitoring**: Monitor prices across multiple exchanges like Binance and Kraken.
- **Arbitrage Opportunities**: Detect and execute arbitrage opportunities between exchanges.

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

MIT License - see LICENSE file for details.