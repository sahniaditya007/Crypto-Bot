# Crypto Bot

A comprehensive cryptocurrency analysis and trading bot that combines sentiment analysis and iceberg order detection.

## Project Structure

```
crypto-bot/
├── sentiment/                 # Sentiment analysis module
│   ├── data_collection.py     # Data collection from various sources
│   ├── sentiment_analysis.py  # Sentiment analysis implementation
│   ├── market_analysis.py     # Market trend analysis
│   ├── prediction_generation.py # Price prediction generation
│   ├── recommendation_system.py # Investment recommendations
│   ├── model_training.py      # Model training utilities
│   ├── evaluation_metrics.py  # Model evaluation metrics
│   └── main.py               # Main entry point for sentiment analysis
│
├── iceberg/                   # Iceberg order detection module
│   └── iceberg_detector.py    # Iceberg order detection implementation
│
├── data/                      # Data storage directory
│   ├── market_data/          # Market data
│   ├── news_data/            # News articles
│   └── social_data/          # Social media data
│
├── models/                    # Trained models storage
│   ├── sentiment/            # Sentiment analysis models
│   └── iceberg/              # Iceberg detection models
│
├── logs/                      # Log files
├── cache/                     # Cache files
├── requirements.txt           # Project dependencies
├── .env.example              # Example environment variables
└── README.md                 # This file
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
```bash
cp .env.example .env.sentiment
cp .env.example .env.iceberg
```
Edit both .env files with your API keys and configuration.

5. Run the sentiment analysis:
```bash
python -m sentiment.main
```

6. Run the iceberg detector:
```bash
python -m iceberg.iceberg_detector
```

## Features

- Sentiment Analysis
  - News article analysis
  - Social media sentiment
  - Market trend analysis
  - Price predictions
  - Investment recommendations

- Iceberg Order Detection
  - Order book analysis
  - Hidden order detection
  - Market manipulation detection

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

MIT License - see LICENSE file for details 