# Crypto Bot

A comprehensive cryptocurrency analysis and trading bot that combines sentiment analysis and iceberg order detection.

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
│   └── iceberg/             # Iceberg order detection module
│       └── iceberg_detector.py    # Iceberg order detection implementation
│
├── config/                   # Configuration files
│   ├── .env.iceberg        # Iceberg module environment variables
│   └── .env.sentiment      # Sentiment module environment variables
│
├── data/                    # Data storage directory
│   ├── market_data/        # Market data
│   ├── news_data/          # News articles
│   └── social_data/        # Social media data
│
├── models/                  # Trained models storage
│   ├── sentiment/          # Sentiment analysis models
│   └── iceberg/            # Iceberg detection models
│
├── logs/                    # Log files
├── cache/                   # Cache files
├── requirements.txt         # Project dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
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
The project uses two separate environment files for different modules:
- `config/.env.sentiment` for sentiment analysis configuration
- `config/.env.iceberg` for iceberg detection configuration

Make sure to set up your API keys and configuration in both files.

5. Run the sentiment analysis:
```bash
python -m src.sentiment.main
```

6. Run the iceberg detector:
```bash
python -m src.iceberg.iceberg_detector
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