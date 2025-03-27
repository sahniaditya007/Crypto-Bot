# Crypto Trading Bot

A sophisticated cryptocurrency trading bot that combines market data analysis, sentiment analysis, and machine learning to generate trading predictions.

## Features

- **Data Collection**: Gathers data from multiple sources including:
  - Market data (CoinGecko API)
  - News articles (CoinDesk, CryptoPanic, NewsAPI)
  - Social media sentiment (Twitter)

- **Sentiment Analysis**: 
  - Processes news articles and social media posts
  - Uses NLTK and TextBlob for natural language processing
  - Implements parallel processing for improved performance

- **Market Analysis**:
  - Calculates technical indicators
  - Analyzes market trends
  - Implements caching for efficient data processing

- **Machine Learning**:
  - Random Forest Classifier for prediction
  - Hyperparameter tuning with GridSearchCV
  - Feature importance analysis
  - Comprehensive model evaluation metrics

- **Performance Optimization**:
  - Caching mechanisms for API calls
  - Parallel processing for sentiment analysis
  - Efficient data structures and algorithms
  - Comprehensive error handling and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-bot.git
cd crypto-bot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
COINDESK_API_URL=your_coindesk_api_url
CRYPTOPANIC_API_URL=your_cryptopanic_api_url
NEWSAPI_URL=your_newsapi_url
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET_KEY=your_twitter_api_secret_key
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
COINGECKO_API_URL=your_coingecko_api_url
HISTORICAL_DATA_FILE=path_to_historical_data.json
```

## Project Structure

```
crypto-bot/
├── data_collection.py      # Data gathering from various sources
├── sentiment_analysis.py   # Sentiment analysis of news and social media
├── market_analysis.py      # Market data analysis and trend detection
├── model_training.py       # Model training and hyperparameter tuning
├── prediction_generation.py # Generate trading predictions
├── evaluation_metrics.py   # Model evaluation and visualization
├── main.py                # Main orchestrator
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
├── models/               # Trained models and artifacts
├── metrics/              # Model evaluation metrics and plots
├── cache/                # Cached data
└── README.md            # Project documentation
```

## Usage

1. Run the complete pipeline:
```bash
python main.py
```

2. Train the model:
```bash
python model_training.py
```

3. Generate predictions:
```bash
python prediction_generation.py
```

4. Evaluate model performance:
```bash
python evaluation_metrics.py
```

## Model Training

The model training process includes:
- Data preprocessing and validation
- Feature scaling
- Hyperparameter tuning using GridSearchCV
- Model evaluation with multiple metrics
- Feature importance analysis
- Model persistence with timestamps

## Performance Metrics

The bot evaluates model performance using:
- Accuracy, Precision, Recall, F1 Score
- ROC Curve and AUC
- Precision-Recall Curve
- Confusion Matrix
- Feature Importance Analysis

## Caching

The bot implements caching for:
- API responses (configurable TTL)
- Sentiment analysis results
- Market data processing
- Model predictions

## Error Handling

Comprehensive error handling includes:
- API request retries with exponential backoff
- Data validation and cleaning
- Exception logging
- Graceful degradation
- Recovery mechanisms

## Logging

All operations are logged to `crypto_bot.log` with:
- Timestamps
- Log levels (INFO, WARNING, ERROR)
- Detailed error messages
- Performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This bot is for educational purposes only. Cryptocurrency trading involves significant risks, and past performance does not guarantee future results. Always do your own research and never invest more than you can afford to lose. 