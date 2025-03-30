"""
Crypto Bot Sentiment Analysis Package
This package contains the core functionality for the crypto bot including
sentiment analysis, market analysis, prediction generation, and recommendation system.
"""

from .main import CryptoBot
from .sentiment_analysis import process_news_data, process_twitter_data
from .market_analysis import analyze_market_trends
from .prediction_generation import generate_predictions
from .evaluation_metrics import evaluate_model
from .recommendation_system import CryptoRecommender

__all__ = [
    'CryptoBot',
    'process_news_data',
    'process_twitter_data',
    'analyze_market_trends',
    'generate_predictions',
    'evaluate_model',
    'CryptoRecommender'
] 