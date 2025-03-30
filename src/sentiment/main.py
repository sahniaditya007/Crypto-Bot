import logging
import time
from datetime import datetime
from typing import Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv

from .data_collection import DataCollector
from .sentiment_analysis import process_news_data, process_twitter_data
from .market_analysis import analyze_market_trends
from .prediction_generation import generate_predictions
from .evaluation_metrics import evaluate_model
from .recommendation_system import CryptoRecommender

# Define project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# Create necessary directories
for directory in [LOG_DIR, DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(PROJECT_ROOT / 'config' / '.env.sentiment')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'crypto_bot.log'),
        logging.StreamHandler()
    ]
)

class CryptoBot:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.data_collector = DataCollector()
        self.data_dir = DATA_DIR
        self.recommender = CryptoRecommender()

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete crypto bot pipeline."""
        try:
            self.start_time = datetime.now()
            logging.info("Starting crypto bot pipeline")

            # Step 1: Data Collection
            logging.info("Step 1: Collecting data")
            self.data_collector.collect_data()

            # Step 2: Sentiment Analysis
            logging.info("Step 2: Performing sentiment analysis")
            self._run_sentiment_analysis()

            # Step 3: Market Analysis
            logging.info("Step 3: Analyzing market trends")
            market_trends = analyze_market_trends()
            self.results['market_trends'] = market_trends

            # Step 4: Generate Predictions
            logging.info("Step 4: Generating predictions")
            predictions, confidence_scores = generate_predictions()
            self.results['predictions'] = predictions
            self.results['confidence_scores'] = confidence_scores

            # Step 5: Generate Investment Recommendations
            logging.info("Step 5: Generating investment recommendations")
            recommendations = self.recommender.get_top_recommendations()
            self.results['recommendations'] = recommendations

            self.end_time = datetime.now()
            self.results['execution_time'] = (self.end_time - self.start_time).total_seconds()
            
            logging.info("Pipeline completed successfully")
            return self.results

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

    def _run_sentiment_analysis(self):
        """Run sentiment analysis on collected data."""
        try:
            # Process news data
            news_sources = ['coindesk', 'cryptopanic', 'newsapi']
            for source in news_sources:
                news_file = self.data_dir / f"{source}_news.json"
                if not news_file.exists():
                    logging.warning(f"News file not found: {news_file}")
                    continue
                process_news_data(news_file)

            # Process Twitter data
            twitter_file = self.data_dir / "twitter_data.json"
            if twitter_file.exists():
                process_twitter_data(twitter_file)
            else:
                logging.warning("Twitter data file not found")

        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}", exc_info=True)
            raise

def main():
    """Main entry point for the sentiment analysis pipeline."""
    try:
        # Initialize the bot
        bot = CryptoBot()
        
        # Run the pipeline
        results = bot.run_pipeline()
        
        # Log results
        logging.info("Pipeline completed successfully")
        logging.info(f"Results: {results}")
        
    except Exception as e:
        logging.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 