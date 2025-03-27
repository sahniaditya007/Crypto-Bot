import logging
import time
from datetime import datetime
from typing import Dict, Any
import os
from dotenv import load_dotenv

from data_collection import collect_data
from sentiment_analysis import process_news_data, process_twitter_data
from market_analysis import analyze_market_trends
from prediction_generation import generate_predictions
from evaluation_metrics import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_bot.log',
    filemode='a'
)

# Load environment variables
load_dotenv()

class CryptoBot:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {}

    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete crypto bot pipeline."""
        try:
            self.start_time = datetime.now()
            logging.info("Starting crypto bot pipeline")

            # Step 1: Data Collection
            logging.info("Step 1: Collecting data")
            collect_data()

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
            self.results['predictions'] = predictions.tolist()
            self.results['confidence_scores'] = confidence_scores.tolist()

            self.end_time = datetime.now()
            self.results['execution_time'] = (self.end_time - self.start_time).total_seconds()
            
            logging.info("Pipeline completed successfully")
            return self.results

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise

    def _run_sentiment_analysis(self):
        """Run sentiment analysis on collected data."""
        try:
            # Process news data
            news_sources = ['coindesk', 'cryptopanic', 'newsapi']
            for source in news_sources:
                process_news_data(f"{source}_news.json")

            # Process Twitter data
            process_twitter_data("twitter_data.json")

        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}")
            raise

def main():
    """Main entry point for the crypto bot."""
    bot = CryptoBot()
    try:
        results = bot.run_pipeline()
        print("\nPipeline Results:")
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        print(f"Predictions: {results['predictions']}")
        print(f"Confidence Scores: {results['confidence_scores']}")
        print("\nMarket Trends Summary:")
        for key, value in results['market_trends'].items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        logging.error(f"Pipeline execution failed: {str(e)}")

if __name__ == "__main__":
    main() 