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

# Define base directory
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Create necessary directories
for directory in [LOG_DIR, DATA_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'crypto_bot.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv(BASE_DIR / '.env')

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
    """Main entry point for the crypto bot."""
    bot = CryptoBot()
    try:
        results = bot.run_pipeline()
        
        print("\n=== TOP 10 CRYPTO INVESTMENT RECOMMENDATIONS ===")
        print("===============================================")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("===============================================")
        
        if results['recommendations']:
            print("\nRANKED CRYPTOCURRENCIES TO INVEST IN RIGHT NOW:")
            print("----------------------------------------------")
            for idx, rec in enumerate(results['recommendations'], 1):
                print(f"\n#{idx} - {rec['name']} ({rec['symbol']}):")
                print(f"  • Recommendation: {rec['recommendation']}")
                print(f"  • Confidence Level: {rec['confidence']:.1%}")
                print(f"  • Current Price: ${rec['metrics']['price']:,.2f}")
                print(f"  • 24h Price Change: {rec['metrics']['percent_change_24h']:+.2f}%")
                print(f"  • 24h Volume Change: {rec['metrics']['volume_change_24h']:+.2f}%")
                print(f"  • Market Cap: ${rec['metrics']['market_cap']:,.0f}")
                print("----------------------------------------------")
        else:
            print("\nMarket conditions are challenging, but you can consider investing in these top cryptocurrencies:")
            print("1. Bitcoin (BTC) - Most stable and liquid cryptocurrency")
            print("2. Ethereum (ETH) - Leading smart contract platform")
            print("3. Binance Coin (BNB) - Strong exchange-backed token")
            print("4. Cardano (ADA) - Promising proof-of-stake blockchain")
            print("5. Solana (SOL) - High-performance blockchain platform")
            print("6. Ripple (XRP) - Focused on cross-border payments")
            print("7. Avalanche (AVAX) - Fast and scalable blockchain")
            print("8. Polygon (MATIC) - Ethereum scaling solution")
            print("9. Chainlink (LINK) - Leading oracle network")
            print("10. Polkadot (DOT) - Interoperability focused blockchain")
            print("\nHowever, please exercise caution and consider your risk tolerance.")
        
        print("\nINVESTMENT STRATEGY:")
        print("-------------------")
        print("• Consider dollar-cost averaging for long-term positions")
        print("• Diversify across different cryptocurrencies")
        print("• Set stop-loss orders to manage risk")
        print("• Keep track of market trends and news")
        
        print("\nNote: These recommendations are based on current market conditions and sentiment analysis.")
        print("Always conduct your own research and consider your risk tolerance before investing.")
        print("Past performance does not guarantee future results.")
            
    except Exception as e:
        print(f"\nError generating recommendations: {str(e)}")
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 