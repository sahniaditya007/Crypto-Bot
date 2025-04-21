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
CACHE_DIR = PROJECT_ROOT / "cache"

# Create necessary directories
for directory in [LOG_DIR, DATA_DIR, MODEL_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load environment variables
env_path = PROJECT_ROOT / 'config' / '.env.sentiment'
if not env_path.exists():
    env_path = PROJECT_ROOT / '.env'
load_dotenv(env_path)

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
                sentiment_file = self.data_dir / f"{source}_news_sentiment.json"
                
                if not news_file.exists():
                    logging.warning(f"News file not found: {news_file}")
                    continue
                    
                if not sentiment_file.exists():
                    logging.info(f"Processing news data from {news_file}")
                    process_news_data(news_file)
                else:
                    logging.info(f"Sentiment file already exists: {sentiment_file}")

            # Process Twitter data
            twitter_file = self.data_dir / "twitter_data.json"
            twitter_sentiment_file = self.data_dir / "twitter_sentiment.json"
            
            if twitter_file.exists():
                if not twitter_sentiment_file.exists():
                    logging.info("Processing Twitter data")
                    process_twitter_data(twitter_file)
                else:
                    logging.info("Twitter sentiment file already exists")
            else:
                logging.warning("Twitter data file not found")

        except Exception as e:
            logging.error(f"Sentiment analysis failed: {str(e)}", exc_info=True)
            raise

    def format_output(self, results: Dict[str, Any]) -> str:
        """Format the results in a reader-friendly way."""
        output = "\n" + "="*50 + "\n"
        output += "CRYPTO INVESTMENT RECOMMENDATIONS\n"
        output += "="*50 + "\n\n"

        if 'recommendations' in results:
            for i, rec in enumerate(results['recommendations'], 1):
                output += f"{i}. {rec['name']} ({rec['symbol']})\n"
                output += f"   {'='*(len(rec['name'])+len(rec['symbol'])+4)}\n"
                output += f"   Recommendation: {rec['recommendation']}\n"
                output += f"   Risk Level: {rec['risk_level']}\n"
                output += f"   Potential: {rec['potential']}\n"
                output += f"   Current Price: ${rec['metrics']['price']:,.2f}\n"
                output += f"   24h Change: {rec['metrics']['percent_change_24h']:+.2f}%\n"
                output += f"   Market Cap: ${rec['metrics']['market_cap']:,.2f}\n"
                output += f"   Volume (24h): ${rec['metrics']['volume_24h']:,.2f}\n"
                output += f"   Confidence Score: {float(rec['confidence']):,.3f}\n\n"
                output += "   Analysis Metrics:\n"
                output += f"   - Risk Score: {rec['analysis']['risk_score']:,.2f}\n"
                output += f"   - Potential Return: {rec['analysis']['potential_return']:,.2f}\n"
                output += f"   - Market Dominance: {rec['analysis']['market_dominance']:,.2f}\n"
                output += f"   - Volume Stability: {rec['analysis']['volume_stability']:,.2f}\n"
                output += f"   - Price Stability: {rec['analysis']['price_stability']:,.2f}\n\n"
                
                # Add detailed reasoning section with sources
                output += "   Reasoning:\n"
                if rec['recommendation'] == "STRONG BUY":
                    output += f"   The strong buy recommendation for {rec['name']} is based on exceptional\n"
                    output += f"   market metrics with high confidence ({rec['confidence']:.3f}). The asset shows\n"
                    output += f"   strong market dominance ({rec['analysis']['market_dominance']:.2%}), high volume\n"
                    output += f"   stability ({rec['analysis']['volume_stability']:.2%}), and favorable price\n"
                    output += f"   momentum (+{rec['metrics']['percent_change_24h']:.2f}% 24h change).\n\n"
                elif rec['recommendation'] == "BUY":
                    output += f"   {rec['name']} shows positive momentum with a good balance of growth\n"
                    output += f"   potential ({rec['potential']}) and manageable risk ({rec['risk_level']}).\n"
                    output += f"   The price stability ({rec['analysis']['price_stability']:.2f}) and volume\n"
                    output += f"   metrics suggest a healthy trading environment.\n\n"
                elif rec['recommendation'] == "MODERATE BUY":
                    output += f"   While {rec['name']} shows promise, some caution is warranted.\n"
                    output += f"   The moderate buy rating reflects decent market metrics but with\n"
                    output += f"   room for improvement in volume stability ({rec['analysis']['volume_stability']:.2f})\n"
                    output += f"   and market dominance ({rec['analysis']['market_dominance']:.2%}).\n\n"
                elif rec['recommendation'] == "CAUTIOUS BUY":
                    output += f"   {rec['name']} presents a speculative opportunity with higher risk.\n"
                    output += f"   The cautious buy rating acknowledges the potential upside but notes\n"
                    output += f"   increased volatility in price ({rec['analysis']['price_stability']:.2f}) and\n"
                    output += f"   volume metrics. Consider smaller position sizes.\n\n"
                else:  # HOLD
                    output += f"   Current market conditions for {rec['name']} suggest holding positions.\n"
                    output += f"   The metrics indicate uncertainty with moderate risk ({rec['risk_level']})\n"
                    output += f"   and limited short-term upside potential. Wait for clearer signals\n"
                    output += f"   before making new investments.\n\n"
                
                # Add data sources section
                output += "   Data Sources:\n"
                output += "   - Market Data: CoinMarketCap (Real-time price, volume, and market cap)\n"
                if 'sentiment_scores' in rec:
                    if rec['sentiment_scores'].get('news', 0) != 0:
                        output += "   - News Analysis: CryptoPanic, NewsAPI (Market sentiment)\n"
                    if rec['sentiment_scores'].get('social', 0) != 0:
                        output += "   - Social Sentiment: Reddit, Twitter discussions\n"
                    if rec['sentiment_scores'].get('market', 0) != 0:
                        output += "   - Technical Analysis: Price action, volume patterns, market trends\n"
                output += "\n"

        output += "="*50 + "\n"
        output += "Disclaimer: These recommendations are based on technical analysis\n"
        output += "and market sentiment. Please do your own research before making\n"
        output += "any investment decisions.\n"
        output += "="*50 + "\n"
        
        return output

def main():
    """Main entry point for the sentiment analysis pipeline."""
    try:
        # Initialize the bot
        bot = CryptoBot()
        
        # Run the pipeline
        results = bot.run_pipeline()
        
        # Format and display results
        formatted_output = bot.format_output(results)
        print(formatted_output)
        
        # Log completion
        logging.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 