import json
import logging
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Load environment variables
load_dotenv('.env')

class CryptoRecommender:
    def __init__(self):
        # Update data directory to point to the root data folder
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.sentiment_weights = {
            'news': 0.4,
            'social': 0.3,
            'market': 0.3
        }
        self.min_confidence_threshold = 0.6
        self.min_market_cap = 100000000  # $100M minimum market cap
        self.min_volume = 10000000      # $10M minimum 24h volume
        self.setup_logging()

    def setup_logging(self):
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_dir / 'recommendations.log'
        )

    def get_market_data(self) -> Dict:
        """Fetch current market data from CoinMarketCap."""
        try:
            headers = {
                'X-CMC_PRO_API_KEY': os.getenv('COINMARKETCAP_API_KEY'),
                'Accept': 'application/json'
            }
            response = requests.get(
                f"{os.getenv('COINMARKETCAP_API_URL')}/cryptocurrency/listings/latest",
                headers=headers
            )
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            return {}

    def get_social_sentiment(self) -> Dict:
        """Aggregate sentiment from social media sources."""
        try:
            sentiment_data = {}
            
            # Load Twitter sentiment
            twitter_file = self.data_dir / "twitter_sentiment.json"
            if twitter_file.exists():
                with open(twitter_file, 'r') as f:
                    sentiment_data['twitter'] = json.load(f)
                    logging.info(f"Loaded Twitter sentiment data from {twitter_file}")
            else:
                logging.warning(f"Twitter sentiment file not found: {twitter_file}")
                sentiment_data['twitter'] = []

            # Load Reddit sentiment
            reddit_file = self.data_dir / "reddit_sentiment.json"
            if reddit_file.exists():
                with open(reddit_file, 'r') as f:
                    sentiment_data['reddit'] = json.load(f)
                    logging.info(f"Loaded Reddit sentiment data from {reddit_file}")
            else:
                logging.warning(f"Reddit sentiment file not found: {reddit_file}")
                sentiment_data['reddit'] = []

            return sentiment_data
        except Exception as e:
            logging.error(f"Error getting social sentiment: {e}")
            return {'twitter': [], 'reddit': []}

    def get_news_sentiment(self) -> Dict:
        """Aggregate sentiment from news sources."""
        try:
            news_sentiment = {}
            news_sources = ['cryptopanic', 'newsapi']
            
            for source in news_sources:
                file_path = self.data_dir / f"{source}_news_sentiment.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        news_sentiment[source] = json.load(f)
                        logging.info(f"Loaded sentiment data from {file_path}")
                else:
                    logging.warning(f"Sentiment file not found: {file_path}")
                    # Create empty sentiment data for missing files
                    news_sentiment[source] = []
            
            return news_sentiment
        except Exception as e:
            logging.error(f"Error getting news sentiment: {e}")
            return {source: [] for source in news_sources}

    def calculate_aggregate_sentiment(self, sentiment_data: Dict) -> float:
        """Calculate weighted aggregate sentiment score."""
        if not sentiment_data:
            return 0.0

        total_weight = 0
        weighted_sum = 0

        for source, data in sentiment_data.items():
            if isinstance(data, list):
                sentiments = [item.get('sentiment', 0) for item in data]
                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    weight = self.sentiment_weights.get(source, 0.1)
                    weighted_sum += avg_sentiment * weight
                    total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def analyze_market_metrics(self, market_data: Dict) -> Dict:
        """Analyze market metrics for each cryptocurrency."""
        try:
            metrics = {}
            for coin in market_data.get('data', []):
                quote = coin['quote']['USD']
                # Only include coins that meet minimum criteria
                if (quote['market_cap'] >= self.min_market_cap and 
                    quote['volume_24h'] >= self.min_volume):
                    metrics[coin['symbol']] = {
                        'price': quote['price'],
                        'volume_24h': quote['volume_24h'],
                        'market_cap': quote['market_cap'],
                        'percent_change_24h': quote['percent_change_24h'],
                        'volume_change_24h': quote['volume_change_24h'],
                        'name': coin['name']
                    }
            return metrics
        except Exception as e:
            logging.error(f"Error analyzing market metrics: {e}")
            return {}

    def generate_recommendations(self) -> List[Dict]:
        """Generate investment recommendations based on combined analysis."""
        try:
            # Get all necessary data
            market_data = self.get_market_data()
            social_sentiment = self.get_social_sentiment()
            news_sentiment = self.get_news_sentiment()
            market_metrics = self.analyze_market_metrics(market_data)

            # Save market trends data
            market_trends_file = self.data_dir / "market_trends.json"
            with open(market_trends_file, 'w') as f:
                json.dump(market_metrics, f, indent=2)
            logging.info(f"Saved market trends to {market_trends_file}")

            all_candidates = []
            
            # Calculate aggregate sentiment scores
            social_score = self.calculate_aggregate_sentiment(social_sentiment)
            news_score = self.calculate_aggregate_sentiment(news_sentiment)

            # Generate recommendations for each cryptocurrency
            for symbol, metrics in market_metrics.items():
                # Calculate overall score
                market_score = (
                    metrics['percent_change_24h'] / 100 +  # Price change
                    metrics['volume_change_24h'] / 100     # Volume change
                ) / 2

                # Calculate weighted final score
                final_score = (
                    market_score * self.sentiment_weights['market'] +
                    news_score * self.sentiment_weights['news'] +
                    social_score * self.sentiment_weights['social']
                )

                # Calculate risk score (0-1, lower is better)
                risk_score = abs(metrics['volume_change_24h']) / 100

                # Calculate potential return score (0-1, higher is better)
                potential_return = (
                    (metrics['percent_change_24h'] + 100) / 200 +  # Normalize to 0-1
                    (metrics['market_cap'] / max(m['market_cap'] for m in market_metrics.values()))  # Market stability
                ) / 2

                candidate = {
                    'symbol': symbol,
                    'name': metrics['name'],
                    'score': final_score,
                    'confidence': abs(final_score),
                    'risk_level': 'Low' if risk_score < 0.3 else 'Medium' if risk_score < 0.6 else 'High',
                    'potential': 'High' if potential_return > 0.7 else 'Medium' if potential_return > 0.4 else 'Low',
                    'recommendation': self._get_recommendation(final_score, risk_score),
                    'metrics': metrics,
                    'sentiment_scores': {
                        'market': market_score,
                        'news': news_score,
                        'social': social_score
                    },
                    'analysis': {
                        'risk_score': risk_score,
                        'potential_return': potential_return,
                        'market_dominance': metrics['market_cap'] / sum(m['market_cap'] for m in market_metrics.values()),
                        'volume_stability': 1 - (abs(metrics['volume_change_24h']) / 100),
                        'price_stability': 1 - (abs(metrics['percent_change_24h']) / 100)
                    }
                }
                all_candidates.append(candidate)

            # Sort candidates by a combination of factors
            sorted_candidates = sorted(all_candidates, 
                key=lambda x: (
                    x['analysis']['potential_return'] * 0.4 +  # 40% weight to potential return
                    (1 - x['analysis']['risk_score']) * 0.3 +  # 30% weight to lower risk
                    x['analysis']['market_dominance'] * 0.2 +  # 20% weight to market dominance
                    x['confidence'] * 0.1                      # 10% weight to confidence
                ),
                reverse=True
            )

            # Save recommendations
            output_file = os.path.join(self.data_dir, "recommendations.json")
            with open(output_file, 'w') as f:
                json.dump(sorted_candidates[:10], f, indent=2)

            return sorted_candidates[:10]

        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
            return []

    def _get_recommendation(self, score: float, risk_score: float) -> str:
        """Generate detailed recommendation based on score and risk."""
        if score > 0.8 and risk_score < 0.3:
            return "STRONG BUY"
        elif score > 0.5:
            return "BUY"
        elif score > 0.2:
            return "MODERATE BUY"
        elif score > 0:
            return "CAUTIOUS BUY"
        else:
            return "HOLD"

    def get_top_recommendations(self, limit: int = 10) -> List[Dict]:
        """Get top N investment recommendations."""
        recommendations = self.generate_recommendations()
        return recommendations[:limit]

def main():
    """Main function to run the recommendation system."""
    recommender = CryptoRecommender()
    recommendations = recommender.get_top_recommendations()
    
    print("\n=== CRYPTO INVESTMENT RECOMMENDATIONS ===")
    print("=========================================")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=========================================")
    
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            print(f"\n{idx}. {rec['name']} ({rec['symbol']})")
            print(f"Recommendation: {rec['recommendation']}")
            print(f"Risk Level: {rec['risk_level']}")
            print(f"Potential: {rec['potential']}")
            print(f"Current Price: ${rec['metrics']['price']:,.2f}")
            print(f"24h Change: {rec['metrics']['percent_change_24h']:+.2f}%")
            print(f"Market Cap: ${rec['metrics']['market_cap']:,.2f}M")
            print(f"Volume: ${rec['metrics']['volume_24h']:,.2f}M")
            print(f"Confidence Score: {rec['confidence']:.3f}")
            print("Analysis:")
            print(f"Market Dominance: {rec['analysis']['market_dominance']:.2%}")
            print(f"Volume Stability: {rec['analysis']['volume_stability']:.2%}")
            print(f"Price Stability: {rec['analysis']['price_stability']:.2%}")
            print("-----------------------------------------")
    else:
        print("\nNo strong investment opportunities found at this time.")
        print("Consider waiting for better market conditions.")
    
    print("\nDISCLAIMER:")
    print("-----------")
    print("These recommendations are based on technical analysis and market sentiment.")
    print("Always conduct your own research and invest only what you can afford to lose.")
    print("Past performance does not guarantee future results.")

if __name__ == "__main__":
    main() 