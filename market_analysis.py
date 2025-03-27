import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_bot.log'
)

class TrendDirection(Enum):
    UP = 1
    DOWN = -1
    NEUTRAL = 0

@dataclass
class MarketMetrics:
    current_price: float
    price_change_24h: float
    volume_24h: float
    volume_change_24h: float
    market_cap: float
    market_cap_change_24h: float
    price_volatility: float
    volume_volatility: float
    price_trend: TrendDirection
    volume_trend: TrendDirection
    sentiment_trend: TrendDirection

class MarketAnalyzer:
    def __init__(self):
        self.cache_dir = "cache"
        self._setup_cache_dir()
        self._setup_metrics_cache()

    def _setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _setup_metrics_cache(self):
        """Initialize metrics cache."""
        self.metrics_cache = {}
        self.cache_ttl = timedelta(minutes=5)

    def _get_cache_path(self, source: str) -> str:
        """Get cache file path for a data source."""
        return os.path.join(self.cache_dir, f"{source}_cache.json")

    def _is_cache_valid(self, source: str) -> bool:
        """Check if cached data is still valid."""
        cache_path = self._get_cache_path(source)
        if not os.path.exists(cache_path):
            return False
        
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - cache_time < self.cache_ttl

    @lru_cache(maxsize=100)
    def load_sentiment_data(self) -> Dict[str, List[Dict]]:
        """Load sentiment data from all sources with caching."""
        sentiment_data = {}
        files = {
            'coindesk': 'coindesk_sentiment.json',
            'cryptopanic': 'cryptopanic_sentiment.json',
            'newsapi': 'newsapi_sentiment.json',
            'twitter': 'twitter_sentiment.json'
        }
        
        for source, file_path in files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    sentiment_data[source] = json.load(file)
                logging.info(f"Successfully loaded {source} sentiment data")
            except FileNotFoundError:
                logging.error(f"File not found: {file_path}")
                sentiment_data[source] = []
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON in file: {file_path}")
                sentiment_data[source] = []
            except Exception as e:
                logging.error(f"Error loading {source} sentiment data: {e}")
                sentiment_data[source] = []
        
        return sentiment_data

    @lru_cache(maxsize=10)
    def load_market_data(self) -> Optional[pd.DataFrame]:
        """Load and preprocess market data with caching."""
        try:
            with open("market_data.json", 'r', encoding='utf-8') as file:
                market_data = json.load(file)
            
            df = pd.DataFrame(market_data)
            df['timestamp'] = pd.to_datetime(df['last_updated'])
            df = df.sort_values('timestamp')
            
            # Calculate additional metrics
            df['price_change_24h'] = df['price_change_percentage_24h'] / 100
            df['volume_change_24h'] = df['total_volume'].pct_change()
            df['market_cap_change_24h'] = df['market_cap_change_percentage_24h'] / 100
            
            logging.info("Successfully loaded and processed market data")
            return df
        except Exception as e:
            logging.error(f"Error loading market data: {e}")
            return None

    def calculate_sentiment_metrics(self, sentiment_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate sentiment metrics from all sources."""
        metrics = {}
        
        with ThreadPoolExecutor() as executor:
            future_to_source = {
                executor.submit(self._calculate_source_metrics, source, data): source
                for source, data in sentiment_data.items()
            }
            
            for future in future_to_source:
                source = future_to_source[future]
                try:
                    source_metrics = future.result()
                    metrics.update(source_metrics)
                except Exception as e:
                    logging.error(f"Error calculating metrics for {source}: {e}")
        
        # Calculate overall sentiment metrics
        all_sentiments = []
        for data in sentiment_data.values():
            all_sentiments.extend([item.get('sentiment', 0) for item in data])
        
        if all_sentiments:
            metrics['overall_sentiment_mean'] = np.mean(all_sentiments)
            metrics['overall_sentiment_std'] = np.std(all_sentiments)
            metrics['total_sentiment_count'] = len(all_sentiments)
        
        return metrics

    def _calculate_source_metrics(self, source: str, data: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for a single sentiment source."""
        if not data:
            return {}
        
        sentiments = [item.get('sentiment', 0) for item in data]
        return {
            f'{source}_sentiment_mean': np.mean(sentiments),
            f'{source}_sentiment_std': np.std(sentiments),
            f'{source}_sentiment_count': len(sentiments)
        }

    def calculate_market_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market metrics from price and volume data."""
        if df is None or df.empty:
            return {}
        
        metrics = {
            'current_price': df['current_price'].iloc[-1],
            'price_change_24h': df['price_change_24h'].iloc[-1],
            'volume_24h': df['total_volume'].iloc[-1],
            'volume_change_24h': df['volume_change_24h'].iloc[-1],
            'market_cap': df['market_cap'].iloc[-1],
            'market_cap_change_24h': df['market_cap_change_24h'].iloc[-1],
            'price_volatility': df['price_change_24h'].std(),
            'volume_volatility': df['volume_change_24h'].std()
        }
        
        # Calculate moving averages
        for window in [7, 14, 30]:
            metrics[f'price_ma_{window}d'] = df['current_price'].rolling(window=window).mean().iloc[-1]
            metrics[f'volume_ma_{window}d'] = df['total_volume'].rolling(window=window).mean().iloc[-1]
        
        return metrics

    def determine_trends(self, metrics: Dict[str, float]) -> Dict[str, TrendDirection]:
        """Determine market trends based on metrics."""
        trends = {}
        
        # Price trend
        if metrics.get('current_price', 0) > metrics.get('price_ma_7d', 0):
            trends['price_trend'] = TrendDirection.UP
        elif metrics.get('current_price', 0) < metrics.get('price_ma_7d', 0):
            trends['price_trend'] = TrendDirection.DOWN
        else:
            trends['price_trend'] = TrendDirection.NEUTRAL
        
        # Volume trend
        if metrics.get('volume_24h', 0) > metrics.get('volume_ma_7d', 0):
            trends['volume_trend'] = TrendDirection.UP
        elif metrics.get('volume_24h', 0) < metrics.get('volume_ma_7d', 0):
            trends['volume_trend'] = TrendDirection.DOWN
        else:
            trends['volume_trend'] = TrendDirection.NEUTRAL
        
        # Sentiment trend
        sentiment_mean = metrics.get('overall_sentiment_mean', 0)
        if sentiment_mean > 0.1:
            trends['sentiment_trend'] = TrendDirection.UP
        elif sentiment_mean < -0.1:
            trends['sentiment_trend'] = TrendDirection.DOWN
        else:
            trends['sentiment_trend'] = TrendDirection.NEUTRAL
        
        return trends

    def analyze_market_trends(self) -> Dict[str, Any]:
        """Analyze market trends combining sentiment and market data."""
        logging.info("Starting market trend analysis")
        
        try:
            sentiment_data = self.load_sentiment_data()
            market_df = self.load_market_data()
            
            sentiment_metrics = self.calculate_sentiment_metrics(sentiment_data)
            market_metrics = self.calculate_market_metrics(market_df)
            
            # Combine all metrics
            analysis_results = {**sentiment_metrics, **market_metrics}
            
            # Determine trends
            trends = self.determine_trends(analysis_results)
            analysis_results.update({k: v.value for k, v in trends.items()})
            
            # Create MarketMetrics object for type safety
            market_metrics_obj = MarketMetrics(
                current_price=analysis_results.get('current_price', 0),
                price_change_24h=analysis_results.get('price_change_24h', 0),
                volume_24h=analysis_results.get('volume_24h', 0),
                volume_change_24h=analysis_results.get('volume_change_24h', 0),
                market_cap=analysis_results.get('market_cap', 0),
                market_cap_change_24h=analysis_results.get('market_cap_change_24h', 0),
                price_volatility=analysis_results.get('price_volatility', 0),
                volume_volatility=analysis_results.get('volume_volatility', 0),
                price_trend=trends['price_trend'],
                volume_trend=trends['volume_trend'],
                sentiment_trend=trends['sentiment_trend']
            )
            
            # Add market metrics object to results
            analysis_results['market_metrics'] = market_metrics_obj.__dict__
            
            logging.info("Market trend analysis completed")
            return analysis_results
            
        except Exception as e:
            logging.error(f"Error in market trend analysis: {e}")
            raise

def analyze_market_trends() -> Dict[str, Any]:
    """Wrapper function for market trend analysis."""
    analyzer = MarketAnalyzer()
    return analyzer.analyze_market_trends()

if __name__ == "__main__":
    market_trends = analyze_market_trends()
    try:
        with open("market_trends.json", "w", encoding='utf-8') as file:
            json.dump(market_trends, file, indent=2)
        logging.info("Successfully saved market trends")
    except Exception as e:
        logging.error(f"Error saving market trends: {e}")