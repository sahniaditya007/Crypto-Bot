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
from collections import defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crypto_bot.log'),
        logging.StreamHandler()
    ]
)

# Define base directory
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

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
        self.data_dir = "data"
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
            'coindesk': DATA_DIR / 'coindesk_sentiment.json',
            'cryptopanic': DATA_DIR / 'cryptopanic_sentiment.json',
            'newsapi': DATA_DIR / 'newsapi_sentiment.json',
            'twitter': DATA_DIR / 'twitter_sentiment.json'
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
            logging.info("Loading market data from file...")
            with open(os.path.join(self.data_dir, "market_data.json"), 'r', encoding='utf-8') as file:
                market_data = json.load(file)
            
            if not market_data:
                logging.error("No market data available")
                return None
            
            logging.info(f"Loaded {len(market_data)} market data points")
            df = pd.DataFrame(market_data)
            
            # Handle timestamps with microseconds
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            df = df.sort_values('timestamp')
            
            logging.info("Calculating price changes...")
            # Calculate price changes
            df['price_change_24h'] = df['current_price'].pct_change(periods=24).fillna(0)
            df['market_cap_change_24h'] = df['market_cap'].pct_change(periods=24).fillna(0)
            df['volume_change_24h'] = df['total_volume'].pct_change(periods=24).fillna(0)
            
            logging.info("Calculating volatility...")
            # Calculate volatility using rolling standard deviation
            df['price_volatility'] = df['current_price'].rolling(window=24).std().pct_change().fillna(0)
            df['volume_volatility'] = df['total_volume'].rolling(window=24).std().pct_change().fillna(0)
            
            logging.info("Calculating moving averages...")
            # Calculate moving averages
            for window in [7, 14, 30]:
                df[f'price_ma_{window}d'] = df['current_price'].rolling(window=window).mean()
                df[f'volume_ma_{window}d'] = df['total_volume'].rolling(window=window).mean()
            
            logging.info("Successfully loaded and processed market data")
            logging.info(f"Latest price: {df['current_price'].iloc[-1]}")
            logging.info(f"Latest market cap: {df['market_cap'].iloc[-1]}")
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
            logging.error("No market data available for metrics calculation")
            return {}
        
        # Get the latest data point
        latest = df.iloc[-1]
        logging.info(f"Calculating metrics for timestamp: {latest['timestamp']}")
        
        metrics = {
            'current_price': float(latest['current_price']),
            'price_change_24h': float(latest['price_change_24h']),
            'volume_24h': float(latest['total_volume']),
            'volume_change_24h': float(latest['volume_change_24h']),
            'market_cap': float(latest['market_cap']),
            'market_cap_change_24h': float(latest['market_cap_change_24h']),
            'price_volatility': float(latest['price_volatility']),
            'volume_volatility': float(latest['volume_volatility'])
        }
        
        # Add moving averages
        for window in [7, 14, 30]:
            metrics[f'price_ma_{window}d'] = float(latest[f'price_ma_{window}d'])
            metrics[f'volume_ma_{window}d'] = float(latest[f'volume_ma_{window}d'])
        
        logging.info("Market metrics calculated successfully")
        logging.info(f"Current price: {metrics['current_price']}")
        logging.info(f"24h price change: {metrics['price_change_24h']}")
        return metrics

    def determine_trends(self, metrics: Dict[str, float]) -> Dict[str, TrendDirection]:
        """Determine market trends based on metrics."""
        trends = {}
        
        # Price trend - using multiple timeframes for confirmation
        current_price = metrics.get('current_price', 0)
        price_ma_7d = metrics.get('price_ma_7d', 0)
        price_ma_14d = metrics.get('price_ma_14d', 0)
        price_ma_30d = metrics.get('price_ma_30d', 0)
        
        # Strong trend if price is above all moving averages
        if current_price > price_ma_7d > price_ma_14d > price_ma_30d:
            trends['price_trend'] = TrendDirection.UP
        # Strong downtrend if price is below all moving averages
        elif current_price < price_ma_7d < price_ma_14d < price_ma_30d:
            trends['price_trend'] = TrendDirection.DOWN
        else:
            trends['price_trend'] = TrendDirection.NEUTRAL
        
        # Volume trend - using multiple timeframes and volume change
        current_volume = metrics.get('volume_24h', 0)
        volume_ma_7d = metrics.get('volume_ma_7d', 0)
        volume_ma_14d = metrics.get('volume_ma_14d', 0)
        volume_ma_30d = metrics.get('volume_ma_30d', 0)
        volume_change = metrics.get('volume_change_24h', 0)
        
        # Strong volume trend if current volume is significantly above average
        if current_volume > volume_ma_7d * 1.2 and volume_change > 0.1:
            trends['volume_trend'] = TrendDirection.UP
        elif current_volume < volume_ma_7d * 0.8 and volume_change < -0.1:
            trends['volume_trend'] = TrendDirection.DOWN
        else:
            trends['volume_trend'] = TrendDirection.NEUTRAL
        
        # Sentiment trend - using weighted average of different sources
        sentiment_mean = metrics.get('overall_sentiment_mean', 0)
        sentiment_std = metrics.get('overall_sentiment_std', 0)
        sentiment_count = metrics.get('total_sentiment_count', 0)
        
        # Strong sentiment if we have enough data points and clear direction
        if sentiment_count >= 10:
            if sentiment_mean > 0.2 and sentiment_std < 0.5:
                trends['sentiment_trend'] = TrendDirection.UP
            elif sentiment_mean < -0.2 and sentiment_std < 0.5:
                trends['sentiment_trend'] = TrendDirection.DOWN
            else:
                trends['sentiment_trend'] = TrendDirection.NEUTRAL
        else:
            trends['sentiment_trend'] = TrendDirection.NEUTRAL
        
        return trends

    def analyze_market_trends(self) -> Dict[str, Any]:
        """Analyze market trends combining sentiment and market data."""
        logging.info("Starting market trend analysis")
        
        try:
            sentiment_data = self.load_sentiment_data()
            market_df = self.load_market_data()
            
            if market_df is None or market_df.empty:
                logging.error("No market data available for analysis")
                return {
                    'current_price': 0.0,
                    'price_change_24h': 0.0,
                    'volume_24h': 0.0,
                    'volume_change_24h': 0.0,
                    'market_cap': 0.0,
                    'market_cap_change_24h': 0.0,
                    'price_volatility': 0.0,
                    'volume_volatility': 0.0,
                    'price_trend': TrendDirection.NEUTRAL.value,
                    'volume_trend': TrendDirection.NEUTRAL.value,
                    'sentiment_trend': TrendDirection.NEUTRAL.value
                }
            
            logging.info("Calculating sentiment metrics...")
            sentiment_metrics = self.calculate_sentiment_metrics(sentiment_data)
            logging.info("Calculating market metrics...")
            market_metrics = self.calculate_market_metrics(market_df)
            
            # Combine all metrics
            analysis_results = {**sentiment_metrics, **market_metrics}
            
            # Determine trends
            logging.info("Determining market trends...")
            trends = self.determine_trends(analysis_results)
            analysis_results.update({k: v.value for k, v in trends.items()})
            
            # Create MarketMetrics object for type safety
            market_metrics_obj = MarketMetrics(
                current_price=float(analysis_results.get('current_price', 0)),
                price_change_24h=float(analysis_results.get('price_change_24h', 0)),
                volume_24h=float(analysis_results.get('volume_24h', 0)),
                volume_change_24h=float(analysis_results.get('volume_change_24h', 0)),
                market_cap=float(analysis_results.get('market_cap', 0)),
                market_cap_change_24h=float(analysis_results.get('market_cap_change_24h', 0)),
                price_volatility=float(analysis_results.get('price_volatility', 0)),
                volume_volatility=float(analysis_results.get('volume_volatility', 0)),
                price_trend=trends['price_trend'],
                volume_trend=trends['volume_trend'],
                sentiment_trend=trends['sentiment_trend']
            )
            
            logging.info("Market trend analysis completed successfully")
            return {
                'current_price': market_metrics_obj.current_price,
                'price_change_24h': market_metrics_obj.price_change_24h,
                'volume_24h': market_metrics_obj.volume_24h,
                'volume_change_24h': market_metrics_obj.volume_change_24h,
                'market_cap': market_metrics_obj.market_cap,
                'market_cap_change_24h': market_metrics_obj.market_cap_change_24h,
                'price_volatility': market_metrics_obj.price_volatility,
                'volume_volatility': market_metrics_obj.volume_volatility,
                'price_trend': market_metrics_obj.price_trend.value,
                'volume_trend': market_metrics_obj.volume_trend.value,
                'sentiment_trend': market_metrics_obj.sentiment_trend.value
            }
            
        except Exception as e:
            logging.error(f"Error in market trend analysis: {e}")
            return {
                'current_price': 0.0,
                'price_change_24h': 0.0,
                'volume_24h': 0.0,
                'volume_change_24h': 0.0,
                'market_cap': 0.0,
                'market_cap_change_24h': 0.0,
                'price_volatility': 0.0,
                'volume_volatility': 0.0,
                'price_trend': TrendDirection.NEUTRAL.value,
                'volume_trend': TrendDirection.NEUTRAL.value,
                'sentiment_trend': TrendDirection.NEUTRAL.value
            }

def analyze_market_trends() -> Dict[str, Any]:
    """Wrapper function for market trend analysis."""
    analyzer = MarketAnalyzer()
    return analyzer.analyze_market_trends()

def load_market_data() -> List[Dict]:
    """Load market data from file."""
    try:
        with open('data/market_data.json', 'r') as file:
            data = json.load(file)
            logging.info(f"Loaded {len(data)} market data points")
            return data
    except Exception as e:
        logging.error(f"Error loading market data: {e}")
        return []

def calculate_metrics(market_data: List[Dict]) -> Dict:
    """Calculate market metrics for each cryptocurrency."""
    try:
        metrics = {}
        for coin in market_data:
            metrics[coin['symbol']] = {
                'name': coin['name'],
                'current_price': coin['price'],
                'market_cap': coin['market_cap'],
                'volume_24h': coin['volume_24h'],
                'price_change_24h': coin['percent_change_24h'],
                'volume_change_24h': coin['volume_change_24h']
            }
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return {}

def calculate_volatility(price_changes: List[float]) -> float:
    """Calculate price volatility."""
    if not price_changes:
        return 0.0
    return np.std(price_changes)

def determine_trend(values: List[float]) -> int:
    """Determine trend direction (1: up, 0: neutral, -1: down)."""
    if not values or len(values) < 2:
        return 0
    
    # Calculate moving average
    ma = np.mean(values[-5:])  # 5-period moving average
    current = values[-1]
    
    if current > ma * 1.02:  # 2% above MA
        return 1
    elif current < ma * 0.98:  # 2% below MA
        return -1
    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trends = analyze_market_trends()
    print("\nMarket Analysis Results:")
    print("------------------------")
    print(f"Total Market Cap: ${trends.get('market_cap', 0):,.0f}")
    print(f"24h Volume: ${trends.get('volume_24h', 0):,.0f}")
    print(f"Market Sentiment: {trends.get('sentiment_trend', 'Unknown')}")
    
    print("\nTop 10 Cryptocurrencies:")
    print("----------------------")
    for coin in trends.get('top_performers', []):
        print(f"\n{coin['name']} ({coin['symbol']}):")
        print(f"  Price: ${coin['price']:,.2f}")
        print(f"  24h Change: {coin['price_change_24h']:+.2f}%")
        print(f"  Market Cap: ${coin['market_cap']:,.0f}")
        print(f"  Market Dominance: {coin['dominance']:.2f}%")