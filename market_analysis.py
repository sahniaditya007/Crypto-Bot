import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_bot.log'
)

def load_sentiment_data() -> Dict[str, List[Dict]]:
    """Load sentiment data from all sources."""
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

def load_market_data() -> Optional[pd.DataFrame]:
    """Load and preprocess market data."""
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

def calculate_sentiment_metrics(sentiment_data: Dict[str, List[Dict]]) -> Dict[str, float]:
    """Calculate sentiment metrics from all sources."""
    metrics = {}
    
    for source, data in sentiment_data.items():
        if not data:
            continue
        
        sentiments = [item.get('sentiment', 0) for item in data]
        metrics[f'{source}_sentiment_mean'] = np.mean(sentiments)
        metrics[f'{source}_sentiment_std'] = np.std(sentiments)
        metrics[f'{source}_sentiment_count'] = len(sentiments)
    
    # Calculate overall sentiment metrics
    all_sentiments = []
    for data in sentiment_data.values():
        all_sentiments.extend([item.get('sentiment', 0) for item in data])
    
    if all_sentiments:
        metrics['overall_sentiment_mean'] = np.mean(all_sentiments)
        metrics['overall_sentiment_std'] = np.std(all_sentiments)
        metrics['total_sentiment_count'] = len(all_sentiments)
    
    return metrics

def calculate_market_metrics(df: pd.DataFrame) -> Dict[str, float]:
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

def analyze_market_trends() -> Dict[str, float]:
    """Analyze market trends combining sentiment and market data."""
    logging.info("Starting market trend analysis")
    
    sentiment_data = load_sentiment_data()
    market_df = load_market_data()
    
    sentiment_metrics = calculate_sentiment_metrics(sentiment_data)
    market_metrics = calculate_market_metrics(market_df)
    
    # Combine all metrics
    analysis_results = {**sentiment_metrics, **market_metrics}
    
    # Calculate trend indicators
    if market_df is not None and not market_df.empty:
        # Price trend
        analysis_results['price_trend'] = 1 if market_metrics['current_price'] > market_metrics['price_ma_7d'] else -1
        
        # Volume trend
        analysis_results['volume_trend'] = 1 if market_metrics['volume_24h'] > market_metrics['volume_ma_7d'] else -1
        
        # Sentiment trend
        analysis_results['sentiment_trend'] = 1 if analysis_results.get('overall_sentiment_mean', 0) > 0 else -1
    
    logging.info("Market trend analysis completed")
    return analysis_results

if __name__ == "__main__":
    market_trends = analyze_market_trends()
    try:
        with open("market_trends.json", "w", encoding='utf-8') as file:
            json.dump(market_trends, file, indent=2)
        logging.info("Successfully saved market trends")
    except Exception as e:
        logging.error(f"Error saving market trends: {e}")