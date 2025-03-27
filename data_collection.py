import requests
import tweepy
import json
import os
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, Timeout, RequestException
from ratelimit import limits, sleep_and_retry
from functools import lru_cache
from typing import Dict, Optional, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_bot.log'
)

# Load environment variables
load_dotenv()

# API Configuration
API_CONFIG = {
    'coindesk': {
        'url': os.getenv("COINDESK_API_URL"),
        'cache_ttl': 3600  # 1 hour
    },
    'cryptopanic': {
        'url': os.getenv("CRYPTOPANIC_API_URL"),
        'cache_ttl': 1800  # 30 minutes
    },
    'newsapi': {
        'url': os.getenv("NEWSAPI_URL"),
        'cache_ttl': 1800  # 30 minutes
    },
    'coingecko': {
        'url': os.getenv("COINGECKO_API_URL"),
        'cache_ttl': 300  # 5 minutes
    }
}

# Twitter API Configuration
TWITTER_CONFIG = {
    'api_key': os.getenv("TWITTER_API_KEY"),
    'api_secret_key': os.getenv("TWITTER_API_SECRET_KEY"),
    'access_token': os.getenv("TWITTER_ACCESS_TOKEN"),
    'access_token_secret': os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
}

class DataCollector:
    def __init__(self):
        self.cache_dir = "cache"
        self._setup_cache_dir()
        self._setup_twitter_api()

    def _setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _setup_twitter_api(self):
        """Initialize Twitter API client."""
        try:
            auth = tweepy.OAuth1UserHandler(
                TWITTER_CONFIG['api_key'],
                TWITTER_CONFIG['api_secret_key'],
                TWITTER_CONFIG['access_token'],
                TWITTER_CONFIG['access_token_secret']
            )
            self.twitter_api = tweepy.API(auth)
        except Exception as e:
            logging.error(f"Failed to initialize Twitter API: {e}")
            self.twitter_api = None

    def _get_cache_path(self, source: str) -> str:
        """Get cache file path for a data source."""
        return os.path.join(self.cache_dir, f"{source}_cache.json")

    def _is_cache_valid(self, source: str) -> bool:
        """Check if cached data is still valid."""
        cache_path = self._get_cache_path(source)
        if not os.path.exists(cache_path):
            return False
        
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        ttl = API_CONFIG[source]['cache_ttl']
        return datetime.now() - cache_time < timedelta(seconds=ttl)

    @sleep_and_retry
    @limits(calls=30, period=60)  # 30 calls per minute
    def _fetch_data(self, url: str, retries: int = 3) -> Optional[Dict]:
        """Fetch data from API with retry mechanism."""
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except ConnectionError:
                logging.error(f"Connection error to {url}. Attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
            except Timeout:
                logging.error(f"Request to {url} timed out. Attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except RequestException as e:
                logging.error(f"Request error for {url}: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return None

    def _save_to_cache(self, source: str, data: Dict):
        """Save data to cache file."""
        cache_path = self._get_cache_path(source)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.error(f"Failed to save cache for {source}: {e}")

    def _load_from_cache(self, source: str) -> Optional[Dict]:
        """Load data from cache file."""
        cache_path = self._get_cache_path(source)
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load cache for {source}: {e}")
            return None

    def fetch_news_data(self, source: str) -> Optional[Dict]:
        """Fetch news data with caching."""
        if self._is_cache_valid(source):
            logging.info(f"Using cached data for {source}")
            return self._load_from_cache(source)

        try:
            data = self._fetch_data(API_CONFIG[source]['url'])
            if data:
                self._save_to_cache(source, data)
                return data
        except Exception as e:
            logging.error(f"Failed to fetch {source} data: {e}")
        return None

    def fetch_twitter_data(self, query: str, count: int = 100) -> List[Dict]:
        """Fetch Twitter data with error handling."""
        if not self.twitter_api:
            logging.error("Twitter API not initialized")
            return []

        try:
            tweets = self.twitter_api.search_tweets(q=query, count=count)
            return [tweet._json for tweet in tweets]
        except tweepy.TweepyException as e:
            logging.error(f"Twitter API error: {e}")
            return []

    def fetch_market_data(self, crypto_id: str) -> Optional[Dict]:
        """Fetch market data with caching."""
        if self._is_cache_valid('coingecko'):
            logging.info("Using cached market data")
            return self._load_from_cache('coingecko')

        try:
            market_data_url = f"{API_CONFIG['coingecko']['url']}/coins/markets?vs_currency=usd&ids={crypto_id}"
            data = self._fetch_data(market_data_url)
            if data:
                self._save_to_cache('coingecko', data)
                return data
        except Exception as e:
            logging.error(f"Failed to fetch market data: {e}")
        return None

    def collect_data(self):
        """Collect all required data."""
        logging.info("Starting data collection")
        
        collector = DataCollector()
        
        # Collect news data
        for source in ['coindesk', 'cryptopanic', 'newsapi']:
            data = collector.fetch_news_data(source)
            if data:
                with open(f"{source}_news.json", "w") as file:
                    json.dump(data, file)
                logging.info(f"Successfully collected {source} news")
        
        # Collect Twitter data
        twitter_data = collector.fetch_twitter_data(query="cryptocurrency")
        if twitter_data:
            with open("twitter_data.json", "w") as file:
                json.dump(twitter_data, file)
            logging.info("Successfully collected Twitter data")
        
        # Collect market data
        market_data = collector.fetch_market_data(crypto_id="bitcoin")
        if market_data:
            with open("market_data.json", "w") as file:
                json.dump(market_data, file)
            logging.info("Successfully collected market data")
        
        logging.info("Data collection completed")

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()