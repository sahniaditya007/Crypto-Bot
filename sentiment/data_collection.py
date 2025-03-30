import requests
import tweepy
import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, Timeout, RequestException
from ratelimit import limits, sleep_and_retry
from functools import lru_cache
from typing import Dict, Optional, List, Any

# Define base directory
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
DATA_DIR = BASE_DIR / "data"

# Create necessary directories
for directory in [CACHE_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / '.env')

# API Configuration
API_CONFIG = {
    'newsapi': {
        'base_url': os.getenv("NEWSAPI_URL"),  # Using direct NewsAPI URL
        'sources': ['bbc-news', 'cnn', 'fox-news', 'google-news'],
        'cache_ttl': 1800  # 30 minutes
    },
    'coingecko': {
        'url': os.getenv("COINGECKO_API_URL"),
        'cache_ttl': 300,  # 5 minutes
        'proxies': {
            'http': os.getenv('HTTP_PROXY'),
            'https': os.getenv('HTTPS_PROXY')
        }
    }
}

class DataCollector:
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.data_dir = DATA_DIR
        self._setup_cache_dir()
        self._setup_twitter_api()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoBot/1.0',
            'Accept': 'application/json'
        })
        self.setup_logging()
        self.setup_apis()

    def _setup_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _setup_twitter_api(self):
        """Initialize Twitter API client."""
        try:
            auth = tweepy.OAuth1UserHandler(
                os.getenv("TWITTER_API_KEY"),
                os.getenv("TWITTER_API_SECRET_KEY"),
                os.getenv("TWITTER_ACCESS_TOKEN"),
                os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            )
            self.twitter_api = tweepy.API(auth)
            logging.info("Twitter API initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Twitter API: {e}")
            self.twitter_api = None

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='logs/crypto_bot.log'
        )

    def setup_apis(self):
        """Initialize API connections and verify credentials."""
        try:
            # CoinMarketCap API setup
            self.cmc_headers = {
                'X-CMC_PRO_API_KEY': os.getenv('COINMARKETCAP_API_KEY'),
                'Accept': 'application/json'
            }
            
            # Verify CoinMarketCap API
            response = requests.get(
                f"{os.getenv('COINMARKETCAP_API_URL')}/cryptocurrency/listings/latest",
                headers=self.cmc_headers
            )
            if response.status_code == 200:
                logging.info("CoinMarketCap API initialized successfully")
            else:
                logging.error(f"CoinMarketCap API initialization failed: {response.status_code}")

            # Other API setups can be added here
            
        except Exception as e:
            logging.error(f"Error setting up APIs: {e}")

    @sleep_and_retry
    @limits(calls=50, period=60)
    def _fetch_coingecko_data(self, coin_id: str = "bitcoin") -> Optional[Dict]:
        """Fetch current and historical data from CoinGecko."""
        try:
            # Get current data with retry mechanism
            current_url = f"{API_CONFIG['coingecko']['url']}/coins/{coin_id}"
            for attempt in range(3):
                try:
                    current_response = self.session.get(
                        current_url,
                        timeout=10,
                        proxies=API_CONFIG['coingecko']['proxies']
                    )
                    current_response.raise_for_status()
                    current_data = current_response.json()
                    break
                except (ConnectionError, Timeout) as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff

            # Get historical data with retry mechanism
            history_url = f"{API_CONFIG['coingecko']['url']}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '30',
                'interval': 'daily'
            }
            for attempt in range(3):
                try:
                    history_response = self.session.get(
                        history_url,
                        params=params,
                        timeout=10,
                        proxies=API_CONFIG['coingecko']['proxies']
                    )
                    history_response.raise_for_status()
                    history_data = history_response.json()
                    break
                except (ConnectionError, Timeout) as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff

            # Validate data before processing
            if not current_data or not history_data:
                logging.error("Invalid data received from CoinGecko")
                return None

            # Process and combine data
            market_data = []
            for i in range(len(history_data['prices'])):
                timestamp = datetime.fromtimestamp(history_data['prices'][i][0] / 1000)
                market_data.append({
                    'timestamp': timestamp.isoformat(),
                    'current_price': history_data['prices'][i][1],
                    'market_cap': history_data['market_caps'][i][1],
                    'total_volume': history_data['total_volumes'][i][1],
                    'last_updated': timestamp.isoformat()
                })

            # Add current data with validation
            try:
                current_price = current_data['market_data']['current_price']['usd']
                market_cap = current_data['market_data']['market_cap']['usd']
                total_volume = current_data['market_data']['total_volume']['usd']
                price_change_24h = current_data['market_data']['price_change_percentage_24h']
                market_cap_change_24h = current_data['market_data']['market_cap_change_percentage_24h']

                market_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'current_price': current_price,
                    'market_cap': market_cap,
                    'total_volume': total_volume,
                    'price_change_percentage_24h': price_change_24h,
                    'market_cap_change_percentage_24h': market_cap_change_24h,
                    'last_updated': datetime.now().isoformat()
                })
            except KeyError as e:
                logging.error(f"Missing required data in CoinGecko response: {e}")
                return None

            return market_data

        except Exception as e:
            logging.error(f"Error fetching CoinGecko data: {e}")
            return None

    @sleep_and_retry
    @limits(calls=30, period=60)
    def _fetch_news_data(self, source: str) -> Optional[Dict]:
        """Fetch news data from various sources."""
        try:
            url = f"{API_CONFIG['newsapi']['base_url']}/everything/{source}.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching {source} data: {e}")
            return None

    def fetch_twitter_data(self, query: str = "bitcoin", count: int = 100) -> List[Dict]:
        """Fetch Twitter data with error handling."""
        if not self.twitter_api:
            logging.error("Twitter API not initialized")
            return []

        try:
            tweets = self.twitter_api.search_tweets(q=query, count=count, tweet_mode="extended")
            return [{
                'id': tweet.id_str,
                'text': tweet.full_text,
                'created_at': tweet.created_at.isoformat(),
                'user': tweet.user.screen_name,
                'retweet_count': tweet.retweet_count,
                'favorite_count': tweet.favorite_count
            } for tweet in tweets]
        except Exception as e:
            logging.error(f"Twitter API error: {e}")
            return []

    def collect_market_data(self):
        """Collect market data from CoinMarketCap."""
        try:
            url = f"{os.getenv('COINMARKETCAP_API_URL')}/cryptocurrency/listings/latest"
            params = {
                'start': '1',
                'limit': '100',
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=self.cmc_headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("Invalid response format from CoinMarketCap API")
            
            market_data = []
            for crypto in data['data']:
                try:
                    quote = crypto['quote']['USD']
                    market_data.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': crypto['symbol'],
                        'name': crypto['name'],
                        'price': quote['price'],
                        'market_cap': quote['market_cap'],
                        'volume': quote['volume_24h'],
                        'percent_change_24h': quote['percent_change_24h'],
                        'volume_change_24h': quote.get('volume_change_24h', 0),
                        'market_cap_change_24h': quote.get('market_cap_change_24h', 0)
                    })
                except KeyError as e:
                    logging.warning(f"Skipping cryptocurrency due to missing data: {e}")
                    continue
            
            # Save market data
            with open(os.path.join(self.data_dir, "market_data.json"), "w") as f:
                json.dump(market_data, f)
            
            logging.info(f"Successfully collected market data for {len(market_data)} cryptocurrencies")
            
        except Exception as e:
            logging.error(f"Error collecting market data: {e}")
            raise

    def create_historical_data(self):
        """Create historical data file for model training."""
        try:
            # Load market data
            with open(os.path.join(self.data_dir, "market_data.json"), "r") as file:
                market_data = json.load(file)

            if not market_data:
                logging.warning("No market data available for creating historical data")
                return

            historical_data = []
            for i in range(1, len(market_data)):
                current = market_data[i]
                previous = market_data[i-1]

                try:
                    historical_data.append({
                        'timestamp': current['timestamp'],
                        'price': current['price'],
                        'market_cap': current['market_cap'],
                        'volume': current['volume'],
                        'price_change_24h': current['percent_change_24h'],
                        'market_cap_change_24h': current['market_cap_change_24h'],
                        'volume_change_24h': current['volume_change_24h']
                    })
                except KeyError as e:
                    logging.warning(f"Skipping data point due to missing field: {e}")
                    continue

            # Save historical data
            with open(os.path.join(self.data_dir, "historical_data.json"), "w") as f:
                json.dump(historical_data, f)
            logging.info("Successfully created historical data")

        except Exception as e:
            logging.error(f"Error creating historical data: {e}")
            raise

    def collect_news_data(self):
        """Collect news data from various sources."""
        try:
            # Collect from CryptoPanic
            cryptopanic_url = os.getenv("CRYPTOPANIC_API_URL")
            response = requests.get(cryptopanic_url, timeout=10)
            response.raise_for_status()
            cryptopanic_data = response.json()
            
            # Save CryptoPanic data
            with open(os.path.join(self.data_dir, "cryptopanic_news.json"), "w") as f:
                json.dump(cryptopanic_data, f)
            logging.info("Successfully collected CryptoPanic news data")

            # Collect from NewsAPI
            newsapi_key = os.getenv("NEWSAPI_URL").split("apiKey=")[1]
            newsapi_url = f"https://newsapi.org/v2/everything?q=cryptocurrency&apiKey={newsapi_key}&language=en&sortBy=publishedAt&pageSize=100"
            
            response = requests.get(newsapi_url, timeout=10)
            response.raise_for_status()
            newsapi_data = response.json()
            
            # Save NewsAPI data
            with open(os.path.join(self.data_dir, "newsapi_news.json"), "w") as f:
                json.dump(newsapi_data, f)
            logging.info("Successfully collected NewsAPI data")

        except Exception as e:
            logging.error(f"Error collecting news data: {e}")

    def collect_social_data(self):
        """Collect social media data."""
        try:
            # Collect Twitter data
            if self.twitter_api:
                try:
                    # Collect tweets about major cryptocurrencies
                    cryptocurrencies = ["bitcoin", "ethereum", "crypto"]
                    all_tweets = []
                    
                    for crypto in cryptocurrencies:
                        try:
                            tweets = self.fetch_twitter_data(query=crypto, count=100)
                            all_tweets.extend(tweets)
                        except Exception as e:
                            logging.warning(f"Error fetching tweets for {crypto}: {e}")
                            continue
                    
                    # Save Twitter data if we have any
                    if all_tweets:
                        with open(os.path.join(self.data_dir, "twitter_data.json"), "w") as f:
                            json.dump(all_tweets, f)
                        logging.info("Successfully collected Twitter data")
                    else:
                        logging.warning("No Twitter data collected")
                except Exception as e:
                    logging.warning(f"Twitter API error: {e}")
                    # Create empty Twitter data file to prevent errors
                    with open(os.path.join(self.data_dir, "twitter_data.json"), "w") as f:
                        json.dump([], f)
            else:
                logging.warning("Twitter API not initialized, skipping Twitter data collection")
                # Create empty Twitter data file to prevent errors
                with open(os.path.join(self.data_dir, "twitter_data.json"), "w") as f:
                    json.dump([], f)

        except Exception as e:
            logging.error(f"Error collecting social data: {e}")
            # Create empty Twitter data file to prevent errors
            with open(os.path.join(self.data_dir, "twitter_data.json"), "w") as f:
                json.dump([], f)

    def collect_data(self):
        """Main method to collect all data."""
        try:
            logging.info("Starting data collection")
            
            # Collect market data
            self.collect_market_data()
            
            # Collect news data
            self.collect_news_data()
            
            # Collect social data
            self.collect_social_data()
            
            # Create historical data
            self.create_historical_data()
            
            logging.info("Data collection completed successfully")
            
        except Exception as e:
            logging.error(f"Error in data collection: {e}")
            raise

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()