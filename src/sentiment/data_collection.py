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

# Define project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

# Load environment variables
load_dotenv(PROJECT_ROOT / 'config' / '.env.sentiment')

# API Configuration
API_CONFIG = {
    'newsapi': {
        'base_url': os.getenv("NEWSAPI_URL"),
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
        self.log_dir = LOG_DIR
        # Initialize session first
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoBot/1.0',
            'Accept': 'application/json'
        })
        # Then setup other components
        self.setup_logging()
        self.setup_apis()
        self._setup_twitter_api()

    def setup_logging(self):
        """Set up logging configuration."""
        log_file = self.log_dir / 'crypto_bot.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _setup_twitter_api(self):
        """Initialize Twitter API client with better error handling."""
        try:
            api_key = os.getenv("TWITTER_API_KEY")
            api_secret = os.getenv("TWITTER_API_SECRET_KEY")
            access_token = os.getenv("TWITTER_ACCESS_TOKEN")
            access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

            if not all([api_key, api_secret, access_token, access_token_secret, bearer_token]):
                logging.warning("Missing Twitter API credentials - Twitter functionality will be disabled")
                self.twitter_api = None
                self.twitter_client = None
                return

            # Use Tweepy v2 API with proper authentication
            auth = tweepy.OAuth1UserHandler(
                api_key,
                api_secret,
                access_token,
                access_token_secret
            )
            
            self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
            self.twitter_client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )
            
            logging.info("Twitter API initialized successfully")
        except Exception as e:
            logging.error(f"Twitter API authentication failed: {e}")
            self.twitter_api = None
            self.twitter_client = None

    def setup_apis(self):
        """Set up API configurations with proper URL formatting."""
        try:
            # CoinMarketCap API setup
            coinmarketcap_api_key = os.getenv("COINMARKETCAP_API_KEY")
            coinmarketcap_base_url = os.getenv("COINMARKETCAP_API_URL", "https://pro-api.coinmarketcap.com/v1")
            
            if coinmarketcap_api_key:
                self.session.headers.update({
                    'X-CMC_PRO_API_KEY': coinmarketcap_api_key
                })
                logging.info("CoinMarketCap API configured successfully")
            else:
                logging.warning("CoinMarketCap API key not found - some features will be disabled")

            # Other API configurations remain the same
            self.api_config = API_CONFIG
            logging.info("API configurations set up successfully")
        except Exception as e:
            logging.error(f"Error setting up APIs: {e}")
            raise

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
            if source == 'cryptopanic':
                url = os.getenv('CRYPTOPANIC_API_URL')
            elif source == 'newsapi':
                url = os.getenv('NEWSAPI_URL')
            else:
                logging.error(f"Unknown news source: {source}")
                return None

            if not url:
                logging.error(f"Missing API URL for source: {source}")
                return None

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching {source} data: {e}")
            return None

    def fetch_twitter_data(self, query: str = "bitcoin", count: int = 100) -> List[Dict]:
        """Fetch tweets using Twitter API v2."""
        if not self.twitter_client:
            logging.warning("Twitter client not initialized - skipping Twitter data collection")
            return []

        try:
            tweets = []
            # Use search_recent_tweets for v2 API
            response = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=count,
                tweet_fields=['created_at', 'public_metrics', 'text'],
                user_fields=['username', 'public_metrics']
            )
            
            if not response.data:
                logging.warning(f"No tweets found for query: {query}")
                return []

            for tweet in response.data:
                tweets.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat(),
                    'metrics': tweet.public_metrics,
                    'query': query
                })

            # Save to file
            twitter_file = self.data_dir / "twitter_data.json"
            with open(twitter_file, 'w') as f:
                json.dump(tweets, f, indent=2)
            
            logging.info(f"Successfully collected {len(tweets)} tweets")
            return tweets

        except Exception as e:
            logging.error(f"Error fetching Twitter data: {e}")
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
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("Invalid response format from CoinMarketCap API")
            
            current_time = datetime.now().isoformat()
            market_data = []
            for crypto in data['data']:
                try:
                    quote = crypto['quote']['USD']
                    market_data.append({
                        'id': crypto['id'],
                        'name': crypto['name'],
                        'symbol': crypto['symbol'],
                        'timestamp': current_time,
                        'metrics': {
                            'price': quote['price'],
                            'volume_24h': quote['volume_24h'],
                            'volume_change_24h': quote['volume_change_24h'],
                            'percent_change_1h': quote['percent_change_1h'],
                            'percent_change_24h': quote['percent_change_24h'],
                            'percent_change_7d': quote['percent_change_7d'],
                            'market_cap': quote['market_cap'],
                            'market_cap_dominance': quote['market_cap_dominance'],
                            'last_updated': quote['last_updated']
                        }
                    })
                except (KeyError, TypeError) as e:
                    logging.warning(f"Error processing crypto data for {crypto.get('name', 'Unknown')}: {e}")
                    continue
            
            return market_data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching market data: {e}")
            return None
        except ValueError as e:
            logging.error(f"Error processing market data: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in market data collection: {e}")
            return None

    def create_historical_data(self):
        """Create historical data from collected market data."""
        try:
            market_data_file = self.data_dir / "market_data.json"
            if not market_data_file.exists():
                logging.error("Market data file not found")
                return

            with open(market_data_file, 'r') as f:
                market_data = json.load(f)

            if not market_data:
                logging.error("No market data available")
                return

            historical_data = []
            for crypto in market_data:
                try:
                    historical_entry = {
                        'timestamp': crypto['timestamp'],
                        'symbol': crypto['symbol'],
                        'name': crypto['name'],
                        'price': crypto['metrics']['price'],
                        'market_cap': crypto['metrics']['market_cap'],
                        'volume_24h': crypto['metrics']['volume_24h'],
                        'percent_change_24h': crypto['metrics']['percent_change_24h']
                    }
                    historical_data.append(historical_entry)
                except KeyError as e:
                    logging.warning(f"Missing data for historical entry: {e}")
                    continue

            if historical_data:
                historical_file = self.data_dir / "historical_data.json"
                with open(historical_file, 'w') as f:
                    json.dump(historical_data, f, indent=4)
                logging.info("Successfully created historical data")
            else:
                logging.warning("No valid historical data entries created")

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
        """Collect all required data and store it in the data directory."""
        logging.info("Starting data collection")
        
        try:
            # Collect market data
            market_data = self.collect_market_data()
            if market_data:
                market_data_file = self.data_dir / "market_data.json"
                with open(market_data_file, 'w') as f:
                    json.dump(market_data, f, indent=4)
                logging.info(f"Successfully collected market data for {len(market_data)} cryptocurrencies")
            
            # Collect news data from different sources
            news_sources = {
                'cryptopanic': self._fetch_news_data('cryptopanic'),
                'newsapi': self._fetch_news_data('newsapi')
            }
            
            for source, data in news_sources.items():
                if data:
                    news_file = self.data_dir / f"{source}_news.json"
                    with open(news_file, 'w') as f:
                        json.dump(data, f, indent=4)
                    logging.info(f"Successfully collected {source} news data")
            
            # Collect Twitter data for major cryptocurrencies
            crypto_queries = ["bitcoin", "ethereum", "cryptocurrency"]
            twitter_data = []
            for query in crypto_queries:
                tweets = self.fetch_twitter_data(query)
                twitter_data.extend(tweets)
            
            if twitter_data:
                twitter_file = self.data_dir / "twitter_data.json"
                with open(twitter_file, 'w') as f:
                    json.dump(twitter_data, f, indent=4)
                logging.info(f"Successfully collected {len(twitter_data)} tweets")
            
            # Create historical data
            self.create_historical_data()
            logging.info("Successfully created historical data")
            
            logging.info("Data collection completed successfully")
            
        except Exception as e:
            logging.error(f"Error in data collection: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()