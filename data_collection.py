import requests
import tweepy
import json
import os
import time
import logging
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, Timeout, RequestException
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_bot.log'
)

# Load environment variables
load_dotenv()

COINDESK_API_URL = os.getenv("COINDESK_API_URL")
CRYPTOPANIC_API_URL = os.getenv("CRYPTOPANIC_API_URL")
NEWSAPI_URL = os.getenv("NEWSAPI_URL")
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET_KEY = os.getenv("TWITTER_API_SECRET_KEY")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
COINGECKO_API_URL = os.getenv("COINGECKO_API_URL")

# Rate limiting decorators
@sleep_and_retry
@limits(calls=30, period=60)  # 30 calls per minute
def fetch_data(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except ConnectionError:
            logging.error(f"Connection error to {url}. Attempt {attempt + 1}/{retries}")
            if attempt == retries - 1:
                logging.error(f"Failed to connect to {url} after {retries} attempts")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
        except Timeout:
            logging.error(f"Request to {url} timed out. Attempt {attempt + 1}/{retries}")
            if attempt == retries - 1:
                logging.error(f"Request to {url} timed out after {retries} attempts")
                return None
            time.sleep(2 ** attempt)
        except RequestException as e:
            logging.error(f"Request error for {url}: {e}")
            if attempt == retries - 1:
                logging.error(f"Failed to fetch data from {url} after {retries} attempts")
                return None
            time.sleep(2 ** attempt)
    return None

def fetch_coindesk_news():
    return fetch_data(COINDESK_API_URL)

def fetch_cryptopanic_news():
    return fetch_data(CRYPTOPANIC_API_URL)

def fetch_newsapi_news():
    return fetch_data(NEWSAPI_URL)

def fetch_twitter_data(query, count=100):
    auth = tweepy.OAuth1UserHandler(
        TWITTER_API_KEY, TWITTER_API_SECRET_KEY, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET
    )
    api = tweepy.API(auth)
    try:
        tweets = api.search_tweets(q=query, count=count)
        return [tweet._json for tweet in tweets]
    except tweepy.TweepyException as e:
        print(f"An error occurred: {e}")
    return []

def fetch_market_data(crypto_id):
    market_data_url = f"{COINGECKO_API_URL}/coins/markets?vs_currency=usd&ids={crypto_id}"
    return fetch_data(market_data_url)

def collect_data():
    logging.info("Starting data collection")
    
    coindesk_news = fetch_coindesk_news()
    if coindesk_news:
        with open("coindesk_news.json", "w") as file:
            file.write(json.dumps(coindesk_news))
        logging.info("Successfully collected Coindesk news")
    
    cryptopanic_news = fetch_cryptopanic_news()
    if cryptopanic_news:
        with open("cryptopanic_news.json", "w") as file:
            file.write(json.dumps(cryptopanic_news))
        logging.info("Successfully collected Cryptopanic news")
    
    newsapi_news = fetch_newsapi_news()
    if newsapi_news:
        with open("newsapi_news.json", "w") as file:
            file.write(json.dumps(newsapi_news))
        logging.info("Successfully collected NewsAPI news")
    
    twitter_data = fetch_twitter_data(query="cryptocurrency")
    if twitter_data:
        with open("twitter_data.json", "w") as file:
            file.write(json.dumps(twitter_data))
        logging.info("Successfully collected Twitter data")
    
    market_data = fetch_market_data(crypto_id="bitcoin")
    if market_data:
        with open("market_data.json", "w") as file:
            file.write(json.dumps(market_data))
        logging.info("Successfully collected market data")
    
    logging.info("Data collection completed")

if __name__ == "__main__":
    collect_data()