import requests
import tweepy
import json
import os
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, Timeout, RequestException

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

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except ConnectionError:
        print(f"Error connecting to {url}. Please check your network connection.")
    except Timeout:
        print(f"Request to {url} timed out. Please try again later.")
    except RequestException as e:
        print(f"An error occurred: {e}")
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
    coindesk_news = fetch_coindesk_news()
    cryptopanic_news = fetch_cryptopanic_news()
    newsapi_news = fetch_newsapi_news()
    twitter_data = fetch_twitter_data(query="cryptocurrency")
    market_data = fetch_market_data(crypto_id="bitcoin")

    if coindesk_news:
        with open("coindesk_news.json", "w") as file:
            file.write(json.dumps(coindesk_news))
    if cryptopanic_news:
        with open("cryptopanic_news.json", "w") as file:
            file.write(json.dumps(cryptopanic_news))
    if newsapi_news:
        with open("newsapi_news.json", "w") as file:
            file.write(json.dumps(newsapi_news))
    if twitter_data:
        with open("twitter_data.json", "w") as file:
            file.write(json.dumps(twitter_data))
    if market_data:
        with open("market_data.json", "w") as file:
            file.write(json.dumps(market_data))

if __name__ == "__main__":
    collect_data()