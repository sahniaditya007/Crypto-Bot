import json
import pandas as pd

def load_sentiment_data():
    with open("coindesk_sentiment.json", "r") as file:
        coindesk_sentiment = json.load(file)
    with open("cryptopanic_sentiment.json", "r") as file:
        cryptopanic_sentiment = json.load(file)
    with open("newsapi_sentiment.json", "r") as file:
        newsapi_sentiment = json.load(file)
    with open("twitter_sentiment.json", "r") as file:
        twitter_sentiment = json.load(file)
    return coindesk_sentiment, cryptopanic_sentiment, newsapi_sentiment, twitter_sentiment

def load_market_data():
    with open("market_data.json", "r") as file:
        market_data = json.load(file)
    return market_data

def analyze_market_trends():
    coindesk_sentiment, cryptopanic_sentiment, newsapi_sentiment, twitter_sentiment = load_sentiment_data()
    market_data = load_market_data()

    # Combine sentiment data
    all_sentiments = coindesk_sentiment + cryptopanic_sentiment + newsapi_sentiment + twitter_sentiment
    sentiment_df = pd.DataFrame(all_sentiments)

    # Calculate average sentiment
    average_sentiment = sentiment_df["sentiment"].mean()

    # Analyze market data
    market_df = pd.DataFrame(market_data)
    trading_volume = market_df["total_volume"].mean()
    price_movement = market_df["price_change_percentage_24h"].mean()

    return {
        "average_sentiment": average_sentiment,
        "trading_volume": trading_volume,
        "price_movement": price_movement,
    }

if __name__ == "__main__":
    market_trends = analyze_market_trends()
    with open("market_trends.json", "w") as file:
        file.write(json.dumps(market_trends))