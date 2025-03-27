import json
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def process_news_data(file_path):
    with open(file_path, "r") as file:
        news_data = json.load(file)
    for article in news_data:
        article["sentiment"] = analyze_sentiment(article.get("content", article.get("title", "")))
    return news_data

def process_twitter_data(file_path):
    with open(file_path, "r") as file:
        twitter_data = json.load(file)
    for tweet in twitter_data:
        tweet["sentiment"] = analyze_sentiment(tweet["text"])
    return twitter_data

if __name__ == "__main__":
    coindesk_sentiment = process_news_data("coindesk_news.json")
    cryptopanic_sentiment = process_news_data("cryptopanic_news.json")
    newsapi_sentiment = process_news_data("newsapi_news.json")
    twitter_sentiment = process_twitter_data("twitter_data.json")

    with open("coindesk_sentiment.json", "w") as file:
        file.write(json.dumps(coindesk_sentiment))
    with open("cryptopanic_sentiment.json", "w") as file:
        file.write(json.dumps(cryptopanic_sentiment))
    with open("newsapi_sentiment.json", "w") as file:
        file.write(json.dumps(newsapi_sentiment))
    with open("twitter_sentiment.json", "w") as file:
        file.write(json.dumps(twitter_sentiment))