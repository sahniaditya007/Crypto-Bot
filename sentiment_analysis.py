import json
import logging
import re
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_bot.log'
)

def preprocess_text(text):
    """Preprocess text for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def analyze_sentiment(text):
    """Analyze sentiment of preprocessed text."""
    try:
        processed_text = preprocess_text(text)
        if not processed_text:
            return 0.0
        
        blob = TextBlob(processed_text)
        return blob.sentiment.polarity
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return 0.0

def process_news_data(file_path):
    """Process news data and add sentiment scores."""
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            news_data = json.load(file)
        
        for article in news_data:
            content = article.get("content", article.get("title", ""))
            article["sentiment"] = analyze_sentiment(content)
            article["processed_text"] = preprocess_text(content)
        
        return news_data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error processing news data: {e}")
        return []

def process_twitter_data(file_path):
    """Process Twitter data and add sentiment scores."""
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            twitter_data = json.load(file)
        
        for tweet in twitter_data:
            text = tweet.get("text", "")
            tweet["sentiment"] = analyze_sentiment(text)
            tweet["processed_text"] = preprocess_text(text)
        
        return twitter_data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error processing Twitter data: {e}")
        return []

if __name__ == "__main__":
    logging.info("Starting sentiment analysis")
    
    coindesk_sentiment = process_news_data("coindesk_news.json")
    if coindesk_sentiment:
        with open("coindesk_sentiment.json", "w", encoding='utf-8') as file:
            json.dump(coindesk_sentiment, file, ensure_ascii=False, indent=2)
        logging.info("Processed Coindesk sentiment")
    
    cryptopanic_sentiment = process_news_data("cryptopanic_news.json")
    if cryptopanic_sentiment:
        with open("cryptopanic_sentiment.json", "w", encoding='utf-8') as file:
            json.dump(cryptopanic_sentiment, file, ensure_ascii=False, indent=2)
        logging.info("Processed Cryptopanic sentiment")
    
    newsapi_sentiment = process_news_data("newsapi_news.json")
    if newsapi_sentiment:
        with open("newsapi_sentiment.json", "w", encoding='utf-8') as file:
            json.dump(newsapi_sentiment, file, ensure_ascii=False, indent=2)
        logging.info("Processed NewsAPI sentiment")
    
    twitter_sentiment = process_twitter_data("twitter_data.json")
    if twitter_sentiment:
        with open("twitter_sentiment.json", "w", encoding='utf-8') as file:
            json.dump(twitter_sentiment, file, ensure_ascii=False, indent=2)
        logging.info("Processed Twitter sentiment")
    
    logging.info("Sentiment analysis completed")