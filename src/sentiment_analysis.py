import json
import logging
import re
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import numpy as np
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_bot.log'
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize sentiment cache
        self.sentiment_cache = {}
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.special_chars_pattern = re.compile(r'[^\w\s]')

    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis with caching."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Remove special characters and numbers
        text = self.special_chars_pattern.sub('', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of preprocessed text with caching."""
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return 0.0
            
            # Check cache first
            if processed_text in self.sentiment_cache:
                return self.sentiment_cache[processed_text]
            
            blob = TextBlob(processed_text)
            sentiment = blob.sentiment.polarity
            
            # Cache the result
            self.sentiment_cache[processed_text] = sentiment
            return sentiment
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            return 0.0

    def process_item(self, item: Dict) -> Dict:
        """Process a single item (news article or tweet)."""
        try:
            content = item.get("content", item.get("title", item.get("text", "")))
            item["sentiment"] = self.analyze_sentiment(content)
            item["processed_text"] = self.preprocess_text(content)
            return item
        except Exception as e:
            logging.error(f"Error processing item: {e}")
            return item

    def process_batch(self, items: List[Dict]) -> List[Dict]:
        """Process a batch of items in parallel."""
        processed_items = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(self.process_item, item): item for item in items}
            for future in as_completed(future_to_item):
                try:
                    processed_item = future.result()
                    processed_items.append(processed_item)
                except Exception as e:
                    logging.error(f"Error processing batch item: {e}")
        return processed_items

    def process_news_data(self, file_path: str) -> List[Dict]:
        """Process news data with parallel processing."""
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                news_data = json.load(file)
            
            if not isinstance(news_data, list):
                news_data = [news_data]
            
            processed_data = self.process_batch(news_data)
            
            # Calculate aggregate metrics
            sentiments = [item.get("sentiment", 0) for item in processed_data]
            if sentiments:
                aggregate_metrics = {
                    "mean_sentiment": np.mean(sentiments),
                    "std_sentiment": np.std(sentiments),
                    "max_sentiment": max(sentiments),
                    "min_sentiment": min(sentiments),
                    "total_items": len(sentiments)
                }
                logging.info(f"Aggregate metrics for {file_path}: {aggregate_metrics}")
            
            return processed_data
            
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in file: {file_path}")
            return []
        except Exception as e:
            logging.error(f"Error processing news data: {e}")
            return []

    def process_twitter_data(self, file_path: str) -> List[Dict]:
        """Process Twitter data with parallel processing."""
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                twitter_data = json.load(file)
            
            if not isinstance(twitter_data, list):
                twitter_data = [twitter_data]
            
            processed_data = self.process_batch(twitter_data)
            
            # Calculate aggregate metrics
            sentiments = [item.get("sentiment", 0) for item in processed_data]
            if sentiments:
                aggregate_metrics = {
                    "mean_sentiment": np.mean(sentiments),
                    "std_sentiment": np.std(sentiments),
                    "max_sentiment": max(sentiments),
                    "min_sentiment": min(sentiments),
                    "total_tweets": len(sentiments)
                }
                logging.info(f"Aggregate metrics for {file_path}: {aggregate_metrics}")
            
            return processed_data
            
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in file: {file_path}")
            return []
        except Exception as e:
            logging.error(f"Error processing Twitter data: {e}")
            return []

def process_news_data(file_path: str) -> List[Dict]:
    """Wrapper function for processing news data."""
    analyzer = SentimentAnalyzer()
    return analyzer.process_news_data(file_path)

def process_twitter_data(file_path: str) -> List[Dict]:
    """Wrapper function for processing Twitter data."""
    analyzer = SentimentAnalyzer()
    return analyzer.process_twitter_data(file_path)

if __name__ == "__main__":
    logging.info("Starting sentiment analysis")
    
    analyzer = SentimentAnalyzer()
    
    # Process news data
    news_sources = ['coindesk', 'cryptopanic', 'newsapi']
    for source in news_sources:
        processed_data = analyzer.process_news_data(f"{source}_news.json")
        if processed_data:
            with open(f"{source}_sentiment.json", "w", encoding='utf-8') as file:
                json.dump(processed_data, file, ensure_ascii=False, indent=2)
            logging.info(f"Processed {source} sentiment")
    
    # Process Twitter data
    processed_twitter = analyzer.process_twitter_data("twitter_data.json")
    if processed_twitter:
        with open("twitter_sentiment.json", "w", encoding='utf-8') as file:
            json.dump(processed_twitter, file, ensure_ascii=False, indent=2)
        logging.info("Processed Twitter sentiment")
    
    logging.info("Sentiment analysis completed")