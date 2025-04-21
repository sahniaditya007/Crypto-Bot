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
import os
from pathlib import Path
from datetime import datetime

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

def setup_nltk():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger')

class SentimentAnalyzer:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.data_dir = DATA_DIR
        self.log_dir = LOG_DIR
        setup_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize sentiment cache
        self.sentiment_cache = {}
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.special_chars_pattern = re.compile(r'[^\w\s]')
        
        # Setup logging
        log_file = self.log_dir / 'sentiment_analysis.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis with caching."""
        if not isinstance(text, str):
            return ""
        
        try:
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
        except Exception as e:
            logging.error(f"Error in text preprocessing: {e}")
            return ""

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
            if not content:
                return item
                
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

    def process_news_file(self, file_path: Path) -> None:
        """Process a news data file and save sentiment analysis results."""
        try:
            if not file_path.exists():
                logging.warning(f"News file not found: {file_path}")
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            processed_data = []
            if 'articles' in data:  # NewsAPI format
                for article in data['articles']:
                    try:
                        # Combine title and description for sentiment analysis
                        content = f"{article['title']} {article['description']}"
                        sentiment = self.analyze_sentiment(content)
                        processed_data.append({
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'published_at': article['publishedAt'],
                            'source': article['source']['name'],
                            'sentiment': sentiment,
                        })
                    except Exception as e:
                        logging.warning(f"Error processing article: {e}")
                        continue

            elif 'results' in data:  # CryptoPanic format
                for result in data['results']:
                    try:
                        content = result['title']
                        sentiment = self.analyze_sentiment(content)
                        processed_data.append({
                            'title': result['title'],
                            'url': result['url'],
                            'published_at': result['published_at'],
                            'source': result['source']['title'],
                            'sentiment': sentiment,
                        })
                    except Exception as e:
                        logging.warning(f"Error processing result: {e}")
                        continue

            # Calculate aggregate metrics
            sentiments = [item.get("sentiment", 0) for item in processed_data]
            if sentiments:
                aggregate_metrics = {
                    "mean_sentiment": float(np.mean(sentiments)),
                    "std_sentiment": float(np.std(sentiments)),
                    "max_sentiment": float(max(sentiments)),
                    "min_sentiment": float(min(sentiments)),
                    "total_items": len(sentiments)
                }
            else:
                aggregate_metrics = {
                    "mean_sentiment": 0.0,
                    "std_sentiment": 0.0,
                    "max_sentiment": 0.0,
                    "min_sentiment": 0.0,
                    "total_items": 0
                }

            # Save sentiment data
            output_file = self.data_dir / f"{file_path.stem}_sentiment.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(aggregate_metrics, f, indent=4)

            logging.info(f"Aggregate metrics for {file_path}: {aggregate_metrics}")
            logging.info(f"Saved sentiment data to {output_file}")

        except Exception as e:
            logging.error(f"Error processing news file {file_path}: {e}")
            raise

    def process_twitter_file(self, file_path: Path) -> None:
        """Process Twitter data file and save sentiment analysis results."""
        try:
            if not file_path.exists():
                logging.warning(f"Twitter file not found: {file_path}")
                return

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            processed_data = []
            for tweet in data:
                try:
                    content = tweet['text']
                    sentiment = self.analyze_sentiment(content)
                    processed_data.append({
                        'id': tweet['id'],
                        'text': content,
                        'created_at': tweet['created_at'],
                        'metrics': tweet['metrics'],
                        'sentiment': sentiment,
                    })
                except Exception as e:
                    logging.warning(f"Error processing tweet: {e}")
                    continue

            # Calculate aggregate metrics
            sentiments = [item.get("sentiment", 0) for item in processed_data]
            if sentiments:
                aggregate_metrics = {
                    "mean_sentiment": float(np.mean(sentiments)),
                    "std_sentiment": float(np.std(sentiments)),
                    "max_sentiment": float(max(sentiments)),
                    "min_sentiment": float(min(sentiments)),
                    "total_items": len(sentiments)
                }
            else:
                aggregate_metrics = {
                    "mean_sentiment": 0.0,
                    "std_sentiment": 0.0,
                    "max_sentiment": 0.0,
                    "min_sentiment": 0.0,
                    "total_items": 0
                }

            # Save sentiment data
            output_file = self.data_dir / "twitter_sentiment.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(aggregate_metrics, f, indent=4)

            logging.info(f"Aggregate metrics for Twitter data: {aggregate_metrics}")
            logging.info(f"Saved Twitter sentiment data to {output_file}")

        except Exception as e:
            logging.error(f"Error processing Twitter file {file_path}: {e}")
            raise

def process_news_data(file_path: str) -> List[Dict]:
    """Process news data from file."""
    analyzer = SentimentAnalyzer()
    file_path = Path(file_path)
    analyzer.process_news_file(file_path)
    return []

def process_twitter_data(file_path: str) -> List[Dict]:
    """Process Twitter data from file."""
    analyzer = SentimentAnalyzer()
    file_path = Path(file_path)
    analyzer.process_twitter_file(file_path)
    return []

if __name__ == "__main__":
    logging.info("Starting sentiment analysis")
    
    analyzer = SentimentAnalyzer()
    
    # Process news data
    news_sources = ['coindesk', 'cryptopanic', 'newsapi']
    for source in news_sources:
        processed_data = analyzer.process_news_data(os.path.join(analyzer.data_dir, f"{source}_news.json"))
        if processed_data:
            with open(os.path.join(analyzer.data_dir, f"{source}_sentiment.json"), "w", encoding='utf-8') as file:
                json.dump(processed_data, file, ensure_ascii=False, indent=2)
            logging.info(f"Processed {source} sentiment")
    
    # Process Twitter data
    processed_twitter = analyzer.process_twitter_data(os.path.join(analyzer.data_dir, "twitter_data.json"))
    if processed_twitter:
        with open(os.path.join(analyzer.data_dir, "twitter_sentiment.json"), "w", encoding='utf-8') as file:
            json.dump(processed_twitter, file, ensure_ascii=False, indent=2)
        logging.info("Processed Twitter sentiment")
    
    logging.info("Sentiment analysis completed")