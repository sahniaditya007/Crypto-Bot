import pickle
import json
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Load environment variables
load_dotenv()

# Define base directory
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

class PredictionConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class PredictionResult:
    prediction: int
    confidence_score: float
    confidence_level: PredictionConfidence
    timestamp: datetime
    features: Dict[str, float]

class PredictionGenerator:
    def __init__(self):
        self.model_dir = "models"
        self.data_dir = "data"
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self._load_latest_model()

    def _load_latest_model(self):
        """Load the latest trained model and related artifacts."""
        try:
            # Get list of model files
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_")]
            if not model_files:
                raise FileNotFoundError("No trained models found")

            # Get latest model file
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.model_dir, x)))
            timestamp = latest_model.split("_")[1].split(".")[0]

            # Load model
            model_path = os.path.join(self.model_dir, latest_model)
            with open(model_path, "rb") as file:
                self.model = pickle.load(file)

            # Load scaler
            scaler_path = os.path.join(self.model_dir, f"scaler_{timestamp}.pkl")
            with open(scaler_path, "rb") as file:
                self.scaler = pickle.load(file)

            # Load feature importance
            importance_path = os.path.join(self.model_dir, f"feature_importance_{timestamp}.json")
            self.feature_importance = pd.read_json(importance_path)

            logging.info(f"Loaded model from {latest_model}")
            logging.info(f"Model features: {list(self.feature_importance['feature'])}")

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def _validate_input_data(self, data: Dict[str, float]) -> pd.DataFrame:
        """Validate and preprocess input data."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data])

            # Check for missing features
            required_features = list(self.feature_importance['feature'])
            missing_features = set(required_features) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Check for extra features
            extra_features = set(df.columns) - set(required_features)
            if extra_features:
                logging.warning(f"Extra features found: {extra_features}")
                df = df[required_features]

            # Scale features
            df_scaled = self.scaler.transform(df)
            return pd.DataFrame(df_scaled, columns=df.columns)

        except Exception as e:
            logging.error(f"Error validating input data: {e}")
            raise

    def _determine_confidence_level(self, confidence_score: float) -> PredictionConfidence:
        """Determine confidence level based on score."""
        if confidence_score >= 0.8:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.6:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW

    def _get_feature_contributions(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature contributions to prediction."""
        try:
            contributions = {}
            for feature in data.columns:
                importance = self.feature_importance[
                    self.feature_importance['feature'] == feature
                ]['importance'].iloc[0]
                value = data[feature].iloc[0]
                contributions[feature] = importance * value
            return contributions
        except Exception as e:
            logging.error(f"Error calculating feature contributions: {e}")
            return {}

    def generate_predictions(self, market_trends: Dict[str, float]) -> List[PredictionResult]:
        """Generate predictions with confidence scores and feature analysis."""
        try:
            # Validate and preprocess input data
            input_data = self._validate_input_data(market_trends)

            # Generate predictions
            predictions = self.model.predict(input_data)
            confidence_scores = self.model.predict_proba(input_data)[:, 1]

            # Create prediction results
            results = []
            for pred, conf_score in zip(predictions, confidence_scores):
                result = PredictionResult(
                    prediction=int(pred),
                    confidence_score=float(conf_score),
                    confidence_level=self._determine_confidence_level(conf_score),
                    timestamp=datetime.now(),
                    features=self._get_feature_contributions(input_data)
                )
                results.append(result)

            return results

        except Exception as e:
            logging.error(f"Error generating predictions: {e}")
            raise

    def save_predictions(self, results: List[PredictionResult], file_path: str):
        """Save prediction results to file."""
        try:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_results.append({
                    "prediction": result.prediction,
                    "confidence_score": result.confidence_score,
                    "confidence_level": result.confidence_level.value,
                    "timestamp": result.timestamp.isoformat(),
                    "features": result.features
                })

            # Save to file
            with open(file_path, "w") as file:
                json.dump(serializable_results, file, indent=2)

            logging.info(f"Saved predictions to {file_path}")

        except Exception as e:
            logging.error(f"Error saving predictions: {e}")
            raise

def generate_predictions() -> Tuple[List[Dict], List[float]]:
    """Generate predictions based on market trends and sentiment analysis."""
    try:
        # Load market trends
        trends_file = DATA_DIR / "market_trends.json"
        if not trends_file.exists():
            logging.error(f"Error in prediction generation: {trends_file}")
            return [], []
            
        with open(trends_file, 'r') as file:
            trends = json.load(file)
            
        if not trends:
            logging.error("No market trends available for prediction")
            return [], []
            
        # Generate predictions
        predictions = []
        confidence_scores = []
        
        # Process each cryptocurrency
        for symbol in trends['price_trends'].keys():
            try:
                prediction, confidence = predict_crypto(symbol, trends)
                if prediction:
                    predictions.append(prediction)
                    confidence_scores.append(confidence)
            except Exception as e:
                logging.warning(f"Error predicting {symbol}: {e}")
                continue
                
        # Sort by confidence score
        sorted_indices = np.argsort(confidence_scores)[::-1]
        predictions = [predictions[i] for i in sorted_indices]
        confidence_scores = [confidence_scores[i] for i in sorted_indices]
        
        return predictions, confidence_scores
        
    except Exception as e:
        logging.error(f"Error in prediction generation: {e}")
        return [], []

def predict_crypto(symbol: str, trends: Dict) -> Tuple[Optional[Dict], float]:
    """Generate prediction for a single cryptocurrency."""
    try:
        # Get relevant data
        price_trend = trends['price_trends'].get(symbol, {})
        volume_trend = trends['volume_trends'].get(symbol, {})
        mcap_trend = trends['market_cap_trends'].get(symbol, {})
        sentiment_trends = trends['sentiment_trends']
        
        # Calculate confidence score
        confidence = calculate_confidence(
            price_trend,
            volume_trend,
            mcap_trend,
            sentiment_trends
        )
        
        # Generate prediction
        prediction = {
            'symbol': symbol,
            'prediction': 'BUY' if confidence > 0.6 else 'HOLD' if confidence > 0.4 else 'SELL',
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'price_change': price_trend.get('price_change', 0),
                'volume_change': volume_trend.get('volume_change', 0),
                'market_cap_change': mcap_trend.get('market_cap_change', 0),
                'sentiment_score': np.mean([t.get('mean_sentiment', 0) for t in sentiment_trends.values()])
            }
        }
        
        return prediction, confidence
        
    except Exception as e:
        logging.error(f"Error predicting {symbol}: {e}")
        return None, 0.0

def calculate_confidence(
    price_trend: Dict,
    volume_trend: Dict,
    mcap_trend: Dict,
    sentiment_trends: Dict
) -> float:
    """Calculate confidence score for prediction."""
    try:
        # Price trend weight
        price_weight = 0.4
        price_score = min(1.0, max(0.0, price_trend.get('price_change', 0) + 0.5))
        
        # Volume trend weight
        volume_weight = 0.2
        volume_score = min(1.0, max(0.0, volume_trend.get('volume_change', 0) + 0.5))
        
        # Market cap trend weight
        mcap_weight = 0.2
        mcap_score = min(1.0, max(0.0, mcap_trend.get('market_cap_change', 0) + 0.5))
        
        # Sentiment trend weight
        sentiment_weight = 0.2
        sentiment_scores = [t.get('mean_sentiment', 0) for t in sentiment_trends.values()]
        sentiment_score = min(1.0, max(0.0, np.mean(sentiment_scores) + 0.5))
        
        # Calculate weighted average
        confidence = (
            price_weight * price_score +
            volume_weight * volume_score +
            mcap_weight * mcap_score +
            sentiment_weight * sentiment_score
        )
        
        return confidence
        
    except Exception as e:
        logging.error(f"Error calculating confidence: {e}")
        return 0.0

if __name__ == "__main__":
    predictions, confidence_scores = generate_predictions()
    for pred, conf in zip(predictions, confidence_scores):
        print(f"Recommendation: {pred}, Confidence Score: {conf:.2f}")