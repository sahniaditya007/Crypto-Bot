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

# Load environment variables
load_dotenv()

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

def generate_predictions() -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper function for prediction generation."""
    try:
        # Load market trends
        with open(os.path.join("data", "market_trends.json"), "r") as file:
            market_trends = json.load(file)

        # Initialize generator
        generator = PredictionGenerator()

        # Generate predictions
        results = generator.generate_predictions(market_trends)

        # Save predictions
        generator.save_predictions(results, os.path.join("data", "predictions.json"))

        # Extract predictions and confidence scores
        predictions = np.array([r.prediction for r in results])
        confidence_scores = np.array([r.confidence_score for r in results])

        # Convert numpy types to Python native types
        predictions = predictions.tolist()
        confidence_scores = confidence_scores.tolist()

        return predictions, confidence_scores

    except Exception as e:
        logging.error(f"Error in prediction generation: {e}")
        raise

if __name__ == "__main__":
    predictions, confidence_scores = generate_predictions()
    for pred, conf in zip(predictions, confidence_scores):
        print(f"Recommendation: {pred}, Confidence Score: {conf:.2f}")