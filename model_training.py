import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='crypto_bot.log'
)

# Load environment variables
load_dotenv()

class ModelTrainer:
    def __init__(self):
        self.model_dir = "models"
        self._setup_model_dir()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_params = None
        self.feature_importance = None

    def _setup_model_dir(self):
        """Create model directory if it doesn't exist."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate training data."""
        try:
            with open(os.getenv("HISTORICAL_DATA_FILE"), "r") as file:
                historical_data = json.load(file)
            
            df = pd.DataFrame(historical_data)
            
            # Validate data
            if df.empty:
                raise ValueError("Empty dataset")
            
            if "target" not in df.columns:
                raise ValueError("Missing target column")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logging.warning(f"Missing values found: {missing_values[missing_values > 0]}")
                df = df.dropna()
            
            # Separate features and target
            X = df.drop(columns=["target"])
            y = df["target"]
            
            logging.info(f"Loaded {len(df)} samples with {len(X.columns)} features")
            return X, y
            
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            raise

    def preprocess_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            return pd.DataFrame(X_scaled, columns=X.columns)
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """Train model with hyperparameter tuning."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Define parameter grid for GridSearchCV
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Initialize base model
            base_model = RandomForestClassifier(random_state=42)
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='f1',
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Evaluate model
            y_pred = self.best_model.predict(X_test)
            metrics = self.evaluate_model(y_test, y_pred)
            
            logging.info(f"Best parameters: {self.best_params}")
            logging.info(f"Model metrics: {metrics}")
            logging.info(f"Top 5 important features: {self.feature_importance.head()}")
            
            return self.best_model
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise

    def evaluate_model(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }

    def save_model(self):
        """Save trained model and related artifacts."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.pkl")
            with open(model_path, "wb") as file:
                pickle.dump(self.best_model, file)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"scaler_{timestamp}.pkl")
            with open(scaler_path, "wb") as file:
                pickle.dump(self.scaler, file)
            
            # Save feature importance
            importance_path = os.path.join(self.model_dir, f"feature_importance_{timestamp}.json")
            self.feature_importance.to_json(importance_path)
            
            # Save best parameters
            params_path = os.path.join(self.model_dir, f"best_params_{timestamp}.json")
            with open(params_path, "w") as file:
                json.dump(self.best_params, file)
            
            logging.info(f"Saved model and artifacts with timestamp: {timestamp}")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def train_and_save(self):
        """Complete training pipeline."""
        try:
            # Load data
            X, y = self.load_training_data()
            
            # Preprocess data
            X_scaled = self.preprocess_data(X)
            
            # Train model
            self.train_model(X_scaled, y)
            
            # Save model and artifacts
            self.save_model()
            
            logging.info("Model training completed successfully")
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise

def train_model():
    """Wrapper function for model training."""
    trainer = ModelTrainer()
    trainer.train_and_save()

if __name__ == "__main__":
    train_model()