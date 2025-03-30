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
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

class ModelTrainer:
    def __init__(self):
        self.model_dir = "models"
        self.data_dir = "data"
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
            with open(os.path.join(self.data_dir, "historical_data.json"), "r") as file:
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

    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model with cross-validation and hyperparameter tuning."""
        try:
            # Validate data size
            n_samples = len(X)
            if n_samples < 2:
                raise ValueError(f"Not enough samples for training. Got {n_samples} samples, need at least 2.")
            
            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            logging.info(f"Class distribution: {dict(zip(unique, counts))}")
            
            # If any class has less than 2 samples, use simple random split
            if np.any(counts < 2):
                logging.warning("Using simple random split due to insufficient class samples")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                # Use stratified split if we have enough samples in each class
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            
            # Adjust cross-validation based on data size
            if n_samples < 5:
                logging.warning("Using Leave-One-Out cross-validation due to small dataset")
                cv = n_samples
            else:
                cv = min(5, n_samples // 2)  # Use at most 5 folds, but ensure enough samples per fold
            
            # Define parameter grid for Random Forest
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            # Initialize Random Forest model
            rf_model = RandomForestClassifier(random_state=42)
            
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator=rf_model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                scoring='accuracy',
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            
            # Log best parameters
            logging.info(f"Best parameters: {grid_search.best_params_}")
            logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Test set accuracy: {accuracy:.4f}")
            
            # Calculate and log feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logging.info("\nFeature Importance:")
            for _, row in feature_importance.head(10).iterrows():
                logging.info(f"{row['feature']}: {row['importance']:.4f}")
            
            # Save feature importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
            plt.title('Top 10 Most Important Features')
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'feature_importance.png'))
            plt.close()
            
        except Exception as e:
            logging.error(f"Error in model training: {e}")
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
            # Use YYYYMMDD format for timestamp
            timestamp = datetime.now().strftime("%Y%m%d")
            
            # Save model
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.pkl")
            with open(model_path, "wb") as file:
                pickle.dump(self.model, file)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"scaler_{timestamp}.pkl")
            with open(scaler_path, "wb") as file:
                pickle.dump(self.scaler, file)
            
            # Save feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importance_path = os.path.join(self.model_dir, f"feature_importance_{timestamp}.json")
                feature_importance_dict = {
                    'feature': list(self.X.columns),
                    'importance': list(self.model.feature_importances_)
                }
                with open(importance_path, 'w') as file:
                    json.dump(feature_importance_dict, file)
            
            # Save best parameters
            params_path = os.path.join(self.model_dir, f"best_params_{timestamp}.json")
            with open(params_path, "w") as file:
                json.dump(self.model.get_params(), file)
            
            logging.info(f"Saved model and artifacts with timestamp: {timestamp}")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def train_and_save(self):
        """Complete training pipeline."""
        try:
            # Load data
            X, y = self.load_training_data()
            
            # Store feature names
            self.X = X
            
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