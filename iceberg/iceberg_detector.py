import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import hashlib
import hmac
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "iceberg" / "iceberg_detector.h5"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"

# Create necessary directories
for directory in [DATA_DIR, LOG_DIR, MODEL_PATH.parent]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'iceberg_detector_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv(PROJECT_ROOT / '.env.iceberg')

# Get API credentials from environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_API_SECRET')

if not API_KEY or not SECRET_KEY:
    raise ValueError("API credentials not found in .env.iceberg file. Please check your configuration.")

# Binance API Base URL
BASE_URL = "https://api.binance.us"

# Configure TensorFlow for local machine
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

def generate_signature(params):
    """Generate Binance API signature."""
    query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params)])
    return hmac.new(SECRET_KEY.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def get_order_book(symbol, depth=100):
    """Fetch order book data from Binance."""
    url = f"{BASE_URL}/api/v3/depth"
    params = {"symbol": symbol.upper(), "limit": depth}
    headers = {"X-MBX-APIKEY": API_KEY}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return data if 'bids' in data and 'asks' in data else None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching order book: {e}")
        return None

def extract_features(order_book):
    """Extract features from order book data."""
    if order_book is None:
        return None

    try:
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'], dtype=float)

        features = {
            "avg_bid_price": bids["price"].mean(),
            "avg_ask_price": asks["price"].mean(),
            "avg_bid_size": bids["quantity"].mean(),
            "avg_ask_size": asks["quantity"].mean(),
            "bid_ask_spread": asks["price"].min() - bids["price"].max(),
        }

        return list(features.values())
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def generate_training_data(symbol="BTCUSDT", samples=500):
    """Generate training data from order book."""
    X, y = [], []
    logging.info(f"Generating training data for {symbol} with {samples} samples")
    
    for i in range(samples):
        if i % 100 == 0:
            logging.info(f"Progress: {i}/{samples} samples collected")
            
        order_book = get_order_book(symbol)
        if order_book is None:
            continue

        features = extract_features(order_book)
        if features is None:
            continue

        X.append(features)
        y.append(np.random.randint(0, 2))  # Dummy labels for now
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)

    return np.array(X), np.array(y)

def train_model():
    """Train and save the model."""
    logging.info("Starting model training...")
    X, y = generate_training_data()
    
    if len(X) == 0:
        logging.error("No training data collected. Check API response.")
        return

    # Preprocess Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define Neural Network Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train Model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train Model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Save Model
    model.save(MODEL_PATH)
    logging.info(f"Model training complete & saved to: {MODEL_PATH}")
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(DATA_DIR / 'training_history.csv', index=False)

def load_model():
    """Load the trained model."""
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded successfully!")
        return model
    else:
        logging.warning("No saved model found. Training a new one...")
        train_model()
        return tf.keras.models.load_model(MODEL_PATH)

def predict_iceberg(symbol="BTCUSDT"):
    """Predict iceberg orders."""
    model = load_model()
    order_book = get_order_book(symbol)
    features = extract_features(order_book)

    if features is None:
        logging.warning("No valid features extracted.")
        return

    # Reshape input to match model's expected format
    X_input = np.array(features).reshape(1, -1)

    # Predict
    prediction = model.predict(X_input)[0][0]
    if prediction > 0.5:
        logging.info(f"Symbol: {symbol} | Prediction: ICEBERG ORDER DETECTED 🚨")
    else:
        logging.info(f"Symbol: {symbol} | Prediction: No iceberg order detected ✅")

if __name__ == "__main__":
    try:
        # Train and save the model
        train_model()
        # Predict iceberg orders
        predict_iceberg("BTCUSDT")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise 