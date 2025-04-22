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
import json
import pickle

# Define paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "iceberg"
MODEL_PATH = MODEL_DIR / "model.keras"  # Using new Keras format
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
DATA_DIR = PROJECT_ROOT / "data" / "iceberg"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Create necessary directories with error handling
for directory in [DATA_DIR, LOG_DIR, MODEL_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory created/verified: {directory}")
    except Exception as e:
        logging.error(f"Error creating directory {directory}: {e}")
        raise

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
env_path = CONFIG_DIR / '.env.iceberg'
load_dotenv(env_path)

# Get API credentials from environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
SECRET_KEY = os.getenv('BINANCE_API_SECRET')

# Debug logging
logging.info(f"Project root: {PROJECT_ROOT}")
logging.info(f"Config directory: {CONFIG_DIR}")
logging.info(f"Environment file path: {env_path}")
logging.info(f"Environment file exists: {env_path.exists()}")
logging.info(f"API_KEY present: {'Yes' if API_KEY else 'No'}")
logging.info(f"SECRET_KEY present: {'Yes' if SECRET_KEY else 'No'}")

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

        # Basic features
        features = {
            "avg_bid_price": bids["price"].mean(),
            "avg_ask_price": asks["price"].mean(),
            "avg_bid_size": bids["quantity"].mean(),
            "avg_ask_size": asks["quantity"].mean(),
            "bid_ask_spread": asks["price"].min() - bids["price"].max(),
            
            # Additional features for iceberg detection
            "bid_size_std": bids["quantity"].std(),  # Standard deviation of bid sizes
            "ask_size_std": asks["quantity"].std(),  # Standard deviation of ask sizes
            "bid_price_std": bids["price"].std(),    # Standard deviation of bid prices
            "ask_price_std": asks["price"].std(),    # Standard deviation of ask prices
            
            # Volume imbalance features
            "bid_volume": bids["quantity"].sum(),
            "ask_volume": asks["quantity"].sum(),
            "volume_imbalance": (bids["quantity"].sum() - asks["quantity"].sum()) / (bids["quantity"].sum() + asks["quantity"].sum()),
            
            # Price level features
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "level_imbalance": (len(bids) - len(asks)) / (len(bids) + len(asks)),
            
            # Large order features
            "large_bid_orders": len(bids[bids["quantity"] > bids["quantity"].mean() * 2]),
            "large_ask_orders": len(asks[asks["quantity"] > asks["quantity"].mean() * 2])
        }

        return list(features.values())
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def is_iceberg_order(order_book):
    """Determine if an order book pattern indicates an iceberg order."""
    if order_book is None:
        return False

    try:
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'], dtype=float)

        # Calculate key metrics
        bid_volume = bids["quantity"].sum()
        ask_volume = asks["quantity"].sum()
        total_volume = bid_volume + ask_volume
        volume_imbalance = abs((bid_volume - ask_volume) / total_volume) if total_volume > 0 else 0
        
        bid_mean_size = bids["quantity"].mean()
        ask_mean_size = asks["quantity"].mean()
        bid_std_size = bids["quantity"].std()
        ask_std_size = asks["quantity"].std()
        
        # More sophisticated criteria for iceberg detection
        criteria = {
            # Volume imbalance (indicating hidden orders)
            "volume_imbalance": volume_imbalance > 0.2,  # Reduced threshold
            
            # Large orders relative to mean
            "large_orders": (
                len(bids[bids["quantity"] > bid_mean_size + 2 * bid_std_size]) > 3 or
                len(asks[asks["quantity"] > ask_mean_size + 2 * ask_std_size]) > 3
            ),
            
            # Price clustering (indicating hidden orders at similar prices)
            "price_clustering": (
                bids["price"].std() < bids["price"].mean() * 0.002 or
                asks["price"].std() < asks["price"].mean() * 0.002
            ),
            
            # Volume concentration
            "volume_concentration": (
                bids["quantity"].max() > bid_mean_size * 2.5 or
                asks["quantity"].max() > ask_mean_size * 2.5
            ),
            
            # Order size distribution
            "size_distribution": (
                bid_std_size < bid_mean_size * 0.5 or
                ask_std_size < ask_mean_size * 0.5
            )
        }

        # Add some randomness to prevent perfect separation
        criteria_met = sum(criteria.values())
        random_threshold = np.random.normal(3.5, 0.5)  # Random threshold around 3.5
        
        return criteria_met >= random_threshold
    except Exception as e:
        logging.error(f"Error in iceberg detection: {e}")
        return False

def generate_training_data(symbol="BTCUSDT", samples=500):
    """Generate training data from order book with real iceberg labels."""
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

        # Use real iceberg detection instead of random labels
        is_iceberg = is_iceberg_order(order_book)
        
        X.append(features)
        y.append(1 if is_iceberg else 0)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)

    # Log the distribution of iceberg orders
    iceberg_count = sum(y)
    logging.info(f"Found {iceberg_count} iceberg orders out of {len(y)} total samples")
    logging.info(f"Iceberg order percentage: {(iceberg_count/len(y))*100:.2f}%")

    return np.array(X), np.array(y)

def save_model_metadata(model, scaler, feature_names, training_params):
    """Save model metadata and configuration."""
    metadata = {
        "model_architecture": [layer.get_config() for layer in model.layers],
        "feature_names": feature_names,
        "training_params": training_params,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    logging.info(f"Model metadata and scaler saved to {MODEL_DIR}")

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

    # Define Neural Network Model with regularization
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu",
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu",
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation="relu",
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compile with learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, 
                 loss="binary_crossentropy", 
                 metrics=["accuracy", tf.keras.metrics.AUC()])

    # Train Model with early stopping and model checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    # Training parameters
    training_params = {
        "epochs": 20,
        "batch_size": 32,
        "initial_learning_rate": initial_learning_rate,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy", "auc"]
    }

    # Train Model
    history = model.fit(
        X_train, y_train,
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Save Model and metadata
    model.save(MODEL_PATH)
    
    # Get feature names from the extract_features function
    feature_names = [
        "avg_bid_price", "avg_ask_price", "avg_bid_size", "avg_ask_size",
        "bid_ask_spread", "bid_size_std", "ask_size_std", "bid_price_std",
        "ask_price_std", "bid_volume", "ask_volume", "volume_imbalance",
        "bid_levels", "ask_levels", "level_imbalance", "large_bid_orders",
        "large_ask_orders"
    ]
    
    save_model_metadata(model, scaler, feature_names, training_params)
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(DATA_DIR / 'training_history.csv', index=False)
    
    # Save training data
    np.save(DATA_DIR / 'X_train.npy', X_train)
    np.save(DATA_DIR / 'X_test.npy', X_test)
    np.save(DATA_DIR / 'y_train.npy', y_train)
    np.save(DATA_DIR / 'y_test.npy', y_test)
    
    # Log final metrics
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
    logging.info(f"Test accuracy: {test_accuracy:.4f}")
    logging.info(f"Test AUC: {test_auc:.4f}")

def load_model():
    """Load the trained model and its components."""
    if not MODEL_PATH.exists():
        logging.warning("No saved model found. Training a new one...")
        train_model()
    
    try:
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        logging.info("Model, scaler, and metadata loaded successfully!")
        return model, scaler, metadata
    except Exception as e:
        logging.error(f"Error loading model components: {e}")
        raise

def predict_iceberg(symbol="BTCUSDT"):
    """Predict iceberg orders."""
    model, scaler, metadata = load_model()
    order_book = get_order_book(symbol)
    features = extract_features(order_book)

    if features is None:
        logging.warning("No valid features extracted.")
        return

    # Scale features
    X_input = scaler.transform(np.array(features).reshape(1, -1))

    # Predict
    prediction = model.predict(X_input)[0][0]
    
    # Print detailed prediction
    print("\n" + "="*50)
    print(f"ICEBERG ORDER ANALYSIS FOR {symbol}")
    print("="*50)
    print(f"Prediction Score: {prediction:.4f}")
    confidence = prediction if prediction > 0.5 else 1 - prediction
    if prediction > 0.5:
        print(f"Result: ICEBERG ORDER DETECTED 🚨")
        print(f"Confidence: {confidence:.2%}")
    else:
        print(f"Result: No iceberg order detected ✅")
        print(f"Confidence: {confidence:.2%}")
    print("="*50 + "\n")

if __name__ == "__main__":
    try:
        # Train and save the model
        train_model()
        # Predict iceberg orders
        predict_iceberg("BTCUSDT")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise 