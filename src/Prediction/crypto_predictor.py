import numpy as np
import pandas as pd
import os
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import time
from pathlib import Path

@tf.keras.saving.register_keras_serializable()
class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)
        
    def call(self, inputs):
        query = inputs[0]
        value = inputs[1]
        attention_scores = tf.matmul(query, value, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

class CryptoPredictor:
    def __init__(self, api_key=None, api_secret=None, model_path=None, time_steps=60):
        # Use default API keys from config if not provided
        if api_key is None or api_secret is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / '.env.iceberg'
            if not config_path.exists():
                raise ValueError("API credentials not found in config/.env.iceberg")
            
            with open(config_path) as f:
                for line in f:
                    if line.startswith('BINANCE_API_KEY='):
                        api_key = line.split('=')[1].strip()
                    elif line.startswith('BINANCE_API_SECRET='):
                        api_secret = line.split('=')[1].strip()
        
        if not api_key or not api_secret:
            raise ValueError("API credentials not found")
            
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = Client(api_key, api_secret, tld='com', testnet=True)
        self.time_steps = time_steps
        self.scaler = MinMaxScaler()
        self.model = None
        self.model_path = model_path or str(Path(__file__).parent.parent.parent / 'models' / 'lstm_trading_model.keras')
        
    def get_historical_data(self, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, lookback='365 days ago UTC'):
        try:
            klines = self.client.get_historical_klines(symbol, interval, lookback)
            df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                           'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                           'Taker buy quote asset volume', 'Ignore'])
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def create_sequences(self, data, time_steps=None):
        if time_steps is None:
            time_steps = self.time_steps
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
            y.append(data[i + time_steps, 3])  # Predicting 'Close' price
        return np.array(X), np.array(y)

    def build_model(self):
        inputs = Input(shape=(self.time_steps, 5))
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        attention = CustomAttention()([x, x])
        x = Bidirectional(LSTM(64, return_sequences=False))(attention)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_model(self, data=None, epochs=50, batch_size=32):
        if data is None:
            data = self.get_historical_data()
            
        if data is None:
            raise ValueError("No data available for training")
            
        data_scaled = self.scaler.fit_transform(data)
        X, y = self.create_sequences(data_scaled)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train model
        self.model = self.build_model()
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        
        # Save model
        self.model.save(self.model_path, save_format='keras')
        print(f"Model trained and saved at: {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print("Model loaded successfully!")
            return True
        return False

    def predict_next_price(self, symbol='BTCUSDT'):
        try:
            if self.model is None:
                if not self.load_model():
                    raise ValueError("No model available. Please train or load a model first.")
                    
            latest_data = self.get_historical_data(symbol).values[-self.time_steps:]
            
            if latest_data.shape[0] != self.time_steps:
                raise ValueError(f"Expected shape ({self.time_steps}, 5), but got {latest_data.shape}")

            latest_scaled = self.scaler.transform(latest_data)
            latest_scaled = np.expand_dims(latest_scaled, axis=0)
            
            prediction = self.model.predict(latest_scaled)
            
            # Inverse transform the prediction
            predicted_prices = self.scaler.inverse_transform(
                np.hstack((np.zeros((1, latest_data.shape[1] - 1)), prediction.reshape(-1, 1)))
            )[:, -1]

            return predicted_prices[0]
        except Exception as e:
            print(f"Error predicting next price: {e}")
            return None 