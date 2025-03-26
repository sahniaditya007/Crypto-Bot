

import numpy as np
import pandas as pd
import time
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Binance API Credentials (Replace with your own keys)
API_KEY = "SqQl1AFmjzYiIKvnt1QQoSb8MUn5ussbH1KgkRZiAuHbxCR0SUfeM4SkOHOywU9G"
API_SECRET = "kIEa5qbDAf1LVHO84orgNPR5ADqx6pJwcPNgFdlvxKIv9r2E7Ibpuhc1M1DTqlSn"
client = Client(API_KEY, API_SECRET)

# Function to fetch historical OHLCV data
def get_historical_data(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, lookback='30 days ago UTC'):
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    return df

# Fetch data
data = get_historical_data()

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Function to create time-series sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 3])  # Predicting 'Close' price
    return np.array(X), np.array(y)

# Prepare data
TIME_STEPS = 60
X, y = create_sequences(data_scaled, TIME_STEPS)
X_train, X_test, y_train, y_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):], y[:int(0.8*len(y))], y[int(0.8*len(y)):] 

# Build LSTM Model
def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(TIME_STEPS, 5)),
        LSTM(50),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model()

# Train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return model

model = train_model(model, X_train, y_train, X_test, y_test)

# Function for real-time prediction
def predict_next_close():
    latest_data = get_historical_data().values[-TIME_STEPS:]
    latest_scaled = scaler.transform(latest_data)
    latest_scaled = np.expand_dims(latest_scaled, axis=0)
    prediction = model.predict(latest_scaled)
    predicted_price = scaler.inverse_transform([[0, 0, 0, prediction[0][0], 0]])[0][3]
    return predicted_price

# Function for semi-automated trade execution with stop-loss & take-profit
def execute_trade(symbol='BTCUSDT', trade_size=0.001, stop_loss_pct=0.02, take_profit_pct=0.05):
    predicted_price = predict_next_close()
    current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    stop_loss = current_price * (1 - stop_loss_pct)
    take_profit = current_price * (1 + take_profit_pct)
    
    print(f"Predicted Price: {predicted_price}, Current Price: {current_price}")
    print(f"Stop-Loss Price: {stop_loss}, Take-Profit Price: {take_profit}")
    
    decision = input("Enter 'BUY' or 'SELL' or 'SKIP': ").strip().upper()
    
    if decision == 'BUY':
        order = client.order_market_buy(symbol=symbol, quantity=trade_size)
        print("Buy Order Executed:", order)
    elif decision == 'SELL':
        order = client.order_market_sell(symbol=symbol, quantity=trade_size)
        print("Sell Order Executed:", order)
    else:
        print("No trade executed.")

# Function to monitor trade and enforce stop-loss/take-profit
def monitor_trade(symbol='BTCUSDT', trade_size=0.001, stop_loss_pct=0.02, take_profit_pct=0.05):
    stop_loss = float(client.get_symbol_ticker(symbol=symbol)['price']) * (1 - stop_loss_pct)
    take_profit = float(client.get_symbol_ticker(symbol=symbol)['price']) * (1 + take_profit_pct)
    while True:
        current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        if current_price <= stop_loss:
            print("Stop-Loss triggered! Selling position...")
            client.order_market_sell(symbol=symbol, quantity=trade_size)
            break
        elif current_price >= take_profit:
            print("Take-Profit reached! Selling position...")
            client.order_market_sell(symbol=symbol, quantity=trade_size)
            break
        time.sleep(5)

# Running real-time prediction & trading loop
while True:
    execute_trade()
    time.sleep(60)  # Wait 1 minute before next execution
