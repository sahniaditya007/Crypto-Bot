import pandas as pd
from binance.client import Client
import ta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

config = {
    'symbol': 'BTCUSDT',
    'interval': Client.KLINE_INTERVAL_1HOUR,
    'limit': 100,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
}

def get_historical_data(symbol, interval, limit):
    try:
        klines = client.get_historical_klines(
            symbol,
            interval,
            limit=limit
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def generate_signals(df, current_price):
    """Generate trading signals based on technical indicators."""
    signals = {
        'should_buy': False,
        'should_sell': False,
        'reason': ''
    }
    
    # Get the latest indicators
    latest_rsi = df['rsi'].iloc[-1]
    latest_sma_20 = df['sma_20'].iloc[-1]
    latest_sma_50 = df['sma_50'].iloc[-1]
    
    # RSI strategy
    if latest_rsi < config['rsi_oversold']:
        signals['should_buy'] = True
        signals['reason'] = f"RSI oversold ({latest_rsi:.2f})"
    elif latest_rsi > config['rsi_overbought']:
        signals['should_sell'] = True
        signals['reason'] = f"RSI overbought ({latest_rsi:.2f})"
    
    # Moving average crossover strategy
    if latest_sma_20 > latest_sma_50:
        signals['should_buy'] = True
        signals['reason'] += " | Golden Cross"
    elif latest_sma_20 < latest_sma_50:
        signals['should_sell'] = True
        signals['reason'] += " | Death Cross"
    
    return signals

def execute_trades(client, signals, current_price):
    """Execute trades based on signals."""
    if not api_key or not api_secret:
        print("API credentials not found. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        return
        
    if signals['should_buy']:
        print(f"Buy signal detected! Reason: {signals['reason']}")
        print(f"Current price: {current_price}")
        # Implement your buy logic here
        
    elif signals['should_sell']:
        print(f"Sell signal detected! Reason: {signals['reason']}")
        print(f"Current price: {current_price}")
        # Implement your sell logic here
        
    else:
        print("No trading signals detected.")

def main():
    try:
        # Initialize Binance client
        client = Client(api_key, api_secret)
        
        # Get historical data
        symbol = config['symbol']
        interval = config['interval']
        limit = config['limit']
        
        print(f"Fetching historical data for {symbol}...")
        df = get_historical_data(symbol, interval, limit)
        
        if df is None:
            print("Failed to fetch historical data. Exiting...")
            return
            
        print(f"Successfully fetched {len(df)} historical data points")
        
        # Calculate indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # Get current price
        current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        
        # Generate signals
        signals = generate_signals(df, current_price)
        
        # Execute trades based on signals
        execute_trades(client, signals, current_price)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 