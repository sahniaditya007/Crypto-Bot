import ccxt
import os
import websocket
import json
import threading
import boto3
from botocore.exceptions import ClientError

# Function to fetch secrets from AWS Secrets Manager
def get_secret():
    secret_name = "arn:aws:secretsmanager:us-east-1:799854597846:secret:prod/cryptopilot-MA71Q3"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    # Parse and return the secret as a dictionary
    return json.loads(get_secret_value_response['SecretString'])

# Load secrets
secrets = get_secret()

# Initialize Binance (for placing trades)
binance = ccxt.binance({
    'apiKey': secrets['BINANCE_API_KEY'],  # Replace with the key from your secret
    'secret': secrets['BINANCE_SECRET_KEY'],  # Replace with the key from your secret
})

# Initialize Kraken (for placing trades)
kraken = ccxt.kraken({
    'apiKey': secrets['KRAKEN_API_KEY'],  # Replace with the key from your secret
    'apiSecret': secrets['KRAKEN_API_SECRET'],  # Replace with the key from your secret
})

# Global variables to store latest prices
binance_bid, binance_ask = None, None
kraken_bid, kraken_ask = None, None

# Binance WebSocket URL
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth5"
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

# WebSocket Handlers
def binance_ws_message(ws, message):
    global binance_bid, binance_ask
    data = json.loads(message)
    binance_bid = float(data['bids'][0][0])
    binance_ask = float(data['asks'][0][0])

def kraken_ws_message(ws, message):
    global kraken_bid, kraken_ask
    data = json.loads(message)
    if isinstance(data, list) and len(data) > 1 and isinstance(data[1], dict):
        order_book = data[1]
        if 'b' in order_book and 'a' in order_book:
            kraken_bid = float(order_book['b'][0][0])
            kraken_ask = float(order_book['a'][0][0])

# Start WebSockets
def start_binance_ws():
    ws = websocket.WebSocketApp(BINANCE_WS_URL, on_message=binance_ws_message)
    ws.run_forever()

def start_kraken_ws():
    def on_open(ws):
        ws.send(json.dumps({"event": "subscribe", "pair": ["BTC/USDT"], "subscription": {"name": "book"}}))
    
    ws = websocket.WebSocketApp(KRAKEN_WS_URL, on_message=kraken_ws_message, on_open=on_open)
    ws.run_forever()

# Fetch latest prices before arbitrage check
def update_prices():
    global binance_bid, binance_ask, kraken_bid, kraken_ask
    try:
        binance_order_book = binance.fetch_order_book('BTC/USDT')
        binance_bid = binance_order_book['bids'][0][0] if binance_order_book['bids'] else None
        binance_ask = binance_order_book['asks'][0][0] if binance_order_book['asks'] else None

        kraken_order_book = kraken.fetch_order_book('BTC/USDT')
        kraken_bid = kraken_order_book['bids'][0][0] if kraken_order_book['bids'] else None
        kraken_ask = kraken_order_book['asks'][0][0] if kraken_order_book['asks'] else None
    except Exception as e:
        print(f"❌ Error fetching prices: {e}")

# Place limit orders
def place_limit_order(exchange, symbol, side, amount, price):
    try:
        order = exchange.create_limit_order(symbol, side, amount, price)
        print(f"✅ {side.upper()} limit order placed on {exchange.name} at {price}: {order}")
    except Exception as e:
        print(f"❌ Error placing {side} limit order on {exchange.name}: {e}")

# Check for arbitrage opportunity
def check_arbitrage():
    if binance_bid and binance_ask and kraken_bid and kraken_ask:
        if binance_ask < kraken_bid:
            profit = kraken_bid - binance_ask
            print(f"💰 Arbitrage Opportunity: Buy on Binance at {binance_ask} and sell on Kraken at {kraken_bid} (Profit: {profit:.2f} USDT)")
            return ('binance', 'buy', binance_ask, 'kraken', 'sell', kraken_bid)
        elif kraken_ask < binance_bid:
            profit = binance_bid - kraken_ask
            print(f"💰 Arbitrage Opportunity: Buy on Kraken at {kraken_ask} and sell on Binance at {binance_bid} (Profit: {profit:.2f} USDT)")
            return ('kraken', 'buy', kraken_ask, 'binance', 'sell', binance_bid)
    return None

# Main user-interaction loop
def main_loop():
    print("Ready? Press 's' to start.")

    while input().strip().lower() != 's':
        pass  # Wait for user to press 's'
    
    print("🔄 Arbitrage Scanner Started!")

    while True:
        update_prices()  # Fetch latest prices
        opportunity = check_arbitrage()
        
        if opportunity:
            ex_buy, side_buy, price_buy, ex_sell, side_sell, price_sell = opportunity
            key = input("Execute trade? (y/n): ").strip().lower()  # Get user input for execution
            
            if key == 'y':
                print("⚡ Executing trades...")
                place_limit_order(globals()[ex_buy], 'BTC/USDT', side_buy, 0.0001, price_buy)
                place_limit_order(globals()[ex_sell], 'BTC/USDT', side_sell, 0.0001, price_sell)

# Run WebSocket Threads and Main Loop
if __name__ == "__main__":
    binance_thread = threading.Thread(target=start_binance_ws, daemon=True)
    kraken_thread = threading.Thread(target=start_kraken_ws, daemon=True)
    binance_thread.start()
    kraken_thread.start()
    
    main_loop()
