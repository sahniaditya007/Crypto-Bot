import ccxt

# Initialize Binance
binance = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True
})

# Function to fetch order book
def get_order_book(pair):
    order_book = binance.fetch_order_book(pair)
    best_bid = order_book['bids'][0][0] if order_book['bids'] else None
    best_ask = order_book['asks'][0][0] if order_book['asks'] else None
    return best_bid, best_ask

# Example pairs
pairs = ['BTC/USDT', 'ETH/USDT', 'ETH/BTC']

for pair in pairs:
    bid, ask = get_order_book(pair)
    print(f"{pair} - Bid: {bid}, Ask: {ask}")
