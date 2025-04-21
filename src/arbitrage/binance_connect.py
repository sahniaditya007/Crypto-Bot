import os
import ccxt
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Connect to Binance
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'options': {'adjustForTimeDifference': True}
})

# Fetch live BTC/USDT price
ticker = exchange.fetch_ticker('BTC/USDT')
print(f"BTC/USDT Price: {ticker['last']}")
