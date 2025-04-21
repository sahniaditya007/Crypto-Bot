from crypto_predictor import CryptoPredictor
from trading_bot import TradingBot

def train_new_model():
    """Example of training a new model"""
    predictor = CryptoPredictor()  # Will automatically use API keys from config
    predictor.train_model(epochs=50, batch_size=32)

def run_trading_bot():
    """Example of running the trading bot"""
    bot = TradingBot(symbol='BTCUSDT')
    bot.execute_trade()

if __name__ == "__main__":
    # Uncomment the following line to train a new model
    train_new_model()
    
    # Run the trading bot
    run_trading_bot() 