import os
from crypto_predictor import CryptoPredictor
import time
from pathlib import Path

class TradingBot:
    def __init__(self, symbol='BTCUSDT'):
        self.symbol = symbol
        self.predictor = CryptoPredictor()  # Will automatically use API keys from config
        
    def calculate_trade_signals(self, current_price, predicted_price):
        """Calculate trading signals based on predicted price"""
        price_change_percent = ((predicted_price - current_price) / current_price) * 100
        
        # Define thresholds
        BUY_THRESHOLD = 0.5  # Buy if predicted price is 0.5% higher
        SELL_THRESHOLD = -0.5  # Sell if predicted price is 0.5% lower
        
        if price_change_percent > BUY_THRESHOLD:
            return 'BUY'
        elif price_change_percent < SELL_THRESHOLD:
            return 'SELL'
        return 'HOLD'
    
    def calculate_stop_loss_take_profit(self, entry_price, side='BUY'):
        """Calculate stop-loss and take-profit levels"""
        if side == 'BUY':
            stop_loss = entry_price * 0.98  # 2% below entry
            take_profit = entry_price * 1.05  # 5% above entry
        else:  # SELL
            stop_loss = entry_price * 1.02  # 2% above entry
            take_profit = entry_price * 0.95  # 5% below entry
        return stop_loss, take_profit
    
    def execute_trade(self):
        """Execute trading strategy"""
        try:
            # Get current market data
            current_price = float(self.predictor.client.get_symbol_ticker(symbol=self.symbol)['price'])
            
            # Get price prediction
            predicted_price = self.predictor.predict_next_price(self.symbol)
            
            if predicted_price is None:
                print("Could not get price prediction")
                return
            
            # Get trading signal
            signal = self.calculate_trade_signals(current_price, predicted_price)
            
            # Calculate stop-loss and take-profit levels
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(current_price, signal)
            
            print(f"Predicted Price: {predicted_price}, Current Price: {current_price}")
            print(f"Stop-Loss Price: {stop_loss}, Take-Profit Price: {take_profit}")
            
            # In simulation mode, ask for manual confirmation
            action = input("\nEnter 'BUY' or 'SELL' or 'SKIP': ").upper()
            
            if action in ['BUY', 'SELL']:
                print(f"Simulated {action} order executed at {current_price}")
                print(f"Stop-Loss set at: {stop_loss}")
                print(f"Take-Profit set at: {take_profit}")
            
        except Exception as e:
            print(f"Error executing trade: {e}")

def main():
    # Initialize trading bot
    bot = TradingBot()
    
    try:
        while True:
            bot.execute_trade()
            time.sleep(60)  # Wait 1 minute before next execution
    except KeyboardInterrupt:
        print("\nTrading bot stopped.")

if __name__ == "__main__":
    main() 