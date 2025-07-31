import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
from prediction import PricePredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    size: float
    horizon: int
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = 'OPEN'  # 'OPEN', 'CLOSED', 'STOPPED'

@dataclass
class TradingSignal:
    """Represents a trading signal."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    horizon: int
    confidence: float
    probability: float
    timestamp: datetime
    strength: str  # 'WEAK', 'MEDIUM', 'STRONG'

class RiskManager:
    """Handles risk management calculations."""
    
    def __init__(self, max_risk_per_trade=0.02, max_total_risk=0.1):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_total_risk = max_total_risk  # 10% total exposure
        
    def calculate_position_size(self, account_balance, confidence, risk_multiplier=1.0):
        """Calculate position size based on confidence and risk parameters."""
        base_risk = self.max_risk_per_trade * risk_multiplier
        confidence_adjusted_risk = base_risk * confidence
        return min(confidence_adjusted_risk, self.max_risk_per_trade)
    
    def should_open_position(self, current_positions, max_positions=5):
        """Determine if new position should be opened."""
        open_positions = [p for p in current_positions if p.status == 'OPEN']
        return len(open_positions) < max_positions

class TradingBot:
    """Main trading bot class."""
    
    def __init__(self, model_dir='./', config_file='trading_config.json'):
        self.predictor = PricePredictor(model_dir)
        self.risk_manager = RiskManager()
        self.positions: List[Position] = []
        self.account_balance = 10000.0  # Starting balance
        self.config = self.load_config(config_file)
        
        # Trading parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.65)
        self.min_horizon = self.config.get('min_horizon', 3)
        self.max_positions = self.config.get('max_positions', 5)
        self.check_interval = self.config.get('check_interval', 60)  # seconds
        
    def load_config(self, config_file):
        """Load trading configuration."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {config_file} not found, using defaults")
            return {}
    
    def initialize(self):
        """Initialize the trading bot."""
        logging.info("Initializing trading bot...")
        
        # Load models
        horizons = self.config.get('horizons', [1, 3, 5, 10])
        self.predictor.load_models(horizons)
        
        logging.info(f"Bot initialized with {len(self.predictor.models)} models")
        logging.info(f"Configuration: confidence_threshold={self.confidence_threshold}, "
                    f"min_horizon={self.min_horizon}, max_positions={self.max_positions}")
    
    def get_market_data(self, symbol='EURUSD'):
        """
        Get current market data. 
        In real implementation, this would connect to your broker's API.
        For demo, we'll use the existing data file.
        """
        try:
            # This is a placeholder - replace with real market data API
            data_path = self.config.get('data_path', './eurusd.json')
            predictions = self.predictor.predict_from_file(data_path)
            return predictions
        except Exception as e:
            logging.error(f"Failed to get market data: {e}")
            return None
    
    def generate_signals(self, predictions) -> List[TradingSignal]:
        """Generate trading signals from predictions."""
        signals = []
        
        for horizon, pred in predictions.items():
            if horizon >= self.min_horizon and pred['confidence'] >= self.confidence_threshold:
                
                # Determine signal strength
                if pred['confidence'] >= 0.8:
                    strength = 'STRONG'
                elif pred['confidence'] >= 0.7:
                    strength = 'MEDIUM'
                else:
                    strength = 'WEAK'
                
                signal = TradingSignal(
                    symbol='EURUSD',
                    action='BUY' if pred['direction'] == 'UP' else 'SELL',
                    horizon=horizon,
                    confidence=pred['confidence'],
                    probability=pred['probability'],
                    timestamp=datetime.now(),
                    strength=strength
                )
                
                signals.append(signal)
                
        return signals
    
    def execute_signal(self, signal: TradingSignal, current_price: float = 1.1500):
        """
        Execute a trading signal.
        In real implementation, this would place orders through broker API.
        """
        try:
            if not self.risk_manager.should_open_position(self.positions, self.max_positions):
                logging.info(f"Max positions reached, skipping signal for {signal.horizon}min")
                return None
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                self.account_balance, 
                signal.confidence
            )
            
            # Calculate stop loss and take profit
            pip_value = 0.0001
            if signal.action == 'BUY':
                stop_loss = current_price - (50 * pip_value)  # 50 pip stop
                take_profit = current_price + (100 * pip_value)  # 100 pip target
            else:
                stop_loss = current_price + (50 * pip_value)
                take_profit = current_price - (100 * pip_value)
            
            position = Position(
                symbol=signal.symbol,
                direction=signal.action,
                entry_price=current_price,
                entry_time=signal.timestamp,
                size=position_size,
                horizon=signal.horizon,
                confidence=signal.confidence,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.positions.append(position)
            
            logging.info(f"EXECUTED: {signal.action} {signal.symbol} | "
                        f"Horizon: {signal.horizon}min | "
                        f"Confidence: {signal.confidence:.2%} | "
                        f"Size: {position_size:.4f} | "
                        f"Price: {current_price:.5f}")
            
            return position
            
        except Exception as e:
            logging.error(f"Failed to execute signal: {e}")
            return None
    
    def manage_positions(self, current_price: float = 1.1500):
        """Manage existing positions - check for stop losses, take profits, or time-based exits."""
        for position in self.positions:
            if position.status != 'OPEN':
                continue
                
            # Check if position should be closed based on time
            time_elapsed = datetime.now() - position.entry_time
            if time_elapsed.total_seconds() / 60 >= position.horizon:
                self.close_position(position, current_price, 'TIME_EXIT')
                continue
            
            # Check stop loss and take profit
            if position.direction == 'BUY':
                if current_price <= position.stop_loss:
                    self.close_position(position, current_price, 'STOP_LOSS')
                elif current_price >= position.take_profit:
                    self.close_position(position, current_price, 'TAKE_PROFIT')
            else:  # SELL
                if current_price >= position.stop_loss:
                    self.close_position(position, current_price, 'STOP_LOSS')
                elif current_price <= position.take_profit:
                    self.close_position(position, current_price, 'TAKE_PROFIT')
    
    def close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position and calculate P&L."""
        position.status = 'CLOSED'
        
        # Calculate P&L
        if position.direction == 'BUY':
            pnl = (exit_price - position.entry_price) * position.size * 100000  # Assuming standard lot
        else:
            pnl = (position.entry_price - exit_price) * position.size * 100000
        
        self.account_balance += pnl
        
        logging.info(f"CLOSED: {position.direction} {position.symbol} | "
                    f"Reason: {reason} | "
                    f"Entry: {position.entry_price:.5f} | "
                    f"Exit: {exit_price:.5f} | "
                    f"P&L: ${pnl:.2f} | "
                    f"Balance: ${self.account_balance:.2f}")
    
    def get_portfolio_summary(self):
        """Get current portfolio summary."""
        open_positions = [p for p in self.positions if p.status == 'OPEN']
        closed_positions = [p for p in self.positions if p.status == 'CLOSED']
        
        total_pnl = sum((p.entry_price - 1.15) * p.size * 100000 for p in closed_positions)  # Simplified P&L
        
        return {
            'account_balance': self.account_balance,
            'open_positions': len(open_positions),
            'total_positions': len(self.positions),
            'total_pnl': total_pnl
        }
    
    def run_strategy(self, max_iterations=None):
        """Main trading loop."""
        logging.info("Starting trading strategy...")
        iteration = 0
        
        try:
            while True:
                iteration += 1
                if max_iterations and iteration > max_iterations:
                    break
                    
                logging.info(f"=== Trading Cycle {iteration} ===")
                
                # Get market predictions
                predictions = self.get_market_data()
                if not predictions:
                    logging.warning("No predictions available, waiting...")
                    time.sleep(self.check_interval)
                    continue
                
                # Generate trading signals
                signals = self.generate_signals(predictions)
                
                # Log predictions
                for horizon, pred in predictions.items():
                    direction_emoji = "📈" if pred['direction'] == 'UP' else "📉"
                    logging.info(f"Prediction {horizon}min: {direction_emoji} {pred['direction']} "
                               f"({pred['confidence']:.1%})")
                
                # Execute signals
                for signal in signals:
                    if signal.strength in ['MEDIUM', 'STRONG']:  # Only trade medium/strong signals
                        self.execute_signal(signal)
                
                # Manage existing positions
                self.manage_positions()
                
                # Log portfolio status
                summary = self.get_portfolio_summary()
                logging.info(f"Portfolio: Balance=${summary['account_balance']:.2f}, "
                           f"Open Positions={summary['open_positions']}")
                
                # Wait for next cycle
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logging.info("Trading bot stopped by user")
        except Exception as e:
            logging.error(f"Trading strategy error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup and save final state."""
        logging.info("Cleaning up trading bot...")
        
        # Close all open positions
        for position in self.positions:
            if position.status == 'OPEN':
                self.close_position(position, 1.1500, 'SHUTDOWN')  # Use current price
        
        # Save trading log
        trading_log = {
            'final_balance': self.account_balance,
            'total_positions': len(self.positions),
            'positions': [
                {
                    'symbol': p.symbol,
                    'direction': p.direction,
                    'entry_time': p.entry_time.isoformat(),
                    'horizon': p.horizon,
                    'confidence': p.confidence,
                    'status': p.status
                }
                for p in self.positions
            ]
        }
        
        with open('trading_results.json', 'w') as f:
            json.dump(trading_log, f, indent=2, default=str)
        
        logging.info(f"Trading session complete. Final balance: ${self.account_balance:.2f}")

def create_default_config():
    """Create default trading configuration file."""
    config = {
        "confidence_threshold": 0.65,
        "min_horizon": 3,
        "max_positions": 5,
        "check_interval": 60,
        "horizons": [1, 3, 5, 10],
        "data_path": "./eurusd.json"
    }
    
    with open('trading_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created default trading_config.json")

def main():
    parser = argparse.ArgumentParser(description='Automated Trading Bot using AI Price Predictions')
    parser.add_argument('--model_dir', default='./', help='Directory containing trained models')
    parser.add_argument('--config', default='trading_config.json', help='Trading configuration file')
    parser.add_argument('--create_config', action='store_true', help='Create default configuration file')
    parser.add_argument('--max_iterations', type=int, help='Maximum trading cycles (for testing)')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
        return
    
    try:
        # Initialize trading bot
        bot = TradingBot(args.model_dir, args.config)
        bot.initialize()
        
        # Run trading strategy
        bot.run_strategy(args.max_iterations)
        
    except Exception as e:
        logging.error(f"Trading bot failed: {e}")
        raise

if __name__ == "__main__":
    main() 