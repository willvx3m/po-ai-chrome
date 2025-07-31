import json
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PricePredictor:
    def __init__(self, model_dir='./'):
        """Load trained models for prediction."""
        self.models = {}
        self.scaler = None
        self.model_dir = model_dir
        self.features = ['open', 'high', 'low', 'close', 'RSI_14', 'EMA_12', 'EMA_26', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'Stoch_K', 'Stoch_D', 'OBV']
        
    def load_models(self, horizons=[1, 3, 5, 10]):
        """Load all trained models and scaler."""
        try:
            # Load scaler
            scaler_path = f'{self.model_dir}/scaler_common.pkl'
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logging.info(f"Scaler loaded from {scaler_path}")
            
            # Load models for each horizon
            for horizon in horizons:
                model_path = f'{self.model_dir}/model_horizon_{horizon}min.h5'
                try:
                    self.models[horizon] = tf.keras.models.load_model(model_path)
                    logging.info(f"Loaded model for {horizon}min horizon from {model_path}")
                except Exception as e:
                    logging.warning(f"Could not load model for {horizon}min: {e}")
                    
        except Exception as e:
            logging.error(f"Failed to load models/scaler: {e}")
            raise
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe."""
        try:
            def calculate_rsi(data, window=14):
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                gain_avg = gain.rolling(window=window).mean()
                loss_avg = loss.rolling(window=window).mean()
                rs = gain_avg / (loss_avg + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                return rsi

            def calculate_macd(data, fast=12, slow=26, signal=9):
                ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
                ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
                macd = ema_fast - ema_slow
                return macd

            def calculate_bollinger_bands(data, window=20):
                sma = data['close'].rolling(window=window).mean()
                std = data['close'].rolling(window=window).std()
                upper = sma + 2 * std
                lower = sma - 2 * std
                return upper, lower

            def calculate_atr(data, window=14):
                high_low = data['high'] - data['low']
                high_close = np.abs(data['high'] - data['close'].shift())
                low_close = np.abs(data['low'] - data['close'].shift())
                tr = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = tr.rolling(window=window).mean()
                return atr

            def calculate_stochastic_kd(data, window=14, smooth=3):
                lowest_low = data['low'].rolling(window=window).min()
                highest_high = data['high'].rolling(window=window).max()
                stoch_k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low + 1e-8)
                stoch_d = stoch_k.rolling(window=smooth).mean()
                return stoch_k, stoch_d

            def calculate_obv(data):
                obv = (np.sign(data['close'].diff()) * data.get('volume', pd.Series(np.ones(len(data))))).cumsum()
                return obv

            df['RSI_14'] = calculate_rsi(df)
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = calculate_macd(df)
            df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df)
            df['ATR'] = calculate_atr(df)
            df['Stoch_K'], df['Stoch_D'] = calculate_stochastic_kd(df)
            df['OBV'] = calculate_obv(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to add technical indicators: {e}")
            raise
    
    def prepare_data(self, df, lookback=300):
        """Prepare recent data for prediction."""
        try:
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Handle NaN values (same as training)
            df[self.features] = df[self.features].ffill()
            
            # Drop any remaining NaN rows
            df = df.dropna(subset=self.features)
            
            if len(df) < lookback:
                raise ValueError(f"Insufficient data: need {lookback} rows, got {len(df)}")
            
            # Take the last 'lookback' rows
            recent_data = df[self.features].tail(lookback).values
            
            # Scale using the same scaler from training
            scaled_data = np.clip(self.scaler.transform(recent_data), -1e5, 1e5)
            
            # Reshape for model input: (1, lookback, features)
            return scaled_data.reshape(1, lookback, len(self.features))
            
        except Exception as e:
            logging.error(f"Failed to prepare data: {e}")
            raise
    
    def predict(self, input_data):
        """Make predictions for all loaded horizons."""
        predictions = {}
        
        try:
            for horizon, model in self.models.items():
                # Get prediction probability
                prob = model.predict(input_data, verbose=0)[0][0]
                
                # Convert to direction (1=up, 0=down)
                direction = 1 if prob > 0.5 else 0
                confidence = prob if direction == 1 else (1 - prob)
                
                predictions[horizon] = {
                    'direction': 'UP' if direction == 1 else 'DOWN',
                    'probability': prob,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                }
            
            return predictions
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise
    
    def predict_from_file(self, data_path, lookback=300):
        """Make predictions from a data file."""
        try:
            # Load data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df['datetime_point'] = pd.to_datetime(df['datetime_point'])
            df = df.sort_values('datetime_point').reset_index(drop=True)
            
            logging.info(f"Loaded {len(df)} data points from {df['datetime_point'].min()} to {df['datetime_point'].max()}")
            
            # Prepare data and make predictions
            input_data = self.prepare_data(df, lookback)
            predictions = self.predict(input_data)
            
            return predictions
            
        except Exception as e:
            logging.error(f"Failed to predict from file: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Price Direction Prediction using Trained Models')
    parser.add_argument('--data_path', default='./eurusd.json', help='Path to JSON data file')
    parser.add_argument('--model_dir', default='./', help='Directory containing trained models')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 3, 5, 10], help='Prediction horizons to load')
    parser.add_argument('--lookback', type=int, default=300, help='Lookback period in minutes')
    parser.add_argument('--output', help='Output file for predictions (optional)')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = PricePredictor(args.model_dir)
        
        # Load models
        predictor.load_models(args.horizons)
        
        # Make predictions
        predictions = predictor.predict_from_file(args.data_path, args.lookback)
        
        # Display results
        print("\n" + "="*60)
        print("PRICE DIRECTION PREDICTIONS")
        print("="*60)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Prediction Time: {current_time}")
        print(f"Data Source: {args.data_path}")
        print(f"Lookback Period: {args.lookback} minutes")
        print("-"*60)
        
        for horizon in sorted(predictions.keys()):
            pred = predictions[horizon]
            direction_emoji = "📈" if pred['direction'] == 'UP' else "📉"
            confidence_bar = "█" * int(pred['confidence'] * 20)
            
            print(f"{horizon:2d}min: {direction_emoji} {pred['direction']:4s} | "
                  f"Confidence: {pred['confidence']:6.2%} {confidence_bar:20s} | "
                  f"Probability: {pred['probability']:.4f}")
        
        print("-"*60)
        
        # Consensus prediction
        up_votes = sum(1 for p in predictions.values() if p['direction'] == 'UP')
        total_votes = len(predictions)
        consensus = "BULLISH" if up_votes > total_votes / 2 else "BEARISH"
        consensus_emoji = "🐂" if consensus == "BULLISH" else "🐻"
        
        print(f"Consensus: {consensus_emoji} {consensus} ({up_votes}/{total_votes} models predict UP)")
        
        # High confidence predictions
        high_conf_predictions = {h: p for h, p in predictions.items() if p['confidence'] > 0.65}
        if high_conf_predictions:
            print(f"\nHigh Confidence Predictions (>65%):")
            for horizon, pred in high_conf_predictions.items():
                emoji = "📈" if pred['direction'] == 'UP' else "📉"
                print(f"  {horizon}min: {emoji} {pred['direction']} ({pred['confidence']:.1%})")
        
        print("="*60)
        
        # Save predictions if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            print(f"Predictions saved to {args.output}")
        
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main() 