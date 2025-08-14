#!/usr/bin/env python3
"""
Simple CPU-only training script for RTX 5060 Ti compatibility.
Bypasses GPU detection to avoid compute capability 12.0 issues.
"""

# Force CPU-only mode BEFORE importing TensorFlow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["TF_NUM_INTEROP_THREADS"] = "6" 
os.environ["TF_NUM_INTRAOP_THREADS"] = "12"

import json
import pandas as pd
import numpy as np
import argparse
import gc
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense, BatchNormalization, Attention, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import logging
import psutil

# Try to import enhanced preprocessing
try:
    from enhanced_preprocessing import EnhancedDataProcessor
    ENHANCED_PREPROCESSING_AVAILABLE = True
    logging.info("Enhanced preprocessing module loaded successfully")
except ImportError:
    ENHANCED_PREPROCESSING_AVAILABLE = False
    logging.warning("Enhanced preprocessing module not found, using basic preprocessing")

# CPU Configuration
tf.config.threading.set_inter_op_parallelism_threads(6)
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.set_soft_device_placement(True)

# Set up comprehensive logging (console + file)
def setup_logging():
    """Configure logging to output to both console and file."""
    import os
    from datetime import datetime
    
    # Create logs directory if it doesn't exist
    log_dir = '.'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler (simple format for readability)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (detailed format for debugging)
    log_file = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Log session start
    session_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info("="*80)
    logger.info(f"CPU-ONLY TRAINING SESSION STARTED: {session_start}")
    logger.info(f"Log file: {os.path.abspath(log_file)}")
    logger.info("="*80)
    
    return logger

# Initialize logging
logger = setup_logging()

class AdvancedDataGenerator(Sequence):
    """Memory-efficient data generator with augmentation for CPU training."""
    def __init__(self, sequences, targets, batch_size, shuffle=True, augment=True):
        self.sequences = sequences
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(sequences))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.sequences) / self.batch_size))
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.sequences))
        batch_indices = self.indices[start:end]
        
        X = self.sequences[batch_indices]
        y = self.targets[batch_indices]
        
        # Data augmentation for financial data
        if self.augment and np.random.random() > 0.5:
            # Add small noise (0.1% of std)
            noise = np.random.normal(0, 0.001, X.shape)
            X = X + noise
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

class PricePredictor:
    def __init__(self, data_path, horizons=[1, 3, 5, 10], lookback=300, batch_size=1536, use_enhanced_preprocessing=True):
        """Initialize with CPU-optimized settings."""
        self.data_path = data_path
        self.horizons = horizons
        self.lookback = lookback
        self.batch_size = batch_size
        
        # Enhanced preprocessing setup
        self.use_enhanced_preprocessing = use_enhanced_preprocessing and ENHANCED_PREPROCESSING_AVAILABLE
        if self.use_enhanced_preprocessing:
            self.enhanced_processor = EnhancedDataProcessor(
                outlier_method='modified_zscore',
                outlier_threshold=3.5,
                use_robust_scaler=True
            )
            logging.info("Enhanced preprocessing enabled with RobustScaler")
        else:
            self.scaler = MinMaxScaler()
            logging.info("Using basic preprocessing with MinMaxScaler")
        
        # Initialize data structures
        self.df = None
        self.models = {}
        self.sequences = None
        self.targets = {h: [] for h in horizons}
        self.train_X = None
        self.val_X = None
        self.test_X = None
        self.train_y = {h: [] for h in horizons}
        self.val_y = {h: [] for h in horizons}
        self.test_y = {h: [] for h in horizons}
        self.features = ['open', 'high', 'low', 'close', 'RSI_14', 'EMA_12', 'EMA_26', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'Stoch_K', 'Stoch_D', 'OBV']
        
        # Log configuration
        logging.info(f"PricePredictor initialized for CPU-only training:")
        logging.info(f"  Device: CPU")
        logging.info(f"  Batch size: {self.batch_size}")
        logging.info(f"  Lookback: {self.lookback}")
        logging.info(f"  Horizons: {self.horizons}")
        logging.info(f"  Enhanced preprocessing: {self.use_enhanced_preprocessing}")

    def load_data(self):
        """Load and sort the JSON OHLC data."""
        logging.info("Loading data...")
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
            self.df['datetime_point'] = pd.to_datetime(self.df['datetime_point'])
            self.df = self.df.sort_values('datetime_point').reset_index(drop=True)
            
            logging.info(f"Loaded {len(self.df)} data points from {self.df['datetime_point'].min()} to {self.df['datetime_point'].max()}")
            
            if self.df[['open', 'high', 'low', 'close']].isna().any().any() or \
               np.isinf(self.df[['open', 'high', 'low', 'close']]).any().any():
                logging.error("NaN or infinite values detected in OHLC data")
                raise ValueError("Data contains NaN or infinite values")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def add_technical_indicators(self):
        """Add RSI, EMA, MACD, Bollinger Bands, ATR, Stochastic K/D, and OBV with NaN handling."""
        logging.info("Adding technical indicators...")
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

            self.df['RSI_14'] = calculate_rsi(self.df)
            self.df['EMA_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
            self.df['EMA_26'] = self.df['close'].ewm(span=26, adjust=False).mean()
            self.df['MACD'] = calculate_macd(self.df)
            self.df['BB_upper'], self.df['BB_lower'] = calculate_bollinger_bands(self.df)
            self.df['ATR'] = calculate_atr(self.df)
            self.df['Stoch_K'], self.df['Stoch_D'] = calculate_stochastic_kd(self.df)
            self.df['OBV'] = calculate_obv(self.df)
            
            logging.info(f"Feature stats:\n{self.df[self.features].describe()}")
        except Exception as e:
            logging.error(f"Failed to add technical indicators: {e}")
            raise

    def preprocess(self):
        """Enhanced preprocessing with outlier detection and robust scaling (when available)."""
        logging.info(f"Memory before preprocessing: {psutil.virtual_memory().percent}% used")
        try:
            self.load_data()
            self.add_technical_indicators()
            
            # Choose preprocessing method based on availability and user preference
            if self.use_enhanced_preprocessing:
                # ENHANCED PREPROCESSING PIPELINE
                logging.info("Using enhanced preprocessing pipeline...")
                
                # Apply enhanced preprocessing
                self.df, outlier_stats = self.enhanced_processor.process_data(self.df, self.features)
                
                # Log enhancement statistics
                if outlier_stats:
                    total_improvements = 0
                    if 'price_spikes' in outlier_stats:
                        total_improvements += sum(outlier_stats['price_spikes'].values())
                    if 'technical_indicators' in outlier_stats:
                        total_improvements += sum(outlier_stats['technical_indicators'].values())
                    logging.info(f"Enhanced preprocessing completed: {total_improvements} data points improved")
                
                # Use the enhanced processor's scaler
                self.scaler = self.enhanced_processor.scaler
                
            else:
                # BASIC PREPROCESSING PIPELINE (original)
                logging.info("Using basic preprocessing pipeline...")
                
                # Log initial NaN situation
                initial_nans = self.df[self.features].isna().sum()
                logging.info(f"Initial NaN counts per feature:\n{initial_nans[initial_nans > 0]}")
                
                N = len(self.df)
                logging.info(f"Original dataset size: {N}")
                
                # ROBUST NaN HANDLING: Find the first row where ALL features are valid
                feature_validity = ~self.df[self.features].isna()
                all_features_valid = feature_validity.all(axis=1)
                
                if not all_features_valid.any():
                    raise ValueError("No rows found where all features are valid. Check your technical indicators.")
                
                first_complete_idx = all_features_valid.idxmax()
                
                if first_complete_idx > 0:
                    logging.info(f"Removing first {first_complete_idx} rows with incomplete technical indicators")
                    self.df = self.df.iloc[first_complete_idx:].reset_index(drop=True)
                    N = len(self.df)
                    logging.info(f"Dataset size after removing incomplete rows: {N}")
                
                # Apply forward fill for any remaining isolated NaN values
                self.df[self.features] = self.df[self.features].ffill()
                
                # Remove any remaining NaN rows at the beginning
                first_valid_idx = self.df[self.features].first_valid_index()
                if first_valid_idx is not None and first_valid_idx > 0:
                    self.df = self.df.iloc[first_valid_idx:].reset_index(drop=True)
                    logging.info(f"Removed {first_valid_idx} rows with initial NaN values")
                    N = len(self.df)
                
                # Final check: ensure absolutely no NaN values remain
                if self.df[self.features].isna().any().any():
                    raise ValueError("Critical error: NaN values still present after all cleaning attempts")
            
            # Common processing for both methods
            N = len(self.df)
            train_end = int(0.7 * N)
            val_end = int(0.85 * N)
            logging.info(f"Data split: Train={train_end}, Val={val_end - train_end}, Test={N - val_end}")
            
            # Verify we have enough data left
            min_required = self.lookback + max(self.horizons) + 1000
            if N < min_required:
                raise ValueError(f"Insufficient data after preprocessing: {N} < {min_required} required")
            
            # Scale data based on preprocessing method
            if self.use_enhanced_preprocessing:
                # Enhanced preprocessing already applied scaling
                scaled_data = self.df[self.features].values
            else:
                # Apply basic scaling
                train_data = self.df.iloc[:train_end]
                self.scaler.fit(train_data[self.features])
                scaled_data = np.clip(self.scaler.transform(self.df[self.features]), -1e5, 1e5)
            
            # Validate scaled data
            if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
                logging.error("NaN or Inf detected in scaled data")
                
                # Debug: Provide detailed information about the invalid values
                nan_count = np.isnan(scaled_data).sum()
                inf_count = np.isinf(scaled_data).sum()
                total_count = scaled_data.size
                logging.error(f"Invalid values: {nan_count} NaN, {inf_count} Inf out of {total_count} total")
                
                # Find which features have invalid values
                for i, feature in enumerate(self.features):
                    if i < scaled_data.shape[1]:
                        feature_data = scaled_data[:, i]
                        feature_nan = np.isnan(feature_data).sum()
                        feature_inf = np.isinf(feature_data).sum()
                        if feature_nan > 0 or feature_inf > 0:
                            logging.error(f"  {feature}: {feature_nan} NaN, {feature_inf} Inf")
                
                # Attempt emergency cleanup
                logging.warning("Attempting emergency cleanup of scaled data...")
                scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Verify cleanup worked
                if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
                    logging.error("Emergency cleanup failed - still have invalid values")
                    raise ValueError("Invalid values in scaled data after emergency cleanup")
                else:
                    logging.warning("Emergency cleanup successful - continuing with cleaned data")
            else:
                logging.info("Scaled data validation: PASSED - All values are finite")
            
            # Create sequences
            max_h = max(self.horizons)
            num_sequences = N - self.lookback - max_h + 1
            
            if num_sequences <= 0:
                raise ValueError(f"Cannot create sequences: N={N}, lookback={self.lookback}, max_h={max_h}")
            
            logging.info(f"Creating {num_sequences} sequences with lookback={self.lookback}")
            
            # Optimized sequence creation for CPU
            self.sequences = np.zeros((num_sequences, self.lookback, len(self.features)), dtype=np.float32)
            for i in range(num_sequences):
                self.sequences[i] = scaled_data[i:i + self.lookback, :]
            
            # Create targets
            for h in self.horizons:
                self.targets[h] = [
                    1 if self.df['close'][i + self.lookback - 1 + h] > self.df['close'][i + self.lookback - 1] else 0
                    for i in range(num_sequences)
                ]
            
            # Split data
            num_train = int(0.7 * num_sequences)
            num_val = int(0.15 * num_sequences)
            
            if num_train == 0 or num_val == 0 or (num_sequences - num_train - num_val) == 0:
                raise ValueError(f"Insufficient sequences for splitting: total={num_sequences}")
            
            self.train_X = self.sequences[:num_train]
            self.val_X = self.sequences[num_train:num_train + num_val]
            self.test_X = self.sequences[num_train + num_val:]
            
            for h in self.horizons:
                self.train_y[h] = np.array(self.targets[h][:num_train])
                self.val_y[h] = np.array(self.targets[h][num_train:num_train + num_val])
                self.test_y[h] = np.array(self.targets[h][num_train + num_val:])
                
                # Log class distribution
                train_pos = np.mean(self.train_y[h])
                val_pos = np.mean(self.val_y[h])
                test_pos = np.mean(self.test_y[h])
                logging.info(f"Horizon {h}min - Positive class ratio: Train={train_pos:.3f}, Val={val_pos:.3f}, Test={test_pos:.3f}")
            
            # Log final statistics
            preprocessing_type = "Enhanced" if self.use_enhanced_preprocessing else "Basic"
            logging.info(f"{preprocessing_type} preprocessing completed successfully!")
            logging.info(f"Final dataset: {N} rows, {num_sequences} sequences")
            logging.info(f"Memory after preprocessing: {psutil.virtual_memory().percent}% used")
            
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            raise

    def build_model(self):
        """Build enhanced GRU model optimized for CPU training."""
        try:
            with tf.device('/CPU:0'):
                inputs = Input(shape=(self.lookback, len(self.features)))
                
                # CPU-optimized model architecture
                gru_out = GRU(512, return_sequences=True, kernel_regularizer=l2(0.001))(inputs)
                gru_out = BatchNormalization()(gru_out)
                gru_out = Dropout(0.2)(gru_out)
                
                gru_out = GRU(256, return_sequences=True, kernel_regularizer=l2(0.001))(gru_out)
                gru_out = BatchNormalization()(gru_out)
                gru_out = Dropout(0.2)(gru_out)
                
                gru_out = GRU(128, return_sequences=True)(gru_out)
                gru_out = Dropout(0.1)(gru_out)
                
                # Multi-head attention mechanism
                attention = Attention()([gru_out, gru_out])
                attention_out = Lambda(lambda x: tf.reduce_mean(x, axis=1))(attention)
                
                # Deeper dense layers
                dense_out = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(attention_out)
                dense_out = Dropout(0.3)(dense_out)
                dense_out = Dense(64, activation='relu')(dense_out)
                dense_out = Dropout(0.2)(dense_out)
                
                output = Dense(1, activation='sigmoid')(dense_out)
                model = tf.keras.Model(inputs=inputs, outputs=output)
                
                # CPU-optimized optimizer
                optimizer = Adam(learning_rate=0.002, clipnorm=1.0, amsgrad=True)
                model.compile(loss='binary_crossentropy', optimizer=optimizer, 
                             metrics=['accuracy', 'precision', 'recall'])
                
                logging.info(f"CPU model built successfully. Total parameters: {model.count_params():,}")
                return model
        except Exception as e:
            logging.error(f"Failed to build model: {e}")
            raise

    def train_model(self, h):
        """Enhanced training optimized for CPU."""
        logging.info(f"Training model for horizon {h} minutes...")
        logging.info(f"Memory before training: {psutil.virtual_memory().percent}% used")
        try:
            model = self.build_model()
            train_y_h = self.train_y[h]
            val_y_h = self.val_y[h]
            logging.info(f"Input shapes: train_X={self.train_X.shape}, train_y={train_y_h.shape}")
            
            # Enhanced callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=15, 
                    restore_best_weights=True,
                    min_delta=0.001,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7, 
                    patience=5, 
                    min_lr=1e-6,
                    verbose=1
                ),
                ModelCheckpoint(
                    f'checkpoints/best_model_h{h}.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Create checkpoints directory
            os.makedirs('checkpoints', exist_ok=True)
            
            # Use advanced data generator
            train_gen = AdvancedDataGenerator(
                self.train_X, train_y_h, self.batch_size, shuffle=True, augment=True
            )
            val_gen = AdvancedDataGenerator(
                self.val_X, val_y_h, self.batch_size, shuffle=False, augment=False
            )
            
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=150,
                callbacks=callbacks,
                verbose=1
            )
            
            self.models[h] = model
            
            # Enhanced logging
            best_val_acc = max(history.history['val_accuracy'])
            best_val_loss = min(history.history['val_loss'])
            best_epoch = np.argmax(history.history['val_accuracy']) + 1
            logging.info(f"Horizon {h}: Best Val Accuracy: {best_val_acc:.4f} at epoch {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
            logging.info(f"Memory after training: {psutil.virtual_memory().percent}% used")
            
            # Clean up memory
            gc.collect()
            
        except Exception as e:
            logging.error(f"Training failed for horizon {h}: {e}")
            raise

    def train_models(self):
        """Train models for all horizons sequentially."""
        logging.info(f"Training models for horizons: {self.horizons}")
        for h in self.horizons:
            self.train_model(h)

    def evaluate_models(self):
        """Comprehensive model evaluation with detailed metrics."""
        logging.info("Evaluating models with detailed metrics...")
        
        for h in self.horizons:
            if h not in self.models:
                logging.error(f"No model found for horizon {h}. Skipping evaluation.")
                continue
            try:
                test_y_h = self.test_y[h]
                test_gen = AdvancedDataGenerator(
                    self.test_X, test_y_h, self.batch_size, shuffle=False, augment=False
                )
                
                # Get predictions
                predictions = self.models[h].predict(test_gen, verbose=0)
                y_pred = (predictions > 0.5).astype(int).flatten()
                y_true = test_y_h.flatten()
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)
                
                logging.info(f"Horizon {h} minutes:")
                logging.info(f"  Accuracy:  {accuracy:.4f}")
                logging.info(f"  Precision: {precision:.4f}")
                logging.info(f"  Recall:    {recall:.4f}")
                logging.info(f"  F1 Score:  {f1:.4f}")
                logging.info(f"  Confusion Matrix:\n{cm}")
                
                # Save model
                model_path = f'checkpoints/model_horizon_{h}min.h5'
                self.models[h].save(model_path)
                logging.info(f"Model saved to {model_path}")
                
                # Save scaler
                scaler_path = f'checkpoints/scaler_horizon_{h}min.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logging.info(f"Scaler saved to {scaler_path}")
                
            except Exception as e:
                logging.error(f"Evaluation failed for horizon {h}: {e}")

        # Save common scaler for easy loading
        with open('checkpoints/scaler_common.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        logging.info("All models and scalers saved successfully!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Price Direction Prediction with GRU (CPU-Only)')
    parser.add_argument('--data_path', default='./eurusd.json', help='Path to JSON data file')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 3, 5, 10], help='Prediction horizons in minutes')
    parser.add_argument('--lookback', type=int, default=300, help='Lookback period in minutes')
    parser.add_argument('--batch_size', type=int, default=1536, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=150, help='Maximum number of epochs')
    
    # Enhanced preprocessing options
    parser.add_argument('--no_enhanced_preprocessing', action='store_true', 
                       help='Disable enhanced preprocessing (use basic MinMaxScaler)')
    parser.add_argument('--outlier_method', choices=['iqr', 'zscore', 'modified_zscore', 'isolation_forest'],
                       default='modified_zscore', help='Outlier detection method for enhanced preprocessing')
    parser.add_argument('--outlier_threshold', type=float, default=3.5,
                       help='Outlier detection threshold')
    
    return parser.parse_args()

def close_logging_session(success=True):
    """Close the logging session with a summary."""
    from datetime import datetime
    
    session_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status = "COMPLETED SUCCESSFULLY" if success else "FAILED"
    
    logging.info("="*80)
    logging.info(f"CPU TRAINING SESSION {status}: {session_end}")
    logging.info("="*80)
    
    # Flush all handlers
    for handler in logging.getLogger().handlers:
        handler.flush()

if __name__ == "__main__":
    try:
        args = parse_args()
        
        # Log system information
        logging.info("="*60)
        logging.info("SYSTEM CONFIGURATION (CPU-ONLY MODE)")
        logging.info("="*60)
        logging.info(f"TensorFlow version: {tf.__version__}")
        logging.info(f"Device type: CPU (FORCED)")
        logging.info(f"Available devices: {[d.name for d in tf.config.list_physical_devices()]}")
        logging.info(f"CPU Threads - Inter-op: {tf.config.threading.get_inter_op_parallelism_threads()}")
        logging.info(f"CPU Threads - Intra-op: {tf.config.threading.get_intra_op_parallelism_threads()}")
        logging.info(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        logging.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
        logging.info(f"Enhanced preprocessing: {'Available' if ENHANCED_PREPROCESSING_AVAILABLE else 'Not available'}")
        logging.info("="*60)
        
        # Initialize predictor with enhanced options
        use_enhanced = not args.no_enhanced_preprocessing
        
        predictor = PricePredictor(
            data_path=args.data_path,
            horizons=args.horizons,
            batch_size=args.batch_size,
            lookback=args.lookback,
            use_enhanced_preprocessing=use_enhanced
        )
        
        # Configure enhanced preprocessing if enabled
        if use_enhanced and ENHANCED_PREPROCESSING_AVAILABLE:
            predictor.enhanced_processor.outlier_method = args.outlier_method
            predictor.enhanced_processor.outlier_threshold = args.outlier_threshold
            logging.info(f"Enhanced preprocessing configured: {args.outlier_method} with threshold {args.outlier_threshold}")
        
        # Main training pipeline
        logging.info("="*60)
        logging.info("STARTING CPU TRAINING PIPELINE")
        logging.info("="*60)
        
        predictor.preprocess()
        predictor.train_models()
        logging.info(f"Trained models: {list(predictor.models.keys())}")
        predictor.evaluate_models()
        
        logging.info("="*60)
        logging.info("CPU TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("="*60)
        
        # Close logging session on success
        close_logging_session(success=True)
        
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        
        # Close logging session on failure
        close_logging_session(success=False)
        raise 