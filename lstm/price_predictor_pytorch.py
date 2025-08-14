#!/usr/bin/env python3
"""
PyTorch LSTM Price Predictor with RTX 5060 Ti GPU Acceleration
Converted from TensorFlow version for better GPU compatibility
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging
import os
import json
import time
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./log.txt'),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

class LSTMPricePredictor(nn.Module):
    """LSTM model for price prediction optimized for RTX 5060 Ti"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2, 
                 output_size=1, device='cuda'):
        super(LSTMPricePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers for output
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        out = self.dropout(lstm_out)
        
        # Dense layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

def detect_and_configure_gpu():
    """Detect and configure GPU for RTX 5060 Ti"""
    logging.info("🚀 GPU Detection and Configuration")
    logging.info("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.warning("❌ CUDA not available, using CPU")
        return torch.device('cpu'), False
    
    # Get GPU information
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    logging.info(f"✅ CUDA available: {torch.version.cuda}")
    logging.info(f"✅ GPU count: {device_count}")
    logging.info(f"✅ Current GPU: {device_name}")
    
    # Get detailed GPU properties
    props = torch.cuda.get_device_properties(current_device)
    logging.info(f"✅ Compute capability: {props.major}.{props.minor}")
    logging.info(f"✅ Total memory: {props.total_memory / (1024**3):.1f} GB")
    logging.info(f"✅ Multi-processors: {props.multi_processor_count}")
    
    # Set memory management for RTX 5060 Ti
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    device = torch.device('cuda:0')
    logging.info(f"🎯 Using device: {device}")
    
    return device, True

def load_and_preprocess_data(file_path, lookback_window=120):
    """Load and preprocess the EURUSD data"""
    logging.info(f"📊 Loading data from {file_path}")
    
    try:
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime_point'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        logging.info(f"✅ Loaded {len(df)} records")
        logging.info(f"✅ Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Use close price for prediction
        prices = df['close'].values.reshape(-1, 1)
        
        # Handle outliers using IQR method
        Q1 = np.percentile(prices, 25)
        Q3 = np.percentile(prices, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        prices = np.clip(prices, lower_bound, upper_bound)
        
        # Normalize data using RobustScaler (better for financial data)
        scaler = RobustScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        logging.info(f"✅ Price range: {prices.min():.5f} to {prices.max():.5f}")
        logging.info(f"✅ Data preprocessing completed")
        
        return scaled_prices, scaler, df
        
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        raise

def create_sequences(data, lookback_window):
    """Create sequences for LSTM training"""
    logging.info(f"🔄 Creating sequences with lookback window: {lookback_window}")
    
    X, y = [], []
    for i in range(lookback_window, len(data)):
        X.append(data[i-lookback_window:i])
        y.append(data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    logging.info(f"✅ Created {len(X)} sequences")
    logging.info(f"✅ X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y

def train_model(model, train_loader, val_loader, epochs=100, device='cuda', patience=10):
    """Train the LSTM model with early stopping"""
    logging.info(f"🚀 Starting training on {device}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            logging.info(f"Epoch [{epoch+1}/{epochs}] - "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Early stopping
        if patience_counter >= patience:
            logging.info(f"⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    logging.info(f"✅ Training completed in {training_time:.2f} seconds")
    logging.info(f"✅ Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses, training_time

def evaluate_model(model, test_loader, scaler, device='cuda'):
    """Evaluate the trained model"""
    logging.info("📊 Evaluating model performance")
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Inverse transform predictions and actuals
    predictions_original = scaler.inverse_transform(predictions)
    actuals_original = scaler.inverse_transform(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals_original, predictions_original)
    mae = mean_absolute_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals_original, predictions_original)
    
    # Calculate directional accuracy
    actual_direction = np.diff(actuals_original.flatten()) > 0
    pred_direction = np.diff(predictions_original.flatten()) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    logging.info(f"✅ Test MSE: {mse:.8f}")
    logging.info(f"✅ Test MAE: {mae:.8f}")
    logging.info(f"✅ Test RMSE: {rmse:.8f}")
    logging.info(f"✅ Test R²: {r2:.6f}")
    logging.info(f"✅ Directional Accuracy: {directional_accuracy:.2f}%")
    
    return {
        'predictions': predictions_original,
        'actuals': actuals_original,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }

def plot_results(train_losses, val_losses, results, save_plots=True):
    """Plot training results and predictions"""
    logging.info("📈 Generating visualization plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Predictions vs Actuals (time series)
    sample_size = min(500, len(results['actuals']))
    axes[0, 1].plot(results['actuals'][:sample_size], label='Actual', alpha=0.7)
    axes[0, 1].plot(results['predictions'][:sample_size], label='Predicted', alpha=0.7)
    axes[0, 1].set_title(f'Predictions vs Actuals (First {sample_size} samples)')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Scatter plot
    axes[1, 0].scatter(results['actuals'], results['predictions'], alpha=0.6)
    axes[1, 0].plot([results['actuals'].min(), results['actuals'].max()], 
                    [results['actuals'].min(), results['actuals'].max()], 'r--', lw=2)
    axes[1, 0].set_title('Predictions vs Actuals (Scatter)')
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].grid(True)
    
    # Plot 4: Residuals
    residuals = results['actuals'] - results['predictions']
    axes[1, 1].hist(residuals.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Residuals Distribution')
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'checkpoints/pytorch_lstm_results_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        logging.info(f"✅ Plots saved to {plot_filename}")
    
    plt.show()

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='PyTorch LSTM Price Predictor')
    parser.add_argument('--data_file', type=str, default='eurusd.json',
                       help='Path to the JSON data file')
    parser.add_argument('--lookback', type=int, default=120,
                       help='Lookback window for sequences')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    args = parser.parse_args()
    
    logging.info("🚀 PyTorch LSTM Price Predictor with RTX 5060 Ti")
    logging.info("=" * 60)
    
    try:
        # Detect and configure GPU
        device, has_gpu = detect_and_configure_gpu()
        
        # Load and preprocess data
        scaled_data, scaler, df = load_and_preprocess_data(args.data_file, args.lookback)
        
        # Create sequences
        X, y = create_sequences(scaled_data, args.lookback)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
        )
        
        logging.info(f"✅ Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model = LSTMPricePredictor(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=device
        ).to(device)
        
        logging.info(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train model
        train_losses, val_losses, training_time = train_model(
            model, train_loader, val_loader, args.epochs, device
        )
        
        # Evaluate model
        results = evaluate_model(model, test_loader, scaler, device)
        
        # Plot results
        plot_results(train_losses, val_losses, results)
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'checkpoints/pytorch_lstm_model_{timestamp}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'model_config': {
                'input_size': 1,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            },
            'results': results,
            'training_time': training_time
        }, model_filename)
        
        logging.info(f"✅ Model saved to {model_filename}")
        
        # Final summary
        logging.info("\n" + "=" * 60)
        logging.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 60)
        logging.info(f"Device used: {device}")
        logging.info(f"Training time: {training_time:.2f} seconds")
        logging.info(f"Final Test RMSE: {results['rmse']:.8f}")
        logging.info(f"Final Test R²: {results['r2']:.6f}")
        logging.info(f"Directional Accuracy: {results['directional_accuracy']:.2f}%")
        
        if has_gpu:
            logging.info("✅ GPU training successful! Your RTX 5060 Ti is working with PyTorch.")
        else:
            logging.info("⚠️  Training completed on CPU.")
            
        return True
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logging.info("🎯 SUCCESS: Price prediction model trained successfully!")
    else:
        logging.info("❌ FAILED: Training unsuccessful") 