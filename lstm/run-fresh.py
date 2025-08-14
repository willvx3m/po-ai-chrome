#!/usr/bin/env python3
"""
Simple LSTM training script to test RTX 5060 Ti GPU with CUDA.
Generates synthetic time series data and trains a basic LSTM model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

def setup_gpu():
    """Configure GPU settings for RTX 5060 Ti."""
    print("="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available GPUs: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
            
            # Set GPU as the default device
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"✅ Using GPU: {gpus[0].name}")
            
            # Test a simple operation on GPU
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                print(f"✅ GPU test operation successful: {result.numpy()}")
                
            return True
            
        except Exception as e:
            print(f"❌ GPU setup failed: {e}")
            print("Falling back to CPU...")
            return False
    else:
        print("❌ No GPU detected")
        return False

def generate_synthetic_data(num_samples=10000, sequence_length=50, num_features=1):
    """Generate synthetic time series data for LSTM training."""
    print(f"Generating {num_samples} samples with sequence length {sequence_length}...")
    
    # Generate synthetic sine wave with noise
    t = np.linspace(0, 4*np.pi, num_samples + sequence_length)
    
    # Create a more complex pattern: sine + cosine with different frequencies + noise
    data = (np.sin(t) + 
            0.5 * np.sin(2*t) + 
            0.3 * np.cos(3*t) + 
            0.1 * np.random.randn(len(t)))
    
    # Normalize data
    data = (data - np.mean(data)) / np.std(data)
    
    # Create sequences
    X, y = [], []
    for i in range(num_samples):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM input (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], num_features))
    
    print(f"✅ Generated data shapes: X={X.shape}, y={y.shape}")
    return X, y

def create_lstm_model(sequence_length, num_features):
    """Create a simple LSTM model."""
    print("Building LSTM model...")
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("✅ Model compiled successfully")
    print(f"Total parameters: {model.count_params():,}")
    return model

def train_model(model, X_train, y_train, X_val, y_val, device_name):
    """Train the LSTM model."""
    print(f"Starting training on {device_name}...")
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Record training start time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=256,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Record training end time
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    return history, training_time

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("Evaluating model...")
    
    # Make predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    mse = tf.keras.metrics.mean_squared_error(y_test, predictions).numpy().mean()
    mae = tf.keras.metrics.mean_absolute_error(y_test, predictions).numpy().mean()
    
    print(f"✅ Test MSE: {mse:.6f}")
    print(f"✅ Test MAE: {mae:.6f}")
    
    return predictions, mse, mae

def plot_results(history, predictions, y_test, save_plots=True):
    """Plot training results and predictions."""
    print("Generating plots...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Training MAE
    axes[0, 1].plot(history.history['mae'], label='Training MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Model MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Predictions vs Actual (first 200 samples)
    sample_size = min(200, len(y_test))
    axes[1, 0].plot(y_test[:sample_size], label='Actual', alpha=0.7)
    axes[1, 0].plot(predictions[:sample_size], label='Predicted', alpha=0.7)
    axes[1, 0].set_title('Predictions vs Actual (First 200 samples)')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Scatter plot of predictions vs actual
    axes[1, 1].scatter(y_test, predictions, alpha=0.5)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 1].set_title('Predictions vs Actual (Scatter)')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'lstm_training_results_{timestamp}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved to {plot_filename}")
    
    plt.show()

def main():
    """Main training pipeline."""
    print("🚀 LSTM GPU Training Test")
    print("="*60)
    
    # Setup GPU
    gpu_available = setup_gpu()
    device_name = "GPU" if gpu_available else "CPU"
    
    # Generate synthetic data
    sequence_length = 50
    num_features = 1
    num_samples = 10000
    
    X, y = generate_synthetic_data(num_samples, sequence_length, num_features)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"✅ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create and train model
    if gpu_available:
        with tf.device('/GPU:0'):
            model = create_lstm_model(sequence_length, num_features)
            history, training_time = train_model(model, X_train, y_train, X_val, y_val, device_name)
    else:
        model = create_lstm_model(sequence_length, num_features)
        history, training_time = train_model(model, X_train, y_train, X_val, y_val, device_name)
    
    # Evaluate model
    predictions, mse, mae = evaluate_model(model, X_test, y_test)
    
    # Plot results
    plot_results(history, predictions, y_test)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Device used: {device_name}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final Test MSE: {mse:.6f}")
    print(f"Final Test MAE: {mae:.6f}")
    
    if gpu_available:
        print("✅ GPU training successful! Your RTX 5060 Ti is working with CUDA.")
    else:
        print("⚠️  Training completed on CPU. GPU may have compatibility issues.")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'lstm_model_{timestamp}.h5'
    model.save(model_filename)
    print(f"✅ Model saved to {model_filename}")
    
    return gpu_available, training_time, mse, mae

if __name__ == "__main__":
    try:
        success, time_taken, final_mse, final_mae = main()
        
        if success:
            print("\n🎯 SUCCESS: Your RTX 5060 Ti GPU is working perfectly with TensorFlow!")
        else:
            print("\n⚠️  INFO: Training completed on CPU. Check GPU drivers and CUDA installation.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Training failed. Please check your setup.")
        raise