# 🚀 AI-Powered LSTM Price Prediction Trading System

A complete end-to-end system for cryptocurrency/forex price direction prediction using deep learning LSTM networks, optimized for Intel i7-8700 CPU with 32GB RAM.

## 📊 System Overview

This system combines advanced machine learning with practical trading applications:

- **🧠 Deep Learning**: GRU neural networks with attention mechanisms
- **📈 Technical Analysis**: 14 advanced indicators (RSI, MACD, Bollinger Bands, etc.)
- **🤖 Automated Trading**: Risk-managed trading bot with position tracking
- **📊 Performance Monitoring**: Real-time model performance tracking and alerts
- **⚡ CPU Optimized**: Fully optimized for multi-core CPU training (no GPU required)

## 🎯 Key Features

### ✅ **Fixed Data Leakage Issues**
- Proper forward-fill only (no future data leakage)
- Robust NaN handling for technical indicators
- Clean train/validation/test splits

### ⚡ **CPU Performance Optimized**
- **Intel i7-8700 optimized**: 6 cores/12 threads fully utilized
- **Memory efficient**: Uses 10-14GB of your 32GB RAM
- **Batch optimization**: 1536 batch size for optimal throughput

### 🎯 **Multiple Prediction Horizons**
- **1 minute**: Scalping opportunities
- **3 minutes**: Short-term momentum
- **5 minutes**: Medium-term trends  
- **10 minutes**: Swing trading signals

### 📈 **Advanced Model Architecture**
- **GRU Layers**: 512→256→128 neurons with attention
- **Regularization**: Dropout, BatchNorm, L2 regularization
- **Technical Indicators**: 14 features including RSI, MACD, Bollinger Bands
- **Expected Accuracy**: 54-60% directional prediction

## 📁 System Components

```
lstm/
├── run-cpu.py           # 🧠 Main training script (FIXED)
├── prediction.py        # 🔮 Make predictions with trained models
├── trading_bot.py       # 🤖 Automated trading bot
├── monitor.py           # 📊 Performance monitoring and alerts
├── example_usage.py     # 🚀 Complete workflow example
└── README.md           # 📖 This documentation
```

## 🛠️ Installation & Setup

### 1. **Environment Setup**
```bash
# Activate your TensorFlow environment
cd /mnt/WORK/Project/MegaVX/po-ai
source venv_tf_cpu/bin/activate
cd lstm
```

### 2. **Verify Dependencies**
The system requires:
- ✅ TensorFlow 2.17+
- ✅ Pandas, NumPy, Scikit-learn
- ✅ Your `eurusd.json` data file
- ✅ 32GB RAM (will use ~10-14GB)

## 🚀 Quick Start

### **Option 1: Complete Workflow (Recommended)**
```bash
# Run everything automatically
python example_usage.py --mode all --quick

# Or full training (4-6 hours)
python example_usage.py --mode all
```

### **Option 2: Step-by-Step**

#### **Step 1: Train Models**
```bash
# Ultra-quick mode (1 hour) - for testing
python run-cpu.py --horizons 1 --batch_size 2048 --lookback 120

# Quick test (2-3 hours per horizon)
python run-cpu.py --horizons 1 3 --batch_size 1536 --lookback 180

# Balanced performance (4-6 hours per horizon)  
python run-cpu.py --horizons 1 3 5 10 --batch_size 1536 --lookback 300

# Maximum quality (6-8 hours per horizon)
python run-cpu.py --batch_size 1024 --lookback 480
```

#### **Step 2: Make Predictions**
```bash
# Generate predictions for all horizons
python prediction.py --output predictions.json

# Custom horizons
python prediction.py --horizons 1 5 10 --lookback 300
```

#### **Step 3: Run Trading Bot (Demo)**
```bash
# Create default configuration
python trading_bot.py --create_config

# Run demo trading (limited cycles)
python trading_bot.py --max_iterations 10

# Full trading bot (use with caution!)
python trading_bot.py
```

#### **Step 4: Monitor Performance**
```bash
# Single performance check
python monitor.py --output monitoring_report.json

# Continuous monitoring
python monitor.py --continuous --interval 3600
```

## 📊 Expected Performance

### **Training Time (Intel i7-8700)**
| Mode | Time per Horizon | Total (4 horizons) | Accuracy |
|------|------------------|---------------------|----------|
| **Ultra-Quick** | 1 hour | 4 hours | 52-55% |
| **Quick** | 2-3 hours | 8-12 hours | 54-57% |
| **Balanced** | 4-6 hours | 16-24 hours | 55-58% |
| **Maximum** | 6-8 hours | 24-32 hours | 56-60% |

### **Resource Usage**
- **CPU**: 95-100% utilization (all 12 threads)
- **Memory**: 10-14GB RAM during training
- **Storage**: ~50MB per trained model

## 📁 Output Files

After training and running, you'll have:

### **Model Files**
```
model_horizon_1min.h5      # 1-minute prediction model
model_horizon_3min.h5      # 3-minute prediction model  
model_horizon_5min.h5      # 5-minute prediction model
model_horizon_10min.h5     # 10-minute prediction model
scaler_common.pkl          # Data preprocessing scaler
```

### **Results Files**
```
predictions.json           # Latest predictions
trading_results.json       # Trading bot results
monitoring_report.json     # Performance analysis
trading_bot.log           # Trading activity log
performance_history.json   # Historical performance
```

## 🔮 Making Predictions

The prediction output shows:

```
============================================================
PRICE DIRECTION PREDICTIONS
============================================================
Prediction Time: 2025-07-30 13:45:00
Data Source: ./eurusd.json
Lookback Period: 300 minutes
------------------------------------------------------------
 1min: 📈 UP   | Confidence:  67.3% ████████████████████ | Probability: 0.6732
 3min: 📉 DOWN | Confidence:  72.1% ████████████████████ | Probability: 0.2791
 5min: 📈 UP   | Confidence:  58.9% ████████████████████ | Probability: 0.5889
10min: 📈 UP   | Confidence:  81.2% ████████████████████ | Probability: 0.8123
------------------------------------------------------------
Consensus: 🐂 BULLISH (3/4 models predict UP)

High Confidence Predictions (>65%):
  10min: 📈 UP (81.2%)
  3min: 📉 DOWN (72.1%)
============================================================
```

## 🤖 Trading Bot Features

### **Risk Management**
- **Position Sizing**: Based on confidence levels
- **Stop Loss**: 50 pip automatic stops
- **Take Profit**: 100 pip targets
- **Max Positions**: Configurable limit
- **Time-based Exits**: Close at horizon expiry

### **Signal Generation**
- **Confidence Threshold**: Only trade high-confidence signals (>65%)
- **Multiple Horizons**: Combine different timeframe signals
- **Signal Strength**: WEAK/MEDIUM/STRONG classification

### **Configuration (trading_config.json)**
```json
{
  "confidence_threshold": 0.65,
  "min_horizon": 3,
  "max_positions": 5,
  "check_interval": 60,
  "horizons": [1, 3, 5, 10],
  "data_path": "./eurusd.json"
}
```

## 📊 Performance Monitoring

The monitor tracks:

- **Model Accuracy**: Directional prediction accuracy
- **Confidence Levels**: Average prediction confidence
- **Performance Degradation**: Alerts when accuracy drops
- **Pattern Analysis**: Market bias detection
- **Retraining Alerts**: When models need updates

### **Monitoring Alerts**
```
⚠️  Performance degradation detected for 5min model:
    Current=0.534, Historical=0.587, Drop=0.053
    
📊 Recommendations:
    - CAUTION: Low overall accuracy
    - Consider retraining models with MEDIUM severity alerts
```

## 🛡️ Risk Warnings

### **⚠️ Important Disclaimers**

1. **Educational Purpose**: This system is for educational and research purposes
2. **No Financial Advice**: Not financial advice, trade at your own risk
3. **Past Performance**: Historical results don't guarantee future performance
4. **Market Risk**: Cryptocurrency/forex trading involves significant risk
5. **Demo Mode**: Test thoroughly before any real trading

### **🔒 Safety Features**
- **Demo Mode**: Default trading is simulation only
- **Position Limits**: Maximum position controls
- **Stop Losses**: Automatic loss protection
- **Monitoring**: Performance degradation alerts

## 🔧 Troubleshooting

### **Common Issues**

1. **Memory Error**
   ```bash
   # Reduce batch size
   python run-cpu.py --batch_size 1024
   ```

2. **NaN Values Error**
   ```bash
   # Use debug mode to investigate
   python run-cpu.py --debug
   ```

3. **Model Loading Error**
   ```bash
   # Check if models exist
   ls -la *.h5 *.pkl
   ```

4. **Poor Performance**
   ```bash
   # Check monitoring report
   python monitor.py
   ```

## 📈 Advanced Usage

### **Custom Training**
```bash
# Train only specific horizons
python run-cpu.py --horizons 5 10 --lookback 480

# Adjust for your system
python run-cpu.py --batch_size 2048 --epochs 100
```

### **Real-time Integration**
```python
# In your trading application
from prediction import PricePredictor

predictor = PricePredictor('./')
predictor.load_models([1, 3, 5, 10])

# Get live predictions
predictions = predictor.predict_from_file('live_data.json')
```

### **Custom Monitoring**
```bash
# Monitor specific horizons only
python monitor.py --horizons 1 5 --interval 1800

# Generate performance plots
python monitor.py --output custom_report.json
```

## 📚 Technical Details

### **Model Architecture**
```
Input (300, 14) → GRU(512) → BatchNorm → Dropout(0.2) →
GRU(256) → BatchNorm → Dropout(0.2) →
GRU(128) → Dropout(0.1) →
Attention → Dense(128) → Dropout(0.3) →
Dense(64) → Dropout(0.2) → Dense(1, sigmoid)
```

### **Technical Indicators**
1. **RSI (14)** - Relative Strength Index
2. **EMA (12, 26)** - Exponential Moving Averages
3. **MACD** - Moving Average Convergence Divergence
4. **Bollinger Bands** - Volatility indicator
5. **ATR** - Average True Range
6. **Stochastic K/D** - Momentum oscillators
7. **OBV** - On-Balance Volume

### **CPU Optimization**
```python
# Thread configuration for i7-8700
OMP_NUM_THREADS = "12"
TF_NUM_INTEROP_THREADS = "6" 
TF_NUM_INTRAOP_THREADS = "12"
```

## 🤝 Contributing

Improvements welcome! Areas for enhancement:
- Additional technical indicators
- Alternative model architectures
- Real broker API integration
- Enhanced risk management
- Multi-asset support

## 📞 Support

For issues:
1. Check the troubleshooting section
2. Run with `--debug` flag for detailed logs
3. Verify your system meets requirements
4. Ensure data file is properly formatted

## 📄 License

This project is for educational purposes. Use responsibly and at your own risk.

---

**🚀 Happy Trading! Remember: Past performance doesn't guarantee future results.** 