Based on your ultimate goal of establishing a binary trading strategy by predicting price movements at specific future intervals (3, 5, 8, and 10 minutes) using 1-minute OHLC data for EURUSD OTC, I’ll outline a structured approach to kick off training an LSTM model. Your focus is on predicting the direction of price movement (up or down) as the first step, making this a binary classification problem. Below, I’ll detail the key elements—data preprocessing, features, model specifications, training, and evaluation—starting with simple steps and providing a roadmap for improvement.

1. Data Details and Preprocessing

Data Overview: You have 2 months of 1-minute OHLC data (without volume), totaling approximately 60 days × 24 hours × 60 minutes = 86,400 data points. No missing timestamps is a great starting point, as it simplifies preprocessing.
Cleaning: While timestamps are intact, check for outliers (e.g., extreme price spikes) that could skew the model. If detected, cap or smooth these anomalies to maintain data integrity.
Scaling: Apply MinMaxScaler to normalize OHLC values (and additional features) between 0 and 1. This ensures stable training for the LSTM model. Fit the scaler only on the training data to avoid data leakage.


2. Prediction Target

Objective: Predict whether the closing price will go up or down at 3, 5, 8, and 10 minutes from the current time point.

1 = Closing price at the future time is higher than the current closing price (up).
0 = Closing price is lower or equal (down).


Approach: Train separate models for each prediction horizon (3, 5, 8, 10 minutes) to keep the setup simple and modular.


3. Feature Engineering

Base Features: Use the OHLC values (Open, High, Low, Close) directly as the core inputs.
Technical Indicators: Since you’ve experimented with RSI and EMA, include them initially:

RSI: 14-period (a common choice for short-term trading).
EMA: 12-period and 26-period (to capture short- and medium-term trends).


Condition: Retain these indicators unless they drastically reduce model performance (e.g., lower directional accuracy). You can test their impact during evaluation.
Sequence Length: Use a 60-minute lookback period (60 time steps) for each prediction. This means each input sequence contains the past 60 minutes of OHLC and indicator data.


4. Model Specifications

Architecture: Start with a simple LSTM model:

Input Layer: Accepts sequences of shape (60, number of features), where features include OHLC + RSI + EMA (e.g., 6 features if using two EMAs).
LSTM Layer: 50 units (a balanced starting point for capturing patterns).
Dropout Layer: 0.2 dropout rate to reduce overfitting.
Output Layer: Dense layer with sigmoid activation for binary classification (up/down).


Loss Function: Use binary cross-entropy, standard for binary classification.
Optimizer: Use Adam with a learning rate of 0.001 for efficient convergence.


5. Training Setup

Data Split: Split the data sequentially (since it’s time series):

Training: 70% (~42 days, first 60,480 data points).
Validation: 15% (~9 days, next 12,960 data points).
Test: 15% (~9 days, last 12,960 data points).


Batch Size: Set to 64 for a good balance between training speed and gradient stability.
Epochs: Train for 50 epochs, using early stopping (patience=5) to halt training if validation loss stops improving.


6. Evaluation Metrics

Primary Focus: Directional Accuracy—the percentage of correct predictions (up or down) for each horizon (3, 5, 8, 10 minutes).
Secondary Metrics: Track precision, recall, and F1-score to assess the model’s balance, especially if up/down movements are uneven in the data.
Evaluation Process: Test the model on the held-out test set and report accuracy for each prediction horizon.


7. Implementation Preferences

Tools: Use Python with TensorFlow/Keras for implementation due to their robust support for LSTM models.
Hardware: A standard CPU is sufficient for this initial setup, but a GPU will speed up training if available.
Workflow:

Load and preprocess the 1-minute OHLC data.
Compute RSI and EMA indicators.
Create 60-minute sequences with corresponding targets (up/down) for each horizon.
Scale features, split data, and train the LSTM models.
Evaluate performance on the test set.




8. Roadmap for Improvement
After evaluating the initial model, refine and enhance it with these steps:

Hyperparameter Tuning:

Adjust lookback period (e.g., 30, 90, 120 minutes).
Vary LSTM units (e.g., 100, 150) or dropout rates (e.g., 0.3, 0.4).


Feature Refinement:

Test additional indicators (e.g., MACD, Bollinger Bands) or remove RSI/EMA if they hurt accuracy.


Advanced Models:

Try stacked LSTMs (multiple layers) or alternatives like GRU or CNN-LSTM for potentially better performance.


Ensemble Approach:

Combine predictions from multiple models (e.g., one per horizon) to boost reliability.


Validation:

Use walk-forward validation to simulate real-time performance and assess robustness.


Retraining:

Periodically retrain the model with new data to adapt to evolving market conditions.


Backtesting:

Integrate predictions into a trading strategy and backtest on historical data to estimate profitability and risk.




Summary and Next Steps

Kick-Off: Build a simple LSTM model with a 60-minute lookback, using OHLC, RSI, and EMA as features. Train separate models for 3, 5, 8, and 10-minute horizons, focusing on directional accuracy.
Evaluate: Assess performance on the test set and check if RSI/EMA improve or hinder results.
Iterate: Refine the model based on accuracy, experimenting with hyperparameters, features, or architectures as needed.
Future: Once satisfied, backtest the predictions in a trading strategy to optimize for max-profit/min-risk.

This approach provides a practical starting point for your goal, balancing simplicity with flexibility for improvement. Let me know if you’d like a sample code snippet or further details on any step!