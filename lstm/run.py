import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class PricePredictor:
    def __init__(self, data_path, horizons=[3, 5, 8, 10], lookback=60):
        """Initialize the PricePredictor with data path, prediction horizons, and lookback period."""
        self.data_path = data_path
        self.horizons = horizons
        self.lookback = lookback
        self.df = None
        self.scaler = MinMaxScaler()
        self.models = {}
        self.sequences = None
        self.targets = {h: [] for h in horizons}
        self.train_X = None
        self.val_X = None
        self.test_X = None
        self.train_y = {h: [] for h in horizons}
        self.val_y = {h: [] for h in horizons}
        self.test_y = {h: [] for h in horizons}
        self.features = ['open', 'high', 'low', 'close', 'RSI_14', 'EMA_12', 'EMA_26']

    def load_data(self):
        """Load and sort the JSON OHLC data."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        self.df = pd.DataFrame(data)
        self.df['datetime_point'] = pd.to_datetime(self.df['datetime_point'])
        self.df = self.df.sort_values('datetime_point').reset_index(drop=True)

    def add_technical_indicators(self):
        """Add RSI and EMA indicators to the DataFrame."""
        def calculate_rsi(data, window=14):
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        self.df['RSI_14'] = calculate_rsi(self.df)
        self.df['EMA_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['EMA_26'] = self.df['close'].ewm(span=26, adjust=False).mean()

    def preprocess(self):
        """Preprocess data: load, add indicators, scale, and create sequences."""
        self.load_data()
        self.add_technical_indicators()
        
        # Define data splits
        N = len(self.df)
        train_end = int(0.7 * N)
        val_end = int(0.85 * N)
        
        # Scale features using training data
        train_data = self.df.iloc[:train_end]
        self.scaler.fit(train_data[self.features])
        scaled_data = self.scaler.transform(self.df[self.features])
        
        # Create sequences
        max_h = max(self.horizons)
        num_sequences = N - self.lookback - max_h + 1
        self.sequences = np.array([scaled_data[i:i + self.lookback, :] for i in range(num_sequences)])
        
        # Create targets for each horizon
        for h in self.horizons:
            self.targets[h] = [
                1 if self.df['close'][i + self.lookback - 1 + h] > self.df['close'][i + self.lookback - 1] else 0
                for i in range(num_sequences)
            ]
        
        # Split sequences and targets
        num_train = int(0.7 * num_sequences)
        num_val = int(0.15 * num_sequences)
        self.train_X = self.sequences[:num_train]
        self.val_X = self.sequences[num_train:num_train + num_val]
        self.test_X = self.sequences[num_train + num_val:]
        
        for h in self.horizons:
            self.train_y[h] = self.targets[h][:num_train]
            self.val_y[h] = self.targets[h][num_train:num_train + num_val]
            self.test_y[h] = self.targets[h][num_train + num_val:]

    def build_model(self):
        """Build and compile the LSTM model."""
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.lookback, len(self.features))))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def train_models(self):
        """Train an LSTM model for each prediction horizon."""
        for h in self.horizons:
            model = self.build_model()
            train_y_h = np.array(self.train_y[h])
            val_y_h = np.array(self.val_y[h])
            model.fit(
                self.train_X, train_y_h,
                validation_data=(self.val_X, val_y_h),
                epochs=50, batch_size=64,
                callbacks=[EarlyStopping(patience=5)],
                verbose=1
            )
            self.models[h] = model

    def evaluate_models(self):
        """Evaluate each model on the test set and print accuracy."""
        for h in self.horizons:
            test_y_h = np.array(self.test_y[h])
            loss, accuracy = self.models[h].evaluate(self.test_X, test_y_h, verbose=0)
            print(f"Horizon {h} minutes: Test Directional Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    # Example usage
    predictor = PricePredictor('../backtest/module/eurusd-full.json')
    predictor.preprocess()
    predictor.train_models()
    predictor.evaluate_models()