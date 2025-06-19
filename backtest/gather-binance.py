import ccxt
import pandas as pd
import time
from datetime import datetime

# Initialize Binance exchange
exchange = ccxt.binance({
    'enableRateLimit': True,  # Avoid hitting API rate limits
})

# Define assets and timeframe
# symbols = ['BTC/USDT', 'EUR/USDT']  # Use EUR/USDT or EUR/USDT-PERP for futures
# symbols = ['EUR/USDT']  # Use EUR/USDT or EUR/USDT-PERP for futures
# symbols = ['BTC/USDT']  # Use EUR/USDT or EUR/USDT-PERP for futures
symbols = ['CNY/USDT']
timeframe = '1m'  # 5-minute candles (adjust as needed: '1m', '5m', '1h', etc.)
since = exchange.parse8601('2025-06-01 00:00:00')  # Start date
limit = 1000  # Max candles per request
output_file = f'{symbols[0].replace("/", "_")}_ohlcv.csv'

# Function to fetch OHLCV data
def fetch_ohlcv(symbol, timeframe, since, limit):
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Update timestamp for next batch
            time.sleep(1)  # Respect rate limits
            if len(ohlcv) < limit:
                break
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            break
    return all_ohlcv

# Function to adjust prices for OTC differences
def adjust_for_otc(ohlcv_df, spread_multiplier=1.001, offset=0):
    """
    Adjust OHLCV prices to account for Pocket Option's OTC quotes.
    - spread_multiplier: Increase/decrease prices to simulate OTC spread (e.g., 1.001 for +0.1%).
    - offset: Add/subtract a fixed value to prices (e.g., 0.0001 for EUR/USD).
    """
    ohlcv_df['open'] = ohlcv_df['open'] * spread_multiplier + offset
    ohlcv_df['high'] = ohlcv_df['high'] * spread_multiplier + offset
    ohlcv_df['low'] = ohlcv_df['low'] * spread_multiplier + offset
    ohlcv_df['close'] = ohlcv_df['close'] * spread_multiplier + offset
    return ohlcv_df

# Fetch and process data for each symbol
all_data = []
for symbol in symbols:
    print(f"Fetching OHLCV data for {symbol}...")
    ohlcv = fetch_ohlcv(symbol, timeframe, since, limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['symbol'] = symbol
    
    # Adjust for OTC price differences (example values)
    # For EUR/USD, OTC spreads might be ~0.1-0.5% wider; for BTC/USDT, ~0.2-1%
    spread_multiplier = 1.001 if 'EUR' in symbol else 1.002
    offset = 0.0001 if 'EUR' in symbol else 10  # Example offset for EUR/USD or BTC/USDT
    spread_multiplier = 1
    offset = 0
    df = adjust_for_otc(df, spread_multiplier, offset)
    
    all_data.append(df)

# Combine data and save to CSV
combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")

# Display sample data
print(combined_df.head())