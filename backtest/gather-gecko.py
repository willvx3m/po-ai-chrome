import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Define the API endpoint
# network = 'base'
# pool_address = '0x7b2c99188d8ec7b82d6b3b3b1c1002095f1b8498'  # EURC/USDC pool
# pair = 'eurc_usdc'
network = 'solana'
pool_address = 'CsPcrAP4gL2JGJxRBLvw4fTBkGdpRDqiSSCfJ8WcTiUk'  # CHF/USDC pool
pair = 'chf_usdc'
timeframe = 'minute'
aggregate = 1  # 1-minute candles

# Set the end timestamp (current time: June 19, 2025, 10:20 PM CST)
current_time = datetime(2025, 6, 19, 22, 20)
before_timestamp = int(current_time.timestamp())  # Convert to Unix seconds

# Calculate the start timestamp (1 month ago)
start_time = current_time - timedelta(days=30)
start_timestamp = int(start_time.timestamp())

# Initialize lists to store all OHLCV data
all_ohlcv = []

# Set headers
headers = {'accept': 'application/json'}

try:
    # Paginate through data until we reach the start timestamp
    while True:
        # Construct the API URL with before_timestamp
        url = f'https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}?aggregate={aggregate}&before_timestamp={before_timestamp}&limit=1000'
        
        # Make the API request
        response = requests.get(url, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract OHLCV data
            ohlcv_list = data['data']['attributes']['ohlcv_list']
            
            if not ohlcv_list:
                print("No more data available.")
                break
                
            # Append to the main list
            all_ohlcv.extend(ohlcv_list)
            
            # Update before_timestamp to the earliest timestamp in the current batch
            earliest_timestamp = ohlcv_list[-1][0]
            if earliest_timestamp <= start_timestamp:
                print("Reached data before the start timestamp.")
                break
                
            before_timestamp = earliest_timestamp
            
            # Respect rate limits (30 calls/minute ~ 2 calls/second)
            time.sleep(0.5)
            
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)
            break
            
    # Convert to pandas DataFrame
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        
        # Filter data within the desired 1-month range
        df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= current_time)]
        
        # Set timestamp as index
        df.set_index('Timestamp', inplace=True)
        
        # Sort by timestamp (ascending)
        df.sort_index(inplace=True)
        
        # Display the first few rows
        print(df.head())
        print(f"\nTotal rows: {len(df)}")
        
        # Save to CSV
        df.to_csv(f'gecko-{pair}_ohlcv_1m.csv')
        print(f"Data saved to gecko-{pair}_ohlcv_1m.csv")
        
    else:
        print("No data retrieved.")
        
except Exception as e:
    print(f"An error occurred: {e}")