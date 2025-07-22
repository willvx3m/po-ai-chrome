import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import argparse
import os
import shutil
from pathlib import Path

def process_file(file_path, save_image_path):
    result = json.load(open(file_path, 'r'))
    print(f"File: {file_path}, Candle Length: {len(result)}")

    # for item in result:
    #     print(item)

    if len(result) <= 10:
        print("Less than 10 candles detected. Likely a failed attempt to read the chart.")
        return
    
    candle_array = result

    # Optional: Draw final chart directly from candle_array
    if candle_array:
        fig, ax = plt.subplots(figsize=(20, 12), label=file_path.split('/')[-1])
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        x_positions = [c['x'] for c in candle_array]
        if x_positions:
            min_x = min(x_positions)
            max_x = max(x_positions)
            time_range = max_x - min_x if max_x > min_x else 1  # Avoid division by zero

            for i, c in enumerate(candle_array):
                # Map x-position to time_steps using interpolation
                time_pos = i
                # if time_range > 0:
                #     time_pos = (x - min_x) * len(candle_array) / time_range
                # else:
                #     time_pos = x # Fallback to index if range is zero

                # Draw wicks
                ax.plot([time_pos, time_pos], [c['low'], c['high']], color='k', linewidth=1)  # Wicks
                # ax.plot([time_pos - 0.2, time_pos + 0.2], [open_price, open_price], color='k', linewidth=2)  # Open
                # ax.plot([time_pos - 0.2, time_pos + 0.2], [close_price, close_price], color='k', linewidth=2)  # Close

                # Determine bullish/bearish based on y-coordinates (higher y = lower price due to inversion)
                if c['open'] <= c['close']:  # Bullish if open is higher (lower y) than close
                    color = 'g'
                    body = plt.Rectangle((time_pos - 0.2, min(c['open'], c['close'])), 0.4, abs(c['close'] - c['open']), facecolor=color)
                else:  # Bearish if open is lower (higher y) than close
                    color = 'r'
                    body = plt.Rectangle((time_pos - 0.2, min(c['open'], c['close'])), 0.4, abs(c['close'] - c['open']), facecolor=color)
                ax.add_patch(body)

        ax.set_xlim(- 1, len(candle_array) + 1)
        min_price = min(c['low'] for c in candle_array)
        max_price = max(c['high'] for c in candle_array)
        ax.set_ylim(min_price - (max_price - min_price) * 0.05, max_price + (max_price - min_price) * 0.05)  # Use high_y and low_y for y-limits
        ax.set_title('Candlestick Chart from Extracted Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        plt.xticks([t for t in range(len(candle_array))], [c['time_label'] for c in candle_array], rotation=45)
        plt.grid(True)
        plt.savefig(save_image_path)
        # plt.show()
        plt.close()

def copy_file(source_path, destination_path):
    try:
        # Convert to Path objects to handle Windows paths
        src = Path(source_path)
        dst = Path(destination_path)
        
        # Copy the file
        shutil.copy(src, dst)
        print(f"Copied {src} to {dst}")
    except FileNotFoundError:
        print(f"Error: Source file {src} not found")
    except PermissionError:
        print(f"Error: Permission denied for {src} or {dst}")
    except Exception as e:
        print(f"Error: {e}")

def move_file(source_path, destination_path):
    try:
        # Convert to Path objects to handle Windows paths
        src = Path(source_path)
        dst = Path(destination_path)
        
        # Move the file
        shutil.move(src, dst)
        print(f"Moved {src} to {dst}")
    except FileNotFoundError:
        print(f"Error: Source file {src} not found")
    except PermissionError:
        print(f"Error: Permission denied for {src} or {dst}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw charts from json files.')
    parser.add_argument('--path', type=str, help='Path to the json file')
    parser.add_argument('--source-dir', type=str, help='Source directory')
    parser.add_argument('--destination-dir', type=str, help='Destination directory')
    parser.add_argument('--draw-chart', type=bool, default=False, help='Draw chart')
    args = parser.parse_args()

    if not args.path and not args.source_dir:
        print("No json path or source directory provided. Exiting.")
        exit()

    all_files = []

    if args.source_dir:
        for file in os.listdir(args.source_dir):
            if file.endswith('.json'):
                all_files.append(os.path.join(args.source_dir, file))

    if args.path:
        all_files.append(args.path)

    if not all_files:
        print("No files found in source directory. Exiting.")
        exit()

    for file_path in all_files:
        result = json.load(open(file_path, 'r'))
        # for item in result:
        #     print(item)
        process_file(file_path, f"{args.destination_dir}/{file_path.split('/')[-1]}.png")

# python sp-draw-charts.py --source-dir {SOURCE_DIR} --destination-dir {DESTINATION_DIR}
# python sp-draw-charts.py --path {JSON_PATH} --destination-dir {DESTINATION_DIR}