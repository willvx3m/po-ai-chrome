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

def change_candles(candles, type, value, direction):
    return candles

def increment_date(year, month, day, change):
    if "+" in change:
        day = int(day) + int(change.replace("+", "").replace("d", ""))
    elif "-" in change:
        day = int(day) - int(change.replace("-", "").replace("d", ""))
    odd_months = [1, 3, 5, 7, 8, 10]
    even_months = [4, 6, 9, 11]
    if month in odd_months and day > 31:
        month = month + 1
        day = day - 31
    elif month in even_months and day > 30:
        month = month + 1
        day = day - 30
    elif month == 12 and day > 31:
        year = year + 1
        month = 1
        day = day - 31
    elif month == 2 and day > 28:
        month = 3
        day = day - 28
    return year, month, day

def process_file(file_path, destination_dir):
    result = json.load(open(file_path, 'r'))
    print(f"File: {file_path}, Candle Length: {len(result)}")

    file_name = file_path.split('/')[-1].replace('.json', '')
    [year, month, day] = file_name.split('-')[0].split('.')
    # print(f"=> Default date: {year}.{month:02d}.{day:02d}")
    changes_needed = file_name.split('_')[1:]
    stacked_time_adjustments = []
    stacked_price_adjustments = []

    for change in changes_needed:
        if "m" in change:
            if "+" in change:
                # print(f"=> Add {int(change.replace("+", "").replace("m", ""))} minutes")
                stacked_time_adjustments.append(timedelta(minutes=int(change.replace("+", "").replace("m", ""))))
            elif "-" in change:
                # print(f"=> Subtract {int(change.replace("-", "").replace("m", ""))} minutes")
                stacked_time_adjustments.append(timedelta(minutes=-int(change.replace("-", "").replace("m", ""))))
            else:
                print(f"=> Invalid")
                exit()
        elif "h" in change:
            if "+" in change:
                # print(f"=> Add {int(change.replace("+", "").replace("h", ""))} hours")
                stacked_time_adjustments.append(timedelta(hours=int(change.replace("+", "").replace("h", ""))))
            elif "-" in change:
                # print(f"=> Subtract {int(change.replace("-", "").replace("h", ""))} hours")
                stacked_time_adjustments.append(timedelta(hours=-int(change.replace("-", "").replace("h", ""))))
            else:
                print(f"=> Invalid")
                exit()
        elif "p" in change:
            if "+" in change:
                # print(f"=> Add {int(change.replace("+", "").replace("p", ""))} price points")
                stacked_price_adjustments.append(int(change.replace("+", "").replace("p", "")))
            elif "-" in change:
                # print(f"=> Subtract {int(change.replace("-", "").replace("p", ""))} price points")
                stacked_price_adjustments.append(-int(change.replace("-", "").replace("p", "")))
            else:
                print(f"=> Invalid")
                exit()
        elif "d" in change:
            if "+" in change:
                # print(f"=> Add {int(change.replace("+", "").replace("d", ""))} days")
                stacked_time_adjustments.append(timedelta(days=int(change.replace("+", "").replace("d", ""))))
            elif "-" in change:
                # print(f"=> Subtract {int(change.replace("-", "").replace("d", ""))} days")
                stacked_time_adjustments.append(timedelta(days=-int(change.replace("-", "").replace("d", ""))))
            else:
                print(f"=> Invalid")
                exit()
        elif len(change) == 8:
            year = int(change[:4])
            month = int(change[4:6])
            day = int(change[6:])
            # print(f"=> New date: {year}.{month:02d}.{day:02d}")
        else:
            print(f"=> Invalid")
            exit()

    for index, candle in enumerate(result):
        candle_time = datetime(year=int(year), month=int(month), day=int(day), hour=int(candle["time_label"][:2]), minute=int(candle["time_label"][3:5]))
        for time_adjustment in stacked_time_adjustments:
            candle_time = candle_time + time_adjustment
        for price_adjustment in stacked_price_adjustments:
            candle['open'] = candle['open'] + price_adjustment
            candle['high'] = candle['high'] + price_adjustment
            candle['low'] = candle['low'] + price_adjustment
            candle['close'] = candle['close'] + price_adjustment
        result[index] = candle
        candle["time_label"] = candle_time.strftime("%H:%M")
        if index == len(result) - 1:
            year = candle_time.year
            month = candle_time.month
            day = candle_time.day
            # print(f"=> Final date: {year}.{month:02d}.{day:02d}")

    if destination_dir:
        start_label = result[0]["time_label"]
        end_label = result[-1]["time_label"]
        save_path = f"{destination_dir}/{year}.{month:02d}.{day:02d}-{start_label}-{end_label}.png.json"
        with open(save_path, 'w') as f:
            json.dump(result, f)

        print(f"=> Saved to {save_path}")

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
    parser = argparse.ArgumentParser(description='Merge json files.')
    parser.add_argument('--source-dir', type=str, help='Source directory')
    args = parser.parse_args()

    if not args.source_dir:
        print("No source directory provided. Exiting.")
        exit()

    all_files = []

    if args.source_dir:
        for file in os.listdir(args.source_dir):
            if file.endswith('.json'):
                all_files.append(os.path.join(args.source_dir, file))

    if not all_files:
        print("No files found in source directory. Exiting.")
        exit()

    all_files.sort()

    merged_result = []
    unique_points = []

    for file_path in all_files:
        result = json.load(open(file_path, 'r'))
        print(f"File: {file_path}, Candle Length: {len(result)}")

        date = datetime.strptime(file_path.split('/')[-1].split('-')[0], '%Y.%m.%d')
        first_time = result[0]['time_label']
        last_time = result[-1]['time_label']
        first_date = date
        if '23:' in first_time and '00:' in last_time:
            first_date = (date + timedelta(days=-1))
            print(f"First date: {first_date.strftime('%Y-%m-%d')}, Original date: {date.strftime('%Y-%m-%d')}, File: {file_path}")
        for candle in result:
            candle_date = first_date if '23:' in candle['time_label'] else date
            candle_time = candle['time_label']
            datetime_point = f"{candle_date.strftime('%Y-%m-%d')} {candle_time}"
            candle['date'] = candle_date.strftime("%Y-%m-%d")
            candle['datetime_point'] = datetime_point
            if datetime_point not in unique_points:
                unique_points.append(datetime_point)
                merged_result.append(candle)

    with open('merged.json', 'w') as f:
        json.dump(merged_result, f)

    print(f"Total unique points: {len(unique_points)}")

# python sp-merge.py --source-dir {SOURCE_DIR}