import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import argparse
import shutil
from pathlib import Path
import os
import time

IMAGE_WIDTH = 1580  # Width of the image
IMAGE_HEIGHT = 820  # Height of the image

# TIME LABEL EXTRACTION CONSTANTS
WINDOW_WIDTH = 32  # Width of the time label window in pixels
WINDOW_HEIGHT = 20  # Height of the time label window in pixels
SEARCH_STEP = 2    # Step size to move the window rightward in pixels
LABEL_SPACING = 26.4  # Distance between consecutive time labels in pixels
# 25 for aed/cny - 1H view, 4m interval on time labels
# 24.425 for eur/usd - 1H view, 4m interval on time labels
# 26.4 (1056/40) - 1H view (normal)
TIME_INCREMENT = timedelta(minutes=1)  # Increment time by 1 minute
CONFIDENCE_THRESHOLD_LABEL = 75  # Confidence threshold for label correctness
TIME_LABEL_CHAR_LENGTH = 5

# DATE LABEL EXTRACTION CONSTANTS
DATE_LABEL_WINDOW_WIDTH = 100  # Width of the date label window in pixels
DATE_LABEL_WINDOW_HEIGHT = 20  # Height of the date label window in pixels
DATE_LABEL_SEARCH_STEP = 6    # Step size to move the window rightward in pixels
DATE_LABEL_CONFIDENCE_THRESHOLD = 0  # Confidence threshold for date label correctness
DATE_LABEL_CHAR_LENGTH = 10

# TESSERACT CONFIG
TIME_LABEL_TESSERACT_CONFIG = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789:'

def extract_date(img_rgb):
    """Extract date from the bottom of the image by searching for DD.MM.YYYY HH:MM"""
    # Extract image dimensions
    height, width = img_rgb.shape[:2]
    bottom_region_y = height - DATE_LABEL_WINDOW_HEIGHT  # Start from the bottom

    # Search for the first valid date label
    for x in range(0, width - DATE_LABEL_WINDOW_WIDTH + 1, DATE_LABEL_SEARCH_STEP):
        # Define the search window
        window = img_rgb[bottom_region_y:bottom_region_y + DATE_LABEL_WINDOW_HEIGHT, x:x + DATE_LABEL_WINDOW_WIDTH]
        # print(f"Window: {window.shape}, x: {x}")
        data = pytesseract.image_to_data(window, output_type=pytesseract.Output.DICT)
        # text = pytesseract.image_to_string(window, config=TIME_LABEL_TESSERACT_CONFIG).strip()
        # print(f"Text: {text}")

        for i in range(len(data['text'])):
            # print(f"Text: {data['text'][i]}")
            # print(f"Confidence: {data['conf'][i]}")
            # print(f"Left: {data['left'][i]}")
            # print(f"Top: {data['top'][i]}")
            # print(f"Width: {data['width'][i]}")
            # print(f"Height: {data['height'][i]}")

            if data['conf'][i] < DATE_LABEL_CONFIDENCE_THRESHOLD:
                continue

            text = data['text'][i][:DATE_LABEL_CHAR_LENGTH]

            # Validate DD.MM.YYYY HH:MM format
            if len(text) == DATE_LABEL_CHAR_LENGTH and '.' in text:
                try:
                    day, month, year = map(int, text.split('.'))
                    if 0 <= day <= 31 and 0 <= month <= 12 and 0 <= year <= 9999:
                        print(f"First valid date label found at x={x}: {text} {day}.{month}.{year}")
                        return [day, month, year]
                except ValueError:
                    continue
    return None


def extract_time_labels(img_rgb):
    """Extract time labels from the bottom of the image by searching for the first valid HH:MM label and populating an array based on x-position and time increment."""
    # Extract image dimensions
    height, width = img_rgb.shape[:2]
    bottom_region_y = height - WINDOW_HEIGHT  # Start from the bottom

    # Search for the first valid time label
    first_label = None
    for x in range(0, width - WINDOW_WIDTH + 1, SEARCH_STEP):
        # Define the search window
        window = img_rgb[bottom_region_y:bottom_region_y + WINDOW_HEIGHT, x:x + WINDOW_WIDTH]
        # print(f"Window: {window.shape}, x: {x}")
        data = pytesseract.image_to_data(window, output_type=pytesseract.Output.DICT)
        # text = pytesseract.image_to_string(window, config=TIME_LABEL_TESSERACT_CONFIG).strip()
        # print(f"Text: {text}")

        for i in range(len(data['text'])):
            # print(f"Text: {data['text'][i]}")
            # print(f"Confidence: {data['conf'][i]}")
            # print(f"Left: {data['left'][i]}")
            # print(f"Top: {data['top'][i]}")
            # print(f"Width: {data['width'][i]}")
            # print(f"Height: {data['height'][i]}")

            if data['conf'][i] < CONFIDENCE_THRESHOLD_LABEL:
                continue

            text = data['text'][i]

            # Validate HH:MM format
            if len(text) == TIME_LABEL_CHAR_LENGTH and ':' in text:
                try:
                    hour, minute = map(int, text.split(':'))
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        first_label = (x, text)  # (left_x, time_text)
                        print(f"First valid time label found at x={x}: {text}")
                        break
                except ValueError:
                    continue
        
        if first_label:
            break

    # Populate time label array
    time_labels = []
    if first_label:
        start_x, start_time_text = first_label
        start_time = datetime.strptime(start_time_text, '%H:%M').time()
        start_full = datetime.combine(datetime.today(), start_time)

        # Roll back to the first label that fits within the viewport
        for i in range(60, -1, -1):  # Include the first label at i=0
            if start_x > LABEL_SPACING:
                start_x -= LABEL_SPACING
                start_full -= TIME_INCREMENT
                start_time = start_full.time()
            else:
                break

        # Calculate the number of labels based on image width
        max_x = width - WINDOW_WIDTH
        num_labels = min((max_x - start_x) // LABEL_SPACING + 1, width // LABEL_SPACING)  # Ensure within image bounds

        # Generate time labels
        for i in range(int(num_labels)):
            x_pos = start_x + i * LABEL_SPACING
            if x_pos + WINDOW_WIDTH <= width:  # Stay within image bounds
                current_time = start_full + i * TIME_INCREMENT
                time_labels.append([int(x_pos + WINDOW_WIDTH // 2), current_time.time()])

    # if time_labels:
    #     print(f"Extracted time labels: {[[t[0], t[1].strftime('%H:%M')] for t in time_labels]}")

    return time_labels
    
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

def process_file(file_path, processed_dir, destination_dir):
    print(f"Processing file: {file_path}")
    img = cv2.imread(file_path)
    if img is None:
        print(f"Image file {file_path} not found or unreadable.") 
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    time_steps = extract_time_labels(img_rgb)
    # print(time_steps)

    date = extract_date(img_rgb)

    if date is None:
        # date = [13, 7, 2025] # for aed/cny only, GET THIS FROM THE FILE NAME
        date_part = file_path.split('/')[-1].split('_')[0]
        if len(date_part) == 8:
            date = [int(date_part[6:8]), int(date_part[4:6]), int(date_part[0:4])]
        else:
            print("File does not have valid date format.")
            return

    print(date)

    if time_steps is None or len(time_steps) == 0 or date is None:
        print("No time steps or date found.")
        return

    first_time_step = time_steps[0]
    last_time_step = time_steps[-1]
    datarange = f"{date[2]:4d}.{date[1]:02d}.{date[0]:02d}-{first_time_step[1].strftime('%H.%M')}-{last_time_step[1].strftime('%H.%M')}"
    print(f"Data range: {datarange}")

    if destination_dir:
        copy_file(file_path, f"{destination_dir}/{datarange}.png")
        time.sleep(3)
    if processed_dir:
        filename = file_path.split('/')[-1]
        move_file(file_path, f"{processed_dir}/{filename}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read candles from a screenshot.')
    parser.add_argument('--source-dir', type=str, help='Source directory')
    parser.add_argument('--processed-dir', type=str, help='Processed directory')
    parser.add_argument('--destination-dir', type=str, help='Destination directory')
    parser.add_argument('--path', type=str, help='Path to the image file')
    args = parser.parse_args()

    if not args.path and not args.source_dir:
        print("No image path or source directory provided. Exiting.")
        exit()

    # dynamic LABEL_SPACING
    # if (args.source_dir is not None and '1h' in args.source_dir) or (args.path is not None and '1h' in args.path):
    #     LABEL_SPACING = 26.4
    # elif (args.source_dir is not None and 'aedcny' in args.source_dir) or (args.path is not None and 'aedcny' in args.path):
    #     LABEL_SPACING = 25
    # else:
    #     LABEL_SPACING = 24.425

    # print(f"LABEL_SPACING: {LABEL_SPACING}")

    all_files = []

    if args.source_dir:
        for file in os.listdir(args.source_dir):
            if file.endswith('.png'):
                all_files.append(os.path.join(args.source_dir, file))

    if args.path:
        all_files.append(args.path)

    if not all_files:
        print("No files found in source directory. Exiting.")
        exit()

    for file_path in all_files:
        process_file(file_path, args.processed_dir, args.destination_dir)


# Running command:
# python sp-read-daterange.py --source-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-aedcny --processed-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-aedcny-processed --destination-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-aedcny-final

# python sp-read-daterange.py --source-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/1 --processed-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/processed --destination-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/final
# python sp-read-daterange.py --source-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/2 --processed-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/processed --destination-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/final
# python sp-read-daterange.py --source-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/3 --processed-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/processed --destination-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-eurusd/final

# python sp-read-daterange.py --source-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/1 --processed-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/processed --destination-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/final
# python sp-read-daterange.py --source-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/2 --processed-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/processed --destination-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/final
# python sp-read-daterange.py --source-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/3 --processed-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/processed --destination-dir /Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots-chfjpy/final