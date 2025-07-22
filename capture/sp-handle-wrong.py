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
    parser = argparse.ArgumentParser(description='Handle wrong json files.')
    parser.add_argument('--source-dir', type=str, help='Source json directory')
    parser.add_argument('--image-dir', type=str, help='Source image directory')
    args = parser.parse_args()

    if not args.source_dir or not args.image_dir:
        print("No source or image directory provided. Exiting.")
        exit()

    all_json_files = []
    all_image_files = []

    for file in os.listdir(args.source_dir):
        if file.endswith('.json'):
            all_json_files.append(os.path.join(args.source_dir, file))

    for file in os.listdir(args.image_dir):
        if file.endswith('.png'):
            all_image_files.append(os.path.join(args.image_dir, file))

    if not all_json_files or not all_image_files:
        print("No files found in source directory. Exiting.")
        exit()

    images_to_handle = []

    for json_file in all_json_files:
        found_image = False
        image_name = json_file.split('/')[-1].replace('_wrong.json', '')
        # print(f"Image name: {image_name}")
        for image_file in all_image_files:
            if image_name in image_file:
                # print(f"Image file found: {image_file}")
                images_to_handle.append(image_file)
                found_image = True
                break
        if not found_image:
            print(f"Image file not found: {image_name}")
            exit()
    
    if len(images_to_handle) != len(all_json_files):
        print(f"Number of images to handle: {len(images_to_handle)} does not match number of json files: {len(all_json_files)}")
        exit()

    for image_file in images_to_handle:
        print(f"Image file: {image_file}")
        file_name = image_file.split('/')[-1].replace('.png', '')
        copy_file(image_file, f"{args.destination_dir}/{file_name}.png")