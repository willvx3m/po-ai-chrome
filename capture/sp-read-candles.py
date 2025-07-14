import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import argparse

IMAGE_WIDTH = 1580  # Width of the image
IMAGE_HEIGHT = 820  # Height of the image

# PRICE LABEL EXTRACTION CONSTANTS
IMAGE_RIGHT_MARGIN = 0.05  # 10% from the right
CONFIDENCE_THRESHOLD = 30  # Minimum confidence for OCR
PRICE_LABEL_LINE_MARGIN = 10  # Offset y-position downward
BG_COLOR_TOLERANCE = 20  # Tolerance for background color match

# TIME LABEL EXTRACTION CONSTANTS
WINDOW_WIDTH = 28  # Width of the time label window in pixels
WINDOW_HEIGHT = 20  # Height of the time label window in pixels
SEARCH_STEP = 2    # Step size to move the window rightward in pixels
LABEL_SPACING = 25  # Distance between consecutive time labels in pixels (Mac: IMAGE_WIDTH / 60, Win: 25)
TIME_INCREMENT = timedelta(minutes=1)  # Increment time by 1 minute
CONFIDENCE_THRESHOLD_LABEL = 85  # Confidence threshold for label correctness

# TESSERACT CONFIG
TIME_LABEL_TESSERACT_CONFIG = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789:'
PRICE_LABEL_TESSERACT_CONFIG = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789.'
TIME_LABEL_CHAR_LENGTH = 5

def extract_price_labels(img_rgb):
    """Extract price levels and their y-positions from the right side of the image, filtering for pure price labels based on the most frequent factor (text color, bg color, left pos, length)."""
    right_side = img_rgb[:, int(1 - IMAGE_RIGHT_MARGIN * img_rgb.shape[1]):]  # Take the right 10% of the image
    data = pytesseract.image_to_data(right_side, output_type=pytesseract.Output.DICT)
    price_data = []
    factor_groups = {}  # Dictionary to store (price, y_pos) lists by factor

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text and data['conf'][i] > CONFIDENCE_THRESHOLD:  # Confidence threshold
            try:
                # Extract bounding box coordinates
                left = data['left'][i]
                top = data['top'][i]
                width = data['width'][i]
                height = data['height'][i]
                right = left + width
                bottom = top + height

                # Sample background color (top-left corner outside the text)
                bg_color = img_rgb[top, left] if top > 0 and left > 0 else [0, 0, 0]  # Fallback to black

                # Define factor as a tuple of quantized attributes
                factor = (
                    tuple(int(c // BG_COLOR_TOLERANCE * BG_COLOR_TOLERANCE) for c in bg_color),  # Quantized bg color
                    int(left // 2),  # Quantized left position (x) in 10px increments
                    len(text)    # Label length
                )

                price = float(text.replace(',', '.'))
                y_pos = data['top'][i] + data['height'][i] // 2 + PRICE_LABEL_LINE_MARGIN  # Offset y-position

                # Add to factor group
                if factor not in factor_groups:
                    factor_groups[factor] = []
                factor_groups[factor].append((price, y_pos))
            except ValueError:
                continue

    # Find the factor group with the most matches
    if factor_groups:
        most_frequent_factor = max(factor_groups.items(), key=lambda x: len(x[1]))[0]
        print(f"Most frequent factor: {most_frequent_factor}, count: {len(factor_groups[most_frequent_factor])}")
        price_data = factor_groups[most_frequent_factor]
    else:
        print("No valid factor groups found.")

    price_data.sort(key=lambda x: x[1])  # Sort by y-position
    price_levels = [p[0] for p in price_data]
    y_positions = [p[1] for p in price_data]
    print(f"Extracted price levels with y-positions: {list(zip(price_levels, y_positions))}")
    return price_levels, y_positions

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

    if time_labels:
        print(f"Extracted time labels: {[[t[0], t[1].strftime('%H:%M')] for t in time_labels]}")

    return time_labels

def extract_candle_array(img_rgb, price_levels, price_y_positions, time_steps, draw_flag=False):
    """Extract candle array with OHLCV as position data (x, y-coordinates) using contours and wicks, optionally drawing extracted areas."""
    # Convert to HSV for color-based segmentation
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask_green, mask_red)

    # Detect candle bodies using contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Detected {len(contours)} candle body contours.")

    # Edge detection for wicks to determine full range
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=5, maxLineGap=5)
    print(f"Detected {len(lines) if lines is not None else 0} wick lines.")

    # Calculate pixel_to_price
    pixel_to_price = (max(price_levels) - min(price_levels)) / (max(price_y_positions) - min(price_y_positions))
    base_price = max(price_levels) + (min(price_y_positions) * pixel_to_price)

    candle_array = []
    seen_x = set()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 5 and x not in seen_x:  # Filter noise, ensure width > 20px
            high_y = y  # Top of bounding rect as initial high (highest price)
            low_y = y + h  # Bottom of bounding rect as initial low (lowest price)
            open_y = high_y
            close_y = low_y

            # Find time label
            time_label = None
            for time_step in time_steps:
                if time_step[0] >= x and time_step[0] <= x + w:
                    time_label = time_step[1]
                    break
            if not time_label:
                continue

            # Adjust margins for wick data
            if lines is not None:
                closest_wick = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) < 5:  # Vertical line (wick)
                        mid_x = (x1 + x2) / 2
                        dist = abs(mid_x - (x + w // 2))
                        wick = [min(y1, y2), max(y1, y2)]
                        intercept = (wick[0] <= open_y and open_y <= wick[1]) or (wick[0] <= close_y and close_y <= wick[1])
                        if dist < 5 and intercept:
                            wick[0] = max(wick[0], high_y)
                            wick[1] = min(wick[1], low_y)
                            closest_wick.append(wick)
                if len(closest_wick) > 1:
                    for wick in closest_wick:
                        if wick[0] - high_y < 2 and low_y - wick[1] > 2:
                            open_y = wick[1]
                        if low_y - wick[1] < 2 and wick[0] - high_y > 2:
                            close_y = wick[0]

            # Calculate price from y-position
            open_price = base_price - (open_y * pixel_to_price)
            close_price = base_price - (close_y * pixel_to_price)
            high_price = base_price - (high_y * pixel_to_price)
            low_price = base_price - (low_y * pixel_to_price)

            # Include x-position in the candle array
            candle_array.append([x + w // 2, open_y, close_y, high_y, low_y, open_price, close_price, high_price, low_price, time_label])
            seen_x.add(x)

    # Sort candle_array by x-position to ensure chronological order
    candle_array = sorted(candle_array, key=lambda c: c[0])

    if draw_flag and candle_array:
        for x, open_y, close_y, high_y, low_y, open_price, close_price, high_price, low_price, time_label in candle_array:
            w = w if 'w' in locals() else 20  # Use contour width or approximate
            # Convert back to image coordinates for drawing (y decreases upward in display)
            cv2.rectangle(img_rgb, (x - w // 2, low_y), (x + w // 2, high_y), (255, 255, 0), 2)  # Yellow for High/Low
            cv2.rectangle(img_rgb, (x - w // 2, open_y), (x + w // 2, close_y), (255, 0, 255), 2)  # Blue for Open/Close
        plt.figure(figsize=(10, 5))
        plt.imshow(img_rgb)
        plt.title('Blue: Open/Close Bodies, Yellow: High/Low Full Range')
        plt.show()

    return candle_array

def main(image_path, draw_overlay, draw_chart):
    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file {image_path} not found or unreadable.") 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract data
    price_levels, y_positions = extract_price_labels(img_rgb)
    if not price_levels:
        print("No price levels detected. Exiting.")
        return

    time_steps = extract_time_labels(img_rgb)
    if not time_steps:
        print("No valid time labels found. Exiting.")
        return

    candle_array = extract_candle_array(img_rgb, price_levels, y_positions, time_steps, draw_flag=draw_overlay)
    if not candle_array:
        print("No candle data detected. Exiting.")
        return

    # Optional: Draw final chart directly from candle_array
    if draw_chart and candle_array:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        x_positions = [c[0] for c in candle_array]
        if x_positions:
            min_x = min(x_positions)
            max_x = max(x_positions)
            time_range = max_x - min_x if max_x > min_x else 1  # Avoid division by zero

            for i, (x, open_y, close_y, high_y, low_y, open_price, close_price, high_price, low_price, time_label) in enumerate(candle_array):
                # Map x-position to time_steps using interpolation
                time_pos = i
                # if time_range > 0:
                #     time_pos = (x - min_x) * len(candle_array) / time_range
                # else:
                #     time_pos = x # Fallback to index if range is zero

                # Draw wicks
                ax.plot([time_pos, time_pos], [low_price, high_price], color='k', linewidth=1)  # Wicks
                # ax.plot([time_pos - 0.2, time_pos + 0.2], [open_price, open_price], color='k', linewidth=2)  # Open
                # ax.plot([time_pos - 0.2, time_pos + 0.2], [close_price, close_price], color='k', linewidth=2)  # Close

                # Determine bullish/bearish based on y-coordinates (higher y = lower price due to inversion)
                if open_price <= close_price:  # Bullish if open is higher (lower y) than close
                    color = 'g'
                    body = plt.Rectangle((time_pos - 0.2, min(open_price, close_price)), 0.4, abs(close_price - open_price), facecolor=color)
                else:  # Bearish if open is lower (higher y) than close
                    color = 'r'
                    body = plt.Rectangle((time_pos - 0.2, min(open_price, close_price)), 0.4, abs(close_price - open_price), facecolor=color)
                ax.add_patch(body)

        ax.set_xlim(- 1, len(candle_array) + 1)
        min_price = min(c[6] for c in candle_array)
        max_price = max(c[7] for c in candle_array)
        ax.set_ylim(min_price - (max_price - min_price) * 0.05, max_price + (max_price - min_price) * 0.05)  # Use high_y and low_y for y-limits
        ax.set_title('Candlestick Chart from Extracted Data')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        plt.xticks([t for t in range(len(candle_array))], [c[9].strftime('%H:%M') for c in candle_array], rotation=45)
        plt.grid(True)
        plt.show()

    final_package = []
    for i, (x, open_y, close_y, high_y, low_y, open_price, close_price, high_price, low_price, time_label) in enumerate(candle_array):
        final_package.append({
            'time_label': time_label.strftime('%H:%M'),
            'x': x,
            'open': open_price,
            'close': close_price,
            'high': high_price,
            'low': low_price,
        })
        print(f"Candle {i + 1} at {time_label.strftime('%H:%M')}: x = {x}, Open = {open_price}, Close = {close_price}, "
                f"High = {high_price}, Low = {low_price}")

    return final_package

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read candles from a screenshot.')
    parser.add_argument('--path', type=str, help='Path to the image file')
    parser.add_argument('--draw-overlay', type=bool, default=False, help='Draw overlay on the image')
    parser.add_argument('--draw-chart', type=bool, default=False, help='Draw chart')
    parser.add_argument('--save-json', type=bool, default=True, help='Save to json')
    args = parser.parse_args()

    if not args.path:
        print("No image path provided. Exiting.")
        exit()

    result = main(args.path, args.draw_overlay, args.draw_chart)
    if args.save_json:
        filename = args.path.split('/')[-1].split('.')[0]
        with open(f'json/{filename}.json', 'w') as f:
            json.dump(result, f)