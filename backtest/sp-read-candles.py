import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

# Configuration
show_original = True  # Flag to toggle original image display

# Load the image
image_path = '/Users/million/Downloads/aaa.png'  # Replace with your image path
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image file {image_path} not found or unreadable.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to HSV for color-based segmentation
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

# Define color ranges for green and red candles (adjust as needed)
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Create masks
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask = cv2.bitwise_or(mask_green, mask_red)

# Find contours for candle bodies (Open/Close)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Detected {len(contours)} candle body contours.")

# Edge detection for wicks (to determine full candle range)
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
print(f"Detected {len(lines) if lines is not None else 0} wick lines.")

# Extract price levels and their y-positions from the right side using OCR
right_side = img_rgb[:, int(0.9 * img.shape[1]):]  # Take the right 10% of the image
# Use image_to_data for bounding box info
data = pytesseract.image_to_data(right_side, output_type=pytesseract.Output.DICT)
price_data = []
for i in range(len(data['text'])):
    if data['text'][i].strip() and data['conf'][i] > 50:  # Confidence threshold
        try:
            price = float(data['text'][i].replace(',', '.'))
            y_pos = data['top'][i] + data['height'][i] // 2 + 10  # Offset y-position by 10px downward
            price_data.append((price, y_pos))
        except ValueError:
            continue
price_data.sort(key=lambda x: x[1])  # Sort by y-position
price_levels = [p[0] for p in price_data]
y_positions = [p[1] for p in price_data]
print(f"Extracted price levels with y-positions: {list(zip(price_levels, y_positions))}")

if not price_levels:
    print("No price levels detected. Adjust OCR settings or image.")
else:
    height = img.shape[0]
    # Calculate pixel_to_price based on actual y-positions
    if len(price_levels) > 1:
        pixel_to_price = (max(price_levels) - min(price_levels)) / (max(y_positions) - min(y_positions))
    else:
        pixel_to_price = 0  # Fallback if only one price level
        print("Warning: Only one price level detected, pixel_to_price set to 0.")

    # Process candle bodies (Open/Close) - thick middle area
    oc_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Filter noise
            # Interpolate price based on y-position and price levels, highest at top
            if pixel_to_price > 0:
                base_price = max(price_levels) + (min(y_positions) * pixel_to_price) # Highest price at top (min y)
                middle_y = y + h // 2  # Middle of the thick body
                close_price = base_price - (middle_y * pixel_to_price)
                open_price = base_price - ((middle_y + h) * pixel_to_price)  # Adjust for body height
            else:
                close_price = max(price_levels) - (y * (price_range / height)) if price_levels else 0
                open_price = max(price_levels) - ((y + h) * (price_range / height)) if price_levels else 0
            oc_data.append([x, y, w, h, open_price, close_price])

    # Process wicks (High/Low) to determine full candle range
    hl_data = {}
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 5:  # Vertical line (wick)
                mid_x = (x1 + x2) / 2
                if mid_x not in hl_data or min(y1, y2) < hl_data[mid_x][0]:
                    high_y = min(y1, y2)
                    low_y = max(y1, y2)
                    if pixel_to_price > 0:
                        base_price = max(price_levels) + (min(y_positions) * pixel_to_price) # Highest price at top
                        high_price = base_price - (high_y * pixel_to_price)
                        low_price = base_price - (low_y * pixel_to_price)
                    else:
                        high_price = max(price_levels) - (high_y * (price_range / height)) if price_levels else 0
                        low_price = max(price_levels) - (low_y * (price_range / height)) if price_levels else 0
                    hl_data[mid_x] = [high_y, low_y, high_price, low_price]

    # Combine data with improved matching
    candle_data = []
    for x_oc, y_oc, w_oc, h_oc, open_price, close_price in oc_data:
        mid_x_oc = x_oc + w_oc / 2
        min_dist = float('inf')
        closest_hl = None
        for mid_x_hl, (high_y, low_y, high_price, low_price) in hl_data.items():
            dist = abs(mid_x_hl - mid_x_oc)
            if dist < min_dist and dist < w_oc * 1.5:  # Match within 1.5x body width
                min_dist = dist
                closest_hl = [high_y, low_y, high_price, low_price]
        if closest_hl:
            high_y, low_y, high_price, low_price = closest_hl
            candle_data.append([open_price, close_price, high_price, low_price])

    print(f"Combined candle data length: {len(candle_data)}")

    # Display the original image with detected rectangles (if flag is True)
    if show_original:
        # for x_oc, y_oc, w_oc, h_oc, _, _ in oc_data:
            # cv2.rectangle(img_rgb, (x_oc, y_oc), (x_oc + w_oc, y_oc + h_oc), (0, 255, 0), 2)  # Green for Open/Close (thick middle)
        for x_oc, _, w_oc, _, _, _ in oc_data:
            mid_x_oc = x_oc + w_oc / 2
            if mid_x_oc in hl_data:
                high_y, low_y, _, _ = hl_data[mid_x_oc]
                cv2.rectangle(img_rgb, (x_oc, low_y), (x_oc + w_oc, high_y), (0, 0, 255), 2)  # Blue for High/Low (full range)
        plt.figure(figsize=(10, 5))
        plt.imshow(img_rgb)
        plt.title('Green: Open/Close Bodies, Blue: High/Low Full Range')
        plt.show()

    # Create candlestick chart with extracted data
    if candle_data:
        n_candles = len(candle_data)
        time_steps = np.arange(n_candles)  # Use indices as x-axis

        opens = [data[0] for data in candle_data]
        closes = [data[1] for data in candle_data]
        highs = [data[2] for data in candle_data]
        lows = [data[3] for data in candle_data]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot candlesticks
        for i in range(n_candles):
            if closes[i] >= opens[i]:  # Bullish candle
                color = 'g'
                body = plt.Rectangle((i - 0.2, min(opens[i], closes[i])), 0.4, abs(closes[i] - opens[i]), facecolor=color)
            else:  # Bearish candle
                color = 'r'
                body = plt.Rectangle((i - 0.2, min(opens[i], closes[i])), 0.4, abs(closes[i] - opens[i]), facecolor=color)
            ax.add_patch(body)
            ax.plot([i, i], [lows[i], highs[i]], color='k', linewidth=1)  # Wicks
            ax.plot([i - 0.2, i + 0.2], [opens[i], opens[i]], color='k', linewidth=2)  # Open
            ax.plot([i - 0.2, i + 0.2], [closes[i], closes[i]], color='k', linewidth=2)  # Close

        ax.set_xlim(-0.5, n_candles - 0.5)
        ax.set_ylim(min(lows) - 0.001, max(highs) + 0.001)
        ax.set_title('Candlestick Chart from Extracted Data')
        ax.set_xlabel('Candle Index')
        ax.set_ylabel('Price')
        plt.xticks(range(n_candles))
        plt.grid(True)
        # plt.show()

        # Print combined candle data
        for i, (open_price, close_price, high_price, low_price) in enumerate(candle_data):
            print(f"Candle {i + 1}: Open = {open_price:.4f}, Close = {close_price:.4f}, "
                  f"High = {high_price:.4f}, Low = {low_price:.4f}")
    else:
        print("No candle data detected. Adjust parameters or check image.")