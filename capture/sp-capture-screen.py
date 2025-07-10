import pyautogui
import mss
import numpy as np
import cv2
import time
import argparse

def capture_screen_region(left=0, top=0, width=None, height=None, name=None):
    """
    Capture a screenshot of a specified region or the entire screen.
    
    Args:
        left (int): Left coordinate of the region (default: 0).
        top (int): Top coordinate of the region (default: 0).
        width (int): Width of the region (default: None, uses full screen width).
        height (int): Height of the region (default: None, uses full screen height).
    
    Returns:
        numpy.ndarray: Screenshot as a BGR image array.
    """
    # Get screen dimensions if not specified
    if width is None or height is None:
        screen_width, screen_height = pyautogui.size()
        width = width if width is not None else screen_width
        height = height if height is not None else screen_height

    # Ensure coordinates are within screen bounds
    screen_width, screen_height = pyautogui.size()
    left = max(0, min(left, screen_width - 1))
    top = max(0, min(top, screen_height - 1))
    width = min(width, screen_width - left)
    height = min(height, screen_height - top)

    # Define the region to capture
    region = {'left': int(left), 'top': int(top), 'width': int(width), 'height': int(height)}

    # Use mss for fast screenshot
    sct = mss.mss()
    screenshot = sct.grab(region)

    # Convert to numpy array and BGR format for OpenCV
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Capture a screenshot of a specified screen region.')
    parser.add_argument('--left', type=int, default=0, help='Left coordinate of the region (default: 0)')
    parser.add_argument('--top', type=int, default=0, help='Top coordinate of the region (default: 0)')
    parser.add_argument('--width', type=int, help='Width of the region (default: full screen width)')
    parser.add_argument('--height', type=int, help='Height of the region (default: full screen height)')
    parser.add_argument('--name', type=str, help='Name of the screenshot')

    args = parser.parse_args()

    print(f"Capturing region: left={args.left}, top={args.top}, width={args.width}, height={args.height}, name={args.name}")
    
    # Capture the screen region
    screenshot = capture_screen_region(args.left, args.top, args.width, args.height, args.name)
    
    # Display the captured image
    # cv2.imshow('Screenshot', screenshot)
    # cv2.waitKey(0)  # Wait for a key press to close
    # cv2.destroyAllWindows()
    
    # Optionally save the screenshot
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'screenshots/{args.name}_{timestamp}.png' if args.name else f'screenshots/{timestamp}.png'
    cv2.imwrite(filename, screenshot)
    print(f"Screenshot saved as {filename}")

if __name__ == "__main__":
    # Add a small delay to give time to switch to the target window
    time.sleep(2)
    main()