## PREREQUISITES
Install python modules:
```shell
python -m pip install numpy opencv-python pytesseract pyautogui mss matplotlib Pillow
```

### Additional Dependency (MacOS)
Requires the Tesseract OCR engine to be installed on your system. On macOS, install via Homebrew:
```bash
brew install tesseract
```

Ensure Tesseract is installed and its path is accessible. On macOS, after installing with Homebrew, you may need to add it to your PATH or specify the path in your script (e.g., pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract').

### Additional Dependency (Windows)

#### Tesseract OCR EnginePurpose
Required by pytesseract to perform OCR on images.

##### Installation
Download the installer from UB Mannheim Tesseract at UB Mannheim or use the official site.
Run the installer (e.g., tesseract-ocr-setup-5.3.0.exe) and follow the prompts. Default installation path is typically C:\Program Files\Tesseract-OCR.

##### Configuration
After installation, set the Tesseract path in your script or environment:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

##### Verification
Run `tesseract -v` in Command Prompt to check the version.

#### Visual C++ RedistributablePurpose
Required by opencv-python and other compiled packages.

##### Installation
Download from Microsoft’s official site (x64 version recommended). Install if not already present.

##### Verification 
Check in "Programs and Features" or reinstall if issues arise with OpenCV.

## GET READY PO

1. Open PO platform and maximize it
2. Go to settings and remove any unncessary ONs including NO BACKGROUND
3. Set the chart range to 1H
4. Run the following javscript in console:
```javascript
document.querySelector('div.mfp-bg')?.remove(),
document.querySelector('div.mfp-wrap')?.remove(),
document.querySelector('div.zoom-controls')?.remove(),
document.querySelector('div.scroll-to-end')?.remove(),
document.querySelector('div.deposit-btn-wrap div.h-btn__text').innerText = document.querySelector('span.current-symbol')?.innerText,
document.querySelector('div.top-left-block')?.remove(),
document.querySelector('div.control-wrap')?.remove()
```

## CAPTURE SCREENSHOTS
**Make sure to place the browser in the correct place so that the command correctly captures "necessary" part of screen.

In Mac, run the following command:
```shell
python sp-capture-screen.py --left 87 --top 210 --width 1580 --height 820
```

In Win, run the following command:
```shell
python sp-capture-screen.py --left 87 --top 210 --width 1580 --height 830
```

## READ CANDLES
GIVE VALID IMAGE PATH FOR THE DATE/TIME RANGE. Run the following command:
```shell
python sp-read-candles.py --path {IMG_PATH} --draw-overlay True --draw-chart True --save-json True
```

### Constants
There are sevearl constants that define the performance of candle read. Make sure to adjust them unless the reading is not successful.

```
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
LABEL_SPACING = IMAGE_WIDTH / 60  # Distance between consecutive time labels in pixels (Mac: IMAGE_WIDTH / 60, Win: 25)
TIME_INCREMENT = timedelta(minutes=1)  # Increment time by 1 minute
CONFIDENCE_THRESHOLD_LABEL = 85  # Confidence threshold for label correctness

# TESSERACT CONFIG
TIME_LABEL_TESSERACT_CONFIG = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789:'
PRICE_LABEL_TESSERACT_CONFIG = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789.'
TIME_LABEL_CHAR_LENGTH = 5
```

## NEXT STEPS

### PO Auto Scroller
This is a helper (Chrome extension) that scrolls the price chart automatically so that screen capture can run automatically.

**Key points**
- remove any overlapping elements (eg. like claiming prize)
- avoid skipping any timeframe
- **disable auto-jump to current time**

### Automatic Candle Reader
Let the candle reader work by himself with new screenshots.

### Merge Json
- read json files and merge them
- add-up missing price values for missing time frames
- refine data (no clear direction yet)

## MISCELLANOUS

### TEST FILES

**OLD ONES**

/Users/million/Downloads/aaa.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_100911-ok.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_100850.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_100830-cur-price-top.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_100807-half-full.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_104532.png\

**NEW ONES (fine tuned with constants)**

/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_114341-start.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_114349.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_114355.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_114402.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_114408.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_114415.png\
/Volumes/WORK/Project/MegaVX/po-ai/capture/screenshots/screenshot_20250710_114426.png