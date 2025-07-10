## PREREQUISITES

1. Open PO platform and maximize it
2. Go to settings and remove any unncessary ONs including NO BACKGROUND
3. Run the following javscript in console:
```javascript
document.querySelector('div.zoom-controls')?.remove(),
document.querySelector('div.scroll-to-end')?.remove(),
document.querySelector('div.top-left-block')?.remove(),
document.querySelector('div.control-wrap')?.remove()
```

## CAPTURE SCREENSHOTS
**Make sure to place the browser in the correct place so that the command correctly captures "necessary" part of screen.

In Mac, run the following command:
```shell
python sp-capture-screen.py --left 87 --top 210 --width 1580 --height 820
```

In Mac, run the following command:
```shell
python sp-capture-screen.py --left 87 --top 210 --width 1580 --height 820
```

## READ CANDLES
Run the following command:
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
LABEL_SPACING = IMAGE_WIDTH / 60  # Distance between consecutive time labels in pixels
TIME_INCREMENT = timedelta(minutes=1)  # Increment time by 1 minute
CONFIDENCE_THRESHOLD_LABEL = 85  # Confidence threshold for label correctness

# TESSERACT CONFIG
TIME_LABEL_TESSERACT_CONFIG = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789:'
PRICE_LABEL_TESSERACT_CONFIG = r'--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789.'
TIME_LABEL_CHAR_LENGTH = 5
```

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