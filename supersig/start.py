import pytesseract
from PIL import Image
import re

# Note: Install pytesseract and Pillow: pip install pytesseract pillow
# Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract

# Path to your screenshot image
image_path = 'sample.png'  # Replace with your actual image path

# Extract text from the image
image = Image.open(image_path)
text = pytesseract.image_to_string(image)

# Split into lines and clean up
lines = [line.strip() for line in text.splitlines() if line.strip()]

# Group messages by timestamp, starting from latest (bottom)
messages = []
current_group = []
timestamp_pattern = re.compile(r'\d{1,2}:\d{2} [AP]M')

for line in reversed(lines):  # Reverse for bottom-up (latest to earliest)
    if line:
        current_group.append(line)
        if timestamp_pattern.search(line):
            messages.append(' - '.join(reversed(current_group)))  # Reverse to original order
            current_group = []

if current_group:
    messages.append(' - '.join(reversed(current_group)))

# Adjust SIGNAL arrows
for i, msg in enumerate(messages):
    if 'SIGNAL' in msg:
        if '↑' in msg or 'UP' in msg:
            messages[i] = msg.replace('SIGNAL', 'SIGNAL ↑').replace('UP', '').replace('^', '')
        elif '↓' in msg or 'DOWN' in msg:
            messages[i] = msg.replace('SIGNAL', 'SIGNAL ↓').replace('DOWN', '').replace('v', '')

# Output results
for msg in messages:
    print('-', msg)