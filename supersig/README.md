# Screenshot Message Reader with Grok-3 Vision

This Python script reads messages from screenshot images using Grok-3's advanced vision model, groups them by sender/receiver, and identifies UP/DOWN arrows, converting them to text.

## Features

- **Grok-3 Vision Analysis**: Uses Grok-3's state-of-the-art vision model for superior text extraction
- **Intelligent Arrow Detection**: Automatically identifies UP/DOWN arrows and converts them to text
- **Smart Message Grouping**: Groups messages by sender/receiver with enhanced context
- **Trading Signal Extraction**: Automatically extracts trading information (assets, payouts, accuracy, expiration)
- **Comprehensive Logging**: Logs to both console and file
- **Structured Output**: Results saved to JSON file for further processing

## Prerequisites

### System Dependencies

1. **Grok-3 API Access**: You need access to Grok-3 API (x.ai)
   - Sign up at [x.ai](https://x.ai) to get API access
   - Obtain your API key from the dashboard

2. **Python 3.7+**: Required for running the script

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Get Grok-3 API Key

1. Visit [x.ai](https://x.ai) and sign up for an account
2. Navigate to the API section in your dashboard
3. Generate a new API key
4. Copy the API key

### 2. Set Environment Variable

Set your Grok-3 API key as an environment variable:

```bash
# Linux/macOS
export GROK3_API_KEY='your_api_key_here'

# Windows (PowerShell)
$env:GROK3_API_KEY='your_api_key_here'

# Windows (Command Prompt)
set GROK3_API_KEY=your_api_key_here
```

### 3. Place Your Screenshot

Put your screenshot image (e.g., `sample.png`) in the same directory as the script.

## Usage

1. **Set your API key** (see setup above)

2. **Run the script**:
   ```bash
   python run.py
   ```

3. **View results**: The script will:
   - Display enhanced analysis in the console
   - Save results to `grok3_extracted_messages.json`
   - Also save to `extracted_messages.json` for compatibility
   - Create a log file `log.txt` with processing details

## How It Works

### 1. Grok-3 Vision Analysis
- Sends image to Grok-3's vision model via API
- Uses specialized prompt for trading signal analysis
- Receives structured JSON response with extracted information

### 2. Enhanced Processing
- Processes Grok-3's response for better message grouping
- Extracts trading information (assets, payouts, accuracy, expiration)
- Identifies arrow directions in text content
- Enhances message context with trading details

### 3. Intelligent Output
- Groups messages by sender/receiver
- Identifies trading signals and their parameters
- Provides arrow direction analysis
- Creates structured, searchable output

## Output Format

The script generates:

1. **Console Output**: Formatted display of enhanced analysis
2. **JSON Files**: Structured data with trading signals and enhanced messages
3. **Log File**: Detailed processing information and any errors

### JSON Structure
```json
{
  "extracted_text": "Complete extracted text from Grok-3",
  "arrows": ["UP", "DOWN", "UP", ...],
  "trading_signals": [
    {
      "asset": "GBPAUD",
      "payout": "92%",
      "accuracy": "88%",
      "expiration": "H2",
      "direction": "UP"
    }
  ],
  "enhanced_messages": [
    {
      "sender": "system",
      "content": "SIGNAL Asset: GBPAUD Payout: 92%",
      "trading_info": {
        "asset": "GBPAUD",
        "payout": "92%",
        "accuracy": "88%",
        "expiration": "H2"
      },
      "arrows": ["UP"]
    }
  ],
  "timestamp": "2024-01-01T12:00:00",
  "image_path": "sample.png"
}
```

## Advantages of Grok-3

- **Superior Text Recognition**: Better than traditional OCR for complex layouts
- **Context Understanding**: Understands trading terminology and signal structures
- **Arrow Detection**: Automatically identifies and classifies directional indicators
- **Structured Output**: Returns organized, parseable data instead of raw text
- **Multi-language Support**: Handles various text formats and languages

## Customization

### Grok-3 Prompt
Modify the prompt in `analyze_image_with_grok3()` method to:
- Change the analysis focus
- Adjust output structure
- Add specific trading signal requirements

### Message Enhancement
Customize the `enhance_message_grouping()` method to:
- Add more trading information extraction
- Modify message grouping logic
- Enhance arrow detection algorithms

## Troubleshooting

### Common Issues

1. **API Key Not Set**: Ensure `GROK3_API_KEY` environment variable is set
2. **API Rate Limits**: Grok-3 may have rate limits; check your plan
3. **Image Format**: Ensure image is in supported format (PNG, JPG, etc.)
4. **API Errors**: Check log file for detailed error messages

### Error Messages

- **"GROK3_API_KEY environment variable not set"**: Set your API key
- **"API request failed"**: Check API key validity and rate limits
- **"Failed to process image"**: Check image format and API response

### Logging
Check `log.txt` for detailed error messages and processing information.

## Dependencies

- `opencv-python`: Image processing utilities
- `numpy`: Numerical computing
- `Pillow`: Image handling
- `requests`: HTTP requests for Grok-3 API

## API Costs

- Grok-3 API usage incurs costs based on your plan
- Check [x.ai pricing](https://x.ai/pricing) for current rates
- Vision model calls typically cost more than text-only calls

## License

This script is provided as-is for educational and development purposes. 