#!/usr/bin/env python3
"""
Screenshot Message Reader for Signal Analysis using Grok-3
Reads messages from a screenshot image and groups them by sender/receiver.
Uses Grok-3 vision model for better text extraction and arrow detection.
"""

import cv2
import numpy as np
from PIL import Image
import re
import os
import logging
from datetime import datetime
import base64
import io
import json
import requests
from typing import Optional, Dict, List, Any
from pathlib import Path

# Configure logging to both console and file
def setup_logging():
    """Setup logging to both console and file in the same directory as the script"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "log.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_env_file():
    """Load environment variables from .env file in the same directory as the script"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(script_dir, ".env")
    
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                            logging.getLogger(__name__).info(f"Loaded environment variable: {key.strip()}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not load .env file: {str(e)}")
    else:
        logging.getLogger(__name__).info("No .env file found, using system environment variables")

class Grok3VisionAnalyzer:
    """Class to handle Grok-3 vision model interactions"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.x.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image to base64 string for API transmission"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {str(e)}")
            return None
    
    def analyze_image_with_grok3(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Analyze image using Grok-3 vision model"""
        try:
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return None
            
            # Prepare the prompt for Grok-3
            prompt = """Analyze this screenshot image and extract the following information:

1. All text content visible in the image
2. Identify any UP or DOWN arrows and mark them clearly
3. Group messages by sender/receiver if applicable
4. Identify trading signals, asset names, payouts, accuracy, expiration times
5. Extract balance information and trade IDs

Please provide a structured analysis with clear identification of:
- Text content
- Arrow directions (UP/DOWN)
- Message grouping
- Trading information

Format the response as JSON with the following structure:
{
  "extracted_text": "all text content",
  "arrows": ["UP", "DOWN", "UP", ...],
  "trading_signals": [
    {
      "asset": "asset name",
      "payout": "percentage",
      "accuracy": "percentage", 
      "expiration": "time",
      "direction": "UP or DOWN"
    }
  ],
  "messages": [
    {
      "sender": "sender name or 'system'",
      "content": "message content"
    }
  ]
}"""

            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "grok-4-0709",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1
            }
            
            # Make the API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Try to parse JSON response
                try:
                    parsed_content = json.loads(content)
                    return parsed_content
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw content
                    return {
                        "extracted_text": content,
                        "arrows": [],
                        "trading_signals": [],
                        "messages": [],
                        "raw_response": content
                    }
            else:
                self.logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error analyzing image with Grok-3: {str(e)}")
            return None

class ScreenshotMessageReader:
    def __init__(self, image_path: str, grok3_api_key: str):
        self.image_path = image_path
        self.grok3_analyzer = Grok3VisionAnalyzer(grok3_api_key)
        self.logger = setup_logging()
        
    def process_image_with_grok3(self) -> Optional[Dict[str, Any]]:
        """Process image using Grok-3 vision model"""
        try:
            self.logger.info(f"Processing image with Grok-3: {self.image_path}")
            
            # Check if image exists
            if not os.path.exists(self.image_path):
                self.logger.error(f"Image file not found: {self.image_path}")
                return None
            
            # Analyze image with Grok-3
            analysis_result = self.grok3_analyzer.analyze_image_with_grok3(self.image_path)
            
            if analysis_result:
                self.logger.info("Grok-3 analysis completed successfully")
                
                # Add timestamp
                analysis_result['timestamp'] = datetime.now().isoformat()
                analysis_result['image_path'] = self.image_path
                
                return analysis_result
            else:
                self.logger.error("Grok-3 analysis failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing image with Grok-3: {str(e)}")
            return None
    
    def enhance_message_grouping(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance message grouping based on Grok-3 analysis"""
        if 'messages' not in analysis_result:
            return analysis_result
            
        # Process messages to improve grouping
        enhanced_messages = []
        
        for msg in analysis_result['messages']:
            # Clean up message content
            content = msg.get('content', '')
            
            # Identify trading signals in content
            if 'SIGNAL' in content.upper():
                # Extract trading information
                trading_info = self.extract_trading_info(content)
                if trading_info:
                    msg['trading_info'] = trading_info
            
            # Identify arrow directions in content
            arrows_in_content = self.extract_arrows_from_text(content)
            if arrows_in_content:
                msg['arrows'] = arrows_in_content
            
            enhanced_messages.append(msg)
        
        analysis_result['enhanced_messages'] = enhanced_messages
        return analysis_result
    
    def extract_trading_info(self, text: str) -> Optional[Dict[str, str]]:
        """Extract trading information from text"""
        trading_info = {}
        
        # Extract asset
        asset_match = re.search(r'Asset:\s*([A-Z]+/[A-Z]+)', text, re.IGNORECASE)
        if asset_match:
            trading_info['asset'] = asset_match.group(1)
        
        # Extract payout
        payout_match = re.search(r'Payout:\s*(\d+%)', text, re.IGNORECASE)
        if payout_match:
            trading_info['payout'] = payout_match.group(1)
        
        # Extract accuracy
        accuracy_match = re.search(r'Accuracy:\s*(\d+%)', text, re.IGNORECASE)
        if accuracy_match:
            trading_info['accuracy'] = accuracy_match.group(1)
        
        # Extract expiration
        expiration_match = re.search(r'Expiration:\s*([A-Z0-9]+)', text, re.IGNORECASE)
        if expiration_match:
            trading_info['expiration'] = expiration_match.group(1)
        
        return trading_info if trading_info else None
    
    def extract_arrows_from_text(self, text: str) -> List[str]:
        """Extract arrow directions from text content"""
        arrows = []
        
        # Look for UP/DOWN indicators
        up_matches = re.findall(r'\bUP\b', text, re.IGNORECASE)
        down_matches = re.findall(r'\bDOWN\b', text, re.IGNORECASE)
        
        arrows.extend(['UP'] * len(up_matches))
        arrows.extend(['DOWN'] * len(down_matches))
        
        return arrows
    
    def print_results(self, results: Dict[str, Any]):
        """Print the results in a formatted way"""
        if not results:
            print("No results to display.")
            return
        
        print("=" * 60)
        print("GROK-3 SCREENSHOT ANALYSIS")
        print("=" * 60)
        print(f"Processed at: {results.get('timestamp', 'Unknown')}")
        print(f"Image: {results.get('image_path', 'Unknown')}")
        print()
        
        # Display trading signals
        if 'trading_signals' in results and results['trading_signals']:
            print("TRADING SIGNALS:")
            print("-" * 40)
            for i, signal in enumerate(results['trading_signals'], 1):
                print(f"{i}. Asset: {signal.get('asset', 'N/A')}")
                print(f"   Payout: {signal.get('payout', 'N/A')}")
                print(f"   Accuracy: {signal.get('accuracy', 'N/A')}")
                print(f"   Expiration: {signal.get('expiration', 'N/A')}")
                print(f"   Direction: {signal.get('direction', 'N/A')}")
                print()
        
        # Display enhanced messages
        if 'enhanced_messages' in results and results['enhanced_messages']:
            print("ENHANCED MESSAGES:")
            print("-" * 40)
            for i, msg in enumerate(results['enhanced_messages'], 1):
                print(f"{i}. Sender: {msg.get('sender', 'Unknown')}")
                print(f"   Content: {msg.get('content', 'N/A')}")
                
                if 'trading_info' in msg:
                    print(f"   Trading Info: {msg['trading_info']}")
                
                if 'arrows' in msg and msg['arrows']:
                    print(f"   Arrows: {', '.join(msg['arrows'])}")
                print()
        
        # Display arrows summary
        if 'arrows' in results and results['arrows']:
            print("ARROW SUMMARY:")
            print("-" * 40)
            up_count = results['arrows'].count('UP')
            down_count = results['arrows'].count('DOWN')
            print(f"UP arrows: {up_count}")
            print(f"DOWN arrows: {down_count}")
            print(f"Total arrows: {len(results['arrows'])}")
            print()
        
        # Display extracted text
        if 'extracted_text' in results:
            print("EXTRACTED TEXT:")
            print("-" * 40)
            print(results['extracted_text'][:500] + "..." if len(results['extracted_text']) > 500 else results['extracted_text'])
            print()

def main():
    """Main function to run the script"""
    # Load environment variables from .env file first
    load_env_file()
    
    # Configuration
    image_path = "sample.png"  # Relative to the script directory
    
    # Get Grok-3 API key from environment variable (now loaded from .env)
    grok3_api_key = os.getenv('GROK3_API_KEY')
    
    if not grok3_api_key:
        print("ERROR: GROK3_API_KEY environment variable not found!")
        print("Please ensure you have:")
        print("1. Created a .env file in the same directory as this script")
        print("2. Added your API key: GROK3_API_KEY=your_api_key_here")
        print("3. Or set the environment variable manually:")
        print("   export GROK3_API_KEY='your_api_key_here'")
        return
    
    # Create reader instance
    reader = ScreenshotMessageReader(image_path, grok3_api_key)
    
    # Process the image with Grok-3
    results = reader.process_image_with_grok3()
    
    if results:
        # Enhance results with additional processing
        enhanced_results = reader.enhance_message_grouping(results)
        
        # Display results
        reader.print_results(enhanced_results)
        
        # Save results to JSON file
        output_file = "grok3_extracted_messages.json"
        with open(output_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        print(f"\nEnhanced results saved to: {output_file}")
        
        # Also save to the original filename for compatibility
        with open("extracted_messages.json", 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        print(f"Results also saved to: extracted_messages.json")
    else:
        print("Failed to process image with Grok-3. Check the log for details.")

if __name__ == "__main__":
    main()
