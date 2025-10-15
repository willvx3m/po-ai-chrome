#!/usr/bin/env python3
# Sample script to test Grok-3 connection
import os
import sys

# Load environment variables from .env file
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Check if API key is available
api_key = os.getenv('GROK3_API_KEY')
if not api_key:
    print("ERROR: GROK3_API_KEY not found!")
    print("Please run setup.py first or set the environment variable manually")
    sys.exit(1)

print("✅ Grok-3 API key is configured!")
print(f"Key: {api_key[:8]}...")
print("You can now run: python run.py")
