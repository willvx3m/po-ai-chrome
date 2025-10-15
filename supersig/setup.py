#!/usr/bin/env python3
"""
Setup script for Screenshot Message Reader with Grok-3
Helps configure API key and test connection
"""

import os
import sys
import getpass
import requests
from pathlib import Path

def setup_api_key():
    """Interactive setup for Grok-3 API key"""
    print("=" * 60)
    print("GROK-3 API KEY SETUP")
    print("=" * 60)
    print()
    
    # Check if API key already exists
    existing_key = os.getenv('GROK3_API_KEY')
    if existing_key:
        print(f"Found existing API key: {existing_key[:8]}...")
        use_existing = input("Use existing key? (y/n): ").lower().strip()
        if use_existing == 'y':
            return existing_key
    
    print("To use this script, you need a Grok-3 API key from x.ai")
    print("1. Visit https://x.ai and sign up for an account")
    print("2. Navigate to the API section in your dashboard")
    print("3. Generate a new API key")
    print("4. Copy the API key")
    print()
    
    while True:
        api_key = getpass.getpass("Enter your Grok-3 API key: ").strip()
        
        if not api_key:
            print("API key cannot be empty. Please try again.")
            continue
        
        if len(api_key) < 20:
            print("API key seems too short. Please check and try again.")
            continue
        
        # Test the API key
        print("\nTesting API key...")
        if test_api_key(api_key):
            print("✅ API key is valid!")
            
            # Ask if user wants to save to environment
            save_to_env = input("\nSave API key to environment? (y/n): ").lower().strip()
            if save_to_env == 'y':
                save_api_key_to_env(api_key)
            
            return api_key
        else:
            print("❌ API key test failed. Please check your key and try again.")
            retry = input("Try again? (y/n): ").lower().strip()
            if retry != 'y':
                sys.exit(1)

def test_api_key(api_key):
    """Test if the API key is valid by making a simple API call"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Simple test call to check authentication
        response = requests.get(
            "https://api.x.ai/v1/models",
            headers=headers,
            timeout=10
        )
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error testing API key: {str(e)}")
        return False

def save_api_key_to_env(api_key):
    """Save API key to environment file"""
    try:
        # Create .env file
        env_file = Path(".env")
        
        # Read existing content
        existing_content = ""
        if env_file.exists():
            with open(env_file, 'r') as f:
                existing_content = f.read()
        
        # Check if GROK3_API_KEY already exists
        if "GROK3_API_KEY=" in existing_content:
            # Replace existing line
            lines = existing_content.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith("GROK3_API_KEY="):
                    new_lines.append(f"GROK3_API_KEY={api_key}")
                else:
                    new_lines.append(line)
            new_content = '\n'.join(new_lines)
        else:
            # Add new line
            new_content = existing_content + f"\nGROK3_API_KEY={api_key}"
        
        # Write to .env file
        with open(env_file, 'w') as f:
            f.write(new_content.strip() + '\n')
        
        print(f"✅ API key saved to {env_file}")
        print("Note: You may need to restart your terminal or run 'source .env' to load the key")
        
    except Exception as e:
        print(f"Warning: Could not save to .env file: {str(e)}")
        print("Please manually set the environment variable:")
        print(f"export GROK3_API_KEY='{api_key}'")

def create_sample_script():
    """Create a sample script to test the setup"""
    sample_script = """#!/usr/bin/env python3
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
"""
    
    with open("test_setup.py", "w") as f:
        f.write(sample_script)
    
    print("✅ Created test_setup.py - run it to verify your configuration")

def main():
    """Main setup function"""
    print("Welcome to Screenshot Message Reader with Grok-3 Setup!")
    print()
    
    try:
        # Setup API key
        api_key = setup_api_key()
        
        print("\n" + "=" * 60)
        print("SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"API Key: {api_key[:8]}...")
        print()
        
        # Create test script
        create_sample_script()
        
        print("\nNext steps:")
        print("1. Run: python test_setup.py (to verify setup)")
        print("2. Place your screenshot image in this directory")
        print("3. Run: python run.py")
        print()
        print("Happy analyzing! 🚀")
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 