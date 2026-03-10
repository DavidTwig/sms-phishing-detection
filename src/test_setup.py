"""
Test that all packages and API key are working correctly.
"""

import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

print("Testing your setup...\n")

# Test 1: Check API key exists
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print("✓ Groq API key loaded successfully")
else:
    print("✗ Groq API key not found - check your .env file")
    exit()

# Test 2: Test core packages
try:
    import requests
    print("✓ requests")
except ImportError:
    print("✗ requests")

try:
    from bs4 import BeautifulSoup
    print("✓ beautifulsoup4")
except ImportError:
    print("✗ beautifulsoup4")

try:
    import whois
    print("✓ python-whois")
except ImportError:
    print("✗ python-whois")

try:
    import pandas
    print("✓ pandas")
except ImportError:
    print("✗ pandas")

try:
    import nltk
    print("✓ nltk")
except ImportError:
    print("✗ nltk")

try:
    from groq import Groq
    print("✓ groq")
except ImportError:
    print("✗ groq")

# Test 3: Test Groq API connection
print("\nTesting Groq API connection...")
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say 'Hello, SMS Phishing Detection Project!' and nothing else."}],
        max_tokens=50
    )
    print(f"✓ Groq API working! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Groq API error: {e}")

print("\n✅ Setup complete! You're ready to start building.")




## Run the test with python src/test_setup.py in venv

