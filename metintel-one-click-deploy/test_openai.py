#!/usr/bin/env python3
"""Test script to verify OpenAI API integration"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv('.env')

def test_openai_connection():
    """Test OpenAI API connection and generate a sample prediction"""
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"API Key found: {'Yes' if api_key else 'No'}")
    
    if not api_key:
        print("❌ OpenAI API key not found in environment variables")
        return False
    
    # Mask the API key for security
    masked_key = api_key[:10] + "..." + api_key[-10:] if len(api_key) > 20 else "***"
    print(f"API Key (masked): {masked_key}")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        
        # Test with a simple prediction request
        prompt = """
        You are a precious metals market analyst. Based on current market conditions, 
        provide a brief price prediction for Gold (XAU/USD) for the next 30 days.
        
        Include:
        1. Predicted price range
        2. Confidence level (1-100%)
        3. Key factors influencing the prediction
        
        Keep the response concise and professional.
        """
        
        print("🔄 Testing OpenAI API call...")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional precious metals market analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        prediction = response.choices[0].message.content
        print("✅ OpenAI API call successful!")
        print("\n📊 Sample Gold Prediction:")
        print("-" * 50)
        print(prediction)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing OpenAI API Integration for PreciousAI")
    print("=" * 60)
    
    success = test_openai_connection()
    
    if success:
        print("\n🎉 OpenAI integration test PASSED!")
        print("✅ Your PreciousAI application is ready for live AI predictions!")
    else:
        print("\n❌ OpenAI integration test FAILED!")
        print("Please check your API key and try again.")

