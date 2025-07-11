#!/usr/bin/env python3
"""Comprehensive test script for both OpenAI and Perplexity AI integrations"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv('.env')

def test_openai_integration():
    """Test OpenAI API integration"""
    print("🔍 Testing OpenAI Integration...")
    print("-" * 40)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found")
        return False
    
    masked_key = api_key[:10] + "..." + api_key[-10:]
    print(f"✅ OpenAI API Key: {masked_key}")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Test prediction generation
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precious metals market analyst."},
                {"role": "user", "content": "Provide a brief gold price prediction for next week with confidence level."}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        prediction = response.choices[0].message.content
        print("✅ OpenAI API call successful!")
        print(f"📊 Sample prediction: {prediction[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI error: {str(e)}")
        return False

def test_perplexity_integration():
    """Test Perplexity AI API integration"""
    print("\n🔍 Testing Perplexity AI Integration...")
    print("-" * 40)
    
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        print("❌ Perplexity API key not found")
        return False
    
    masked_key = api_key[:10] + "..." + api_key[-10:]
    print(f"✅ Perplexity API Key: {masked_key}")
    
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial market analyst focused on precious metals."
                },
                {
                    "role": "user", 
                    "content": "What are the latest market trends affecting gold prices this week?"
                }
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            market_analysis = result['choices'][0]['message']['content']
            print("✅ Perplexity AI API call successful!")
            print(f"📈 Market analysis: {market_analysis[:100]}...")
            return True
        else:
            print(f"❌ Perplexity API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Perplexity error: {str(e)}")
        return False

def test_combined_prediction():
    """Test combined AI prediction using both services"""
    print("\n🤖 Testing Combined AI Prediction...")
    print("-" * 40)
    
    # This simulates how the actual application will work
    try:
        # Step 1: Get market context from Perplexity
        perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if not (perplexity_key and openai_key):
            print("❌ Missing API keys for combined test")
            return False
        
        print("🔄 Getting market context from Perplexity...")
        
        # Perplexity for market context
        perplexity_url = "https://api.perplexity.ai/chat/completions"
        perplexity_headers = {
            "Authorization": f"Bearer {perplexity_key}",
            "Content-Type": "application/json"
        }
        
        context_data = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "user",
                    "content": "What are the current economic factors affecting precious metals prices? Include inflation, geopolitical events, and market sentiment."
                }
            ],
            "max_tokens": 300
        }
        
        context_response = requests.post(perplexity_url, headers=perplexity_headers, json=context_data, timeout=30)
        
        if context_response.status_code != 200:
            print(f"❌ Failed to get market context: {context_response.status_code}")
            return False
        
        market_context = context_response.json()['choices'][0]['message']['content']
        print("✅ Market context retrieved")
        
        print("🔄 Generating prediction with OpenAI...")
        
        # OpenAI for prediction based on context
        openai_client = OpenAI(api_key=openai_key)
        
        prediction_prompt = f"""
        Based on the following current market context, provide a detailed precious metals price prediction:
        
        Market Context: {market_context}
        
        Please provide:
        1. Gold price prediction for next 30 days
        2. Confidence level (1-100%)
        3. Key risk factors
        4. Investment recommendation
        
        Format as JSON with clear structure.
        """
        
        prediction_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert precious metals analyst. Provide structured, data-driven predictions."},
                {"role": "user", "content": prediction_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        prediction = prediction_response.choices[0].message.content
        print("✅ AI prediction generated successfully!")
        
        print("\n🎯 Combined AI Analysis Result:")
        print("=" * 50)
        print(f"Market Context: {market_context[:150]}...")
        print(f"\nPrediction: {prediction[:200]}...")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Combined prediction error: {str(e)}")
        return False

def main():
    """Run all AI integration tests"""
    print("🚀 PreciousAI - AI Integration Test Suite")
    print("=" * 60)
    
    # Test individual services
    openai_success = test_openai_integration()
    perplexity_success = test_perplexity_integration()
    
    # Test combined functionality
    combined_success = False
    if openai_success and perplexity_success:
        combined_success = test_combined_prediction()
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    print(f"OpenAI Integration:     {'✅ PASS' if openai_success else '❌ FAIL'}")
    print(f"Perplexity Integration: {'✅ PASS' if perplexity_success else '❌ FAIL'}")
    print(f"Combined AI Prediction: {'✅ PASS' if combined_success else '❌ FAIL'}")
    
    if openai_success and perplexity_success and combined_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your PreciousAI application is ready for live AI-powered predictions!")
        print("🚀 Both OpenAI and Perplexity AI are working perfectly!")
    else:
        print("\n⚠️  Some tests failed. Please check the API keys and try again.")
    
    return openai_success and perplexity_success and combined_success

if __name__ == "__main__":
    main()

