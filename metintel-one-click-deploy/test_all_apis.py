#!/usr/bin/env python3
"""
Comprehensive API Testing Script for PreciousAI
Tests all API integrations: OpenAI, Perplexity AI, and MetalpriceAPI
"""

import os
import sys
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
METALS_API_KEY = os.getenv('METALS_API_KEY')

# Test Results
test_results = {
    'timestamp': datetime.now().isoformat(),
    'tests': []
}

def log_test(test_name, status, details=None, error=None):
    """Log test result"""
    result = {
        'test': test_name,
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    if details:
        result['details'] = details
    if error:
        result['error'] = str(error)
    
    test_results['tests'].append(result)
    
    status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_emoji} {test_name}: {status}")
    if details:
        print(f"   Details: {details}")
    if error:
        print(f"   Error: {error}")
    print()

def test_environment_variables():
    """Test if all required environment variables are set"""
    print("🔧 Testing Environment Variables...")
    
    # Test OpenAI API Key
    if OPENAI_API_KEY and len(OPENAI_API_KEY) > 20:
        log_test("OpenAI API Key", "PASS", f"Key length: {len(OPENAI_API_KEY)} characters")
    else:
        log_test("OpenAI API Key", "FAIL", "Missing or invalid API key")
    
    # Test Perplexity API Key
    if PERPLEXITY_API_KEY and len(PERPLEXITY_API_KEY) > 20:
        log_test("Perplexity API Key", "PASS", f"Key length: {len(PERPLEXITY_API_KEY)} characters")
    else:
        log_test("Perplexity API Key", "FAIL", "Missing or invalid API key")
    
    # Test MetalpriceAPI Key
    if METALS_API_KEY and len(METALS_API_KEY) > 20:
        log_test("MetalpriceAPI Key", "PASS", f"Key length: {len(METALS_API_KEY)} characters")
    else:
        log_test("MetalpriceAPI Key", "FAIL", "Missing or invalid API key")

def test_openai_api():
    """Test OpenAI API connection and functionality"""
    print("🤖 Testing OpenAI API...")
    
    if not OPENAI_API_KEY:
        log_test("OpenAI API Connection", "SKIP", "No API key provided")
        return
    
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Test with a simple completion request
        data = {
            'model': 'gpt-4',
            'messages': [
                {
                    'role': 'user',
                    'content': 'What is the current trend in gold prices? Respond in one sentence.'
                }
            ],
            'max_tokens': 50,
            'temperature': 0.7
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            log_test("OpenAI API Connection", "PASS", f"Response: {content[:100]}...")
            
            # Test token usage
            usage = result.get('usage', {})
            log_test("OpenAI Token Usage", "PASS", f"Tokens used: {usage.get('total_tokens', 'unknown')}")
            
        else:
            log_test("OpenAI API Connection", "FAIL", f"HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        log_test("OpenAI API Connection", "FAIL", error=e)

def test_perplexity_api():
    """Test Perplexity AI API connection and functionality"""
    print("🔍 Testing Perplexity AI API...")
    
    if not PERPLEXITY_API_KEY:
        log_test("Perplexity API Connection", "SKIP", "No API key provided")
        return
    
    try:
        headers = {
            'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Test with a market analysis request
        data = {
            'model': 'sonar-pro',
            'messages': [
                {
                    'role': 'user',
                    'content': 'What are the current factors affecting gold prices today? Provide a brief analysis.'
                }
            ],
            'max_tokens': 100,
            'temperature': 0.2
        }
        
        response = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            log_test("Perplexity API Connection", "PASS", f"Response: {content[:100]}...")
            
            # Test usage information
            usage = result.get('usage', {})
            log_test("Perplexity Token Usage", "PASS", f"Tokens used: {usage.get('total_tokens', 'unknown')}")
            
        else:
            log_test("Perplexity API Connection", "FAIL", f"HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        log_test("Perplexity API Connection", "FAIL", error=e)

def test_metalpriceapi():
    """Test MetalpriceAPI connection and functionality"""
    print("🥇 Testing MetalpriceAPI...")
    
    if not METALS_API_KEY:
        log_test("MetalpriceAPI Connection", "SKIP", "No API key provided")
        return
    
    try:
        # Test current prices endpoint
        url = f"https://api.metalpriceapi.com/v1/latest?api_key={METALS_API_KEY}&base=USD&currencies=XAU,XAG,XPT,XPD"
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                rates = result.get('rates', {})
                log_test("MetalpriceAPI Connection", "PASS", f"Retrieved {len(rates)} metal prices")
                
                # Test individual metal prices
                metals = ['XAU', 'XAG', 'XPT', 'XPD']
                for metal in metals:
                    if metal in rates:
                        price = rates[metal]
                        # Convert from per gram to per troy ounce (multiply by 31.1035)
                        troy_ounce_price = price * 31.1035
                        log_test(f"MetalpriceAPI {metal} Price", "PASS", f"${troy_ounce_price:.2f}/oz")
                    else:
                        log_test(f"MetalpriceAPI {metal} Price", "FAIL", "Price not available")
                
            else:
                log_test("MetalpriceAPI Connection", "FAIL", f"API returned error: {result.get('error', 'Unknown error')}")
                
        else:
            log_test("MetalpriceAPI Connection", "FAIL", f"HTTP {response.status_code}: {response.text}")
            
    except Exception as e:
        log_test("MetalpriceAPI Connection", "FAIL", error=e)

def test_local_backend_api():
    """Test local backend API endpoints"""
    print("🖥️ Testing Local Backend API...")
    
    base_url = "http://localhost:5003/api"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            log_test("Backend Health Check", "PASS", "Backend is running")
        else:
            log_test("Backend Health Check", "FAIL", f"HTTP {response.status_code}")
    except Exception as e:
        log_test("Backend Health Check", "FAIL", error=e)
    
    # Test current prices endpoint
    try:
        response = requests.get(f"{base_url}/prices/current", timeout=10)
        if response.status_code == 200:
            data = response.json()
            log_test("Backend Current Prices", "PASS", f"Retrieved prices for {len(data.get('data', {}).get('prices', []))} metals")
        else:
            log_test("Backend Current Prices", "FAIL", f"HTTP {response.status_code}")
    except Exception as e:
        log_test("Backend Current Prices", "FAIL", error=e)
    
    # Test unit conversion
    try:
        response = requests.get(f"{base_url}/prices/current?unit=gram", timeout=10)
        if response.status_code == 200:
            log_test("Backend Unit Conversion", "PASS", "Unit conversion working")
        else:
            log_test("Backend Unit Conversion", "FAIL", f"HTTP {response.status_code}")
    except Exception as e:
        log_test("Backend Unit Conversion", "FAIL", error=e)

def test_exchange_rates():
    """Test exchange rate API for currency conversion"""
    print("💱 Testing Exchange Rate API...")
    
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            rates = data.get('rates', {})
            
            # Test major currencies
            major_currencies = ['EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']
            available_currencies = [curr for curr in major_currencies if curr in rates]
            
            log_test("Exchange Rate API", "PASS", f"Retrieved rates for {len(available_currencies)} major currencies")
            
            for currency in available_currencies:
                rate = rates[currency]
                log_test(f"Exchange Rate USD/{currency}", "PASS", f"Rate: {rate}")
                
        else:
            log_test("Exchange Rate API", "FAIL", f"HTTP {response.status_code}")
            
    except Exception as e:
        log_test("Exchange Rate API", "FAIL", error=e)

def generate_test_report():
    """Generate and save test report"""
    print("📊 Generating Test Report...")
    
    # Count results
    total_tests = len(test_results['tests'])
    passed_tests = len([t for t in test_results['tests'] if t['status'] == 'PASS'])
    failed_tests = len([t for t in test_results['tests'] if t['status'] == 'FAIL'])
    skipped_tests = len([t for t in test_results['tests'] if t['status'] == 'SKIP'])
    
    # Create summary
    summary = {
        'total_tests': total_tests,
        'passed': passed_tests,
        'failed': failed_tests,
        'skipped': skipped_tests,
        'success_rate': f"{(passed_tests / (total_tests - skipped_tests) * 100):.1f}%" if total_tests > skipped_tests else "0%"
    }
    
    test_results['summary'] = summary
    
    # Save to file
    with open('/home/ubuntu/api_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 API TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⚠️ Skipped: {skipped_tests}")
    print(f"Success Rate: {summary['success_rate']}")
    print("="*60)
    
    if failed_tests > 0:
        print("\n❌ FAILED TESTS:")
        for test in test_results['tests']:
            if test['status'] == 'FAIL':
                print(f"  • {test['test']}: {test.get('error', 'Unknown error')}")
    
    print(f"\n📄 Full report saved to: /home/ubuntu/api_test_results.json")

def main():
    """Run all API tests"""
    print("🚀 Starting Comprehensive API Testing for PreciousAI")
    print("="*60)
    
    # Run all tests
    test_environment_variables()
    test_openai_api()
    test_perplexity_api()
    test_metalpriceapi()
    test_local_backend_api()
    test_exchange_rates()
    
    # Generate report
    generate_test_report()

if __name__ == "__main__":
    main()

