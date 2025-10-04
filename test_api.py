#!/usr/bin/env python3
"""
Test script for ExoHunter FastAPI application with database logging.
"""

import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8001"

def test_prediction_endpoint():
    """Test the prediction endpoint and verify database logging."""
    url = f"{BASE_URL}/predict"
    
    payload = {
        "features": [3.5, 0.01, 4.2, 15.3, 0.95, 1.2, 0.8, 2.1]
    }
    
    print("Testing prediction endpoint...")
    print(f"POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the FastAPI server is running on port 8000.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_lightcurve_endpoint():
    """Test the light curve prediction endpoint."""
    url = f"{BASE_URL}/predict/lightcurve"
    
    # Generate some sample light curve data
    import numpy as np
    time_data = np.linspace(0, 10, 100)
    flux_data = 1.0 + 0.01 * np.sin(2 * np.pi * time_data) + 0.005 * np.random.randn(100)
    
    payload = {
        "time": time_data.tolist(),
        "flux": flux_data.tolist()
    }
    
    print("\nTesting light curve endpoint...")
    print(f"POST {url}")
    print(f"Light curve with {len(time_data)} data points")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_health_endpoint():
    """Test the health check endpoint."""
    url = f"{BASE_URL}/health"
    
    print("\nTesting health endpoint...")
    print(f"GET {url}")
    
    try:
        response = requests.get(url, timeout=5)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction_history():
    """Test the prediction history endpoint."""
    url = f"{BASE_URL}/predictions/history"
    
    print("\nTesting prediction history endpoint...")
    print(f"GET {url}")
    
    try:
        response = requests.get(url, timeout=5)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_api_stats():
    """Test the API statistics endpoint."""
    url = f"{BASE_URL}/admin/stats"
    
    print("\nTesting API statistics endpoint...")
    print(f"GET {url}")
    
    try:
        response = requests.get(url, timeout=5)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("ExoHunter API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Prediction Endpoint", test_prediction_endpoint),
        ("Light Curve Prediction", test_lightcurve_endpoint),
        ("Prediction History", test_prediction_history),
        ("API Statistics", test_api_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*20} Test Summary {'='*20}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Check the server is running and database is configured.")

if __name__ == "__main__":
    main()
