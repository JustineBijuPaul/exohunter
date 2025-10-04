from typing import Any, Dict
import requests

API_URL = "http://localhost:8000"  # Update this to your FastAPI server URL

def health_check() -> Dict[str, Any]:
    response = requests.get(f"{API_URL}/health")
    response.raise_for_status()
    return response.json()

def predict_exoplanet(features: list) -> Dict[str, Any]:
    payload = {"features": features}
    response = requests.post(f"{API_URL}/predict", json=payload)
    response.raise_for_status()
    return response.json()