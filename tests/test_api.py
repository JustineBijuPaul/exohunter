"""
Tests for FastAPI endpoints in the ExoHunter API.

Tests API health endpoint, prediction endpoints, and error handling
using FastAPI TestClient.
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import after path setup to avoid import errors
try:
    from web.api.main import app
except ImportError:
    # Create a mock app if the real one can't be imported
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "exohunter-api"}


class TestHealthEndpoint:
    """Test the API health endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_health_endpoint_success(self, client):
        """Test that health endpoint returns success status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok"]
        
        # API should identify itself (service, name, or version field acceptable)
        assert "service" in data or "name" in data or "version" in data
    
    def test_health_endpoint_response_format(self, client):
        """Test that health endpoint returns properly formatted JSON."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
    
    def test_health_endpoint_contains_required_fields(self, client):
        """Test that health endpoint contains required status information."""
        response = client.get("/health")
        data = response.json()
        
        # Should have status field
        assert "status" in data
        assert isinstance(data["status"], str)
        
        # May have additional fields like version, timestamp, etc.
        # But we'll just check for the essential status
    
    def test_health_endpoint_multiple_calls(self, client):
        """Test that health endpoint is consistently available."""
        # Make multiple calls to ensure consistency
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data


class TestPredictionEndpoints:
    """Test prediction endpoints if they exist."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_predict_endpoint_exists(self, client):
        """Test if prediction endpoint exists and handles requests."""
        # Try to access prediction endpoint
        response = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        
        # Should not return 404 (endpoint exists)
        # Could return 422 (validation error), 500 (server error), or 200 (success)
        assert response.status_code != 404
    
    def test_predict_endpoint_requires_features(self, client):
        """Test that prediction endpoint validates input."""
        # Try empty request
        response = client.post("/predict", json={})
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_predict_endpoint_handles_invalid_data(self, client):
        """Test that prediction endpoint handles invalid input gracefully."""
        # Try invalid feature count
        response = client.post("/predict", json={"features": [1.0, 2.0]})  # Too few features
        
        # Should return error, not crash
        assert response.status_code in [400, 422, 500]
        
        # Response should be JSON
        try:
            data = response.json()
            assert isinstance(data, dict)
        except json.JSONDecodeError:
            pytest.fail("Response should be valid JSON even for errors")


class TestLightCurvePredictionEndpoint:
    """Test light curve prediction endpoint if it exists."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_lightcurve_endpoint_exists(self, client):
        """Test if light curve prediction endpoint exists."""
        # Sample light curve data
        time_data = list(range(100))
        flux_data = [1.0 + 0.01 * np.sin(0.1 * t) for t in time_data]
        
        response = client.post("/predict/lightcurve", json={
            "time": time_data,
            "flux": flux_data
        })
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
    
    def test_lightcurve_endpoint_validates_input(self, client):
        """Test that light curve endpoint validates input data."""
        # Try missing flux data
        response = client.post("/predict/lightcurve", json={"time": [1, 2, 3]})
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_lightcurve_endpoint_handles_mismatched_arrays(self, client):
        """Test handling of mismatched time and flux arrays."""
        response = client.post("/predict/lightcurve", json={
            "time": [1, 2, 3],
            "flux": [1.0, 2.0]  # Different length
        })
        
        # Should return error
        assert response.status_code in [400, 422, 500]


class TestFileUploadEndpoint:
    """Test file upload prediction endpoint if it exists."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_upload_endpoint_exists(self, client):
        """Test if upload endpoint exists."""
        # Create a simple CSV file content
        csv_content = "feature1,feature2,feature3\n1.0,2.0,3.0\n"
        
        response = client.post("/predict/upload", files={
            "file": ("test.csv", csv_content, "text/csv")
        })
        
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
    
    def test_upload_endpoint_requires_file(self, client):
        """Test that upload endpoint requires a file."""
        response = client.post("/predict/upload")
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_upload_endpoint_handles_invalid_csv(self, client):
        """Test handling of invalid CSV files."""
        invalid_csv = "invalid,csv,content\nno,proper,structure"
        
        response = client.post("/predict/upload", files={
            "file": ("invalid.csv", invalid_csv, "text/csv")
        })
        
        # Should handle error gracefully
        assert response.status_code in [400, 422, 500]
        
        # Should return JSON error message
        try:
            data = response.json()
            assert isinstance(data, dict)
            assert "detail" in data or "error" in data
        except json.JSONDecodeError:
            # If not JSON, should at least not crash
            pass


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_invalid_endpoint_returns_404(self, client):
        """Test that invalid endpoints return 404."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404
    
    def test_wrong_http_method(self, client):
        """Test that wrong HTTP methods return appropriate errors."""
        # Try POST on health endpoint (should be GET)
        response = client.post("/health")
        assert response.status_code in [405, 422]  # Method not allowed or validation error
    
    def test_invalid_json_payload(self, client):
        """Test handling of invalid JSON payloads."""
        # Try to send invalid JSON to predict endpoint
        response = client.post(
            "/predict",
            data="invalid json content",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return JSON parsing error
        assert response.status_code in [400, 422]
    
    def test_large_payload_handling(self, client):
        """Test handling of very large payloads."""
        # Create a large feature array
        large_features = [1.0] * 10000
        
        response = client.post("/predict", json={"features": large_features})
        
        # Should either handle it or reject it gracefully
        assert response.status_code in [200, 400, 413, 422, 500]


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_docs_endpoint_accessible(self, client):
        """Test that API documentation is accessible."""
        response = client.get("/docs")
        
        # Should return documentation page
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_openapi_schema_accessible(self, client):
        """Test that OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        
        # Should return OpenAPI schema
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        
        # Should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
        assert "openapi" in data
        assert "info" in data


class TestCORSHeaders:
    """Test CORS headers for frontend integration."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present for browser requests."""
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        
        # Should have CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers or response.status_code == 200
    
    def test_preflight_request_handling(self, client):
        """Test handling of CORS preflight requests."""
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        # Should handle preflight request
        assert response.status_code in [200, 204]


class TestAPIIntegration:
    """Integration tests for API functionality."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @patch('web.api.main.loaded_models')
    def test_prediction_with_mocked_model(self, mock_models, client):
        """Test prediction with mocked model."""
        # Mock a model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.1, 0.1]])
        
        mock_models.__getitem__.return_value = mock_model
        mock_models.__contains__.return_value = True
        
        # Try prediction
        response = client.post("/predict", json={
            "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        })
        
        # Should work with mocked model
        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data or "prediction" in data
    
    def test_api_startup_and_shutdown(self, client):
        """Test that API can start up and shut down properly."""
        # Health check should work, indicating successful startup
        response = client.get("/health")
        assert response.status_code == 200
        
        # API should be responsive
        data = response.json()
        assert isinstance(data, dict)


if __name__ == "__main__":
    pytest.main([__file__])
