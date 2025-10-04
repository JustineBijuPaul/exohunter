"""
Pytest configuration for ExoHunter test suite.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data_path():
    """Provide path to sample test data."""
    return project_root / "tests" / "data"


@pytest.fixture
def models_dir():
    """Provide path to models directory."""
    return project_root / "models"


@pytest.fixture
def mock_kepler_data():
    """Mock Kepler dataset for testing."""
    import pandas as pd
    import numpy as np
    
    # Create synthetic data that mimics Kepler format
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'kepid': range(1000000, 1000000 + n_samples),
        'koi_period': np.random.uniform(1, 400, n_samples),
        'koi_depth': np.random.uniform(0.001, 0.1, n_samples),
        'koi_duration': np.random.uniform(1, 10, n_samples),
        'koi_slogg': np.random.uniform(3.5, 5.0, n_samples),
        'koi_srad': np.random.uniform(0.5, 2.0, n_samples),
        'ra': np.random.uniform(0, 360, n_samples),
        'dec': np.random.uniform(-90, 90, n_samples),
        'koi_kepmag': np.random.uniform(10, 17, n_samples),
        'koi_disposition': np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_lightcurve_data():
    """Mock light curve data for testing."""
    import numpy as np
    
    np.random.seed(42)
    time = np.linspace(0, 27.4, 1000)  # ~27 day period
    
    # Create a simple transit signal
    transit_depth = 0.01
    transit_duration = 0.2
    period = 3.5
    
    flux = np.ones_like(time)
    for i in range(8):  # 8 transits
        transit_center = i * period
        mask = np.abs((time - transit_center) % period) < transit_duration / 2
        flux[mask] -= transit_depth
    
    # Add noise
    flux += np.random.normal(0, 0.001, len(time))
    
    return {
        'time': time,
        'flux': flux
    }


@pytest.fixture
def sample_features():
    """Sample feature array for testing."""
    import numpy as np
    np.random.seed(42)
    return np.random.uniform(0, 1, 8)  # 8 features as commonly used


@pytest.fixture(scope="session")
def test_database():
    """Create a test database for testing database operations."""
    import tempfile
    import sqlite3
    
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def api_client():
    """Create a test client for FastAPI testing."""
    from fastapi.testclient import TestClient
    from web.api.main import app
    
    return TestClient(app)


# Configure pytest to run tests in a specific order if needed
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance/smoke test"
    )


# Add custom command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true", 
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="run performance smoke tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--run-performance"):
        skip_performance = pytest.mark.skip(reason="need --run-performance option to run")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)
