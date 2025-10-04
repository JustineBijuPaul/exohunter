# ExoHunter Developer Guide

This guide provides comprehensive information for developers working with ExoHunter, including local development setup, model training, API customization, and deployment strategies.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Local Development](#local-development)
4. [API Development](#api-development)
5. [Model Training](#model-training)
6. [Data Pipeline](#data-pipeline)
7. [Testing](#testing)
8. [Database Integration](#database-integration)
9. [Frontend Development](#frontend-development)
10. [Deployment](#deployment)
11. [Contributing](#contributing)

## Development Environment Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Docker & Docker Compose**
- **Git**
- **PostgreSQL** (for database features)

### Initial Setup

1. **Clone and setup repository:**
   ```bash
   git clone https://github.com/your-repo/exohunter.git
   cd exohunter
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

2. **Setup pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Environment configuration:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Development Dependencies

Install additional tools for development:

```bash
pip install pytest pytest-cov black flake8 mypy pre-commit
npm install -g @commitlint/cli @commitlint/config-conventional
```

## Project Structure

```
exohunter/
├── exohunter/              # Core Python package
│   ├── __init__.py
│   ├── db/                 # Database models and utilities
│   └── data/               # Data processing modules
├── web/                    # Web applications
│   ├── api/                # FastAPI backend
│   │   ├── main.py         # API entry point
│   │   ├── models.py       # Pydantic models
│   │   └── endpoints/      # API route modules
│   ├── frontend/           # React frontend
│   │   ├── src/
│   │   ├── package.json
│   │   └── Dockerfile
│   └── streamlit/          # Streamlit interface
├── data/                   # Data processing scripts
│   ├── ingest.py           # Data ingestion
│   └── labels.py           # Label mapping
├── models/                 # ML model training
│   └── train_baseline.py   # Training scripts
├── tests/                  # Test suite
├── deploy/                 # Deployment configurations
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks
├── Dockerfile              # Main application container
├── docker-compose.yml      # Multi-service orchestration
├── requirements.txt        # Python dependencies
└── deploy.sh              # Deployment script
```

### Key Components

- **`web/api/`**: FastAPI backend with ML inference endpoints
- **`web/frontend/`**: React-based web interface
- **`exohunter/`**: Core library with data processing and ML utilities
- **`models/`**: Model training and evaluation scripts
- **`data/`**: Data ingestion and preprocessing modules
- **`tests/`**: Comprehensive test suite

## Local Development

### Running Services

#### Option 1: Docker Compose (Recommended)
```bash
# Start all services in development mode
docker-compose -f docker-compose.dev.yml up

# Or use the deployment script
./deploy.sh start
```

#### Option 2: Manual Setup

**Start the API server:**
```bash
# Activate virtual environment
source venv/bin/activate

# Start with hot reload
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload

# With debug logging
DEBUG=true uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

**Start the frontend:**
```bash
cd web/frontend
npm install
npm run dev  # Starts on port 3000
```

**Start Streamlit interface (optional):**
```bash
streamlit run web/streamlit/app.py --server.port 8501
```

### Development Workflow

1. **Feature development:**
   ```bash
   git checkout -b feature/your-feature-name
   # Make changes
   git add .
   git commit -m "feat: add your feature description"
   ```

2. **Run tests:**
   ```bash
   pytest tests/ -v --cov=exohunter
   ```

3. **Code quality checks:**
   ```bash
   black .                    # Format code
   flake8 .                   # Lint code
   mypy exohunter/            # Type checking
   ```

4. **Submit changes:**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request
   ```

## API Development

### FastAPI Application Structure

The API is built with FastAPI and follows these patterns:

```python
# web/api/main.py - Main application
from fastapi import FastAPI, HTTPException, Depends
from .models import PredictionRequest, PredictionResponse
from .endpoints import predictions, health, admin

app = FastAPI(
    title="ExoHunter API",
    description="Exoplanet classification API",
    version="1.0.0"
)

# Include routers
app.include_router(predictions.router, prefix="/predict")
app.include_router(health.router, prefix="/health")
app.include_router(admin.router, prefix="/admin")
```

### Adding New Endpoints

1. **Create endpoint module:**
   ```python
   # web/api/endpoints/my_endpoint.py
   from fastapi import APIRouter, HTTPException
   from ..models import MyRequest, MyResponse
   
   router = APIRouter()
   
   @router.post("/my-endpoint", response_model=MyResponse)
   async def my_endpoint(request: MyRequest):
       # Implementation
       return MyResponse(result="success")
   ```

2. **Define Pydantic models:**
   ```python
   # web/api/models.py
   from pydantic import BaseModel, Field
   
   class MyRequest(BaseModel):
       input_data: str = Field(..., description="Input data")
   
   class MyResponse(BaseModel):
       result: str = Field(..., description="Result")
   ```

3. **Include router:**
   ```python
   # web/api/main.py
   from .endpoints import my_endpoint
   app.include_router(my_endpoint.router, prefix="/api")
   ```

### Request/Response Models

All API endpoints use Pydantic models for validation:

```python
class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        description="Numerical features for classification",
        min_items=1
    )
    
    @validator('features')
    def validate_features(cls, v):
        if not all(np.isfinite(f) for f in v):
            raise ValueError("All features must be finite")
        return v

class PredictionResponse(BaseModel):
    predicted_label: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence: str
    all_probabilities: Dict[str, float]
    model_version: str
```

### Error Handling

```python
from fastapi import HTTPException

# Standard error responses
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": "validation_error"}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "type": "not_found"}
    )
```

### API Testing

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from web.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={"features": [365.25, 1.0, 4.5, 5778, 1.0, 4.44]}
    )
    assert response.status_code == 200
    assert "predicted_label" in response.json()
```

## Model Training

### Training Script Usage

The baseline training script supports multiple datasets and models:

```bash
# Train on local data
python models/train_baseline.py --data data/your_dataset.csv

# Train with custom parameters
python models/train_baseline.py \
    --data data/your_dataset.csv \
    --model random_forest \
    --test-size 0.2 \
    --cv-folds 5 \
    --output models/my_model.pkl

# Train XGBoost model
python models/train_baseline.py \
    --data data/your_dataset.csv \
    --model xgboost \
    --xgb-n-estimators 200 \
    --xgb-max-depth 8
```

### Custom Model Training

Create your own training script:

```python
# models/train_custom.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_custom_model(data_path: str, output_path: str):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Feature engineering
    features = [
        'orbital_period', 'planet_radius', 'transit_duration',
        'stellar_teff', 'stellar_radius', 'stellar_logg'
    ]
    
    X = df[features].fillna(df[features].median())
    y = df['disposition']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_custom_model("data/dataset.csv", "models/custom_model.pkl")
```

### Model Integration

To use a custom model in the API:

1. **Save model with metadata:**
   ```python
   import joblib
   
   model_info = {
       'model': trained_model,
       'features': feature_names,
       'classes': class_names,
       'version': '1.0.0',
       'metrics': {'accuracy': 0.85}
   }
   
   joblib.dump(model_info, 'models/my_model.pkl')
   ```

2. **Update model loading in API:**
   ```python
   # web/api/main.py
   def load_models():
       global loaded_models
       model_path = "models/my_model.pkl"
       
       if os.path.exists(model_path):
           loaded_models['my_model'] = joblib.load(model_path)
           logger.info(f"Loaded model: my_model")
   ```

### Feature Engineering

Common feature engineering patterns:

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for better model performance."""
    
    # Planetary characteristics
    df['planet_density'] = df['planet_mass'] / (df['planet_radius'] ** 3)
    df['equilibrium_temp'] = df['stellar_teff'] * np.sqrt(
        df['stellar_radius'] / (2 * df['orbital_period'])
    )
    
    # Stellar characteristics
    df['stellar_mass'] = np.power(10, 0.4 * (4.83 - df['stellar_logg']))
    
    # Transit characteristics
    df['transit_depth'] = (df['planet_radius'] / df['stellar_radius']) ** 2
    df['impact_parameter'] = df['orbital_inclination'] / 90.0
    
    return df
```

### Model Evaluation

```python
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation."""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_names = ['orbital_period', 'planet_radius', 'transit_duration',
                        'stellar_teff', 'stellar_radius', 'stellar_logg']
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.show()
```

## Data Pipeline

### Data Ingestion

The data ingestion module supports multiple astronomical catalogs:

```python
# data/ingest.py usage
from data.ingest import download_dataset, load_kepler_koi

# Download datasets
success = download_dataset(
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=...",
    "data/kepler_koi.csv"
)

# Load and preprocess
df = load_kepler_koi("data/kepler_koi.csv")
```

### Custom Data Sources

Add support for new data sources:

```python
def load_custom_catalog(file_path: str) -> pd.DataFrame:
    """Load custom exoplanet catalog."""
    
    df = pd.read_csv(file_path)
    
    # Standardize column names
    column_mapping = {
        'your_period_col': 'orbital_period',
        'your_radius_col': 'planet_radius',
        # Add more mappings
    }
    
    df = df.rename(columns=column_mapping)
    
    # Clean data
    df = df.dropna(subset=['orbital_period', 'planet_radius'])
    df = df[df['orbital_period'] > 0]
    
    return df
```

### Data Validation

```python
from pydantic import BaseModel, validator
from typing import List

class ExoplanetData(BaseModel):
    """Data validation schema."""
    
    orbital_period: float
    planet_radius: float
    transit_duration: float
    stellar_teff: float
    stellar_radius: float
    stellar_logg: float
    
    @validator('orbital_period')
    def validate_period(cls, v):
        if not 0.1 <= v <= 10000:
            raise ValueError('Orbital period must be between 0.1 and 10000 days')
        return v
    
    @validator('planet_radius')
    def validate_radius(cls, v):
        if not 0.1 <= v <= 20:
            raise ValueError('Planet radius must be between 0.1 and 20 Earth radii')
        return v

def validate_dataset(df: pd.DataFrame) -> List[str]:
    """Validate entire dataset and return errors."""
    errors = []
    
    for idx, row in df.iterrows():
        try:
            ExoplanetData(**row.to_dict())
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
    
    return errors
```

## Testing

### Test Structure

```
tests/
├── conftest.py                 # Pytest configuration
├── test_api.py                # API endpoint tests
├── test_models.py             # Model tests
├── test_data_processing.py    # Data pipeline tests
├── test_integration.py        # Integration tests
└── fixtures/                  # Test data files
    └── sample_data.csv
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=exohunter --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_prediction_endpoint -v

# Run tests with marks
pytest -m "not slow"  # Skip slow tests
```

### Test Configuration

```python
# conftest.py
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from web.api.main import app

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def sample_data():
    """Sample dataset for testing."""
    return pd.DataFrame({
        'orbital_period': [365.25, 12.34, 45.67],
        'planet_radius': [1.0, 1.15, 2.34],
        'transit_duration': [4.5, 3.2, 5.8],
        'stellar_teff': [5778, 5750, 6200],
        'stellar_radius': [1.0, 0.89, 1.23],
        'stellar_logg': [4.44, 4.52, 4.15],
        'disposition': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
    })

@pytest.fixture
def sample_features():
    """Sample feature vector."""
    return [365.25, 1.0, 4.5, 5778, 1.0, 4.44]
```

### API Testing

```python
# tests/test_api.py
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_prediction_endpoint(client, sample_features):
    response = client.post("/predict", json={"features": sample_features})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert "probability" in data
    assert 0 <= data["probability"] <= 1

def test_invalid_features(client):
    response = client.post("/predict", json={"features": [float('inf')]})
    assert response.status_code == 422  # Validation error
```

### Model Testing

```python
# tests/test_models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def test_model_training(sample_data):
    features = ['orbital_period', 'planet_radius', 'transit_duration',
                'stellar_teff', 'stellar_radius', 'stellar_logg']
    
    X = sample_data[features]
    y = sample_data['disposition']
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in y.unique() for pred in predictions)

def test_model_persistence(tmp_path, sample_data):
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = sample_data[['orbital_period', 'planet_radius', 'transit_duration',
                     'stellar_teff', 'stellar_radius', 'stellar_logg']]
    y = sample_data['disposition']
    model.fit(X, y)
    
    # Save model
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(model, model_path)
    
    # Load and test
    loaded_model = joblib.load(model_path)
    predictions = loaded_model.predict(X)
    assert len(predictions) == len(y)
```

## Database Integration

### Database Setup

The application uses PostgreSQL with SQLAlchemy:

```python
# exohunter/db/__init__.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/exohunter")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Database Models

```python
# exohunter/db/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    input_features = Column(JSON)
    predicted_label = Column(String)
    probability = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class APIRequest(Base):
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String)
    method = Column(String)
    status_code = Column(Integer)
    response_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Database Operations

```python
# exohunter/db/operations.py
from sqlalchemy.orm import Session
from .models import Prediction, APIRequest

def save_prediction(db: Session, features: list, prediction: dict) -> Prediction:
    """Save prediction to database."""
    db_prediction = Prediction(
        input_features=features,
        predicted_label=prediction["predicted_label"],
        probability=prediction["probability"],
        model_version=prediction["model_version"]
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def log_api_request(db: Session, endpoint: str, method: str, 
                   status_code: int, response_time: float):
    """Log API request."""
    db_request = APIRequest(
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time=response_time
    )
    db.add(db_request)
    db.commit()
```

## Frontend Development

### React Application Structure

```
web/frontend/
├── src/
│   ├── components/          # Reusable components
│   │   ├── PredictionForm.jsx
│   │   ├── ResultsDisplay.jsx
│   │   └── FileUpload.jsx
│   ├── pages/              # Page components
│   │   ├── Home.jsx
│   │   ├── Predict.jsx
│   │   └── History.jsx
│   ├── hooks/              # Custom React hooks
│   │   └── useAPI.js
│   ├── utils/              # Utility functions
│   │   └── api.js
│   ├── App.jsx            # Main app component
│   └── main.jsx           # Entry point
├── public/
├── package.json
└── vite.config.js
```

### API Integration

```javascript
// src/utils/api.js
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const api = {
  async predict(features) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ features }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
  },

  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/predict/upload`, {
      method: 'POST',
      body: formData,
    });
    
    return response.json();
  },

  async getHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  }
};
```

### Custom Hooks

```javascript
// src/hooks/useAPI.js
import { useState, useEffect } from 'react';
import { api } from '../utils/api';

export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const predict = async (features) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await api.predict(features);
      setResult(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return { predict, loading, result, error };
}
```

### Component Development

```javascript
// src/components/PredictionForm.jsx
import React, { useState } from 'react';
import { usePrediction } from '../hooks/useAPI';

export function PredictionForm() {
  const [features, setFeatures] = useState({
    orbital_period: '',
    planet_radius: '',
    transit_duration: '',
    stellar_teff: '',
    stellar_radius: '',
    stellar_logg: ''
  });
  
  const { predict, loading, result, error } = usePrediction();

  const handleSubmit = (e) => {
    e.preventDefault();
    const featureArray = Object.values(features).map(Number);
    predict(featureArray);
  };

  return (
    <form onSubmit={handleSubmit}>
      {Object.entries(features).map(([key, value]) => (
        <div key={key}>
          <label>{key.replace('_', ' ')}</label>
          <input
            type="number"
            value={value}
            onChange={(e) => setFeatures({
              ...features,
              [key]: e.target.value
            })}
            required
          />
        </div>
      ))}
      
      <button type="submit" disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
      
      {error && <div className="error">{error}</div>}
      {result && <ResultsDisplay result={result} />}
    </form>
  );
}
```

## Deployment

### Docker Deployment

The application includes comprehensive Docker support:

```bash
# Build and start all services
./deploy.sh start

# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up --scale api=3 --scale frontend=2
```

### Environment Configuration

```bash
# .env.production
DEBUG=false
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@db:5432/exohunter
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-secret-key
CORS_ORIGINS=https://yourdomain.com
```

### Cloud Deployment

#### AWS ECS
```bash
# Build and push images
docker build -t your-repo/exohunter-api .
docker push your-repo/exohunter-api

# Deploy with ECS CLI
ecs-cli compose --file docker-compose.aws.yml service up
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/exohunter-api
gcloud run deploy --image gcr.io/PROJECT-ID/exohunter-api --platform managed
```

#### Heroku
```bash
# Deploy with Heroku
heroku create your-app-name
heroku stack:set container
git push heroku main
```

### Production Monitoring

```python
# Add monitoring middleware
from fastapi import Request
import time
import logging

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"Path: {request.url.path} "
        f"Method: {request.method} "
        f"Status: {response.status_code} "
        f"Duration: {process_time:.3f}s"
    )
    
    return response
```

### Health Checks

```python
# Enhanced health check
@app.get("/health")
async def health_check():
    checks = {
        "api": "healthy",
        "database": await check_database(),
        "model": check_model_loaded(),
        "memory": check_memory_usage()
    }
    
    status = "healthy" if all(
        check == "healthy" for check in checks.values()
    ) else "unhealthy"
    
    return {
        "status": status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

## Contributing

### Development Guidelines

1. **Code Style:**
   - Use Black for Python formatting
   - Follow PEP 8 guidelines
   - Use type hints where possible
   - Write docstrings for functions and classes

2. **Testing:**
   - Write tests for new features
   - Maintain >90% code coverage
   - Test edge cases and error conditions
   - Use descriptive test names

3. **Documentation:**
   - Update documentation for API changes
   - Include examples in docstrings
   - Update README for new features

4. **Git Workflow:**
   - Use conventional commit messages
   - Create feature branches for new work
   - Submit pull requests for review
   - Keep commits focused and atomic

### Commit Message Format

```
type(scope): description

Examples:
feat(api): add batch prediction endpoint
fix(model): handle missing features gracefully
docs(readme): update installation instructions
test(api): add integration tests for upload endpoint
```

### Pull Request Process

1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite
4. Update documentation
5. Submit pull request with description
6. Address review feedback
7. Merge after approval

### Issue Reporting

When reporting issues, include:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and logs
- Sample data if applicable

For feature requests, describe:
- Use case and motivation
- Proposed solution
- Alternative approaches considered
- Impact on existing functionality

## Additional Resources

- **API Documentation:** http://localhost:8000/docs (when running)
- **User Guide:** [USER_GUIDE.md](USER_GUIDE.md)
- **Contributing Guidelines:** [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Issue Tracker:** GitHub Issues
- **Discussions:** GitHub Discussions

For questions and support, please use GitHub Issues or Discussions.
