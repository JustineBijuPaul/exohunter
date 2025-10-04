# ExoHunter User Guide

Welcome to ExoHunter, an AI-powered exoplanet classification system that helps identify and classify potential exoplanets from astronomical data. This guide will walk you through using the system for exoplanet analysis.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Running the Application](#running-the-application)
3. [Web Interface](#web-interface)
4. [API Usage](#api-usage)
5. [Dataset Upload](#dataset-upload)
6. [CSV Data Format](#csv-data-format)
7. [Understanding Results](#understanding-results)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Docker and Docker Compose (recommended)
- OR Python 3.11+ with pip

### Quick Start with Docker

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/exohunter.git
   cd exohunter
   ```

2. **Start the application:**
   ```bash
   ./deploy.sh start
   ```

3. **Access the application:**
   - Web Interface: http://localhost:3000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Manual Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server:**
   ```bash
   uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Start the frontend (in another terminal):**
   ```bash
   cd web/frontend
   npm install
   npm run dev
   ```

## Running the Application

### Using Docker (Recommended)

The deployment script provides several convenient commands:

```bash
# Start all services
./deploy.sh start

# View logs
./deploy.sh logs

# Check health status
./deploy.sh health

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart
```

### Development Mode

For development with hot reloading:

```bash
# Start API in development mode
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend in development mode
cd web/frontend && npm run dev

# Start Streamlit interface (optional)
streamlit run web/streamlit/app.py
```

## Web Interface

### Main Features

1. **Single Prediction:** Upload individual data points for classification
2. **Batch Upload:** Upload CSV files for bulk prediction
3. **Light Curve Analysis:** Analyze time-series photometric data
4. **Results Visualization:** View prediction results with confidence scores

### Using the Web Interface

1. **Navigate to the web interface** at http://localhost:3000
2. **Choose your prediction method:**
   - **Manual Input:** Enter stellar and planetary parameters manually
   - **CSV Upload:** Upload a CSV file with multiple objects
   - **Light Curve:** Upload time-series flux data

3. **Review results:** View predictions with confidence scores and explanations

## API Usage

The ExoHunter API provides programmatic access to exoplanet classification.

### Base URL
- Local: `http://localhost:8000`
- Production: Your deployed URL

### Authentication
Currently, no authentication is required for the public endpoints.

### Available Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-03T10:00:00Z",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### 2. Single Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [
    365.25,    // orbital_period (days)
    1.0,       // planet_radius (Earth radii)
    4.5,       // transit_duration (hours)
    5778,      // stellar_teff (Kelvin)
    1.0,       // stellar_radius (Solar radii)
    4.44       // stellar_logg (log g)
  ]
}
```

**Response:**
```json
{
  "predicted_label": "CANDIDATE",
  "probability": 0.85,
  "confidence": "HIGH",
  "all_probabilities": {
    "CANDIDATE": 0.85,
    "FALSE POSITIVE": 0.12,
    "CONFIRMED": 0.03
  },
  "model_version": "random_forest_baseline_v1.0"
}
```

#### 3. Light Curve Prediction
```http
POST /predict/lightcurve
Content-Type: application/json
```

**Request Body:**
```json
{
  "time": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  "flux": [1.0, 0.99, 0.97, 0.99, 1.01, 1.0, 0.98, 1.02, 1.0, 0.99]
}
```

#### 4. File Upload Prediction
```http
POST /predict/upload
Content-Type: multipart/form-data
```

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_data.csv"
```

#### 5. Model Metrics
```http
GET /model/metrics
```

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "accuracy": 0.85,
  "precision": 0.83,
  "recall": 0.87,
  "f1_score": 0.85,
  "feature_count": 6,
  "classes": ["CANDIDATE", "FALSE POSITIVE", "CONFIRMED"]
}
```

#### 6. Prediction History
```http
GET /predictions/history?limit=10&offset=0
```

### Example Usage with Python

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Single prediction
def predict_exoplanet(features):
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"features": features}
    )
    return response.json()

# Example prediction
features = [365.25, 1.0, 4.5, 5778, 1.0, 4.44]
result = predict_exoplanet(features)
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['probability']:.2f}")

# File upload prediction
def predict_from_file(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/predict/upload",
            files={"file": f}
        )
    return response.json()

# Health check
def check_health():
    response = requests.get(f"{BASE_URL}/health")
    return response.json()
```

### Example Usage with cURL

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [365.25, 1.0, 4.5, 5778, 1.0, 4.44]
     }'

# File upload
curl -X POST "http://localhost:8000/predict/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/example_toi.csv"
```

## Dataset Upload

### Supported Formats

- **CSV files** (.csv)
- **Gzipped CSV files** (.csv.gz)
- **Maximum file size:** 50MB

### Upload Methods

1. **Web Interface:** Use the file upload component
2. **API Endpoint:** POST to `/predict/upload`
3. **Drag and Drop:** Drag files directly onto the web interface

### Processing

- Files are automatically validated for format and schema
- Missing values are handled using median imputation
- Outliers beyond 3 standard deviations are capped
- Results include individual predictions for each row

## CSV Data Format

### Required Columns

The CSV file must contain the following columns (column names are case-insensitive):

| Column Name | Description | Units | Example |
|-------------|-------------|-------|---------|
| `orbital_period` or `period` | Orbital period of the planet | days | 365.25 |
| `planet_radius` or `radius` | Planet radius | Earth radii | 1.0 |
| `transit_duration` or `duration` | Transit duration | hours | 4.5 |
| `stellar_teff` | Stellar effective temperature | Kelvin | 5778 |
| `stellar_radius` | Stellar radius | Solar radii | 1.0 |
| `stellar_logg` | Stellar surface gravity | log(cm/s²) | 4.44 |

### Optional Columns

| Column Name | Description | Purpose |
|-------------|-------------|---------|
| `disposition` | Known classification | Validation/comparison |
| `object_id` | Unique identifier | Result tracking |
| `ra` | Right ascension | Astronomical coordinates |
| `dec` | Declination | Astronomical coordinates |

### Alternative Column Names

The system automatically maps common column name variations:

| Standard Name | Alternative Names |
|---------------|-------------------|
| `orbital_period` | `koi_period`, `period`, `toi_period` |
| `planet_radius` | `koi_prad`, `radius`, `toi_prad` |
| `transit_duration` | `koi_duration`, `duration`, `toi_duration` |
| `stellar_teff` | `koi_stemp`, `stellar_teff`, `toi_stemp` |
| `stellar_radius` | `koi_srad`, `stellar_rad`, `toi_srad` |
| `stellar_logg` | `koi_slogg`, `stellar_logg`, `toi_slogg` |

### Example CSV Structure

```csv
object_id,orbital_period,planet_radius,transit_duration,stellar_teff,stellar_radius,stellar_logg,disposition
TOI-1001.01,12.34,1.15,3.2,5750,0.89,4.52,CANDIDATE
TOI-1002.01,45.67,2.34,5.8,6200,1.23,4.15,FALSE POSITIVE
TOI-1003.01,8.90,0.78,2.1,5200,0.67,4.78,CONFIRMED
```

### Data Quality Requirements

- **Finite values:** All numerical values must be finite (no NaN, inf, -inf)
- **Positive values:** Physical parameters should be positive
- **Reasonable ranges:**
  - Orbital period: 0.1 - 10,000 days
  - Planet radius: 0.1 - 20 Earth radii
  - Transit duration: 0.1 - 48 hours
  - Stellar temperature: 2000 - 10,000 K
  - Stellar radius: 0.1 - 10 Solar radii
  - Stellar log g: 2.0 - 6.0

## Understanding Results

### Classification Labels

- **CONFIRMED:** High-confidence exoplanet detection
- **CANDIDATE:** Potential exoplanet requiring further validation
- **FALSE POSITIVE:** Not a planetary signal (stellar activity, binaries, etc.)

### Confidence Levels

- **HIGH (>0.8):** Very reliable prediction
- **MEDIUM (0.6-0.8):** Good prediction with some uncertainty
- **LOW (<0.6):** Uncertain prediction, needs careful review

### Probability Scores

Each prediction includes probability scores for all classes:
- Values sum to 1.0
- Higher values indicate stronger confidence
- Compare relative probabilities between classes

### Interpretation Guidelines

1. **High confidence CONFIRMED:** Strong exoplanet candidate
2. **High confidence CANDIDATE:** Promising target for follow-up
3. **High confidence FALSE POSITIVE:** Likely not a planet
4. **Low confidence results:** Consider additional observations

## Troubleshooting

### Common Issues

#### API Connection Error
```
Error: Connection refused
```
**Solution:** Ensure the API server is running on port 8000

#### File Upload Fails
```
Error: Invalid file format
```
**Solutions:**
- Verify CSV format and encoding (UTF-8)
- Check required columns are present
- Ensure file size < 50MB

#### Empty Predictions
```
Error: No valid features found
```
**Solutions:**
- Check column names match expected format
- Verify data contains finite numerical values
- Remove rows with excessive missing data

#### Model Not Loaded
```
Error: Model not available
```
**Solution:** Wait for model initialization or restart the service

### Getting Help

1. **Check logs:** Use `./deploy.sh logs` to view service logs
2. **Health status:** Visit `/health` endpoint to verify system status
3. **API documentation:** Visit `/docs` for interactive API documentation
4. **GitHub Issues:** Report bugs and feature requests on GitHub

### Performance Tips

- **Batch processing:** Upload multiple objects in CSV files for efficiency
- **Data preprocessing:** Clean and validate data before upload
- **Regular updates:** Keep the application updated for latest models
- **Resource monitoring:** Monitor system resources for large datasets

## Next Steps

- Explore the [Developer Guide](DEVELOPER_GUIDE.md) for customization options
- Learn about training custom models
- Set up monitoring and logging for production use
- Integrate with your existing astronomical workflows

For technical questions and advanced usage, see the [Developer Guide](DEVELOPER_GUIDE.md).
