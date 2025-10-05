# 🚀 ExoHunter Web Applications - Enhanced Version

This directory contains the improved FastAPI backend and Streamlit frontend for ExoHunter exoplanet classification.

## 📋 What's New

### ✨ Major Improvements

#### Backend (FastAPI)
- ✅ **Enhanced Request Validation** - Comprehensive input validation with detailed error messages
- ✅ **Batch Prediction Endpoint** - Process up to 100 predictions at once
- ✅ **Model Metrics API** - Access training metrics and model performance data
- ✅ **Feature Information Endpoint** - Get detailed feature descriptions and requirements
- ✅ **Structured Logging** - JSON logging with file rotation and request tracking
- ✅ **Performance Tracking** - Request timing and performance metrics
- ✅ **Better Error Handling** - Improved error messages and exception handling
- ✅ **CORS Support** - Properly configured for web access
- ✅ **OpenAPI Documentation** - Enhanced API docs with tags and examples

#### Frontend (Streamlit)
- ✅ **Modern UI Design** - Gradient headers, better styling, responsive layout
- ✅ **Multiple Input Methods** - Manual input, CSV upload, quick samples
- ✅ **Rich Visualizations** - Plotly charts for confidence, predictions, and distributions
- ✅ **Prediction History** - Track and visualize past predictions
- ✅ **Batch Processing** - Upload CSV files for bulk predictions
- ✅ **CSV Export** - Download prediction results as CSV
- ✅ **Gauge Charts** - Visual confidence indicators
- ✅ **Better Error Messages** - Helpful troubleshooting information

#### Shared Utilities
- ✅ **Common Model Loading** - Reusable model loading logic
- ✅ **Feature Validation** - Centralized validation functions
- ✅ **Prediction Utilities** - Shared prediction and ensemble calculation
- ✅ **Logging Framework** - Advanced logging with request tracking
- ✅ **Performance Monitoring** - Metrics tracking and performance analysis

## 📁 Structure

```
web/
├── api/                          # FastAPI Backend
│   ├── main.py                  # Enhanced API with new endpoints
│   └── __pycache__/
├── streamlit/                    # Streamlit Frontend
│   └── app.py                   # Improved UI with visualizations
├── shared/                       # Shared Utilities
│   ├── __init__.py
│   ├── config.py                # Configuration
│   ├── schemas.py               # Data schemas
│   ├── utils.py                 # Common utilities (NEW)
│   └── logging_config.py        # Logging configuration (NEW)
├── requirements.txt             # Updated dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Navigate to web directory
cd web

# Install dependencies
pip install -r requirements.txt
```

### Running the Backend

```bash
# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**New API Features:**
- Health check: `http://localhost:8000/health`
- Predict: `http://localhost:8000/predict`
- Batch predict: `http://localhost:8000/predict/batch`
- Model metrics: `http://localhost:8000/models/metrics`
- Feature info: `http://localhost:8000/models/features`
- API docs: `http://localhost:8000/docs`

### Running the Frontend

```bash
# Start Streamlit app
streamlit run streamlit/app.py
```

**New UI Features:**
- 📝 Manual input with live validation
- 📄 CSV file upload for batch predictions
- 📋 Quick test samples
- 📊 Prediction history with charts
- 📥 Export results as CSV

## 📖 API Documentation

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T12:00:00",
  "model_loaded": true,
  "version": "2.0.0",
  "models_available": ["Extra Trees", "LightGBM", "Optimized Random Forest", "Optimized XGBoost"],
  "total_predictions": 42,
  "uptime_seconds": 3600.5
}
```

### Single Prediction

```bash
POST /predict
Content-Type: application/json

{
  "features": [1500, 2.5, 800, 150, 5800, 1.0, 1.0, 4.4, 1, 100, 15, 75, 0.3, 3.5, 400]
}
```

**Response:**
```json
{
  "predictions": {
    "Extra Trees": "CONFIRMED",
    "LightGBM": "CONFIRMED",
    "Optimized Random Forest": "CONFIRMED",
    "Optimized XGBoost": "CONFIRMED"
  },
  "confidence": {
    "Extra Trees": 92.5,
    "LightGBM": 94.2,
    "Optimized Random Forest": 91.8,
    "Optimized XGBoost": 95.1
  },
  "ensemble_prediction": "CONFIRMED",
  "ensemble_confidence": 100.0,
  "processing_time_ms": 45.23
}
```

### Batch Prediction

```bash
POST /predict/batch
Content-Type: application/json

{
  "batch_features": [
    [1500, 2.5, 800, 150, 5800, 1.0, 1.0, 4.4, 1, 100, 15, 75, 0.3, 3.5, 400],
    [800, 1.8, 600, 50, 5200, 0.8, 0.9, 4.5, 2, 30, 8, 25, 0.7, 2.8, 300]
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "predictions": {...},
      "confidence": {...},
      "ensemble_prediction": "CONFIRMED",
      "ensemble_confidence": 100.0,
      "processing_time_ms": 42.1
    },
    {
      "predictions": {...},
      "confidence": {...},
      "ensemble_prediction": "CANDIDATE",
      "ensemble_confidence": 75.0,
      "processing_time_ms": 38.7
    }
  ],
  "total_predictions": 2,
  "processing_time_ms": 95.3
}
```

### Model Metrics

```bash
GET /models/metrics
```

**Response:**
```json
[
  {
    "model_name": "Optimized XGBoost",
    "training_metrics": {
      "cv_mean": 0.8251,
      "cv_std": 0.0087,
      "test_accuracy": 0.8253,
      "classification_report": {...}
    },
    "feature_count": 15,
    "classes": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
    "model_type": "Optimized XGBoost"
  }
]
```

### Feature Information

```bash
GET /models/features
```

**Response:**
```json
{
  "feature_count": 15,
  "features": [
    {
      "index": 0,
      "name": "transit_depth",
      "description": "Feature 1: transit_depth"
    },
    ...
  ]
}
```

## 🎨 Streamlit UI Guide

### Manual Input Tab
- Enter feature values manually
- Live validation with min/max ranges
- Instant prediction with visualizations

### CSV Upload Tab
1. Download the CSV template
2. Fill in your data (up to 100 rows)
3. Upload and predict all at once
4. Download results as CSV

### Quick Samples Tab
- Pre-configured sample data
- One-click loading and prediction
- Test different exoplanet types

### History Tab
- View all past predictions
- Pie chart of prediction distribution
- Clear history option

## 📊 Visualizations

### Gauge Chart
Shows model agreement percentage with color-coded zones:
- 🟢 Green (75-100%): High confidence
- 🟡 Yellow (50-75%): Moderate confidence
- 🔴 Red (0-50%): Low confidence

### Bar Charts
- **Model Votes**: Distribution of predictions across models
- **Confidence Scores**: Confidence level for each model
- **Prediction History**: Overall prediction distribution

## 🔧 Configuration

### Shared Configuration (`shared/config.py`)
```python
class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    API_URL = "http://localhost:8000"
    MODEL_PATH = BASE_DIR / "models" / "trained_models"
```

### Logging Configuration
Logs are automatically created in `logs/` directory:
- `api.log` - General API logs (10MB rotation)
- `requests.log` - Request tracking (10MB rotation)
- `predictions.log` - Prediction history (daily rotation)
- `performance_metrics.json` - Performance statistics

## 🛠️ Utilities

### Model Loading
```python
from web.shared.utils import load_trained_models

models, scaler, features, metrics = load_trained_models()
```

### Feature Validation
```python
from web.shared.utils import validate_features, FeatureValidationError

try:
    validate_features(feature_list)
except FeatureValidationError as e:
    print(f"Validation error: {e}")
```

### Predictions
```python
from web.shared.utils import predict_with_model

prediction, confidence = predict_with_model(
    features, model, "Model Name", scaler
)
```

### Ensemble Calculation
```python
from web.shared.utils import calculate_ensemble_prediction

ensemble_pred, confidence = calculate_ensemble_prediction(predictions)
```

## 📈 Performance

### Optimizations
- **Caching**: Model loading cached with `@st.cache_resource`
- **Batch Processing**: Process multiple predictions efficiently
- **Request Timing**: Track and log processing times
- **Efficient Scaling**: Vectorized feature preprocessing

### Monitoring
Access performance metrics via the health endpoint or check `logs/performance_metrics.json`:
```json
{
  "total_requests": 150,
  "total_predictions": 175,
  "avg_response_time_ms": 45.3,
  "prediction_distribution": {
    "CONFIRMED": 80,
    "CANDIDATE": 60,
    "FALSE POSITIVE": 35
  },
  "error_count": 2,
  "start_time": "2025-10-05T10:00:00"
}
```

## 🐛 Troubleshooting

### Models Not Loading
**Error**: "No models could be loaded"

**Solutions**:
1. Verify models exist: `ls -la models/trained_models/`
2. Check file permissions
3. Review logs in `logs/api.log`
4. Re-run training: `python scripts/optimized_training.py`

### Feature Validation Errors
**Error**: "Expected 15 features, got X"

**Solutions**:
1. Ensure exactly 15 numeric values
2. Check feature order matches documentation
3. Use `/models/features` endpoint for reference

### Port Already in Use
**Error**: "Address already in use"

**Solutions**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Use different port
uvicorn api.main:app --port 8001
```

### CSV Upload Issues
**Error**: "Error processing file"

**Solutions**:
1. Download and use the CSV template
2. Ensure all values are numeric
3. Check for missing values
4. Verify 15 columns exist

## 🔒 Security

### Input Validation
- All features validated for type and range
- NaN and Inf values rejected
- Batch size limited to 100

### Error Handling
- Detailed error messages
- Exception logging with stack traces
- Graceful degradation

### CORS
- Configured for cross-origin requests
- Adjust in `api/main.py` if needed

## 📚 Dependencies

### Core
- **FastAPI**: Web framework for API
- **Streamlit**: Interactive web interface
- **Plotly**: Advanced visualizations
- **Pydantic**: Data validation

### Machine Learning
- **scikit-learn**: Model training and preprocessing
- **XGBoost**: Gradient boosting models
- **LightGBM**: Efficient gradient boosting
- **joblib**: Model serialization

### Utilities
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **python-multipart**: File upload support

## 🎯 Future Enhancements

Potential additions:
- [ ] WebSocket support for real-time predictions
- [ ] User authentication and API keys
- [ ] Rate limiting
- [ ] Model versioning and A/B testing
- [ ] Feature importance visualization
- [ ] SHAP value explanations
- [ ] Database integration for predictions
- [ ] Model retraining pipeline
- [ ] Prometheus metrics export
- [ ] Docker deployment configuration

## 📞 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review API documentation at `/docs`
3. Check model files exist in `models/trained_models/`
4. Verify all dependencies are installed

## 📄 License

Part of the ExoHunter project - Advanced Exoplanet Classification System

---

**Version**: 2.0.0  
**Last Updated**: October 5, 2025  
**Python**: 3.8+
