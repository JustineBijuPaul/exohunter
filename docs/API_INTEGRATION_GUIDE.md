# ExoHunter API - Ultimate Model Integration Guide

## 🎉 Overview

The ExoHunter API has been updated to use the **Ultimate Ensemble Model** (version 3.0), which achieves:

- **92.53% accuracy** on test data
- **97.14% ROC AUC** (excellent discrimination)
- **75.4% high-confidence predictions** (>80% probability)
- **52.9% very high-confidence predictions** (>90% probability)

## 🚀 Quick Start

### Starting the API Server

```bash
# Activate virtual environment
cd /home/linxcapture/Desktop/projects/exohunter
source venv/bin/activate

# Start the API
uvicorn web.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Endpoints**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Testing the API

```bash
# Run the test suite
python web/api/test_api_with_ultimate_model.py
```

## 📊 Model Details

### Ultimate Ensemble Architecture

The Ultimate Model is a sophisticated 5-classifier ensemble using **soft voting**:

| Classifier | Parameters | Weight |
|------------|------------|--------|
| **Random Forest** | 500 trees, max_depth=20 | 3 |
| **Extra Trees** | 500 trees, max_depth=20 | 3 |
| **Gradient Boosting** | 300 estimators | 2 |
| **Logistic Regression** | L2 regularization | 1 |
| **SVM (RBF)** | C=20.0, probability=True | 1 |

### Performance Metrics

```
Accuracy:              92.53%
Precision:             88.29%
Recall:                91.50%
F1-Score:              89.87%
ROC AUC:               97.14%
Cross-Validation:      92.09% ± 0.76%

High Confidence:       75.4% (>80%)
Very High Confidence:  52.9% (>90%)
```

### Required Features (13 total)

All features must be provided in this exact order:

1. **orbital_period** - Orbital period in days
2. **transit_depth** - Transit depth in ppm
3. **planet_radius** - Planet radius in Earth radii
4. **koi_teq** - Equilibrium temperature in Kelvin
5. **koi_insol** - Insolation flux (Earth = 1.0)
6. **stellar_teff** - Stellar effective temperature in Kelvin
7. **stellar_radius** - Stellar radius in Solar radii
8. **koi_smass** - Stellar mass in Solar masses
9. **koi_slogg** - Stellar surface gravity (log g)
10. **transit_duration** - Transit duration in hours
11. **impact_parameter** - Impact parameter (0 = central transit)
12. **koi_max_mult_ev** - Maximum multiple event statistic
13. **koi_num_transits** - Number of transits observed

## 🔌 API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-04T12:00:00",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 2. Model Metrics

**GET** `/model/metrics`

Get comprehensive model performance metrics.

**Response:**
```json
{
  "model_type": "ultimate - 5-classifier ensemble (RF+ET+GB+LR+SVM)",
  "training_accuracy": 0.9923,
  "validation_accuracy": 0.9269,
  "test_accuracy": 0.9253,
  "cross_validation_score": 0.9209,
  "precision": 0.8829,
  "recall": 0.9150,
  "f1_score": 0.8987,
  "roc_auc": 0.9714,
  "feature_count": 13,
  "classes": ["FALSE POSITIVE", "CONFIRMED PLANET"],
  "last_updated": "2025-10-04T12:00:00",
  "high_confidence_pct": 0.7540,
  "very_high_confidence_pct": 0.5290,
  "model_version": "3.0_ultimate_ensemble",
  "feature_names": [
    "orbital_period", "transit_depth", "planet_radius",
    "koi_teq", "koi_insol", "stellar_teff", "stellar_radius",
    "koi_smass", "koi_slogg", "transit_duration",
    "impact_parameter", "koi_max_mult_ev", "koi_num_transits"
  ]
}
```

### 3. Predict Exoplanet

**POST** `/predict`

Classify an exoplanet candidate based on its features.

**Request Body:**
```json
{
  "features": [
    129.9459,  // orbital_period
    200.0,     // transit_depth
    1.17,      // planet_radius
    188.0,     // koi_teq
    0.29,      // koi_insol
    3755.0,    // stellar_teff
    0.54,      // stellar_radius
    0.50,      // koi_smass
    4.66,      // koi_slogg
    6.77,      // transit_duration
    0.35,      // impact_parameter
    82.5,      // koi_max_mult_ev
    15.0       // koi_num_transits
  ]
}
```

**Response:**
```json
{
  "predicted_label": "CONFIRMED PLANET",
  "probability": 0.9234,
  "confidence": "VERY HIGH",
  "all_probabilities": {
    "FALSE POSITIVE": 0.0766,
    "CONFIRMED PLANET": 0.9234
  },
  "model_version": "ultimate"
}
```

**Confidence Levels:**
- **VERY HIGH**: ≥90% probability (most reliable)
- **HIGH**: 80-89% probability (very reliable)
- **MEDIUM**: 70-79% probability (generally reliable)
- **LOW**: 60-69% probability (requires review)
- **VERY LOW**: <60% probability (manual verification needed)

## 💻 Usage Examples

### Python Example

```python
import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

# 1. Check health
response = requests.get(f"{API_URL}/health")
print(f"API Status: {response.json()['status']}")

# 2. Get model metrics
response = requests.get(f"{API_URL}/model/metrics")
metrics = response.json()
print(f"Model Accuracy: {metrics['test_accuracy']:.2%}")
print(f"ROC AUC: {metrics['roc_auc']:.2%}")

# 3. Make a prediction (Earth-like planet)
features = {
    "features": [
        365.25,  # orbital_period (1 year)
        84.0,    # transit_depth
        1.0,     # planet_radius (1 Earth)
        288.0,   # koi_teq (Earth temp)
        1.0,     # koi_insol (1 Earth flux)
        5778.0,  # stellar_teff (Sun-like)
        1.0,     # stellar_radius (1 Solar)
        1.0,     # koi_smass (1 Solar mass)
        4.44,    # koi_slogg (Sun-like)
        13.0,    # transit_duration
        0.0,     # impact_parameter
        100.0,   # koi_max_mult_ev
        20.0     # koi_num_transits
    ]
}

response = requests.post(f"{API_URL}/predict", json=features)
result = response.json()

print(f"\nPrediction: {result['predicted_label']}")
print(f"Confidence: {result['probability']:.2%} ({result['confidence']})")
print(f"Probabilities:")
for label, prob in result['all_probabilities'].items():
    print(f"  {label}: {prob:.2%}")
```

### cURL Example

```bash
# Health check
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/model/metrics

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [129.9459, 200.0, 1.17, 188.0, 0.29, 3755.0, 
                 0.54, 0.50, 4.66, 6.77, 0.35, 82.5, 15.0]
  }'
```

### JavaScript/Fetch Example

```javascript
// Make prediction
const features = {
  features: [
    129.9459, 200.0, 1.17, 188.0, 0.29, 3755.0,
    0.54, 0.50, 4.66, 6.77, 0.35, 82.5, 15.0
  ]
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(features),
})
  .then(response => response.json())
  .then(data => {
    console.log('Prediction:', data.predicted_label);
    console.log('Confidence:', data.confidence);
    console.log('Probability:', data.probability);
  })
  .catch(error => console.error('Error:', error));
```

## 🎯 Integration with Frontend

### Updating the React Frontend

To integrate with the React frontend at `web/frontend`:

1. **Update the API URL** in your frontend config:
   ```javascript
   // src/config.js or .env
   export const API_BASE_URL = 'http://localhost:8000';
   ```

2. **Create an API service** (e.g., `src/services/api.js`):
   ```javascript
   import axios from 'axios';
   
   const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
   
   export const predictExoplanet = async (features) => {
     const response = await axios.post(`${API_URL}/predict`, {
       features
     });
     return response.data;
   };
   
   export const getModelMetrics = async () => {
     const response = await axios.get(`${API_URL}/model/metrics`);
     return response.data;
   };
   
   export const checkHealth = async () => {
     const response = await axios.get(`${API_URL}/health`);
     return response.data;
   };
   ```

3. **Use in components**:
   ```javascript
   import { predictExoplanet } from './services/api';
   
   const handlePredict = async () => {
     try {
       const result = await predictExoplanet(formData.features);
       setResult(result);
     } catch (error) {
       console.error('Prediction failed:', error);
     }
   };
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Model Configuration
MODEL_PATH=models/
DEFAULT_MODEL=ultimate

# Logging
LOG_LEVEL=INFO

# CORS (for production)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Production Deployment

For production deployment:

1. **Disable auto-reload**:
   ```bash
   uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Update CORS settings** in `web/api/main.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=[
           "https://yourdomain.com",
           "https://www.yourdomain.com"
       ],
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

3. **Use a reverse proxy** (nginx example):
   ```nginx
   server {
       listen 80;
       server_name api.yourdomain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## 📈 Performance Considerations

### Batch Predictions

For processing multiple candidates efficiently:

```python
import concurrent.futures
import requests

def predict_candidate(features):
    response = requests.post('http://localhost:8000/predict', 
                           json={'features': features})
    return response.json()

# Batch predict with threading
candidates = [candidate1_features, candidate2_features, ...]

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(predict_candidate, candidates))
```

### Response Times

- **Single prediction**: ~50-100ms
- **Health check**: ~1-5ms
- **Model metrics**: ~1-5ms

### Scaling

For high-throughput scenarios:
- Use multiple workers: `--workers 4`
- Deploy multiple instances with load balancing
- Consider caching for repeated queries
- Use async/await for concurrent requests

## 🐛 Troubleshooting

### Model Not Loading

**Error**: "No models could be loaded"

**Solution**:
1. Verify model files exist in `models/` directory
2. Check file permissions
3. Ensure joblib is installed: `pip install joblib`

### Feature Count Mismatch

**Error**: "Expected 13 features, got X"

**Solution**:
- Ensure exactly 13 features are provided
- Check feature order matches documentation
- Verify all features are numeric

### Low Confidence Predictions

If getting many low confidence predictions:
- Review input data quality
- Check for missing or invalid values
- Ensure features are in correct units
- Compare with known good examples

## 📚 Additional Resources

- **Model Training Details**: See `models/ULTIMATE_MODEL_RESULTS.md`
- **Model Comparison**: Run `python models/compare_all_models.py`
- **Test Predictions**: Run `python models/test_predictions.py`
- **API Tests**: Run `python web/api/test_api_with_ultimate_model.py`

## 🎓 Best Practices

1. **Always check health** before making predictions
2. **Use confidence levels** to filter results
3. **Log all predictions** for model monitoring
4. **Implement retry logic** for production
5. **Monitor response times** and adjust workers
6. **Keep models updated** as new data becomes available

## 📞 Support

For issues or questions:
- Check logs: API logs will show detailed error messages
- Review test scripts for working examples
- Check feature ranges against training data
- Verify all 13 features are provided correctly

---

**Last Updated**: October 4, 2025  
**API Version**: 1.0.0  
**Model Version**: 3.0 (Ultimate Ensemble)
