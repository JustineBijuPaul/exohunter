# 🚀 Quick Start Guide - ExoHunter Enhanced Web Apps

## What Was Improved?

### ✨ Backend (FastAPI)
- ✅ Batch predictions (up to 100 at once)
- ✅ Model metrics API endpoint
- ✅ Enhanced validation and error handling
- ✅ Structured logging with rotation
- ✅ Performance tracking
- ✅ Better API documentation

### ✨ Frontend (Streamlit)
- ✅ Modern UI with gradient styling
- ✅ CSV upload for batch predictions
- ✅ Prediction history with charts
- ✅ Interactive visualizations (Plotly)
- ✅ Multiple input methods (Manual, CSV, Samples)
- ✅ Export results as CSV

## 🏃 Running the Apps

### 1. Install Dependencies
```bash
cd web
pip install -r requirements.txt
```

### 2. Start the Backend
```bash
uvicorn api.main:app --reload --port 8000
```

**Test it:**
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs

### 3. Start the Frontend
```bash
streamlit run streamlit/app.py
```

**Access it:** http://localhost:8501

## 📖 New Features

### Backend API

#### 1. Batch Predictions
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "batch_features": [
      [1500, 2.5, 800, 150, 5800, 1.0, 1.0, 4.4, 1, 100, 15, 75, 0.3, 3.5, 400],
      [800, 1.8, 600, 50, 5200, 0.8, 0.9, 4.5, 2, 30, 8, 25, 0.7, 2.8, 300]
    ]
  }'
```

#### 2. Model Metrics
```bash
curl http://localhost:8000/models/metrics
```

#### 3. Feature Info
```bash
curl http://localhost:8000/models/features
```

### Frontend UI

#### Tab 1: Manual Input
1. Enter 15 feature values
2. Click "Classify Exoplanet"
3. View results with visualizations

#### Tab 2: CSV Upload
1. Download CSV template
2. Fill in your data
3. Upload and predict all
4. Download results

#### Tab 3: Quick Samples
- Click any sample to load
- Instant prediction
- Pre-configured test cases

#### Tab 4: History
- View all predictions
- Pie chart visualization
- Clear history option

## 📊 Visualizations

The new UI includes:
- 📈 **Gauge Chart**: Model agreement (0-100%)
- 📊 **Bar Charts**: Predictions and confidence
- 🥧 **Pie Charts**: Distribution over time
- 📉 **Feature Summary**: All input values

## 🔧 Configuration

Edit `web/shared/config.py`:
```python
class Config:
    API_URL = "http://localhost:8000"  # Change if needed
    MODEL_PATH = BASE_DIR / "models" / "trained_models"
```

## 📁 New Files

```
web/
├── shared/
│   ├── utils.py              # Common utilities (NEW)
│   └── logging_config.py     # Logging framework (NEW)
├── WEB_IMPROVEMENTS.md       # Full documentation (NEW)
└── ENHANCEMENT_SUMMARY.md    # Technical summary (NEW)
```

## 🔍 Troubleshooting

### Models Not Loading?
```bash
# Check if models exist
dir models\trained_models\*.joblib

# If missing, run training:
python scripts\optimized_training.py
```

### Port Already in Use?
```bash
# Find process
netstat -ano | findstr :8000

# Kill it
taskkill /PID <PID> /F

# Or use different port
uvicorn api.main:app --port 8001
```

### Import Errors?
```bash
# Reinstall dependencies
pip install -r web\requirements.txt --upgrade
```

## 📝 Logs

Check logs in `logs/` directory:
- `api.log` - General logs
- `requests.log` - HTTP requests
- `predictions.log` - Prediction history
- `performance_metrics.json` - Statistics

## 🎯 Example Workflow

### Single Prediction (API)
```python
import requests

features = [1500, 2.5, 800, 150, 5800, 1.0, 1.0, 4.4, 1, 100, 15, 75, 0.3, 3.5, 400]

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": features}
)

result = response.json()
print(f"Prediction: {result['ensemble_prediction']}")
print(f"Confidence: {result['ensemble_confidence']}%")
print(f"Time: {result['processing_time_ms']}ms")
```

### Batch Prediction (UI)
1. Open Streamlit at http://localhost:8501
2. Go to "CSV Upload" tab
3. Download template
4. Add your data (up to 100 rows)
5. Upload file
6. Click "Predict All"
7. Download results

## 📚 Documentation

- **Full Guide**: `web/WEB_IMPROVEMENTS.md`
- **Technical Summary**: `web/ENHANCEMENT_SUMMARY.md`
- **API Docs**: http://localhost:8000/docs (when running)

## ✅ What's Better?

| Before | After |
|--------|-------|
| 2 endpoints | 5 endpoints |
| No batch support | 100 predictions at once |
| Basic UI | Rich visualizations |
| No logging | Structured logs + rotation |
| Simple validation | Comprehensive validation |
| No history | Full history with charts |
| No export | CSV export |

## 🎉 You're Ready!

Both the backend and frontend are now significantly improved with:
- Better user experience
- More features
- Better error handling
- Production-ready logging
- Comprehensive documentation

**Enjoy using ExoHunter!** 🌌

---

For detailed documentation, see:
- `WEB_IMPROVEMENTS.md` - Complete user guide
- `ENHANCEMENT_SUMMARY.md` - Technical details
