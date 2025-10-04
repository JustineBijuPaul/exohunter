# ExoHunter 🌟

A comprehensive machine learning pipeline for exoplanet classification using data from Kepler KOI, K2, and TESS TOI missions.

## Overview

ExoHunter combines traditional machine learning methods with deep learning approaches to classify potential exoplanets from astronomical datasets. The system provides both tabular feature classification and light curve analysis capabilities through a production-ready FastAPI interface.

## Features

- **Multi-source Data Ingestion**: Support for Kepler KOI, K2, and TESS TOI datasets
- **Label Standardization**: Unified classification system across different missions
- **Multiple ML Models**: Random Forest, XGBoost, MLP, and CNN implementations
- **Ensemble Methods**: Stacking ensemble for improved robustness
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and calibration analysis
- **Production API**: FastAPI endpoints for real-time predictions
- **Web Interface**: React-based frontend for easy interaction
- **Docker Deployment**: Complete containerized deployment solution
- **Exploratory Analysis**: Jupyter notebooks for data exploration
- **Comprehensive Documentation**: Detailed user and developer guides

## Documentation

- **[User Guide](docs/USER_GUIDE.md)**: Complete guide for using ExoHunter
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Technical documentation for developers
- **[API Documentation](http://localhost:8000/docs)**: Interactive API documentation (when running)
- **[Example Dataset](data/example_toi.csv)**: Sample CSV file for testing

## Quick Start

### Using Docker (Recommended)

The fastest way to get ExoHunter running:

```bash
# Start all services
./deploy.sh start

# Access the application
# Web Interface: http://localhost:3000
# API Documentation: http://localhost:8000/docs
# Health Check: http://localhost:8000/health
```

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/JustineBijuPaul/exoplanet.io.git
cd exohunter
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the API server:
```bash
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing with Example Data

Try the API with our example dataset:

```bash
curl -X POST "http://localhost:8000/predict/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/example_toi.csv"
```

For detailed instructions, see the [User Guide](docs/USER_GUIDE.md).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JustineBijuPaul/exoplanet.io.git
cd exohunter
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
exohunter/
├── data/                    # Data ingestion and preprocessing
│   ├── ingest.py           # Dataset downloading and loading
│   └── labels.py           # Label standardization
├── models/                  # Machine learning models
│   ├── train_baseline.py   # RandomForest and XGBoost
│   ├── advanced.py         # MLP and CNN models
│   ├── ensemble.py         # Ensemble methods
│   └── evaluate.py         # Model evaluation utilities
├── web/                     # FastAPI application
│   └── api/
│       ├── main.py         # API endpoints
│       └── models.py       # Pydantic models
├── notebooks/               # Jupyter notebooks
│   └── EDA.ipynb          # Exploratory data analysis
└── tests/                   # Unit tests
```

## API Usage

### Starting the API Server

```bash
# Option 1: Using uvicorn directly
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload

# Option 2: Using the main function
python -m web.api.main
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### API Endpoints

#### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-03T12:00:00",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Model Metrics
```bash
curl -X GET "http://localhost:8000/model/metrics"
```

Response:
```json
{
  "model_type": "ensemble",
  "training_accuracy": 0.85,
  "validation_accuracy": 0.82,
  "cross_validation_score": 0.80,
  "feature_count": 8,
  "classes": ["CANDIDATE", "FALSE POSITIVE", "CONFIRMED"],
  "last_updated": "2025-10-03T12:00:00"
}
```

#### Prediction from Features
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.2, 0.8, 2.1, 0.95, 1.1, 0.75, 1.8, 0.9]
  }'
```

Response:
```json
{
  "predicted_label": "CANDIDATE",
  "probability": 0.87,
  "confidence": "HIGH",
  "all_probabilities": {
    "CANDIDATE": 0.87,
    "FALSE POSITIVE": 0.10,
    "CONFIRMED": 0.03
  },
  "model_version": "ensemble"
}
```

#### Prediction from Light Curve
```bash
curl -X POST "http://localhost:8000/predict/lightcurve" \
  -H "Content-Type: application/json" \
  -d '{
    "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "flux": [1.0, 0.98, 0.99, 0.97, 1.01, 0.99, 1.0, 0.98, 1.01, 0.99]
  }'
```

#### File Upload Prediction
```bash
# Upload CSV with features
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@exoplanet_features.csv"

# Upload CSV with light curve (time,flux columns)
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@lightcurve_data.csv"
```

### Example CSV Formats

**Tabular Features CSV:**
```csv
period,radius,temp,magnitude,snr,duration,depth,impact
2.5,1.2,5800,12.3,15.2,3.1,0.01,0.5
```

**Light Curve CSV:**
```csv
time,flux
0.0,1.000
0.1,0.998
0.2,0.999
0.3,0.997
0.4,1.001
```

## Model Training

### Train Baseline Models
```python
from exohunter.models.train_baseline import train_random_forest, train_xgboost

# Train Random Forest
rf_model = train_random_forest(X_train, y_train)

# Train XGBoost
xgb_model = train_xgboost(X_train, y_train)
```

### Train Advanced Models
```python
from exohunter.models.advanced import TabularMLP

# Train MLP
mlp = TabularMLP(input_dim=8, num_classes=3)
mlp.train_tabular(X_train, y_train, X_val, y_val)
```

### Train Ensemble
```python
from exohunter.models.ensemble import train_ensemble_suite

# Train all models and ensemble
results = train_ensemble_suite(X_train, y_train, cv_folds=5)
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
This project follows PEP 8 guidelines. Use `black` for formatting:
```bash
pip install black
black .
```

### Adding New Models
1. Implement your model in the `exohunter/models/` directory
2. Add evaluation metrics using `exohunter.models.evaluate`
3. Update the ensemble if needed
4. Add tests in the `tests/` directory

## Data Sources

- **Kepler KOI**: Kepler Objects of Interest catalog
- **K2**: K2 mission candidate catalog
- **TESS TOI**: TESS Objects of Interest catalog

## Model Performance

Current ensemble model performance:
- **Cross-validation Accuracy**: ~80%
- **Precision (Candidate)**: ~82%
- **Recall (Candidate)**: ~78%
- **F1-Score**: ~80%

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA Exoplanet Archive for providing the datasets
- Kepler, K2, and TESS mission teams
- Open source ML community

## Citation

If you use ExoHunter in your research, please cite:

```bibtex
@software{exohunter2025,
  title={ExoHunter: Machine Learning Pipeline for Exoplanet Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/JustineBijuPaul/exoplanet.io}
}
```
