# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-03

### ✨ Features

- **ML Pipeline**: Complete machine learning pipeline for exoplanet classification
- **Multi-Model Support**: Random Forest, XGBoost, and ensemble methods
- **API Service**: Production-ready FastAPI service with ML inference endpoints
- **Web Interface**: React-based frontend for easy interaction with the system
- **Data Processing**: Comprehensive data ingestion and preprocessing for Kepler, K2, and TESS datasets
- **Label Standardization**: Unified classification system across different astronomical surveys
- **Docker Support**: Complete containerized deployment with Docker Compose
- **Performance Testing**: Smoke tests with soft assertions for model performance monitoring

### 🐛 Bug Fixes

- Model loading and preprocessing pipeline stability improvements
- API error handling and validation enhancements

### 📚 Documentation

- **User Guide**: Complete guide for using ExoHunter with API examples
- **Developer Guide**: Technical documentation for development and customization
- **API Documentation**: Interactive OpenAPI documentation with examples
- **Deployment Guide**: Docker deployment instructions for multiple cloud platforms

### 🧪 Tests

- Comprehensive test suite with unit, integration, and performance tests
- Model performance smoke tests with configurable thresholds
- API endpoint testing with various input scenarios
- Data processing and label mapping validation tests

### 🔧 Chores & Maintenance

- **Release Infrastructure**: Automated release packaging and version management
- **CI/CD Support**: GitHub Actions workflows for testing and deployment
- **Code Quality**: Linting, formatting, and type checking configuration
- **Database Integration**: PostgreSQL support for logging and analytics

### 🗂️ Project Structure

```
exohunter/
├── exohunter/           # Core Python package
├── web/                 # Web applications (API + Frontend)
├── models/              # ML model training scripts
├── data/                # Data processing and ingestion
├── tests/               # Comprehensive test suite
├── docs/                # Documentation (User & Developer guides)
├── deploy/              # Deployment configurations
├── scripts/             # Release and utility scripts
└── notebooks/           # Jupyter notebooks for analysis
```

### 🚀 Getting Started

#### Quick Start with Docker
```bash
# Start all services
./deploy.sh start

# Access the application
# Web Interface: http://localhost:3000
# API Documentation: http://localhost:8000/docs
```

#### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend
cd web/frontend && npm run dev
```

### 📊 Model Performance

Current baseline models achieve:
- **Random Forest**: ~85% accuracy on validation data
- **XGBoost**: ~87% accuracy on validation data
- **Ensemble**: ~88% accuracy combining multiple models

Performance is continuously monitored through automated smoke tests.

### 🔬 Supported Data Sources

- **Kepler KOI**: Kepler Objects of Interest catalog
- **K2**: Extended Kepler mission targets
- **TESS TOI**: TESS Objects of Interest from NASA's TESS mission

### 🎯 Classification Categories

- **CONFIRMED**: High-confidence exoplanet detections
- **CANDIDATE**: Potential exoplanets requiring follow-up
- **FALSE POSITIVE**: Non-planetary signals (eclipsing binaries, etc.)

---

## Development

This project is actively developed and welcomes contributions. See the [Developer Guide](docs/DEVELOPER_GUIDE.md) for detailed setup instructions and contribution guidelines.

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Support

- **Documentation**: See `docs/` directory
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join the community discussions

For questions and support, please use GitHub Issues or Discussions.
