# Exoplanet Classification App

This project is an Exoplanet Classification application that utilizes machine learning models to classify exoplanets based on input features. The application consists of a FastAPI backend for handling predictions and a Streamlit frontend for user interaction.

## Project Structure

```
exoplanet-classification-app
├── api                   # FastAPI backend
│   ├── __init__.py
│   ├── main.py          # Entry point for the FastAPI application
│   ├── models.py        # Data models and schemas for API
│   ├── dependencies.py   # Dependency functions for API routes
│   └── routers          # Directory for API routers
│       ├── __init__.py
│       └── predictions.py # Routes related to predictions
├── streamlit_app        # Streamlit frontend
│   ├── __init__.py
│   ├── app.py           # Main entry point for the Streamlit application
│   ├── components       # Directory for Streamlit components
│   │   ├── __init__.py
│   │   ├── prediction_form.py # Form for user input
│   │   └── visualization.py    # Visualization functions
│   └── utils            # Utility functions for the Streamlit app
│       ├── __init__.py
│       └── api_client.py # API client for making requests to FastAPI
├── models               # Directory for trained models
│   └── trained_models
│       ├── extra_trees_20251004_155128.joblib
│       ├── lightgbm_20251004_155128.joblib
│       ├── optimized_rf_20251004_155128.joblib
│       ├── optimized_xgb_20251004_155128.joblib
│       └── scaler_20251004_155128.joblib
├── shared               # Shared components between API and Streamlit
│   ├── __init__.py
│   ├── schemas.py       # Shared data schemas
│   └── config.py        # Configuration settings
├── requirements.txt      # Project dependencies
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile.api        # Dockerfile for FastAPI
├── Dockerfile.streamlit   # Dockerfile for Streamlit
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd exoplanet-classification-app
   ```

2. **Install dependencies:**
   You can install the required Python packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Run the FastAPI backend:**
   You can start the FastAPI application using:
   ```
   uvicorn api.main:app --reload
   ```

4. **Run the Streamlit frontend:**
   You can start the Streamlit application using:
   ```
   streamlit run streamlit/app.py
   ```

5. **Access the applications:**
   - FastAPI documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Streamlit app: [http://localhost:8501](http://localhost:8501)

## Usage

- Use the Streamlit application to input features for exoplanet classification.
- The application will communicate with the FastAPI backend to get predictions based on the input features.
- The results will be displayed in the Streamlit interface.

## Docker Support

This project includes Docker support for both the FastAPI and Streamlit applications. You can use the provided `docker-compose.yml` file to run both services in containers.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.