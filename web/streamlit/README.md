# ExoHunter Streamlit Demo

A quick interactive prototype for demonstrating exoplanet classification capabilities.

## Features

🎯 **Interactive Demo Interface**
- Upload your own exoplanet datasets (CSV format)
- Generate synthetic sample data for testing
- Real-time model training and evaluation

📊 **Data Exploration**
- Dataset overview with summary statistics
- Feature distribution visualizations
- Class balance analysis

🤖 **Machine Learning Models**
- XGBoost, Random Forest, and Ensemble models
- Interactive model training with customizable parameters
- Real-time prediction on uploaded data

📈 **Evaluation & Visualization**
- Confusion matrices with interactive plots
- Feature importance analysis
- Performance metrics (accuracy, precision, recall)
- Model comparison functionality

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install streamlit plotly seaborn
   ```

2. **Run the app:**
   ```bash
   streamlit run web/streamlit/app.py
   ```

3. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

## Usage Modes

### 1. Demo with Sample Data
- Generate synthetic exoplanet data (100-2000 samples)
- Perfect for quick testing and demonstrations
- Includes realistic feature distributions

### 2. Upload Your Dataset
- Support for CSV files with exoplanet features
- Automatic data preprocessing and validation
- Real-time model training on your data

### 3. Model Comparison
- Side-by-side comparison of different algorithms
- Performance benchmarking
- Visual comparison charts

## Supported Features

The app expects these columns in your CSV data:

- **period**: Orbital period (days)
- **radius**: Planet radius (Earth radii)
- **temperature**: Stellar temperature (K)
- **magnitude**: Stellar magnitude
- **snr**: Signal-to-noise ratio
- **duration**: Transit duration (hours)
- **depth**: Transit depth
- **impact**: Impact parameter
- **label**: Classification (CANDIDATE, CONFIRMED, FALSE POSITIVE) - optional for predictions

## Example Data Format

```csv
period,radius,temperature,magnitude,snr,duration,depth,impact,label
2.47,1.2,5800,12.3,15.2,3.1,0.01,0.5,CANDIDATE
1.83,0.9,6200,11.8,18.5,2.8,0.008,0.3,CONFIRMED
4.12,1.8,5400,13.1,12.7,4.2,0.015,0.7,FALSE POSITIVE
```

## Development Notes

- Built with Streamlit for rapid prototyping
- Integrates with the main ExoHunter package
- Designed for demos and educational purposes
- For production use, consider the FastAPI + React frontend

## Troubleshooting

**Import Errors:**
- Ensure you're running from the project root directory
- Check that all dependencies are installed

**Model Training Issues:**
- Verify your data has sufficient samples (>100 recommended)
- Ensure no missing values in critical columns
- Check that labels are properly formatted

**Performance:**
- For large datasets (>10k samples), consider using the main pipeline
- Sample your data for faster interactive exploration
