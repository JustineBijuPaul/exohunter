@echo off
echo ========================================
echo ExoHunter - Complete Model Retraining
echo ========================================
echo.
echo This will:
echo 1. Retrain all 4 models with your current Python environment
echo 2. Fix the pickle compatibility issues
echo 3. Save new models that work with Python 3.12.4
echo 4. This takes about 10-15 minutes
echo.
echo Models to be trained:
echo   - Optimized Random Forest
echo   - Extra Trees
echo   - LightGBM
echo   - Optimized XGBoost
echo.
pause

echo.
echo ========================================
echo Step 1: Training Models
echo ========================================
echo.

python scripts\optimized_training.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: Training failed!
    echo ========================================
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================
echo Step 2: Testing Models
echo ========================================
echo.
echo Running a quick test to verify models work...

python -c "import joblib; from pathlib import Path; models_dir = Path('models/trained_models'); files = list(models_dir.glob('*.joblib')); print(f'Found {len(files)} model files'); latest = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[0] if files else None; print(f'Latest: {latest.name if latest else None}'); model = joblib.load(latest) if latest else None; print('Model loaded successfully!' if model else 'No models found')"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Model test failed, but training completed
    echo You may need to check the model files manually
)

echo.
echo ========================================
echo SUCCESS! Training Complete
echo ========================================
echo.
echo New models saved to: models\trained_models\
echo.
echo Next steps:
echo 1. Update model filenames in web\streamlit\app.py
echo 2. Update model filenames in web\api\main.py
echo 3. Restart your Streamlit and API servers
echo.
echo See MODEL_RETRAIN_GUIDE.md for detailed instructions
echo.
pause
