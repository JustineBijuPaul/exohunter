@echo off
echo ========================================
echo ExoHunter Model Retraining
echo ========================================
echo.
echo This will retrain all models with your current Python environment
echo Python 3.12.4 and scikit-learn 1.7.2
echo.
echo This process will take 10-15 minutes...
echo.
pause

cd /d "%~dp0"

echo.
echo Starting training...
echo.

python scripts\optimized_training.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✅ Training completed successfully!
    echo ========================================
    echo.
    echo New models have been saved to: models\trained_models\
    echo You can now restart the Streamlit app
    echo.
) else (
    echo.
    echo ========================================
    echo ❌ Training failed with error code: %ERRORLEVEL%
    echo ========================================
    echo.
    echo Please check the error messages above
    echo.
)

pause
