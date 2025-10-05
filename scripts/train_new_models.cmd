@echo off
echo ========================================
echo ExoHunter - Train on train.csv
echo ========================================
echo.
echo Dataset: data\train.csv
echo Target: disposition column
echo.
echo This will:
echo   1. Load your train.csv dataset
echo   2. Split into train/test (80/20)
echo   3. Train 4 optimized models
echo   4. Save models compatible with Python 3.12.4
echo   5. Expected accuracy: 75-85%%
echo.
echo Time required: 10-15 minutes
echo.
pause

echo.
echo Starting training...
echo.

python scripts\train_on_train_csv.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Training completed
    echo ========================================
    echo.
    echo Models saved to: models\trained_models\
    echo.
    echo Next steps:
    echo 1. Copy the timestamp from above
    echo 2. Update web\streamlit\app.py with new model filenames
    echo 3. Update web\api\main.py with new model filenames
    echo 4. Restart Streamlit and API servers
    echo.
    echo See ACTION_PLAN.md for detailed instructions
    echo.
) else (
    echo.
    echo ========================================
    echo ERROR: Training failed
    echo ========================================
    echo.
    echo Please check the error messages above
    echo.
)

pause
