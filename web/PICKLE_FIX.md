# 🔧 Pickle Compatibility Issue - Solution Guide

## Problem: KeyError 35

**Error Message**:
```
KeyError: 35
  File "joblib\numpy_pickle.py", line 626, in _unpickle
    obj = unpickler.load()
  File "pickle.py", line 1205, in load
    dispatch[key[0]](self)
```

## Root Cause

This is a **pickle protocol incompatibility** issue that occurs when:

1. **Python Version Mismatch**: Models trained with Python 3.12 but loaded with Python 3.11 (or vice versa)
2. **scikit-learn Version Mismatch**: Different sklearn versions between training and deployment
3. **Pickle Protocol Version**: Models saved with protocol 5 but Python < 3.8 can only read up to protocol 4

The number "35" is a pickle opcode that the current Python/pickle version doesn't recognize.

## Solutions (In Order of Preference)

### Solution 1: Match Python & Library Versions (RECOMMENDED) ✅

**Check training environment**:
```bash
# In the environment where models were trained
python --version
pip show scikit-learn
pip show numpy
pip show joblib
```

**Match in deployment**:
```bash
# Install same versions
pip install scikit-learn==<same_version>
pip install numpy==<same_version>
pip install joblib==<same_version>
```

### Solution 2: Retrain Models (BEST LONG-TERM) ✅

If you have access to the training script and data:

```bash
# Use current Python environment
cd exoplanet-clean
python scripts/optimized_training.py
```

This will create new model files compatible with your current environment.

### Solution 3: Use Safe Loading (IMPLEMENTED) ✅

We've added fallback loading methods that try multiple approaches:

**Changes Made**:
- Added `safe_load_model()` function to both API and Streamlit
- Tries joblib first, then falls back to pickle
- Shows detailed error messages
- Continues loading other models if one fails

### Solution 4: Downgrade Python (TEMPORARY)

If models were trained with Python 3.11:
```bash
# Create new environment with Python 3.11
conda create -n exohunter_py311 python=3.11
conda activate exohunter_py311
pip install -r requirements.txt
```

## Current Implementation

### What We Fixed:

1. **API (`web/api/main.py`)**:
   ```python
   def safe_load_model(model_path: Path):
       try:
           return joblib.load(model_path)
       except (KeyError, Exception) as e:
           # Fallback to pickle
           with open(model_path, 'rb') as f:
               return pickle.load(f)
   ```

2. **Streamlit (`web/streamlit/app.py`)**:
   - Same safe loading function
   - Shows Python and scikit-learn versions in sidebar
   - Continues even if some models fail to load

3. **Error Handling**:
   - Individual model failures don't crash the app
   - Detailed error messages for debugging
   - Falls back to unscaled features if scaler fails

## Debugging Steps

### 1. Check Your Environment
```bash
python --version
pip list | grep -E "(scikit-learn|joblib|numpy)"
```

### 2. Check Model File Info
```bash
# On Windows
dir models\trained_models\*.joblib

# Check file dates - were they created recently or from older training?
```

### 3. Test Loading Manually
```python
import joblib
import pickle
from pathlib import Path

model_path = Path("models/trained_models/scaler_20251004_155128.joblib")

# Try joblib
try:
    scaler = joblib.load(model_path)
    print("✅ Joblib load successful")
except Exception as e:
    print(f"❌ Joblib failed: {e}")
    
    # Try pickle
    try:
        with open(model_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Pickle load successful")
    except Exception as e2:
        print(f"❌ Pickle also failed: {e2}")
```

## Recommended Action Plan

### Immediate (To Get It Working Now):

1. **Restart Streamlit** - The safe loading code is now in place:
   ```bash
   streamlit run web/streamlit/app.py
   ```

2. **Check Versions** - Look at sidebar for Python/sklearn versions

3. **Review Errors** - Sidebar will show which models loaded and which failed

### Short-Term (For Stability):

1. **Retrain models** in your current environment:
   ```bash
   python scripts/optimized_training.py
   ```
   This creates fresh models compatible with your Python version.

### Long-Term (For Production):

1. **Document environment**:
   ```bash
   pip freeze > requirements-frozen.txt
   python --version > python_version.txt
   ```

2. **Use Docker** for consistent environment

3. **Version control** your trained models with metadata:
   ```json
   {
     "python_version": "3.12.0",
     "sklearn_version": "1.3.0",
     "training_date": "2024-10-04",
     "model_files": ["scaler.joblib", "model.joblib"]
   }
   ```

## What Changed

### Files Modified:
1. ✅ `web/api/main.py` - Added `safe_load_model()` function
2. ✅ `web/streamlit/app.py` - Added `safe_load_model()` and version display
3. ✅ Both files - Better error handling for individual model failures

### New Features:
- 📊 Version info displayed in Streamlit sidebar
- 🔄 Fallback loading methods
- ⚠️ Graceful degradation (app works with partial models)
- 📝 Detailed error messages for debugging

## Testing

### Test 1: Verify Safe Loading
```bash
# Start Streamlit
streamlit run web/streamlit/app.py

# Check sidebar for:
# - Python version
# - scikit-learn version
# - Which models loaded successfully
# - Any error messages
```

### Test 2: Verify API
```bash
# Start API
uvicorn api.main:app --reload

# Check console for:
# [OK] or [ERROR] messages
# Which models loaded
```

### Test 3: Make Prediction
If at least one model loads, try making a prediction to verify functionality.

## Expected Outcomes

### Best Case:
- All models load successfully
- Both API and Streamlit work normally

### Moderate Case:
- Some models fail, others succeed
- App works with available models
- Clear error messages shown

### Worst Case:
- All models fail with same error
- **Action**: Retrain models OR match Python version

## Support

If issue persists after trying these solutions:

1. Share your environment info:
   ```bash
   python --version
   pip list
   ```

2. Check when models were trained:
   ```bash
   dir models\trained_models\*.joblib
   ```

3. Try retraining:
   ```bash
   python scripts/optimized_training.py
   ```

---

**Status**: ✅ Safe loading implemented  
**Next Step**: Restart apps and check which models load  
**Best Fix**: Retrain models in current environment
