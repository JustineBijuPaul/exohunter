# ExoHunter Live Demo Script
## 5-6 Minute Presentation Guide

### 🎯 Demo Overview
**Total Time**: 5-6 minutes  
**Audience**: Technical stakeholders, researchers, potential users  
**Goal**: Demonstrate ExoHunter's capability to classify exoplanets using real astronomical data

---

## 🚀 Pre-Demo Setup Commands

### Start the Application Stack
```bash
# Terminal 1: Start the API server
cd /home/linxcapture/Desktop/projects/exohunter
source venv/bin/activate
python web/api/main.py

# Terminal 2: Start the frontend (if available)
cd web/frontend
npm start

# Alternative: Use Docker Compose
docker-compose up -d
```

### Verify Services
```bash
# Check API health
curl http://localhost:8000/health

# Check available endpoints
curl http://localhost:8000/docs

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.2, 0.8, 365.25, 0.05]}'
```

### Demo Data Preparation
```bash
# Ensure example data is available
ls data/example_toi.csv

# Quick data check
head -5 data/example_toi.csv
```

---

## 📝 Demo Script

### **Slide 1: Problem Introduction** *(45 seconds)*

**Script:**
> "Good [morning/afternoon]! I'm excited to show you ExoHunter, an AI-powered system for exoplanet classification. 
> 
> The challenge we're addressing is significant: NASA's TESS mission generates massive amounts of stellar data, and manually identifying exoplanet candidates is extremely time-consuming. Traditional methods require expert astronomers to analyze thousands of light curves by hand.
> 
> ExoHunter automates this process using machine learning to classify potential exoplanets from transit photometry data, helping researchers focus their time on the most promising candidates."

**Visual Cues:**
- Show problem statistics (thousands of TOI candidates)
- Display example light curve showing transit signature

---

### **Slide 2: Exploratory Data Analysis** *(60 seconds)*

**Script:**
> "Let me show you the data we're working with. This is real data from NASA's TESS Objects of Interest catalog."

**Demo Commands:**
```bash
# Show the dataset structure
python -c "
import pandas as pd
import matplotlib.pyplot as plt

# Load example data
df = pd.read_csv('data/example_toi.csv')
print(f'Dataset shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Target distribution:')
print(df['disposition'].value_counts())
"
```

**Script continues:**
> "Our dataset contains key astronomical features:
> - **Stellar magnitude**: Brightness of the host star
> - **Transit depth**: How much light is blocked during transit
> - **Orbital period**: Time for one complete orbit
> - **Transit duration**: Length of the transit event
> 
> The target variable shows the current disposition - whether candidates are confirmed planets, false positives, or still under investigation."

**Visual Cues:**
- Point to feature distributions
- Highlight class imbalance if present

---

### **Slide 3: Live Prediction Demo** *(90 seconds)*

**Script:**
> "Now let's see ExoHunter in action. I'll use a real TOI candidate from our dataset."

**Demo Commands:**
```bash
# Get a sample TOI for demonstration
python -c "
import pandas as pd
import requests
import json

# Load sample data
df = pd.read_csv('data/example_toi.csv')
sample = df.iloc[42]  # Pick an interesting example

print(f'TOI Sample: {sample.name}')
print(f'Stellar Magnitude: {sample[\"mag\"]}')
print(f'Transit Depth: {sample[\"depth\"]} ppm')
print(f'Period: {sample[\"period\"]} days')
print(f'Duration: {sample[\"duration\"]} hours')
print(f'Actual disposition: {sample[\"disposition\"]}')

# Prepare features for prediction
features = [sample['mag'], sample['depth'], sample['period'], sample['duration']]
print(f'Features for ML model: {features}')

# Make API prediction
try:
    response = requests.post(
        'http://localhost:8000/predict',
        json={'features': features},
        timeout=5
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f'Prediction: {result[\"prediction\"]}')
        print(f'Confidence: {result[\"confidence\"]:.3f}')
        print(f'Probabilities: {result[\"probabilities\"]}')
    else:
        print(f'API Error: {response.status_code}')
        
except Exception as e:
    print(f'Connection error: {e}')
    print('Falling back to local prediction...')
    
    # Fallback local prediction
    import joblib
    model = joblib.load('models/exoplanet_classifier.pkl')
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    print(f'Local Prediction: {prediction}')
    print(f'Probabilities: {probabilities}')
"
```

**Script continues:**
> "Excellent! The model predicted [state the prediction] with [X]% confidence. Notice how the model provides both a binary classification and probability scores, giving researchers insight into the prediction certainty.
> 
> This matches/differs from the actual disposition of [state actual], which shows [explain the result - correct prediction, or interesting discrepancy worth investigating]."

**Visual Cues:**
- Show the API response in real-time
- Highlight confidence scores
- Compare with actual disposition

---

### **Slide 4: Model Performance Metrics** *(75 seconds)*

**Script:**
> "Let's look at how well our model performs across the entire test dataset."

**Demo Commands:**
```bash
# Run model evaluation
python -c "
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load model and test data
model = joblib.load('models/exoplanet_classifier.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Load and prepare test data
df = pd.read_csv('data/example_toi.csv')
feature_cols = ['mag', 'depth', 'period', 'duration']
X = df[feature_cols].fillna(df[feature_cols].median())
y = (df['disposition'] == 'CONFIRMED').astype(int)

# Split for evaluation (use last 20% as test)
split_idx = int(len(X) * 0.8)
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

# Scale features
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Print metrics
print('=== Model Performance ===')
print(f'Test set size: {len(y_test)} samples')
print(f'Accuracy: {(y_pred == y_test).mean():.3f}')

print('\\n=== Classification Report ===')
print(classification_report(y_test, y_pred, 
                          target_names=['Not Planet', 'Planet'], 
                          digits=3))

print('\\n=== Confusion Matrix ===')
cm = confusion_matrix(y_test, y_pred)
print(f'True Negatives: {cm[0,0]}  False Positives: {cm[0,1]}')
print(f'False Negatives: {cm[1,0]}  True Positives: {cm[1,1]}')

# Calculate key metrics
precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'\\n=== Key Metrics ===')
print(f'Precision (Planet detection): {precision:.3f}')
print(f'Recall (Planet sensitivity): {recall:.3f}')
print(f'F1-Score: {f1:.3f}')
"
```

**Script continues:**
> "Our model achieves [X]% accuracy on the test set. More importantly for astronomical research:
> 
> - **Precision of [X]%**: When we predict a planet, we're right [X]% of the time
> - **Recall of [X]%**: We successfully identify [X]% of actual planets
> - **Low false positive rate**: Critical for not wasting telescope time on false leads
> 
> The confusion matrix shows we're particularly good at [highlight strength - either minimizing false positives or catching true planets]."

**Visual Cues:**
- Point to key numbers in the confusion matrix
- Emphasize astronomical relevance of metrics

---

### **Slide 5: Interactive UI Demo** *(45 seconds)*

**Script:**
> "For researchers who prefer a visual interface, we've built a web application."

**Demo Actions:**
1. **Navigate to UI**: Open `http://localhost:3000` (or show prepared screenshots)

2. **Input Features**: 
   ```
   Stellar Magnitude: 12.5
   Transit Depth: 1500 ppm
   Orbital Period: 45.2 days
   Transit Duration: 3.1 hours
   ```

3. **Show Results**: Click "Classify" and display:
   - Prediction result
   - Confidence visualization
   - Feature importance plot (if available)

**Script continues:**
> "The interface provides immediate feedback with visual confidence indicators. Researchers can quickly test multiple candidates and sort them by likelihood for follow-up observations."

---

### **Slide 6: Roadmap and Future Work** *(45 seconds)*

**Script:**
> "Looking ahead, we have several exciting developments planned:

**Near-term (3-6 months):**
- **Enhanced Features**: Integration of stellar parameters (temperature, radius, metallicity)
- **Time Series Analysis**: Direct processing of light curve data using neural networks
- **Ensemble Methods**: Combining multiple algorithms for improved accuracy

**Medium-term (6-12 months):**
- **Real-time Processing**: Live integration with TESS data pipeline
- **Multi-mission Support**: Extending to Kepler and future missions
- **Advanced Visualization**: Interactive light curve analysis tools

**Long-term Vision:**
- **Automated Discovery Pipeline**: End-to-end planet discovery with minimal human intervention
- **Collaboration Platform**: Enabling global research teams to share and validate findings
- **Educational Integration**: Tools for astronomy education and citizen science

The ultimate goal is to accelerate exoplanet discovery and help answer the fundamental question: Are we alone?"

---

## 🛠️ Technical Appendix

### Backup Demo Commands

**If API is down, use local prediction:**
```bash
python -c "
import joblib
import numpy as np

# Load model locally
model = joblib.load('models/exoplanet_classifier.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Example prediction
features = [[12.5, 1500, 45.2, 3.1]]  # mag, depth, period, duration
features_scaled = scaler.transform(features)

prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

print(f'Prediction: {\"Planet\" if prediction == 1 else \"Not Planet\"}')
print(f'Confidence: {max(probabilities):.3f}')
"
```

**Quick data visualization:**
```bash
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/example_toi.csv')
print('Dataset Overview:')
print(df.describe())

# Quick plot (if needed)
plt.figure(figsize=(10,6))
df['disposition'].value_counts().plot(kind='bar')
plt.title('Exoplanet Candidate Dispositions')
plt.show()
"
```

### Troubleshooting

**Common Issues:**
- **API not responding**: Use local prediction commands
- **Missing data file**: Copy from backup or use synthetic data
- **Frontend not loading**: Show API responses directly
- **Model file missing**: Use basic sklearn classifier as fallback

**Emergency Backup Plan:**
```bash
# Create minimal synthetic demo data
python -c "
import numpy as np
import pandas as pd

# Generate sample data
np.random.seed(42)
n_samples = 100

data = {
    'mag': np.random.normal(12, 2, n_samples),
    'depth': np.random.lognormal(7, 1, n_samples),
    'period': np.random.lognormal(2, 1, n_samples),
    'duration': np.random.normal(3, 1, n_samples),
    'disposition': np.random.choice(['CONFIRMED', 'FALSE_POSITIVE', 'CANDIDATE'], 
                                  n_samples, p=[0.3, 0.4, 0.3])
}

df = pd.DataFrame(data)
df.to_csv('data/demo_backup.csv', index=False)
print('Backup demo data created!')
"
```

### Post-Demo Q&A Preparation

**Likely Questions:**
1. **"What's the training data size?"** → [State current dataset size and mention ongoing data collection]
2. **"How does this compare to existing methods?"** → [Mention speed improvement and consistency advantages]
3. **"Can it handle different types of transits?"** → [Discuss multi-planet systems and grazing transits]
4. **"What about false positives from stellar activity?"** → [Mention stellar parameter integration in roadmap]
5. **"Is this ready for production use?"** → [Discuss current validation status and deployment plans]

### Demo Success Metrics

**Technical Success:**
- [ ] All commands execute without errors
- [ ] API responds within 2 seconds
- [ ] Model predictions are reasonable
- [ ] UI loads and functions properly

**Presentation Success:**
- [ ] Stays within 6-minute time limit
- [ ] Clearly explains astronomical relevance
- [ ] Demonstrates practical value
- [ ] Engages audience with live interaction
- [ ] Smoothly handles any technical issues

---

## 🎬 Presentation Tips

### Delivery Notes
- **Energy**: Maintain enthusiastic tone about space discovery
- **Timing**: Practice to stay within 6 minutes - have shorter version ready
- **Interaction**: Encourage questions but defer technical details to post-demo
- **Backup**: Always have screenshots ready in case of technical issues

### Visual Aids
- Use dark theme for better visibility
- Increase terminal font size for audience
- Prepare key numbers in advance (accuracy, dataset size, etc.)
- Have backup slides with static results

### Audience Engagement
- Start with relatable space/discovery theme
- Use astronomical terms appropriately for audience level
- Connect technical metrics to real research impact
- End with inspiring vision of automated discovery

**Remember**: The goal is to demonstrate ExoHunter's potential to accelerate exoplanet research and engage the audience with the excitement of space discovery! 🚀🪐
