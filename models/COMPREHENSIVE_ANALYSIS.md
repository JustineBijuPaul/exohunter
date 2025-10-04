# Comprehensive Model Analysis Report

**Date**: 2025-10-04 14:23:15

---

## 📊 Model Performance Summary

- **Test Accuracy**: 73.27%
- **Precision**: 79.18%
- **Recall**: 73.27%
- **F1-Score**: 75.48%
- **Test Samples**: 101

---

## 🎯 Top 10 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `toipfx` | 295.2538 |
| 2 | `stellar_teff` | 227.2828 |
| 3 | `ra` | 221.0731 |
| 4 | `dec` | 219.3716 |
| 5 | `st_tmag` | 189.9074 |
| 6 | `st_dist` | 179.8672 |
| 7 | `transit_duration` | 174.1612 |
| 8 | `st_logg` | 155.6329 |
| 9 | `tid` | 150.6993 |
| 10 | `snr_ratio` | 141.6394 |

---

## 📦 Feature Category Analysis

- **Original**: 111.4762
- **Ratio**: 109.0115
- **Interaction**: 80.1275
- **Polynomial**: 20.4932
- **Normalized**: 16.0896

---

## ⚠️ Error Analysis

- **Total Errors**: 27
- **Error Rate**: 26.73%
- **Correct Predictions**: 74

### Error Patterns

- CANDIDATE → FALSE POSITIVE: 8 (29.6% of errors)
- FALSE POSITIVE → CANDIDATE: 18 (66.7% of errors)
- FALSE POSITIVE → CONFIRMED: 1 (3.7% of errors)

---

## 💡 Key Insights

1. **Feature Engineering Impact**: Normalized and ratio features show high importance
2. **Model Consistency**: Top features show agreement across multiple models
3. **Detection Quality**: SNR-related features are crucial for classification
4. **Physical Properties**: Planet radius and orbital period are fundamental predictors

---

## 📁 Generated Files

- `feature_importance_detailed.csv` - Full feature importance matrix
- `feature_importance_analysis.png` - Visual analysis (4 subplots)
- `error_analysis.csv` - Detailed error cases
- `error_analysis.png` - Error pattern visualization

---

## 🚀 Recommendations

1. **Feature Selection**: Consider using top 50 features for faster inference
2. **Model Optimization**: Remove Deep NN (poor performance), retrain ensemble
3. **Confidence Thresholds**: Use 60% confidence for binary decisions
4. **Error Mitigation**: Focus on improving CANDIDATE vs FALSE POSITIVE separation

