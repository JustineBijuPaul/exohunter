# Performance Test Results

This directory contains performance test results from model smoke tests.

## File Types

- `{model_name}_{timestamp}.json`: Basic performance metrics and pass/fail status
- `{model_name}_detailed_{timestamp}.json`: Detailed analysis including confusion matrices and recommendations
- `model_comparison_{timestamp}.json`: Comparative analysis between models

## Understanding Results

### Basic Results Format
```json
{
  "model_name": "random_forest",
  "metrics": {
    "accuracy": 0.462,
    "precision_macro": 0.346,
    "recall_macro": 0.329,
    "f1_macro": 0.306,
    "precision_weighted": 0.410,
    "recall_weighted": 0.462,
    "f1_weighted": 0.404
  },
  "thresholds": {
    "min_f1_macro": 0.6,
    "min_f1_weighted": 0.6,
    "min_accuracy": 0.65,
    "min_precision_macro": 0.6,
    "min_recall_macro": 0.6
  },
  "passed": false,
  "timestamp": "2025-10-03T11:52:08.419393"
}
```

### Performance Thresholds

Current minimum acceptable performance:
- **F1 Score (macro)**: 0.6
- **F1 Score (weighted)**: 0.6
- **Accuracy**: 0.65
- **Precision (macro)**: 0.6
- **Recall (macro)**: 0.6

### Running Performance Tests

```bash
# Run all performance tests
pytest tests/test_model_performance.py --run-slow --run-performance

# Run specific model test
pytest tests/test_model_performance.py::TestModelPerformance::test_random_forest_performance --run-slow --run-performance

# Run without soft assertions (see warnings)
pytest tests/test_model_performance.py --run-slow --run-performance -v

# Run with detailed output
pytest tests/test_model_performance.py --run-slow --run-performance -s
```

### Interpreting Warnings

When performance is below thresholds, tests will PASS but generate warnings:

```
PERFORMANCE WARNING: random_forest
Model performance below expected thresholds:
  - min_f1_macro: 0.306 < 0.600
  - min_accuracy: 0.462 < 0.650
```

This indicates the model needs investigation but doesn't block CI/CD.

### Action Items for Low Performance

1. **Check data quality**: Ensure training data is representative
2. **Review feature engineering**: Consider additional or different features
3. **Hyperparameter tuning**: Optimize model parameters
4. **Increase dataset size**: More training data may improve performance
5. **Consider ensemble methods**: Combine multiple models
6. **Adjust thresholds**: If consistently low, thresholds may need adjustment

### Files Cleanup

Performance result files accumulate over time. Consider:
- Keeping last N results for trend analysis
- Archiving old results periodically
- Automated cleanup in CI/CD pipelines
