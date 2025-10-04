import React from 'react';

const PredictionResults = ({ results, error }) => {
  if (error) {
    return (
      <div className="card">
        <h2>❌ Error</h2>
        <div className="error">
          {error}
        </div>
      </div>
    );
  }

  if (!results) {
    return null;
  }

  const getConfidenceClass = (confidence) => {
    switch (confidence.toLowerCase()) {
      case 'high':
        return 'confidence-high';
      case 'medium':
        return 'confidence-medium';
      case 'low':
        return 'confidence-low';
      default:
        return 'confidence-medium';
    }
  };

  const getClassificationEmoji = (label) => {
    switch (label.toLowerCase()) {
      case 'candidate':
        return '🌟';
      case 'confirmed':
        return '✅';
      case 'false positive':
        return '❌';
      default:
        return '❓';
    }
  };

  const getClassificationColor = (label) => {
    switch (label.toLowerCase()) {
      case 'candidate':
        return '#28a745';
      case 'confirmed':
        return '#007bff';
      case 'false positive':
        return '#dc3545';
      default:
        return '#6c757d';
    }
  };

  return (
    <div className="card">
      <h2>🔮 Prediction Results</h2>
      
      <div className="results">
        {/* Main prediction */}
        <div 
          className="result-item" 
          style={{ 
            border: `2px solid ${getClassificationColor(results.predicted_label)}`,
            backgroundColor: `${getClassificationColor(results.predicted_label)}15`
          }}
        >
          <div>
            <h3 style={{ margin: '0 0 8px 0', color: getClassificationColor(results.predicted_label) }}>
              {getClassificationEmoji(results.predicted_label)} {results.predicted_label}
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span>Confidence: {(results.probability * 100).toFixed(1)}%</span>
              <span 
                className={`confidence-badge ${getConfidenceClass(results.confidence)}`}
              >
                {results.confidence}
              </span>
            </div>
          </div>
          <div>
            <div className="probability-bar">
              <div 
                className="probability-fill"
                style={{ 
                  width: `${results.probability * 100}%`,
                  background: `linear-gradient(90deg, ${getClassificationColor(results.predicted_label)}, ${getClassificationColor(results.predicted_label)}aa)`
                }}
              ></div>
            </div>
          </div>
        </div>

        {/* All probabilities */}
        <div style={{ marginTop: '24px' }}>
          <h4 style={{ marginBottom: '16px', color: '#555' }}>All Class Probabilities:</h4>
          {Object.entries(results.all_probabilities || {})
            .sort(([,a], [,b]) => b - a)
            .map(([className, probability]) => (
              <div key={className} className="result-item">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>{getClassificationEmoji(className)}</span>
                  <span style={{ fontWeight: className === results.predicted_label ? 'bold' : 'normal' }}>
                    {className}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ minWidth: '60px', textAlign: 'right' }}>
                    {(probability * 100).toFixed(1)}%
                  </span>
                  <div className="probability-bar">
                    <div 
                      className="probability-fill"
                      style={{ 
                        width: `${probability * 100}%`,
                        background: className === results.predicted_label 
                          ? `linear-gradient(90deg, ${getClassificationColor(className)}, ${getClassificationColor(className)}aa)`
                          : '#dee2e6'
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
        </div>

        {/* Model info */}
        <div style={{ 
          marginTop: '24px', 
          padding: '16px', 
          background: '#f8f9fa', 
          borderRadius: '8px',
          fontSize: '14px',
          color: '#6c757d'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
            <span><strong>Model:</strong> {results.model_version}</span>
            <span><strong>Prediction Time:</strong> {new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResults;
