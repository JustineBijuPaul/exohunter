import React, { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import ManualEntry from './components/ManualEntry';
import PredictionResults from './components/PredictionResults';
import LightCurveChart from './components/LightCurveChart';
import { apiService } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [lightCurveData, setLightCurveData] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);

  // Check API health on component mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await apiService.getHealth();
        setApiHealth(health);
      } catch (error) {
        console.error('API health check failed:', error);
        setApiHealth({ status: 'unhealthy', error: error.message });
      }
    };

    checkHealth();
  }, []);

  const handleFileUpload = async (file) => {
    setLoading(true);
    setError(null);
    setResults(null);
    setLightCurveData(null);

    try {
      const prediction = await apiService.predictFromFile(file);
      setResults(prediction);
      
      // If the file contained light curve data, we could extract it here
      // For now, we'll just show the prediction results
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleManualPredict = async (type, data) => {
    setLoading(true);
    setError(null);
    setResults(null);
    setLightCurveData(null);

    try {
      let prediction;
      
      if (type === 'features') {
        prediction = await apiService.predictFromFeatures(data);
      } else if (type === 'lightcurve') {
        prediction = await apiService.predictFromLightCurve(data.time, data.flux);
        // Store light curve data for visualization
        setLightCurveData({
          time: data.time,
          flux: data.flux,
          title: 'Uploaded Light Curve Data'
        });
      }
      
      setResults(prediction);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setResults(null);
    setError(null);
    setLightCurveData(null);
  };

  return (
    <div className="container">
      {/* Header */}
      <div className="header">
        <h1>🌟 ExoHunter</h1>
        <p>Exoplanet Classification using Machine Learning</p>
        
        {/* API Status */}
        <div style={{ 
          marginTop: '16px', 
          padding: '8px 16px', 
          borderRadius: '20px', 
          display: 'inline-block',
          fontSize: '14px',
          background: apiHealth?.status === 'healthy' ? '#d4edda' : '#f8d7da',
          color: apiHealth?.status === 'healthy' ? '#155724' : '#721c24'
        }}>
          {apiHealth?.status === 'healthy' ? '✅ API Online' : '⚠️ API Offline'}
          {apiHealth?.model_loaded && ' • Model Loaded'}
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => { setActiveTab('upload'); clearResults(); }}
        >
          📁 File Upload
        </button>
        <button 
          className={`tab ${activeTab === 'manual' ? 'active' : ''}`}
          onClick={() => { setActiveTab('manual'); clearResults(); }}
        >
          ✍️ Manual Entry
        </button>
      </div>

      {/* Content */}
      {activeTab === 'upload' ? (
        <FileUpload 
          onFileSelect={handleFileUpload}
          loading={loading}
        />
      ) : (
        <ManualEntry 
          onPredict={handleManualPredict}
          loading={loading}
        />
      )}

      {/* Light Curve Visualization */}
      {lightCurveData && (
        <LightCurveChart 
          time={lightCurveData.time}
          flux={lightCurveData.flux}
          title={lightCurveData.title}
        />
      )}

      {/* Results */}
      <PredictionResults 
        results={results}
        error={error}
      />

      {/* Instructions */}
      <div className="card">
        <div className="instructions">
          <h3>How to Use ExoHunter:</h3>
          <ul>
            <li><strong>File Upload:</strong> Upload a CSV file with exoplanet features or light curve data</li>
            <li><strong>Manual Entry:</strong> Enter features manually or provide time/flux arrays</li>
            <li><strong>Features:</strong> Provide 8 numerical features like period, radius, temperature, etc.</li>
            <li><strong>Light Curves:</strong> Provide time-series flux measurements for transit analysis</li>
            <li><strong>Results:</strong> View classification, confidence levels, and probability distributions</li>
          </ul>
        </div>
      </div>

      {/* Footer */}
      <div style={{ 
        textAlign: 'center', 
        padding: '20px', 
        color: 'white', 
        opacity: 0.8, 
        fontSize: '14px' 
      }}>
        <p>
          ExoHunter v1.0.0 • Powered by Machine Learning • 
          {apiHealth?.model_loaded ? ` Using ${results?.model_version || 'ensemble'} model` : ' Demo mode'}
        </p>
      </div>
    </div>
  );
}

export default App;
