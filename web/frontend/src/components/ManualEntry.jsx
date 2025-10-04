import React, { useState } from 'react';

const ManualEntry = ({ onPredict, loading }) => {
  const [features, setFeatures] = useState({
    period: '',
    radius: '',
    temperature: '',
    magnitude: '',
    snr: '',
    duration: '',
    depth: '',
    impact: ''
  });

  const [lightCurve, setLightCurve] = useState({
    time: '',
    flux: ''
  });

  const [mode, setMode] = useState('features'); // 'features' or 'lightcurve'

  const handleFeatureChange = (field, value) => {
    setFeatures(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleLightCurveChange = (field, value) => {
    setLightCurve(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleFeaturesSubmit = (e) => {
    e.preventDefault();
    
    // Convert features to array of numbers
    const featureArray = Object.values(features).map(val => {
      const num = parseFloat(val);
      if (isNaN(num)) {
        throw new Error(`Invalid number: ${val}`);
      }
      return num;
    });

    if (featureArray.length !== 8) {
      alert('Please fill in all 8 feature fields');
      return;
    }

    onPredict('features', featureArray);
  };

  const handleLightCurveSubmit = (e) => {
    e.preventDefault();
    
    try {
      // Parse comma-separated values
      const timeArray = lightCurve.time.split(',').map(val => {
        const num = parseFloat(val.trim());
        if (isNaN(num)) {
          throw new Error(`Invalid time value: ${val}`);
        }
        return num;
      });

      const fluxArray = lightCurve.flux.split(',').map(val => {
        const num = parseFloat(val.trim());
        if (isNaN(num)) {
          throw new Error(`Invalid flux value: ${val}`);
        }
        return num;
      });

      if (timeArray.length !== fluxArray.length) {
        throw new Error('Time and flux arrays must have the same length');
      }

      if (timeArray.length < 10) {
        throw new Error('Light curve must have at least 10 data points');
      }

      onPredict('lightcurve', { time: timeArray, flux: fluxArray });
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  const fillSampleData = () => {
    if (mode === 'features') {
      setFeatures({
        period: '2.47',
        radius: '1.2',
        temperature: '5800',
        magnitude: '12.3',
        snr: '15.2',
        duration: '3.1',
        depth: '0.01',
        impact: '0.5'
      });
    } else {
      setLightCurve({
        time: '0,1,2,3,4,5,6,7,8,9',
        flux: '1.0,0.998,0.999,0.997,1.001,0.999,1.0,0.998,1.001,0.999'
      });
    }
  };

  return (
    <div className="card">
      <h2>✍️ Manual Entry</h2>
      
      <div className="tabs">
        <button 
          className={`tab ${mode === 'features' ? 'active' : ''}`}
          onClick={() => setMode('features')}
        >
          Tabular Features
        </button>
        <button 
          className={`tab ${mode === 'lightcurve' ? 'active' : ''}`}
          onClick={() => setMode('lightcurve')}
        >
          Light Curve Data
        </button>
      </div>

      {mode === 'features' ? (
        <form onSubmit={handleFeaturesSubmit}>
          <div className="instructions">
            <h3>Exoplanet Features:</h3>
            <ul>
              <li>Enter numerical values for each exoplanet characteristic</li>
              <li>All fields are required for accurate prediction</li>
              <li>Use the sample data button to see example values</li>
            </ul>
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label>Orbital Period (days)</label>
              <input
                type="number"
                step="any"
                value={features.period}
                onChange={(e) => handleFeatureChange('period', e.target.value)}
                placeholder="e.g., 2.47"
                required
              />
            </div>

            <div className="form-group">
              <label>Planet Radius (Earth radii)</label>
              <input
                type="number"
                step="any"
                value={features.radius}
                onChange={(e) => handleFeatureChange('radius', e.target.value)}
                placeholder="e.g., 1.2"
                required
              />
            </div>

            <div className="form-group">
              <label>Stellar Temperature (K)</label>
              <input
                type="number"
                step="any"
                value={features.temperature}
                onChange={(e) => handleFeatureChange('temperature', e.target.value)}
                placeholder="e.g., 5800"
                required
              />
            </div>

            <div className="form-group">
              <label>Stellar Magnitude</label>
              <input
                type="number"
                step="any"
                value={features.magnitude}
                onChange={(e) => handleFeatureChange('magnitude', e.target.value)}
                placeholder="e.g., 12.3"
                required
              />
            </div>

            <div className="form-group">
              <label>Signal-to-Noise Ratio</label>
              <input
                type="number"
                step="any"
                value={features.snr}
                onChange={(e) => handleFeatureChange('snr', e.target.value)}
                placeholder="e.g., 15.2"
                required
              />
            </div>

            <div className="form-group">
              <label>Transit Duration (hours)</label>
              <input
                type="number"
                step="any"
                value={features.duration}
                onChange={(e) => handleFeatureChange('duration', e.target.value)}
                placeholder="e.g., 3.1"
                required
              />
            </div>

            <div className="form-group">
              <label>Transit Depth (fraction)</label>
              <input
                type="number"
                step="any"
                value={features.depth}
                onChange={(e) => handleFeatureChange('depth', e.target.value)}
                placeholder="e.g., 0.01"
                required
              />
            </div>

            <div className="form-group">
              <label>Impact Parameter</label>
              <input
                type="number"
                step="any"
                value={features.impact}
                onChange={(e) => handleFeatureChange('impact', e.target.value)}
                placeholder="e.g., 0.5"
                required
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={fillSampleData}
              disabled={loading}
            >
              Fill Sample Data
            </button>
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={loading}
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  Predicting...
                </>
              ) : (
                '🔮 Predict Exoplanet'
              )}
            </button>
          </div>
        </form>
      ) : (
        <form onSubmit={handleLightCurveSubmit}>
          <div className="instructions">
            <h3>Light Curve Data:</h3>
            <ul>
              <li>Enter comma-separated values for time and flux measurements</li>
              <li>Time and flux arrays must have the same length</li>
              <li>Minimum 10 data points required</li>
            </ul>
          </div>

          <div className="form-group">
            <label>Time Values (comma-separated)</label>
            <input
              type="text"
              value={lightCurve.time}
              onChange={(e) => handleLightCurveChange('time', e.target.value)}
              placeholder="e.g., 0,1,2,3,4,5,6,7,8,9"
              required
            />
          </div>

          <div className="form-group">
            <label>Flux Values (comma-separated)</label>
            <input
              type="text"
              value={lightCurve.flux}
              onChange={(e) => handleLightCurveChange('flux', e.target.value)}
              placeholder="e.g., 1.0,0.998,0.999,0.997,1.001,0.999,1.0,0.998,1.001,0.999"
              required
            />
          </div>

          <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={fillSampleData}
              disabled={loading}
            >
              Fill Sample Data
            </button>
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={loading}
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  Predicting...
                </>
              ) : (
                '🔮 Predict from Light Curve'
              )}
            </button>
          </div>
        </form>
      )}
    </div>
  );
};

export default ManualEntry;
