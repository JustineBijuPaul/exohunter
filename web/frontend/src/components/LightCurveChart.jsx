import React from 'react';
import Plot from 'react-plotly.js';

const LightCurveChart = ({ time, flux, title = 'Light Curve' }) => {
  if (!time || !flux || time.length === 0 || flux.length === 0) {
    return null;
  }

  const plotData = [
    {
      x: time,
      y: flux,
      type: 'scatter',
      mode: 'lines+markers',
      marker: {
        size: 4,
        color: '#2a5298',
      },
      line: {
        color: '#2a5298',
        width: 2,
      },
      name: 'Flux',
    },
  ];

  const layout = {
    title: {
      text: title,
      font: {
        size: 16,
        color: '#333',
      },
    },
    xaxis: {
      title: {
        text: 'Time',
        font: { size: 14 },
      },
      gridcolor: '#e1e5e9',
      zerolinecolor: '#e1e5e9',
    },
    yaxis: {
      title: {
        text: 'Relative Flux',
        font: { size: 14 },
      },
      gridcolor: '#e1e5e9',
      zerolinecolor: '#e1e5e9',
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    margin: {
      l: 60,
      r: 20,
      t: 40,
      b: 60,
    },
    showlegend: false,
    hovermode: 'closest',
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: [
      'pan2d',
      'lasso2d',
      'select2d',
      'autoScale2d',
      'hoverClosestCartesian',
      'hoverCompareCartesian',
      'toggleSpikelines',
    ],
    displaylogo: false,
  };

  return (
    <div className="card">
      <h3>📈 Light Curve Visualization</h3>
      <div style={{ width: '100%', height: '400px' }}>
        <Plot
          data={plotData}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
      
      <div style={{ 
        marginTop: '16px', 
        padding: '12px', 
        background: '#f8f9fa', 
        borderRadius: '6px',
        fontSize: '14px',
        color: '#6c757d'
      }}>
        <strong>Data Points:</strong> {time.length} | 
        <strong> Time Range:</strong> {Math.min(...time).toFixed(3)} - {Math.max(...time).toFixed(3)} | 
        <strong> Flux Range:</strong> {Math.min(...flux).toFixed(6)} - {Math.max(...flux).toFixed(6)}
      </div>
    </div>
  );
};

export default LightCurveChart;
