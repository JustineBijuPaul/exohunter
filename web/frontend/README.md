# ExoHunter Frontend

React-based frontend for the ExoHunter exoplanet classification system.

## Features

- 📁 **File Upload**: Drag-and-drop CSV file upload with support for tabular features and light curve data
- ✍️ **Manual Entry**: Form-based input for exoplanet features and light curve time series
- 🔮 **Predictions**: Real-time classification with confidence levels and probability distributions
- 📈 **Visualizations**: Interactive Plotly.js charts for light curve data
- 🎨 **Responsive Design**: Mobile-friendly layout with modern UI/UX
- ⚡ **Real-time API**: Live connection to FastAPI backend with health monitoring

## Quick Start

1. **Install dependencies:**
   ```bash
   cd web/frontend
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open in browser:**
   Navigate to `http://localhost:3000`

## Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Integration

The frontend automatically connects to the FastAPI backend running on `http://localhost:8000`. The proxy is configured in `vite.config.js` to route `/api/*` requests to the backend.

### Supported Data Formats

**Tabular Features CSV:**
```csv
period,radius,temperature,magnitude,snr,duration,depth,impact
2.47,1.2,5800,12.3,15.2,3.1,0.01,0.5
```

**Light Curve CSV:**
```csv
time,flux
0.0,1.000
0.1,0.998
0.2,0.999
0.3,0.997
```

## Components

- **App.jsx** - Main application with routing and state management
- **FileUpload.jsx** - Drag-and-drop file upload with validation
- **ManualEntry.jsx** - Form for manual feature and light curve entry
- **PredictionResults.jsx** - Results display with probability bars
- **LightCurveChart.jsx** - Plotly.js visualization for time series data

## Technology Stack

- **React 18** - Frontend framework
- **Vite** - Build tool and dev server
- **Plotly.js** - Interactive charts
- **Axios** - HTTP client for API calls
- **React Dropzone** - File upload component

## Development

### Environment Variables

Create a `.env` file for custom API endpoints:

```env
VITE_API_BASE_URL=http://localhost:8000
```

### Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory and can be served by any static file server.

## Deployment

The frontend can be deployed to any static hosting service like:

- Vercel
- Netlify
- GitHub Pages
- AWS S3 + CloudFront

Make sure to update the API base URL for production deployment.
