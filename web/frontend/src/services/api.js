import axios from 'axios';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-api-domain.com' 
  : 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Health check
  async getHealth() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error.message}`);
    }
  },

  // Get model metrics
  async getModelMetrics() {
    try {
      const response = await api.get('/model/metrics');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get model metrics: ${error.message}`);
    }
  },

  // Predict from features
  async predictFromFeatures(features) {
    try {
      const response = await api.post('/predict', { features });
      return response.data;
    } catch (error) {
      const message = error.response?.data?.detail || error.message;
      throw new Error(`Prediction failed: ${message}`);
    }
  },

  // Predict from light curve
  async predictFromLightCurve(time, flux) {
    try {
      const response = await api.post('/predict/lightcurve', { time, flux });
      return response.data;
    } catch (error) {
      const message = error.response?.data?.detail || error.message;
      throw new Error(`Light curve prediction failed: ${message}`);
    }
  },

  // Predict from uploaded file
  async predictFromFile(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await api.post('/predict/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      const message = error.response?.data?.detail || error.message;
      throw new Error(`File upload prediction failed: ${message}`);
    }
  },
};

export default apiService;
