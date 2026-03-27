import axios from 'axios';

// The base URL can be controlled via env vars in Vite. 
// Using a fallback for local development if not provided.
const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor: Attach JWT token if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: Global error handling
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      // Logic to handle unauthorized errors
      // E.g., redirecting to login, purging invalid token, etc.
      localStorage.removeItem('auth_token');
      // For now, if we had a router integration we could push to /login, 
      // but reloading state or dispatching to auth store is preferred.
      console.warn('[API] Unauthorized. Local auth state cleared.');
    }
    
    if (error.response?.status === 429) {
      // Rate limit exceeded
      console.warn('[API] Too many requests. Rate limit triggered.');
      // Usually trigger a toast or notification system here
    }
    
    return Promise.reject(error);
  }
);

export default api;
