import axios from 'axios';

// Create axios instance with base URL
// Vite proxy will handle request to /api -> backend:8000/api
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

// Response Types
export interface HealthStatus {
  status: string;
  database: string;
  redis: string;
}

export interface ChatResponse {
  response: string;
  raw?: any;
}

// API Service
export const apiService = {
  checkHealth: async (): Promise<HealthStatus> => {
    const response = await api.get<HealthStatus>('/health');
    return response.data;
  },

  chatWithAI: async (message: string): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/ai/chat', { message });
    return response.data;
  }
};
