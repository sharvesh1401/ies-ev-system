import api from './api';

export interface PredictionParams {
  distance_km: number;
  speed_kmh: number;
  temperature_c: number;
  initial_soc: number;
  initial_soh: number;
  mass_kg?: number;
  drag_coeff?: number;
  model_type?: 'onnx' | 'student' | 'teacher';
}

export interface PredictionResponse {
  energy_kwh: number;
  final_soc: number;
  final_soh: number;
  confidence: number;
  inference_time_ms: number;
  model_used: string;
}

export const predictEnergy = async (params: PredictionParams): Promise<PredictionResponse> => {
  try {
    const response = await api.post('/api/predict/energy', params);
    return response.data;
  } catch (error: any) {
    console.error('Energy prediction failed:', error);
    throw new Error(
      error.response?.data?.detail || 
      'Failed to predict energy. Please try again.'
    );
  }
};

export const checkModelHealth = async () => {
  try {
    const response = await api.get('/api/predict/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw new Error('Backend service unavailable');
  }
};
