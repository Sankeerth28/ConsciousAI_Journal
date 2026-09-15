import { apiClient } from './client';
import type { HealthResponse, ReadyResponse } from './types';

/**
 * Liveness probe: checks if backend server process is running.
 * Endpoint: GET /health
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
}

/**
 * Readiness probe: checks if PostgreSQL and Redis dependencies are connected.
 * Endpoint: GET /ready
 */
export async function checkReadiness(): Promise<ReadyResponse> {
  const response = await apiClient.get<ReadyResponse>('/ready');
  return response.data;
}
