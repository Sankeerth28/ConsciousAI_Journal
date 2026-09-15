import axios from 'axios';
import type { AxiosError, InternalAxiosRequestConfig } from 'axios';
import type { ApiErrorResponse } from './types';

export const TOKEN_STORAGE_KEY = 'consciousai_token';

// Base URL: if VITE_API_BASE_URL is set, use it; otherwise rely on relative URL / Vite proxy
const baseURL = import.meta.env.VITE_API_BASE_URL || '';

export const apiClient = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Request interceptor: inject JWT Bearer token if present
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor: extract error message cleanly & handle auth expiry
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiErrorResponse>) => {
    if (error.response?.status === 401) {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      // Dispatch custom event so AuthContext or app can react smoothly without hard page reload
      window.dispatchEvent(new CustomEvent('consciousai:unauthorized'));
    }
    return Promise.reject(error);
  }
);

/**
 * Helper to parse backend error details into user-friendly strings.
 */
export function formatApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ApiErrorResponse>;
    const data = axiosError.response?.data;

    if (data?.detail) {
      if (typeof data.detail === 'string') {
        return data.detail;
      }
      if (Array.isArray(data.detail) && data.detail.length > 0) {
        return data.detail.map((err) => `${err.loc.join('.')}: ${err.msg}`).join(', ');
      }
    }

    if (axiosError.response?.status === 404) {
      return 'The requested resource was not found.';
    }
    if (axiosError.response?.status === 403) {
      return 'You do not have permission to perform this action.';
    }
    if (axiosError.response?.status === 401) {
      return 'Authentication required or session expired. Please sign in again.';
    }
    if (axiosError.response?.status === 429) {
      return 'Too many requests. Please slow down and try again in a moment.';
    }
    if (axiosError.response?.status === 500) {
      return 'Internal server error. Please try again shortly.';
    }
    if (axiosError.message === 'Network Error') {
      return 'Unable to connect to the ConsciousAI server. Please check your network or ensure backend is running.';
    }
    return axiosError.message || 'An unexpected error occurred.';
  }

  if (error instanceof Error) {
    return error.message;
  }

  return 'An unexpected error occurred.';
}
