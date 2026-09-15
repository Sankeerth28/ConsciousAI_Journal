import { apiClient } from './client';
import type { TokenResponse, UserLogin, UserRead, UserRegister } from './types';

/**
 * Register a new user account.
 * Endpoint: POST /api/v1/auth/register
 */
export async function registerUser(payload: UserRegister): Promise<UserRead> {
  const response = await apiClient.post<UserRead>('/api/v1/auth/register', payload);
  return response.data;
}

/**
 * Authenticate with email & password, returning JWT access token.
 * Endpoint: POST /api/v1/auth/login
 */
export async function loginUser(payload: UserLogin): Promise<TokenResponse> {
  const response = await apiClient.post<TokenResponse>('/api/v1/auth/login', payload);
  return response.data;
}

/**
 * Fetch current authenticated user's profile info.
 * Endpoint: GET /api/v1/auth/me
 */
export async function getMe(): Promise<UserRead> {
  const response = await apiClient.get<UserRead>('/api/v1/auth/me');
  return response.data;
}
