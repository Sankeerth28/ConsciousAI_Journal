import React, { createContext, useEffect, useState, useCallback } from 'react';
import { getMe, loginUser, registerUser } from '../api/auth';
import { TOKEN_STORAGE_KEY } from '../api/client';
import type { UserLogin, UserRead, UserRegister } from '../api/types';

export interface AuthContextType {
  user: UserRead | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (credentials: UserLogin) => Promise<void>;
  register: (credentials: UserRegister) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem(TOKEN_STORAGE_KEY)
  );
  const [user, setUser] = useState<UserRead | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(() => !!localStorage.getItem(TOKEN_STORAGE_KEY));

  const fetchProfile = useCallback(async () => {
    try {
      const userData = await getMe();
      setUser(userData);
    } catch {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      setToken(null);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    if (token) {
      getMe()
        .then((userData) => {
          if (active) {
            setUser(userData);
            setIsLoading(false);
          }
        })
        .catch(() => {
          if (active) {
            localStorage.removeItem(TOKEN_STORAGE_KEY);
            setToken(null);
            setUser(null);
            setIsLoading(false);
          }
        });
    }

    const handleUnauthorized = () => {
      setToken(null);
      setUser(null);
      setIsLoading(false);
    };

    window.addEventListener('consciousai:unauthorized', handleUnauthorized);
    return () => {
      active = false;
      window.removeEventListener('consciousai:unauthorized', handleUnauthorized);
    };
  }, [token]);

  const login = async (credentials: UserLogin) => {
    setIsLoading(true);
    try {
      const tokenData = await loginUser(credentials);
      localStorage.setItem(TOKEN_STORAGE_KEY, tokenData.access_token);
      setToken(tokenData.access_token);
      const userData = await getMe();
      setUser(userData);
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (credentials: UserRegister) => {
    setIsLoading(true);
    try {
      await registerUser(credentials);
      const tokenData = await loginUser({
        email: credentials.email,
        password: credentials.password,
      });
      localStorage.setItem(TOKEN_STORAGE_KEY, tokenData.access_token);
      setToken(tokenData.access_token);
      const userData = await getMe();
      setUser(userData);
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setToken(null);
    setUser(null);
  };

  const refreshUser = async () => {
    if (token) {
      await fetchProfile();
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        isAuthenticated: !!token && !!user,
        login,
        register,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export { useAuth } from './useAuth';
