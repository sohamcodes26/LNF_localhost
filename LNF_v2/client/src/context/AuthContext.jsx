import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

axios.defaults.withCredentials = true;

const AuthContext = createContext(null);

const API_URL = 'http://localhost:8000/apis/lost-and-found/auth';

export const AuthProvider = ({ children }) => {
  const [authState, setAuthState] = useState('loading'); 
  const [userEmail, setUserEmail] = useState(null);

  useEffect(() => {
    const validateUser = async () => {
      try {
        await axios.post(`${API_URL}/validate`);
        setAuthState('authenticated');
      } catch (error) {
        setAuthState('unauthenticated');
      }
    };
    validateUser();
  }, []);

  const signup = async (fullName, email, password) => {

    await axios.post(`${API_URL}/signup`, { email, password, fullName });
    setUserEmail(email);
    setAuthState('pending-verification');
  };

  const verifyOtp = async (otp) => {
    await axios.post(`${API_URL}/otp_verification`, { email: userEmail, otp });
    setAuthState('authenticated');
  };

  const login = async (email, password) => {
    await axios.post(`${API_URL}/signin`, { email, password });
    setUserEmail(email);
    setAuthState('authenticated');
  };

  const logout = async () => {
    await axios.post(`${API_URL}/sign-out`);
    setUserEmail(null);
    setAuthState('unauthenticated');
  };

  const value = {
    authState,
    userEmail,
    isAuthenticated: authState === 'authenticated',
    signup,
    verifyOtp,
    login,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  return useContext(AuthContext);
};
