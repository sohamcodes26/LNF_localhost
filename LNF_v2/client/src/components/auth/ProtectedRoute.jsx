import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

const ProtectedRoute = () => {
  
  const { authState } = useAuth();

  if (authState === 'loading') {
    return null; 
  }


  if (authState === 'authenticated') {
    return <Outlet />;
  }

  return <Navigate to="/" replace />;
};

export default ProtectedRoute;