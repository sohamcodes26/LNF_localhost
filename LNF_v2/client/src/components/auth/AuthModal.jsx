import React from 'react';
import { X } from 'lucide-react';

const AuthModal = ({ children, closeModal }) => {
  return (
    <div 
      className="fixed inset-0 bg-black/60 z-50 flex justify-center items-center"
      onClick={closeModal}
    >
      <div 
        className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md relative"
        onClick={(e) => e.stopPropagation()}
      >
        <button 
          onClick={closeModal}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-800 transition-colors"
        >
          <X size={24} />
        </button>
        {children}
      </div>
    </div>
  );
};

export default AuthModal;