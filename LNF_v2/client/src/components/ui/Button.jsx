import React from 'react';

const Button = ({ children, onClick, className, variant = 'primary' }) => {
  const baseStyles = 'w-full py-3 text-base font-medium rounded-lg shadow-sm hover:shadow-md transition-all duration-300 flex items-center justify-center gap-2';

  const variants = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white',
    secondary: 'border border-blue-300 text-blue-600 hover:bg-blue-50 hover:border-blue-400',
    cta: 'bg-blue-500 text-white font-medium hover:opacity-90',
  };

  return (
    <button onClick={onClick} className={`${baseStyles} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
};

export default Button;
