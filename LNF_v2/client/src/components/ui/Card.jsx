import React from 'react';

const Card = ({ children, className }) => {
  return (
    <div className={`bg-white p-8 rounded-2xl border border-gray-200/80 shadow-sm hover:shadow-lg transition-all duration-300 ${className}`}>
      {children}
    </div>
  );
};

export default Card;