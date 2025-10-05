import React from 'react';
import { User } from 'lucide-react';

const ProfileButton = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="flex items-center justify-center w-10 h-10 bg-gray-200 rounded-full text-gray-600 hover:bg-gray-300 transition-colors"
      aria-label="Open user menu"
    >
      <User size={20} />
    </button>
  );
};

export default ProfileButton;