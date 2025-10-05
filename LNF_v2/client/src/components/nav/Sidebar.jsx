import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { User, Archive, CheckSquare, HelpCircle, LogOut, X, Sparkles , LineChartIcon} from 'lucide-react';

const Sidebar = ({ isOpen, onClose, side = 'left' }) => {
  const { logout, userEmail } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    onClose();
    navigate('/');
  };

  const linkClass = "flex items-center gap-4 px-4 py-3 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors";
  const activeLinkClass = "bg-blue-100 text-blue-600 font-semibold";

  const sidebarPositionClass = side === 'right' ? 'right-0' : 'left-0';
  const sidebarTransformClass = isOpen
    ? 'translate-x-0' 
    : (side === 'right' ? 'translate-x-full' : '-translate-x-full'); 

  return (
    <>
      {}
      <div
        className={`fixed inset-0 bg-black/60 z-40 transition-opacity duration-300 ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        onClick={onClose}
      ></div>

      {}
      <aside
        className={`fixed top-0 h-full w-72 bg-gray-100 shadow-xl z-50 transform transition-transform duration-300
          ${sidebarPositionClass}
          ${sidebarTransformClass}
        `}
      >
        <div className="p-6">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-xl font-bold text-blue-800">My Profile</h2>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
              <X size={24} />
            </button>
          </div>

          <nav className="flex flex-col gap-2">
            <NavLink to="/account" className={({isActive}) => `${linkClass} ${isActive ? activeLinkClass : ''}`} onClick={onClose}>
              <User size={20} />
              <span>My Account</span>
            </NavLink>
            <NavLink to="/my-found-items" className={({isActive}) => `${linkClass} ${isActive ? activeLinkClass : ''}`} onClick={onClose}>
              <Archive size={20} />
              <span>Found Items</span>
            </NavLink>
            <NavLink to="/my-received-items" className={({isActive}) => `${linkClass} ${isActive ? activeLinkClass : ''}`} onClick={onClose}>
              <CheckSquare size={20} />
              <span>My Lost Items</span>
            </NavLink>
            <NavLink to="/results" className={({isActive}) => `${linkClass} ${isActive ? activeLinkClass : ''}`} onClick={onClose}>
              <Sparkles size={20} /> 
              <span>Results</span>
            </NavLink>

            <NavLink to="/statistics" className={({isActive}) => `${linkClass} ${isActive ? activeLinkClass : ''}`} onClick={onClose}>
              <LineChartIcon size={20} /> 
              <span>Statistics</span>
            </NavLink>
            
            <NavLink to="/help" className={({isActive}) => `${linkClass} ${isActive ? activeLinkClass : ''}`} onClick={onClose}>
              <HelpCircle size={20} />
              <span>Help</span>
            </NavLink>
            <button onClick={handleLogout} className={`${linkClass} mt-60 text-red-600 hover:bg-red-100`}>
              <LogOut size={20} />
              <span>Logout</span>
            </button>
          </nav>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
