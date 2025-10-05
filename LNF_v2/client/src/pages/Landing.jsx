import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Search, PlusCircle, LogIn, UserPlus, Mail, Phone, MapPin } from 'lucide-react';
import { useAuth } from '../context/AuthContext'; 

import AuthModal from '../components/auth/AuthModal';
import LoginForm from '../components/auth/LoginForm';
import SignUpForm from '../components/auth/SignUpForm';
import OtpForm from '../components/auth/OtpForm';
import ProfileButton from '../components/nav/ProfileButton';
import Sidebar from '../components/nav/Sidebar'; 

import backgroundImageUrl from '../assets/image.png';

const LandingPage = () => {
  const [modalView, setModalView] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const { isAuthenticated, authState } = useAuth(); 

  const renderModalContent = () => {
    if (authState === 'pending-verification') {
      return <OtpForm />;
    }
    if (modalView === 'login') {
      return <LoginForm />;
    }
    if (modalView === 'signup') {
      return <SignUpForm />;
    }
    return null;
  };

  const isModalOpen = authState === 'pending-verification' || modalView !== null;
  const closeModal = () => setModalView(null);

  const handleProtectedButtonClick = (e, targetPath) => {
    if (!isAuthenticated) {
      e.preventDefault(); 
      setModalView('login'); 
    } else {

    }
  };

  const tooltipMessage = "Login or signup to use services";

  return (
    <div className="min-h-screen bg-gray-100 font-sans">
      {}
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} side="right" />

      {}
      {isModalOpen && !isAuthenticated && (
        <AuthModal closeModal={closeModal}>
          {renderModalContent()}
        </AuthModal>
      )}

      {}
      <header
        className="relative min-h-[60vh] md:min-h-[75vh] bg-cover bg-center text-white flex flex-col"
        style={{ backgroundImage: `url(${backgroundImageUrl})` }}
      >
        <div className="absolute inset-0 bg-black/50 z-0"></div>
        <nav className="relative z-10 w-full p-6 flex justify-between items-center">
          <Link to="/" className="text-2xl font-bold tracking-tight">
            Findit
          </Link>
          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <ProfileButton onClick={() => setIsSidebarOpen(true)} />
            ) : (
              <>
                <button onClick={() => setModalView('login')} className="flex items-center gap-2 px-4 py-2 text-sm font-medium hover:bg-white/10 rounded-lg transition-colors">
                  <LogIn size={16} />
                  <span>Login</span>
                </button>
                <button onClick={() => setModalView('signup')} className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
                  <UserPlus size={16} />
                  <span>Sign Up</span>
                </button>
              </>
            )}
          </div>
        </nav>
        <div className="relative z-10 flex-grow flex flex-col justify-center items-center text-center px-4">
          <h1 className="text-4xl md:text-6xl font-extrabold leading-tight drop-shadow-lg">
            Never Lose Hope,
            <br />
            <span className="text-blue-300">Always Find What's Lost.</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg md:text-xl text-gray-200 drop-shadow-md">
            Connect with your campus community through our AI-powered Lost & Found platform. Post found items, search for lost belongings, and reunite with what matters most.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row gap-4">
            {}
            <Link
              to={isAuthenticated ? "/find-lost-item" : "#"}
              onClick={(e) => handleProtectedButtonClick(e, "/find-lost-item")}
              title={!isAuthenticated ? tooltipMessage : ""}
              className={`flex items-center justify-center gap-2 px-8 py-3 rounded-lg text-lg shadow-lg transition-all duration-300
                ${isAuthenticated
                  ? 'bg-blue-600 text-white hover:bg-blue-700 hover:scale-105'
                  : 'bg-blue-600 text-white cursor-pointer hover:bg-blue-700 hover:scale-105'
                }`}
            >
              <Search size={20} />
              <span>Find My Lost Item</span>
            </Link>

            {}
            <Link
              to={isAuthenticated ? "/post-found" : "#"}
              onClick={(e) => handleProtectedButtonClick(e, "/post-found")}
              title={!isAuthenticated ? tooltipMessage : ""}
              className={`flex items-center justify-center gap-2 px-8 py-3 rounded-lg text-lg shadow-lg transition-all duration-300
                ${isAuthenticated
                  ? 'bg-white/20 backdrop-blur-sm border border-white/30 text-white hover:bg-white/30 hover:scale-105'
                  : 'bg-white/20 backdrop-blur-sm border border-white/30 text-white cursor-pointer hover:bg-white/30 hover:scale-105'
                }`}
            >
              <PlusCircle size={20} />
              <span>Post Found Item</span>
            </Link>
          </div>
        </div>
      </header>

      {}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold text-gray-800 mb-4">How It Works</h2>
          <p className="text-gray-600 mb-12 max-w-2xl mx-auto">Our simple 4-step process makes finding lost items and returning found ones effortless.</p>
          <div className="grid md:grid-cols-4 gap-8">
            <div className="p-6">
              <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-2xl font-bold">1</div>
              <h3 className="text-xl font-semibold mb-2">Post Found Items</h3>
              <p className="text-gray-500">Found something? Snap a picture and post it to help others find their belongings.</p>
            </div>
            <div className="p-6">
              <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-2xl font-bold">2</div>
              <h3 className="text-xl font-semibold mb-2">AI-Powered Search</h3>
              <p className="text-gray-500">Lost something? Describe it, and our AI will find visually similar posted items to help you locate it.</p>
            </div>
            <div className="p-6">
              <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-2xl font-bold">3</div>
              <h3 className="text-xl font-semibold mb-2">Connect & Communicate</h3>
              <p className="text-gray-500">Once you find a match, connect directly with the person who has your item through secure contact.</p>
            </div>
            <div className="p-6">
              <div className="w-16 h-16 mx-auto mb-4 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-2xl font-bold">4</div>
              <h3 className="text-xl font-semibold mb-2">Reunite & Celebrate</h3>
              <p className="text-gray-500">Meet up safely on campus to retrieve your long-lost belongings. Help build our caring community.</p>
            </div>
          </div>
        </div>
      </section>

      {}
      <footer className="bg-slate-800 text-slate-300">
        <div className="container mx-auto px-6 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="md:col-span-2">
              <h3 className="text-xl font-bold text-white mb-4">Lost&Found</h3>
              <p className="max-w-md">Connecting campus communities through AI-powered lost and found services. Helping students reunite with their belongings quickly and safely.</p>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Quick Links</h3>
              <ul className="space-y-2">
                <li><Link to="/find-lost-item" className="hover:text-white transition-colors">Browse Items</Link></li>
                <li><Link to="/post-found" className="hover:text-white transition-colors">Post Found Item</Link></li>
                <li><Link to="/how-it-works" className="hover:text-white transition-colors">How It Works</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Campus</h3>
              <ul className="space-y-2">
                <li className="flex items-center gap-2"><MapPin size={16} /><span>University Center</span></li>
                <li className="flex items-center gap-2"><Mail size={16} /><span>support@lnf.edu</span></li>
                <li className="flex items-center gap-2"><Phone size={16} /><span>(123) 456-7890</span></li>
              </ul>
            </div>
          </div>
          <div className="mt-12 pt-8 border-t border-slate-700 text-center text-sm text-slate-400">
            <p>&copy; {new Date().getFullYear()} Lost&Found. All Rights Reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
