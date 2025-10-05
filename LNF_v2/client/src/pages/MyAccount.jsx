import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { FaArrowLeft } from 'react-icons/fa'; 

const MyAccount = () => {
  const [profile, setProfile] = useState({ fullName: '', email: '' });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [passwordMessage, setPasswordMessage] = useState('');

  useEffect(() => {
    const fetchProfile = async () => {




      const token = localStorage.getItem('token'); 





      const config = {
        headers: {



        },
        withCredentials: true 
      };


      try {
        setLoading(true); 
        const res = await axios.get('http://localhost:8000/apis/lost-and-found/MyAccount/getprofile', config);
        setProfile(res.data);
        setError(''); 
      } catch (err) {
        console.error("Error fetching profile:", err);

        if (err.response && (err.response.status === 401 || err.response.status === 403)) {
            setError('Session expired or not authenticated. Please log in again.');


        } else {
            setError(err.response?.data?.message || 'Failed to load profile. Please try again.');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, []);

  const handleChangePassword = async (e) => {
    e.preventDefault();
    setPasswordMessage(''); 

    if (!oldPassword || !newPassword) {
      setPasswordMessage('Both password fields are required.');
      return;
    }

    if (oldPassword === newPassword) {
      setPasswordMessage('New password cannot be the same as the old password.');
      return;
    }

    const token = localStorage.getItem('token');
    const config = {
      headers: {


      },
      withCredentials: true 
    };

    try {
      const res = await axios.post(
        'http://localhost:8000/apis/lost-and-found/MyAccount/change-password',
        { oldPassword, newPassword },
        config
      );
      setPasswordMessage(res.data.message || 'Password changed successfully!');
      setOldPassword('');
      setNewPassword('');
    } catch (err) {
      console.error("Error changing password:", err);
      setPasswordMessage(err.response?.data?.message || 'Password update failed. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 sm:p-6 relative">
      {}
      <a
        href="/" 
        className="absolute top-4 left-4 sm:top-6 sm:left-6 flex items-center text-blue-600 hover:text-blue-800 transition duration-200"
      >
        <FaArrowLeft className="mr-2" /> Back to Home
      </a>

      <div className="max-w-xl mx-auto mt-12 bg-white rounded-lg shadow-xl overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-blue-800 p-6 text-white text-center">
          <h1 className="text-3xl font-extrabold tracking-tight">My Account</h1>
          <p className="mt-1 text-blue-200">Manage your profile and security settings.</p>
        </div>

        <div className="p-6 space-y-8">
          {loading && (
            <div className="text-center text-blue-600 font-medium">Loading profile...</div>
          )}
          {error && (
            <div className="text-red-600 bg-red-100 border border-red-200 p-3 rounded-md text-center">
              {error}
            </div>
          )}

          {!loading && !error && (
            <>
              {}
              <div className="bg-gray-50 rounded-lg p-5 border border-gray-200">
                <h2 className="text-xl font-semibold mb-3 text-gray-800 flex items-center">
                  <span className="mr-2 text-blue-500">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </span>
                  Profile Information
                </h2>
                <div className="space-y-2 text-gray-700">
                  <p><strong className="font-medium">Full Name:</strong> {profile.fullName}</p>
                  <p><strong className="font-medium">Email:</strong> {profile.email}</p>
                </div>
              </div>

              {}
              <form onSubmit={handleChangePassword} className="bg-gray-50 rounded-lg p-5 border border-gray-200 space-y-4">
                <h2 className="text-xl font-semibold text-gray-800 flex items-center">
                  <span className="mr-2 text-blue-500">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2v5a2 2 0 01-2 2H9a2 2 0 01-2-2V9a2 2 0 012-2h6z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 11.125A3.25 3.25 0 1112.25 8.25 3.25 3.25 0 019 11.125z" />
                    </svg>
                  </span>
                  Change Password
                </h2>

                <div>
                  <label htmlFor="oldPassword" className="block mb-1 text-sm font-medium text-gray-700">Old Password</label>
                  <input
                    id="oldPassword"
                    type="password"
                    value={oldPassword}
                    onChange={(e) => setOldPassword(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                    required
                  />
                </div>

                <div>
                  <label htmlFor="newPassword" className="block mb-1 text-sm font-medium text-gray-700">New Password</label>
                  <input
                    id="newPassword"
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                    required
                  />
                </div>

                <button
                  type="submit"
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition duration-200 font-semibold"
                >
                  Change Password
                </button>

                {passwordMessage && (
                  <p className={`text-center text-sm mt-2 ${passwordMessage.includes('successful') ? 'text-green-600' : 'text-red-600'}`}>
                    {passwordMessage}
                  </p>
                )}
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default MyAccount;