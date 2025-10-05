import React, { useState } from 'react';
import Button from '../ui/Button';
import Label from '../ui/Label';
import { KeyRound } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

const OtpForm = () => {
  const [otp, setOtp] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { verifyOtp, userEmail } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await verifyOtp(otp);
    } catch (err) {
      setError(err.response?.data?.message || 'Invalid OTP. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-3xl font-bold text-center mb-2 text-gray-800">Verify Your Email</h2>
      <p className="text-center text-gray-500 mb-8">
        An OTP has been sent to <span className="font-medium text-blue-600">{userEmail}</span>.
      </p>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <Label htmlFor="otp">Enter OTP</Label>
          <input
            id="otp"
            type="text"
            maxLength="6"
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            placeholder="123456"
            className="w-full px-4 py-2 text-center tracking-[0.5em] text-lg font-semibold text-gray-800 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
            required
          />
        </div>
        {error && <p className="text-red-500 text-sm text-center">{error}</p>}
        <Button type="submit" variant="primary" disabled={loading}>
          <KeyRound size={18} />
          {loading ? 'Verifying...' : 'Verify & Sign Up'}
        </Button>
      </form>
    </div>
  );
};

export default OtpForm;
