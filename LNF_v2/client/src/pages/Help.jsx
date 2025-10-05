import React from 'react';
import { FaArrowLeft, FaQuestionCircle, FaEnvelope, FaPhone, FaInfoCircle } from 'react-icons/fa';

const Help = () => {
  return (
    <div className="min-h-screen bg-gray-200 p-4 sm:p-6 relative flex flex-col">
      {}
      <a
        href="/"
        className="absolute top-4 left-4 sm:top-6 sm:left-6 flex items-center text-blue-700 hover:text-blue-900 transition duration-200 font-semibold z-10"
      >
        <FaArrowLeft className="mr-2" /> Back to Home
      </a>

      <div className="max-w-5xl mx-auto mt-12 bg-white rounded-lg shadow-xl overflow-hidden flex flex-col flex-grow">
        <div className="bg-gradient-to-r from-blue-500 to-blue-700 p-6 text-white text-center flex-shrink-0">
          <h1 className="text-3xl font-extrabold tracking-tight">Help & Support</h1>
          <p className="mt-1 text-blue-100">Find answers to your questions or contact us.</p>
        </div>

        {}
        <div className="p-6 flex-grow grid grid-cols-1 md:grid-cols-2 gap-8 overflow-y-auto custom-scrollbar">
          {}
          <div className="bg-gray-50 rounded-lg p-5 border border-gray-200 shadow-sm flex flex-col">
            <h2 className="text-xl font-semibold mb-4 text-gray-800 flex items-center flex-shrink-0">
              <FaQuestionCircle className="mr-2 text-blue-500" /> Frequently Asked Questions
            </h2>
            <div className="space-y-3 text-gray-700 flex-grow overflow-y-auto pr-2 custom-scrollbar">
              <div>
                <h3 className="font-semibold">How do I report a lost item?</h3>
                <p className="text-sm">Navigate to the "Report Lost Item" section and fill out the form with details about your lost item, including an image if possible.</p>
              </div>
              <div>
                <h3 className="font-semibold">How do I report a found item?</h3>
                <p className="text-sm">Go to the "Report Found Item" section and provide information about the item you found, its location, and date.</p>
              </div>
              <div>
                <h3 className="font-semibold">How can I retrieve my lost item?</h3>
                <p className="text-sm">If your lost item matches a found item, you'll be notified. You can then arrange to verify and retrieve it.</p>
              </div>
              <div>
                <h3 className="font-semibold">What if I found an item but can't find its owner?</h3>
                <p className="text-sm">Report the item as found. Our system will try to match it with lost reports. If no match is found, local authorities might be an option.</p>
              </div>
            </div>
          </div>

          {}
          <div className="bg-gray-50 rounded-lg p-5 border border-gray-200 shadow-sm flex flex-col">
            <h2 className="text-xl font-semibold mb-4 text-gray-800 flex items-center flex-shrink-0">
              <FaEnvelope className="mr-2 text-blue-500" /> Contact Us
            </h2>
            <p className="text-gray-700 mb-4 flex-shrink-0">
              If you can't find an answer to your question, feel free to reach out to us.
            </p>
            <div className="space-y-3 text-gray-700 flex-grow overflow-y-auto pr-2 custom-scrollbar">
              <p className="flex items-center">
                <FaEnvelope className="mr-2 text-gray-600" /> Email: <a href="mailto:support@lostandfound.com" className="ml-1 text-blue-600 hover:underline">support@lostandfound.com</a>
              </p>
              <p className="flex items-center">
                <FaPhone className="mr-2 text-gray-600" /> Phone: <a href="tel:+1234567890" className="ml-1 text-blue-600 hover:underline">+1 (234) 567-890</a>
              </p>
              <p className="flex items-center">
                <FaInfoCircle className="mr-2 text-gray-600" /> Office Hours: Mon-Fri, 9 AM - 5 PM
              </p>
            </div>
          </div>
        </div>
      </div>
      {}
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #888;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #555;
        }
      `}</style>
    </div>
  );
};

export default Help;
