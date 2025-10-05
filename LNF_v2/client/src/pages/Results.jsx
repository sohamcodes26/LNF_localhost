import React, { useState, useEffect } from 'react';
import { MessageSquare, Calendar, Search, X, ArrowLeft, CheckCircle, Loader, ShoppingBag, AlertTriangle, ShieldCheck, XCircle, KeyRound, Info, ChevronLeft, ChevronRight } from 'lucide-react';
import axios from 'axios';
import ChatModal from './ChatModal';

let CURRENT_USER_ID = ''; 

// --- Reusable Modal Components ---
const AlertModal = ({ message, onClose, type = 'info' }) => {
  const icons = {
    info: <Info size={24} className="text-blue-500" />,
    success: <CheckCircle size={24} className="text-green-500" />,
    error: <XCircle size={24} className="text-red-500" />,
  };
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl p-6 md:p-8 w-full max-w-md text-center">
        <div className="flex justify-center mb-4">{icons[type]}</div>
        <p className="text-lg text-gray-700 mb-6">{message}</p>
        <button onClick={onClose} className="px-6 py-2 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors">Close</button>
      </div>
    </div>
  );
};

const InputModal = ({ title, message, onConfirm, onCancel }) => {
  const [inputValue, setInputValue] = useState('');
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl p-6 md:p-8 w-full max-w-md">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">{title}</h2>
        <p className="text-gray-600 mb-6">{message}</p>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          maxLength="6"
          placeholder="6-digit code"
        />
        <div className="flex justify-end gap-4 mt-6">
          <button onClick={onCancel} className="px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300">Cancel</button>
          <button onClick={() => onConfirm(inputValue)} className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700">Confirm</button>
        </div>
      </div>
    </div>
  );
};

const TransferCodeModal = ({ code, onClose, title }) => (
  <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 transition-opacity duration-300" onClick={onClose}>
    <div className="bg-white rounded-2xl p-8 text-center relative" onClick={(e) => e.stopPropagation()}>
      <h2 className="text-2xl font-bold mb-2">Your Transfer Code</h2>
      <p className="text-gray-600 mb-6">Share this code with the item owner to complete the transfer.</p>
      <div className="p-4 bg-gray-100 rounded-lg inline-block">
        <p className="text-4xl font-bold tracking-widest text-indigo-600">{code}</p>
      </div>
      <p className="mt-4 text-sm text-gray-500">Item: <strong>{title}</strong></p>
      <button className="absolute top-4 right-4 text-gray-500 hover:text-gray-800" onClick={onClose}><X size={24} /></button>
    </div>
  </div>
);


// --- NEW: Reusable Carousel for both cards and fullscreen view ---
const ImageCarousel = ({ images, altText, onImageClick, isFullscreen = false }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const goToPrevious = (e) => {
    e.stopPropagation();
    const isFirstSlide = currentIndex === 0;
    const newIndex = isFirstSlide ? images.length - 1 : currentIndex - 1;
    setCurrentIndex(newIndex);
  };

  const goToNext = (e) => {
    e.stopPropagation();
    const isLastSlide = currentIndex === images.length - 1;
    const newIndex = isLastSlide ? 0 : currentIndex + 1;
    setCurrentIndex(newIndex);
  };

  const handleImageClick = () => {
    if (onImageClick) {
      onImageClick(images, currentIndex);
    }
  };

  const imageSrc = images && images.length > 0 ? images[currentIndex] : "https://placehold.co/400x300/E0E0E0/6C757D?text=No+Image";

  return (
    <div className={`w-full relative group bg-gray-200 ${isFullscreen ? 'h-full w-full flex items-center justify-center' : 'h-48'}`} onClick={handleImageClick}>
      <img
        src={imageSrc}
        alt={`${altText} ${currentIndex + 1}`}
        className={isFullscreen ? "max-w-full max-h-full object-contain rounded-lg" : "w-full h-full object-cover transition-all duration-300 group-hover:scale-110 cursor-pointer"}
      />
      {images && images.length > 1 && (
        <>
          <button onClick={goToPrevious} className={`absolute top-1/2 -translate-y-1/2 bg-black/50 text-white rounded-full p-2 transition-opacity ${isFullscreen ? 'left-4 opacity-100' : 'left-2 opacity-0 group-hover:opacity-100'}`}>
            <ChevronLeft size={24} />
          </button>
          <button onClick={goToNext} className={`absolute top-1/2 -translate-y-1/2 bg-black/50 text-white rounded-full p-2 transition-opacity ${isFullscreen ? 'right-4 opacity-100' : 'right-2 opacity-0 group-hover:opacity-100'}`}>
            <ChevronRight size={24} />
          </button>
        </>
      )}
    </div>
  );
};


const ResultsPage = () => {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fullscreenData, setFullscreenData] = useState(null); // UPDATED: for fullscreen carousel
  const [searchQuery, setSearchQuery] = useState('');
  const [chatPartner, setChatPartner] = useState(null);
  const [actionLoading, setActionLoading] = useState({});
  const [transferCodeInfo, setTransferCodeInfo] = useState(null);
  const [alertInfo, setAlertInfo] = useState(null);
  const [verifyModalInfo, setVerifyModalInfo] = useState(null);

  const fetchMatches = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:8000/apis/lost-and-found/results/', { withCredentials: true });
      CURRENT_USER_ID = response.data.userId;
      setMatches(response.data.matches || []);
    } catch (err) {
      setError('Failed to fetch matches. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchMatches(); }, []);

  const updateMatchInState = (updatedMatch) => {
    setMatches(currentMatches =>
      currentMatches.map(m =>
        m._id === updatedMatch._id
          ? { ...m, status: updatedMatch.status }
          : m
      )
    );
  };

  const updateAfterConfirm = (confirmedMatch) => {
    setMatches(currentMatches =>
      currentMatches.map(m => {
        if (m._id === confirmedMatch._id) {
          return { ...m, status: 'confirmed' };
        }
        if ((m.lostQuery?._id === confirmedMatch.lostQuery?._id || m.foundQuery?._id === confirmedMatch.foundQuery?._id) && m.status === 'pending') {
          return { ...m, status: 'rejected' };
        }
        return m;
      })
    );
  }

  const handleMatchAction = async (action, resultId, data = {}) => {
    setActionLoading(prev => ({ ...prev, [resultId]: true }));
    try {
      let response;
      const API_BASE_URL = 'http://localhost:8000/apis/lost-and-found/results';
      switch (action) {
        case 'confirm':
          response = await axios.patch(`${API_BASE_URL}/${resultId}/confirm`, {}, { withCredentials: true });
          updateAfterConfirm(response.data);
          break;
        case 'reject':
          response = await axios.patch(`${API_BASE_URL}/${resultId}/reject`, {}, { withCredentials: true });
          updateMatchInState(response.data);
          break;
        case 'generate-code':
          response = await axios.post(`${API_BASE_URL}/${resultId}/generate-code`, {}, { withCredentials: true });
          const match = matches.find(m => m._id === resultId);
          const itemTitle = match?.lostQuery?.objectName || match?.foundQuery?.objectName;
          setTransferCodeInfo({ code: response.data.transferCode, title: itemTitle });
          break;
        case 'verify-code':
          response = await axios.post(`${API_BASE_URL}/${resultId}/verify-code`, { code: data.code }, { withCredentials: true });
          updateMatchInState(response.data.match);
          setAlertInfo({ message: 'Transfer completed successfully!', type: 'success' });
          break;
        default:
          throw new Error('Invalid action');
      }
    } catch (err) {
      console.error(`Error during ${action}:`, err);
      setAlertInfo({ message: `Action failed: ${err.response?.data?.message || 'Server error'}`, type: 'error' });
    } finally {
      setActionLoading(prev => ({ ...prev, [resultId]: false }));
    }
  };

  const handleVerify = (resultId) => {
    setVerifyModalInfo({
      title: 'Complete Transfer',
      message: 'Please enter the 6-digit transfer code to complete the item return.',
      onConfirm: (code) => {
        if (code) {
          handleMatchAction('verify-code', resultId, { code });
        }
        setVerifyModalInfo(null);
      },
      onCancel: () => setVerifyModalInfo(null)
    });
  }

  const openFullscreen = (images, startIndex = 0) => {
    setFullscreenData({ images, startIndex });
    document.body.style.overflow = 'hidden';
  };

  const closeFullscreen = () => {
    setFullscreenData(null);
    document.body.style.overflow = 'auto';
  };

  const openChatModal = (partnerId, itemTitle) => {
    if (partnerId === CURRENT_USER_ID) {
      setAlertInfo({ message: "You cannot chat with yourself!", type: 'info' });
      return;
    }
    setChatPartner({ id: partnerId, title: itemTitle });
    document.body.style.overflow = 'hidden';
  };

  const closeChatModal = () => {
    setChatPartner(null);
    document.body.style.overflow = 'auto';
  };

  const filteredResults = matches.filter((match) => {
    const query = searchQuery.toLowerCase();
    const itemToShow = match.lostItemOwner?._id === CURRENT_USER_ID ? match.foundQuery : match.lostQuery;
    return (
      itemToShow?.objectName?.toLowerCase().includes(query) ||
      itemToShow?.brand?.toLowerCase().includes(query) ||
      itemToShow?.material?.toLowerCase().includes(query) ||
      itemToShow?.markings?.toLowerCase().includes(query)
    );
  });

  const getStatusInfo = (status) => {
    switch (status) {
      case 'pending': return { text: 'Pending Confirmation', icon: AlertTriangle, color: 'text-yellow-500' };
      case 'confirmed': return { text: 'Match Confirmed', icon: ShieldCheck, color: 'text-blue-500' };
      case 'rejected': return { text: 'Match Rejected', icon: XCircle, color: 'text-red-500' };
      case 'transfer_complete': return { text: 'Item Returned', icon: CheckCircle, color: 'text-green-500' };
      default: return { text: 'Unknown Status', icon: AlertTriangle, color: 'text-gray-500' };
    }
  };

  return (
    <div className="min-h-screen w-full bg-gray-50">
      <header className="relative bg-gradient-to-br from-blue-500 to-indigo-600 pb-32 rounded-b-xl md:rounded-b-3xl">
        <a href="/" className="absolute top-4 left-4 z-20 flex items-center gap-2 px-4 py-2 text-white rounded-full transition-all duration-300 hover:bg-white/20">
          <ArrowLeft size={18} />
          <span>Back to Home</span>
        </a>
        <div className="container mx-auto relative z-10 px-4 sm:px-6 lg:px-8 text-center pt-16 pb-4">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white drop-shadow-lg">Potential Matches Found</h1>
          <p className="mt-3 text-lg text-indigo-100 max-w-2xl mx-auto">Review your matches and take action.</p>
        </div>
      </header>

      <main className="container mx-auto relative z-20 p-4 sm:p-6 lg:p-8 -mt-24">
        <div className="bg-white rounded-2xl shadow-2xl p-6 md:p-8">
          <div className="mb-8 flex justify-center">
            <div className="relative w-full max-w-md">
              <input type="text" placeholder="Refine your search..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="w-full pl-12 pr-4 py-3 rounded-full bg-gray-100 border-2 border-transparent text-gray-800 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-300" />
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            </div>
          </div>

          {loading && <div className="flex flex-col items-center justify-center p-10 text-gray-500"><Loader className="animate-spin mb-4" size={48} /><span className="text-xl font-semibold">Searching for Matches...</span></div>}
          {error && <p className="text-center text-red-500 text-xl p-10">{error}</p>}

          {!loading && !error && (
            <>
              {filteredResults.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {filteredResults.map((match) => {
                    const isOwnerOfLostItem = match.lostItemOwner?._id === CURRENT_USER_ID;
                    const itemToShow = isOwnerOfLostItem ? match.foundQuery : match.lostQuery;
                    const partner = isOwnerOfLostItem ? match.foundItemHolder : match.lostItemOwner;
                    const dateType = isOwnerOfLostItem ? "Found on" : "Lost on";
                    const dateValue = isOwnerOfLostItem ? itemToShow?.dateFound : itemToShow?.dateLost;
                    const StatusIcon = getStatusInfo(match.status).icon;
                    
                    return (
                      <div key={match._id} className="group flex flex-col bg-white rounded-xl overflow-hidden shadow-lg border border-gray-200 hover:scale-[1.03] hover:shadow-2xl transition-transform duration-300">
                        <ImageCarousel images={itemToShow?.images} altText={itemToShow?.objectName} onImageClick={() => openFullscreen(itemToShow?.images)} />
                        <div className="p-5 flex flex-col flex-grow">
                          <h3 className="text-lg font-bold text-gray-800 mb-2">{itemToShow?.objectName}</h3>
                          <div className="space-y-3 mb-5 text-sm text-gray-500">
                            <div className={`flex items-center gap-2 font-semibold ${getStatusInfo(match.status).color}`}>
                              <StatusIcon size={16} />
                              <span>{getStatusInfo(match.status).text}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Calendar size={16} className="text-indigo-500" />
                              <span>{dateType}: <strong>{new Date(dateValue).toLocaleDateString()}</strong></span>
                            </div>
                            {itemToShow?.brand && <p><strong>Brand:</strong> {itemToShow.brand}</p>}
                            {itemToShow?.size && <p><strong>Size:</strong> {itemToShow.size}</p>}
                          </div>

                          <div className="mt-auto pt-4 border-t border-gray-200">
                            {match.status === 'pending' && (
                              <div className="flex gap-2">
                                <button onClick={() => handleMatchAction('confirm', match._id)} disabled={actionLoading[match._id]} className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-green-500 rounded-lg font-semibold text-white text-sm transition-all duration-300 hover:bg-green-600 active:scale-95 disabled:bg-gray-400">
                                  {actionLoading[match._id] ? <Loader size={16} className="animate-spin" /> : <CheckCircle size={16} />} Confirm
                                </button>
                                <button onClick={() => handleMatchAction('reject', match._id)} disabled={actionLoading[match._id]} className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-red-500 rounded-lg font-semibold text-white text-sm transition-all duration-300 hover:bg-red-600 active:scale-95 disabled:bg-gray-400">
                                  {actionLoading[match._id] ? <Loader size={16} className="animate-spin" /> : <XCircle size={16} />} Reject
                                </button>
                              </div>
                            )}
                            {match.status === 'confirmed' && (
                              isOwnerOfLostItem ? (
                                <button onClick={() => handleVerify(match._id)} disabled={actionLoading[match._id]} className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 rounded-lg font-semibold text-white text-base transition-all duration-300 hover:bg-blue-700 active:scale-95 disabled:bg-gray-400">
                                  {actionLoading[match._id] ? <Loader size={18} className="animate-spin" /> : <ShieldCheck size={18} />} Complete Transfer
                                </button>
                              ) : (
                                <button onClick={() => handleMatchAction('generate-code', match._id)} disabled={actionLoading[match._id]} className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-indigo-600 rounded-lg font-semibold text-white text-base transition-all duration-300 hover:bg-indigo-700 active:scale-95 disabled:bg-gray-400">
                                  {actionLoading[match._id] ? <Loader size={18} className="animate-spin" /> : <KeyRound size={18} />} Show Transfer Code
                                </button>
                              )
                            )}
                            <button onClick={() => openChatModal(partner?._id, itemToShow?.objectName)} className="w-full mt-2 flex items-center justify-center gap-2 px-4 py-2 bg-gray-600 rounded-lg font-semibold text-white text-base transition-all duration-300 hover:bg-gray-700 active:scale-95">
                              <MessageSquare size={18} /> Chat with {partner?.name}
                            </button>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="text-center text-gray-500 p-12 bg-gray-50 rounded-xl">
                  <ShoppingBag size={48} className="mx-auto mb-4 text-gray-400" />
                  <h3 className="text-xl font-semibold text-gray-700">No Matches Found Yet</h3>
                  <p className="mt-2 text-base">We'll notify you when a potential match for your items is reported.</p>
                </div>
              )}
            </>
          )}
        </div>
      </main>

      {fullscreenData && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 transition-opacity duration-300" onClick={closeFullscreen}>
          <div className="relative w-full h-full max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <ImageCarousel images={fullscreenData.images} altText="Fullscreen view" isFullscreen={true} />
          </div>
          <button className="absolute top-4 right-4 text-white bg-black/50 rounded-full p-2" onClick={closeFullscreen}><X size={24} /></button>
        </div>
      )}

      {chatPartner && <ChatModal currentUserId={CURRENT_USER_ID} chatPartner={{ id: chatPartner.id, title: chatPartner.title }} onClose={closeChatModal} />}
      {transferCodeInfo && <TransferCodeModal code={transferCodeInfo.code} title={transferCodeInfo.title} onClose={() => setTransferCodeInfo(null)} />}
      {alertInfo && <AlertModal message={alertInfo.message} type={alertInfo.type} onClose={() => setAlertInfo(null)} />}
      {verifyModalInfo && <InputModal {...verifyModalInfo} />}
    </div>
  );
};

export default ResultsPage;