import React, { useEffect, useState } from "react";
import axios from "axios";
import { FaArrowLeft, FaBoxOpen, FaMapMarkerAlt, FaCalendarAlt } from "react-icons/fa"; 
import { CheckCircle, Tag, Droplet, Ruler, Fingerprint, ChevronLeft, ChevronRight, X } from "lucide-react";

// --- Reusable Image Carousel Component (Upgraded to support fullscreen) ---
const ImageCarousel = ({ images, altText, onImageClick, isFullscreen = false, startIndex = 0 }) => {
  const [currentIndex, setCurrentIndex] = useState(startIndex);

  useEffect(() => {
    setCurrentIndex(startIndex); // Sync index when props change
  }, [startIndex]);

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


const FoundItems = () => {
  const [foundItems, setFoundItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [fullscreenData, setFullscreenData] = useState(null); // State for fullscreen view

  useEffect(() => {
    const fetchFoundItems = async () => {
      try {
        const response = await axios.get("http://localhost:8000/apis/lost-and-found/my-items/my-found-items", { withCredentials: true });
        setFoundItems(response.data.foundItems);
      } catch (err) {
        setError(err.response?.data?.message || "Failed to fetch found items.");
      } finally {
        setLoading(false);
      }
    };
    fetchFoundItems();
  }, []);

  // Functions to handle fullscreen modal
  const openFullscreen = (images, startIndex = 0) => {
    setFullscreenData({ images, startIndex });
    document.body.style.overflow = 'hidden';
  };

  const closeFullscreen = () => {
    setFullscreenData(null);
    document.body.style.overflow = 'auto';
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 sm:p-6">
      <a href="/" className="absolute top-4 left-4 flex items-center text-blue-700 hover:text-blue-900 font-semibold z-10"><FaArrowLeft className="mr-2" /> Back to Home</a>
      <div className="max-w-6xl mx-auto mt-12 bg-white rounded-lg shadow-xl overflow-hidden">
        <div className="bg-gradient-to-r from-green-600 to-green-800 p-6 text-white text-center">
          <h1 className="text-3xl font-extrabold">Items You've Found</h1>
          <p className="mt-1 text-green-200">A list of all items you have reported as found.</p>
        </div>
        <div className="p-6">
          {loading && <div className="text-center font-medium py-8">Loading...</div>}
          {error && <div className="text-red-600 bg-red-100 p-3 rounded-md text-center my-4">{error}</div>}
          {!loading && !error && (
            foundItems.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {foundItems.map((item) => (
                  <div key={item._id} className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200 flex flex-col">
                    <ImageCarousel images={item.images} altText={item.objectName} onImageClick={openFullscreen} />
                    <div className="p-4 space-y-3 flex-grow flex flex-col">
                      <h3 className="text-xl font-bold text-gray-800 truncate">{item.objectName || "Unnamed Item"}</h3>
                      <div className="text-gray-700 text-sm space-y-2 flex-grow">
                        {item.brand && <p className="flex items-center gap-2"><Tag size={14} className="text-gray-500" /><strong>Brand:</strong> {item.brand}</p>}
                        {item.size && <p className="flex items-center gap-2"><Ruler size={14} className="text-gray-500" /><strong>Size:</strong> {item.size}</p>}
                        {item.colors && item.colors.length > 0 && <p className="flex items-center gap-2"><Droplet size={14} className="text-gray-500" /><strong>Colors:</strong> {item.colors.join(', ')}</p>}
                        {item.markings && <p className="flex items-center gap-2"><Fingerprint size={14} className="text-gray-500" /><strong>Markings:</strong> {item.markings}</p>}
                      </div>
                      <div className="pt-3 mt-auto border-t border-gray-200 text-sm space-y-1">
                          <p className="flex items-center gap-2"><CheckCircle size={14} className="text-green-500" /><strong>Status:</strong> {item.status}</p>
                          <p className="flex items-center gap-2"><FaMapMarkerAlt className="text-blue-500" /><strong>Location:</strong> {item.locationFound || "N/A"}</p>
                          <p className="flex items-center gap-2"><FaCalendarAlt className="text-purple-500" /><strong>Date:</strong> {new Date(item.dateFound).toLocaleDateString()}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-gray-600 py-8">
                <FaBoxOpen className="mx-auto text-6xl mb-4 text-gray-400" />
                <p className="text-xl font-semibold">No found items reported yet.</p>
              </div>
            )
          )}
        </div>
      </div>
      
      {fullscreenData && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 transition-opacity duration-300" onClick={closeFullscreen}>
          <div className="relative w-full h-full max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <ImageCarousel images={fullscreenData.images} startIndex={fullscreenData.startIndex} altText="Fullscreen view" isFullscreen={true} />
          </div>
          <button className="absolute top-4 right-4 text-white bg-black/50 rounded-full p-2" onClick={closeFullscreen}><X size={24} /></button>
        </div>
      )}
    </div>
  );
};
export default FoundItems;