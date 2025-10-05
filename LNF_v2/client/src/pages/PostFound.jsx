import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Upload, CheckCircle, ArrowLeft, Lightbulb, Loader2, X } from 'lucide-react';
import Button from '../components/ui/Button';
import Label from '../components/ui/Label';
import axios from 'axios';
import walletImage from '../assets/finalbottle.jpg';
import calculatorImage from '../assets/calculator.jpg';
import bottleImage from '../assets/ring.jpg';

// Data for form inputs
const synonymMap = [
    { name: "box", synonyms: ["cardboard", "carton", "rectangular box", "package"] },
  { name: "spectacles", synonyms: ["glasses", "eyewear", "goggles", "frames", "lenses"] },
  { name: "bottle", synonyms: ["waterbottle", "flask", "sipper", "thermos", "hydrator"] },
  { name: "charger", synonyms: ["adapter", "powerbrick", "chargingcable", "usbcharger", "wallcharger"] },
  { name: "laptop", synonyms: ["notebook", "macbook", "chromebook", "thinkpad", "ultrabook"] },
  { name: "earphones", synonyms: ["earbuds", "airpods", "headphones", "earpieces", "in-ear", "buds"] },
  { name: "mobile", synonyms: ["phone", "smartphone", "cellphone", "android", "iphone", "handset", "samsung"] },
  { name: "idcard", synonyms: ["id", "identitycard", "collegeid", "campusid", "badge", "accesscard"] },
  { name: "wallet", synonyms: ["purse", "moneybag", "cardholder", "leatherwallet", "cashholder"] },
  { name: "pen", synonyms: ["ballpen", "gelpen", "inkpen", "stylus", "markerpen"] },
  { name: "notebook", synonyms: ["diary", "journal", "register", "copy", "logbook"] },
  { name: "powerbank", synonyms: ["batterybank", "portablecharger", "externalbattery"] },
  { name: "usbdrive", synonyms: ["pendrive", "flashdrive", "thumbdrive", "memorystick", "stickdrive"] },
  { name: "mouse", synonyms: ["computer mouse", "wirelessmouse", "touchpad", "clicker"] },
  { name: "keyboard", synonyms: ["keypad", "typedevice", "wirelesskeyboard"] },
  { name: "watch", synonyms: ["wristwatch", "smartwatch", "band", "fitband", "timer"] },
  { name: "bag", synonyms: ["backpack", "bagpack", "rucksack", "satchel", "kitbag", "tote"] },
  { name: "calculator", synonyms: ["calc", "scientificcalc", "casio", "mathdevice"] },
  { name: "keys", synonyms: ["lockerkeys", "roomkeys", "keychain", "bikekeys", "housekeys", "keyring"] },
  { name: "umbrella", synonyms: ["rainshade", "parasol", "raincover"] },
  { name: "helmet", synonyms: ["bikehelmet", "safetyhelmet", "headgear"] },
  { name: "cap", synonyms: ["hat", "baseballcap", "headcover", "visor"] },
  { name: "container", synonyms: ["lunchbox", "dabba", "tiffin","basket", "lunchcase"] },
  { name: "shoes", synonyms: ["footwear", "sneakers", "trainers", "sportswear", "loafers"] },
  { name: "slippers", synonyms: ["flipflops", "sandals", "chappals", "slides"] },
  { name: "simcard", synonyms: ["sim", "networkcard", "telecomchip"] },
  { name: "otg", synonyms: ["usbconverter", "usbadapter", "otgcable", "typecconnector"] },
  { name: "remote", synonyms: ["controller", "tvremote", "clicker", "irremote"] },
  { name: "jacket", synonyms: ["coat", "hoodie", "sweater", "zipper", "blazer", "windcheater"] },
  { name: "textbook", synonyms: ["book", "coursebook", "referencebook", "module", "subjectbook"] },
  { name: "marker", synonyms: ["highlighter", "whiteboardpen", "sketchpen", "boardmarker"] },
  { name: "medicines", synonyms: ["tablets", "pills", "capsules", "strip", "drugs"] },
  { name: "earrings", synonyms: ["jewelry", "studs", "jhumkas", "earringset"] },
  { name: "belt", synonyms: ["waistbelt", "leatherbelt", "strap"] },
  { name: "coin", synonyms: ["change", "loosechange", "currencycoin", "moneycoin"] },
  { name: "tripod", synonyms: ["stand", "camerastand", "mobiletripod"] },
  { name: "camera", synonyms: ["dslr", "digicam", "cam", "photodevice"] },
  { name: "tablet", synonyms: ["ipad", "tab", "androidtab", "notetab"] },
  { name: "badge", synonyms: ["tag", "nametag", "idbadge"] },
  { name: "scarf", synonyms: ["dupatta", "stole", "shawl", "wrap"] },
  { name: "usb-cable", synonyms: ["datacable", "cord", "chargerwire", "syncwire"] },
  { name: "harddisk", synonyms: ["hdd", "externaldisk", "backupdrive"] },
  { name: "tripass", synonyms: ["buspass", "idpass", "commutepass"] },
  { name: "stationery", synonyms: ["pen", "pencil", "eraser", "sharpener", "scale", "ruler", "sketchpen"] },
  { name: "helmetlock", synonyms: ["bikelock", "lockchain"] },
  { name: "lens", synonyms: ["dslrlens", "zoomlens", "cameraaccessory"] },
  { name: "hairband", synonyms: ["rubberband", "scrunchie", "band"] },
  { name: "projector", synonyms: ["beamer", "presentationdevice"] },
  { name: "bikekey", synonyms: ["two-wheelerkey", "vehicleremote"] },
  { name: "watchstrap", synonyms: ["bandstrap", "watchbelt"] },
  { name: "perfume", synonyms: ["deodorant", "scent", "spray"] },
  { name: "lunchbag", synonyms: ["tiffinbag", "dabba-bag", "snackbag"] },
  { name: "gloves", synonyms: ["handgloves", "mittens", "ridinggloves"] },
  { name: "sunscreen", synonyms: ["sunblock", "uvlotion", "spfcream"] },
  { name: "folder", synonyms: ["file", "documentsleeve", "paperholder"] },
  { name: "projectfile", synonyms: ["report", "assignmentfile", "submissionfile"] }
];
const allSynonyms = synonymMap.flatMap(({ name, synonyms }) => [name, ...synonyms]);
const SIZES = [
  { value: "Small", label: "Small (fits in pocket)" },
  { value: "Medium", label: "Medium (needs one hand to hold)" },
  { value: "Large", label: "Large (needs two hands to hold)" }
];
const MATERIALS = ["Plastic", "Metal", "Leather", "Fabric", "Wood", "Glass", "Paper", "Rubber", "Other"];
const COLORS = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Black", "White", "Gray", "Brown", "Beige", "Turquoise", "Silver", "Gold"];

const PostFoundPage = () => {
  const [objectName, setObjectName] = useState('');
  const [brand, setBrand] = useState('');
  const [material, setMaterial] = useState('');
  const [size, setSize] = useState('');
  const [markings, setMarkings] = useState('');
  const [colors, setColors] = useState([]);
  const [images, setImages] = useState([]); 
  const [locationFound, setLocationFound] = useState('');
  const [dateFound, setDateFound] = useState('');
  
  const [submitted, setSubmitted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [progressText, setProgressText] = useState(''); // New state for progress text

  const resolveCanonicalName = (input) => {
    const lowerInput = input.toLowerCase();
    for (let entry of synonymMap) {
      if (entry.name === lowerInput || entry.synonyms.includes(lowerInput)) return entry.name;
    }
    return input;
  };
  
  const handleNameChange = (e) => {
    const value = e.target.value;
    setObjectName(value);
    if (value.length > 0) {
      const matches = allSynonyms.filter((syn) => syn.toLowerCase().startsWith(value.toLowerCase()));
      setSuggestions(matches.slice(0, 5));
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (value) => {
    setObjectName(resolveCanonicalName(value));
    setSuggestions([]);
  };

  const handleImageChange = (event) => {
    if (event.target.files) {
      const filesArray = Array.from(event.target.files);
      if (images.length + filesArray.length > 8) {
        setErrorMessage("You can upload a maximum of 8 images.");
        return;
      }
      setImages((prevImages) => [...prevImages, ...filesArray]);
      setErrorMessage('');
    }
  };

  const removeImage = (index) => {
    setImages((prevImages) => prevImages.filter((_, i) => i !== index));
  };
  
  const handleColorChange = (event) => {
    const { value, checked } = event.target;
    setColors(prev => checked ? [...prev, value] : prev.filter(c => c !== value));
  };

  const resetForm = () => {
    setObjectName('');
    setBrand('');
    setMaterial('');
    setSize('');
    setMarkings('');
    setColors([]);
    setImages([]);
    setLocationFound('');
    setDateFound('');
    setSubmitted(true);
  };

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    setErrorMessage('');

    if (colors.length === 0) {
      setErrorMessage('Please select at least one color.');
      return;
    }
    if (images.length < 3) {
      setErrorMessage('Please upload at least 3 images.');
      return;
    }

    setIsLoading(true);
    setProgressText('Uploading images...');
    const formData = new FormData();
    
    formData.append('objectName', objectName);
    formData.append('brand', brand);
    formData.append('material', material);
    formData.append('size', size);
    formData.append('markings', markings);
    colors.forEach(color => formData.append('colors', color));
    formData.append('locationFound', locationFound);
    formData.append('dateFound', dateFound);
    images.forEach(image => formData.append('images', image));

    try {
        const config = {
            headers: { 'Content-Type': 'multipart/form-data' },
            withCredentials: true,
            onUploadProgress: (progressEvent) => {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                if (percentCompleted < 100) {
                    setProgressText(`Uploading images... ${percentCompleted}%`);
                } else {
                    setProgressText('Processing details, please wait...');
                }
            },
        };
      await axios.post('http://localhost:8000/apis/lost-and-found/object-query/report-found', formData, config);
      resetForm();
    } catch (error) {
      console.error('Error uploading item:', error);
      setErrorMessage(error.response?.data?.message || 'Failed to post the item.');
      setSubmitted(false);
    } finally {
      setIsLoading(false);
      setProgressText('');
    }
  };

  return (
    <main className="min-h-screen w-full grid grid-cols-1 md:grid-cols-2">
      <div className="flex flex-col justify-center p-8">
        <div className="w-full max-w-md mx-auto">
          <Link to="/" className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-4 font-medium"><ArrowLeft size={18} />Back to Home</Link>
          <div className="bg-white w-full p-8 rounded-2xl shadow-lg border border-gray-200/80">
            <div className="text-left mb-6">
              <h1 className="text-3xl font-bold text-gray-800">Report a Found Item</h1>
              <p className="text-gray-500 mt-2">Provide detailed information to help us find the owner.</p>
            </div>
            <form onSubmit={handleFormSubmit} className="space-y-4">
              <div>
                <Label htmlFor="object-name">Object Name (e.g., Bottle, Wallet)</Label>
                <input id="object-name" type="text" value={objectName} onChange={handleNameChange} required className="w-full px-4 py-2 mt-2 text-gray-800 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400" />
                {suggestions.length > 0 && (
                  <ul className="absolute z-10 bg-white border border-gray-200 w-full mt-1 rounded-lg shadow-md max-h-40 overflow-y-auto">
                    {suggestions.map((s, i) => <li key={i} onClick={() => handleSuggestionClick(s)} className="px-4 py-2 cursor-pointer hover:bg-blue-50 text-sm text-gray-700">{s}</li>)}
                  </ul>
                )}
              </div>
              
              <div>
                <Label htmlFor="image-upload">Upload Images (3 required, max 8)</Label>
                <div className="mt-2 grid grid-cols-4 gap-2">
                  {images.map((image, index) => (
                    <div key={index} className="relative">
                      <img src={URL.createObjectURL(image)} alt={`preview ${index}`} className="h-20 w-full object-cover rounded-md" />
                      <button type="button" onClick={() => removeImage(index)} className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full p-0.5"><X size={12} /></button>
                    </div>
                  ))}
                  {images.length < 8 && (
                    <label htmlFor="image-upload" className="flex flex-col items-center justify-center w-full h-20 border-2 border-dashed border-gray-300 rounded-xl cursor-pointer bg-gray-50 hover:bg-gray-100 transition">
                      <Upload className="w-6 h-6 text-gray-500" />
                    </label>
                  )}
                </div>
                <input id="image-upload" type="file" accept="image/*" multiple className="hidden" onChange={handleImageChange} />
              </div>

              <div><Label htmlFor="brand">Brand (Optional)</Label><input id="brand" type="text" placeholder="e.g., Apple or type 'Don't Know'" value={brand} onChange={(e) => setBrand(e.target.value)} className="w-full px-4 py-2 mt-2 text-gray-800 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400" /></div>
              
              <div>
                <Label htmlFor="material">Material</Label>
                <select id="material" value={material} onChange={(e) => setMaterial(e.target.value)} required className="w-full px-4 py-2 mt-2 text-gray-800 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white">
                  <option value="">Select Material</option>
                  {MATERIALS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>

              <div>
                <Label htmlFor="size">Size</Label>
                <select id="size" value={size} onChange={(e) => setSize(e.target.value)} required className="w-full px-4 py-2 mt-2 text-gray-800 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white">
                  <option value="">Select Size</option>
                  {SIZES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                </select>
              </div>
              
              <div><Label htmlFor="markings">Unique Markings (Optional)</Label><textarea id="markings" rows={2} placeholder="e.g., 'S' sticker on back, scratch on corner" value={markings} onChange={(e) => setMarkings(e.target.value)} className="w-full px-4 py-2 mt-2 text-gray-800 border border-gray-300 rounded-lg" /></div>
              
              <div>
                <Label>Colors</Label>
                <div className="mt-2 grid grid-cols-3 gap-2 text-sm">
                  {COLORS.map(c => <label key={c} className="flex items-center gap-2"><input type="checkbox" value={c} checked={colors.includes(c)} onChange={handleColorChange} className="rounded" />{c}</label>)}
                </div>
              </div>

              <div>
                <Label htmlFor="location-found">Location Found</Label>
                <select id="location-found" value={locationFound} onChange={(e) => setLocationFound(e.target.value)} className="w-full px-4 py-2 mt-2 text-gray-800 border border-gray-300 rounded-lg" required>
                  <option value="" disabled>Select a location</option>
                  {["Campus", "Grounds", "A Building", "B Building", "C Building", "D Building", "E Building"].map(loc => <option key={loc} value={loc}>{loc}</option>)}
                </select>
              </div>
              <div><Label htmlFor="date-found">Date Found</Label><input id="date-found" type="date" value={dateFound} onChange={(e) => setDateFound(e.target.value)} className="w-full px-4 py-2 mt-2 text-gray-800 border border-gray-300 rounded-lg" required /></div>

              {errorMessage && <p className="text-red-600 text-sm text-center">{errorMessage}</p>}
              {submitted && <p className="text-green-600 font-semibold text-center">✅ Found item posted successfully!</p>}
              
              <Button type="submit" variant="cta" disabled={isLoading}>
                {isLoading ? <><Loader2 className="w-5 h-5 mr-2 animate-spin" />Posting...</> : <><CheckCircle className="w-5 h-5" />Post Found Item</>}
              </Button>

              {isLoading && (
                <div className="text-center mt-2">
                    <p className="text-sm text-gray-600 animate-pulse">{progressText}</p>
                    <p className="text-xs text-gray-500 mt-1">This may take a moment. You can safely navigate away.</p>
                </div>
              )}
            </form>
          </div>
        </div>
      </div>
      <div className="hidden md:flex flex-col justify-center bg-gradient-to-br from-gray-50 via-blue-50 to-gray-100 p-12 lg:p-20">
        <div className="max-w-md">
            <div className="flex items-center gap-3">
              <div className="bg-blue-100 p-3 rounded-full">
                <Lightbulb className="w-6 h-6 text-blue-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-800">Tips for a Good Report</h2>
            </div>
            <p className="mt-4 text-gray-600">Help the owner identify their item by providing clear and accurate information.</p>
            <ul className="mt-6 space-y-4">
              <li className="flex gap-4">
                <strong className="text-blue-600 font-bold">1.</strong>
                <span className="text-gray-700"><strong>Take Clear Photos:</strong> Show the item from a few different angles, including any identifying marks.</span>
              </li>
              <li className="flex gap-4">
                <strong className="text-blue-600 font-bold">2.</strong>
                <span className="text-gray-700"><strong>Be Specific:</strong> Describe any unique features like brand names, scratches, or stickers. Avoid showing sensitive info.</span>
              </li>
              <li className="flex gap-4">
                <strong className="text-blue-600 font-bold">3.</strong>
                <span className="text-gray-700"><strong>Note the Location:</strong> Being precise about where you found the item helps confirm the true owner.</span>
              </li>
            </ul>
            <div className="mt-8 grid grid-cols-3 gap-4">
                <img src={walletImage} alt="Sample Wallet" className="rounded-xl object-cover h-24 w-full" />
                <img src={calculatorImage} alt="Sample Calculator" className="rounded-xl object-cover h-24 w-full" />
                <img src={bottleImage} alt="Sample Bottle" className="rounded-xl object-cover h-24 w-full" />
            </div>
          </div>
      </div>
    </main>
  );
};
export default PostFoundPage;