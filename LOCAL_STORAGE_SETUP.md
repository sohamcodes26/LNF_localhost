# 🖼️ Local Image Storage Setup Guide

## ✅ Changes Made

### 1. **Modified Upload Middleware** (`uploadcareUpload.js`)
   - ❌ Removed: Uploadcare cloud storage
   - ✅ Added: Local disk storage using `multer.diskStorage`
   - 📁 Images saved to: `Images/` folder in workspace root
   - 🔐 Added file validation (images only, 10MB limit)
   - 🎯 Unique filenames: `originalname-timestamp-random.ext`

### 2. **Updated Server Configuration** (`server.js`)
   - ✅ Added static file serving for `/images` route
   - 🌐 Images accessible at: `http://localhost:9000/images/<filename>`

### 3. **Enhanced Route Middleware** (`objectqueryroute.js`)
   - ✅ Added `convertPathsToUrls` middleware
   - 🔄 Converts absolute file paths to HTTP URLs automatically

---

## 📂 File Structure

```
LNF_v2_localhost/
├── Images/                          ← Images stored here
│   ├── image1-1234567890-123.jpg
│   ├── image2-1234567891-456.png
│   └── ...
├── Ai_Pipeline/
│   └── app.py                       ← Flask AI service (port 5000)
└── LNF_v2/
    ├── server/
    │   ├── server.js                ← Modified: serves /images route
    │   ├── middlewares/
    │   │   └── uploadcareUpload.js  ← Modified: local storage
    │   └── routes/
    │       └── objectqueryroute.js  ← Modified: URL conversion
    └── client/
```

---

## 🔄 How It Works

### **Image Upload Flow:**

1. **Frontend** → Sends images via FormData to backend
2. **Multer Middleware** → Saves images to `Images/` folder
3. **convertPathsToUrls** → Converts file paths to URLs
   - Example: `/path/to/Images/photo.jpg` → `http://localhost:9000/images/photo.jpg`
4. **Controller** → Stores URLs in MongoDB
5. **AI Service** → Fetches images from URLs for processing
6. **Frontend** → Displays images using stored URLs

---

## 🚀 Usage

### **Start Services:**

1. **Activate Virtual Environment & Start AI Pipeline:**
   ```powershell
   cd "C:\Users\soham\OneDrive\Desktop\Lost and Found All versions\LNF_v2_localhost\Ai_Pipeline"
   .\venv\Scripts\Activate.ps1
   python app.py
   ```
   ✅ Should see: `Server will run at: http://localhost:5000`

2. **Start Backend Server:**
   ```powershell
   cd "C:\Users\soham\OneDrive\Desktop\Lost and Found All versions\LNF_v2_localhost\LNF_v2\server"
   npm start
   ```
   ✅ Should see: 
   - `Server is running on port 9000`
   - `📁 Serving static images from: <path>/Images`

3. **Start Frontend:**
   ```powershell
   cd "C:\Users\soham\OneDrive\Desktop\Lost and Found All versions\LNF_v2_localhost\LNF_v2\client"
   npm run dev
   ```

---

## 🔗 API Endpoints

### **Backend (Node.js) - Port 9000:**
- Health Check: `http://localhost:9000/`
- Report Lost: `POST http://localhost:9000/apis/lost-and-found/object-query/report-lost`
- Report Found: `POST http://localhost:9000/apis/lost-and-found/object-query/report-found`
- Images: `http://localhost:9000/images/<filename>`

### **AI Service (Flask) - Port 5000:**
- Health Check: `http://localhost:5000/`
- Process Features: `POST http://localhost:5000/process`
- Compare Items: `POST http://localhost:5000/compare`

---

## 📸 Image URL Format

**Stored in Database:**
```
http://localhost:9000/images/calculator-1728123456789-123456789.jpg
```

**Accessible via:**
- Browser: `http://localhost:9000/images/calculator-1728123456789-123456789.jpg`
- AI Service: Fetches from this URL for processing
- Frontend: Displays using this URL

---

## ⚠️ Important Notes

1. **Images Folder Auto-Creation:**
   - The `Images/` folder is created automatically on first upload if it doesn't exist

2. **File Size Limit:**
   - Maximum: 10MB per image
   - Can be adjusted in `uploadcareUpload.js` → `limits.fileSize`

3. **Supported Formats:**
   - All image types: JPG, PNG, GIF, WebP, etc.
   - Non-image files are rejected

4. **No Database Changes:**
   - All existing logic remains the same
   - Only the source of image URLs changed (cloud → local)

5. **Backend Must Be Running:**
   - Images won't load if Node.js server is down
   - Ensure port 9000 is not blocked by firewall

---

## 🔧 Troubleshooting

### **Images not loading:**
```powershell
# Check if Images folder exists
dir "C:\Users\soham\OneDrive\Desktop\Lost and Found All versions\LNF_v2_localhost\Images"

# Check if server is serving static files
# Look for: "📁 Serving static images from: ..." in console
```

### **Upload fails:**
- Check file size (must be < 10MB)
- Ensure file is an image
- Check folder permissions

### **AI Service can't fetch images:**
- Ensure backend server (port 9000) is running
- Check firewall settings
- Verify image URL format in database

---

## ✅ Benefits of Local Storage

1. **No External Dependencies:** No Uploadcare account needed
2. **Faster:** Local disk I/O is faster than cloud uploads
3. **Cost-Free:** No usage limits or charges
4. **Full Control:** Complete ownership of image data
5. **Privacy:** Images stay on your machine
6. **Debugging:** Easy to inspect stored images directly

---

## 🎉 Setup Complete!

Your Lost & Found application now stores all images locally in the `Images/` folder and serves them via HTTP URLs. The AI pipeline can access these images for processing, and your frontend can display them seamlessly!
