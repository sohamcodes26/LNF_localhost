import multer from 'multer';
import { Buffer } from 'buffer';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '../.env') });

// Define the Images folder path - going up to workspace root
const IMAGES_FOLDER = path.join(__dirname, '../../../Images');

// Ensure the Images folder exists
if (!fs.existsSync(IMAGES_FOLDER)) {
  fs.mkdirSync(IMAGES_FOLDER, { recursive: true });
  console.log('📁 Created Images folder at:', IMAGES_FOLDER);
}

// Configure multer to save files locally
const localStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, IMAGES_FOLDER);
  },
  filename: (req, file, cb) => {
    // Create unique filename with timestamp
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    const nameWithoutExt = path.basename(file.originalname, ext);
    const filename = `${nameWithoutExt}-${uniqueSuffix}${ext}`;
    cb(null, filename);
  }
});

const upload = multer({ 
  storage: localStorage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Accept images only
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// Correctly configured to handle an array of up to 8 files from the 'images' field
export const uploadMultipleImages = upload.array('images', 8);

// Middleware to convert file paths to URLs
export const convertPathsToUrls = (req, res, next) => {
  if (req.files && req.files.length > 0) {
    // Convert absolute file paths to URLs
    req.files = req.files.map(file => ({
      ...file,
      path: `http://localhost:${process.env.PORT || 9000}/images/${file.filename}`
    }));
    console.log('✅ Converted file paths to URLs:', req.files.map(f => f.path));
  }
  next();
};
