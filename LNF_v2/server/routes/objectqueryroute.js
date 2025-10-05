import express from 'express';
import { reportLostItem, reportFoundItem } from '../controllers/objectquerycontroller.js';
// Import both upload middleware and URL converter
import { uploadMultipleImages, convertPathsToUrls } from '../middlewares/uploadcareUpload.js'; 

const object_query_router = express.Router();

// Route for lost items, using the multi-image upload middleware then converting paths to URLs
object_query_router.post('/report-lost', uploadMultipleImages, convertPathsToUrls, reportLostItem);

// Route for found items, using the same multi-image upload middleware then converting paths to URLs
object_query_router.post('/report-found', uploadMultipleImages, convertPathsToUrls, reportFoundItem);

export default object_query_router;
