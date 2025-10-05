import express from 'express';
import { getProfile, changePassword } from '../controllers/profilecontroller.js';

const profile_router = express.Router();

profile_router.get('/getprofile',  getProfile);
profile_router.post('/change-password', changePassword);

export default profile_router;
