import express from 'express'
import { otp_verification, resend_otp, sign_in, sign_up, signout, validate } from '../controllers/logincontroller.js';

const loginrouter = express.Router();

loginrouter.post('/signup', sign_up);

loginrouter.post('/signin', sign_in)

loginrouter.post('/otp_verification', otp_verification)

loginrouter.post('/resend-otp', resend_otp)

loginrouter.post('/validate', validate)

loginrouter.post('/sign-out', signout)

export default loginrouter;