// This is a temporary testing version.
// It uses a fixed OTP of "111111" and does not send an email.

import nodemailer from 'nodemailer';

// The random OTP generator functions have been removed as they are no longer needed.

export const sendmail = async (rec_mail) => {
    
    // --- DEVELOPMENT MODE ---
    // Using a fixed OTP instead of a randomly generated one.
    const otp = '111111';

    // We will log the OTP to the console so you can see it in your Render logs.
    console.log('--- 📧 OTP FOR DEVELOPMENT 📧 ---');
    console.log(`Email sending is disabled. Using fixed OTP for ${rec_mail}: ${otp}`);
    console.log('---------------------------------');

    // Return the fixed OTP so your application can continue.
    return otp;
};