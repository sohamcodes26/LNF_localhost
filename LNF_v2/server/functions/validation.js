import bcrypt from "bcryptjs";


export const validatepass = async (details, hashedpass) => {
  
    const len1 = details.email.length;
    const part = len1 << 5;
    const password = details.password + part.toString();

    const res = bcrypt.compare(password, hashedpass);

    return res;
}


export const validateotp = async (details, info) => {
    try {
        
        if (details.otp !== info.otp) { 
            return { message: 'Incorrect OTP', code: 401 };
        }

 
        const OTP_EXPIRY_TIME_MS = 10 * 60 * 1000; 

        
        const now = new Date();
        const otpCreationTime = new Date(info.createdAt); 

        
        const elapsed_ms = now.getTime() - otpCreationTime.getTime();

        
        if (elapsed_ms > OTP_EXPIRY_TIME_MS) {
            return { message: 'OTP Expired', code: 401 };
        }

        
        return { message: 'OTP Verified', code: 200 }; 
    } catch (err) {
        console.error(`Error in validateotp: ${err}`); 
        return { message: 'Internal Server Error', code: 500 };
    }
}
