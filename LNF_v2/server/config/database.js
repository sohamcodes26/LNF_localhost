import mongoose from "mongoose";
import dotenv from "dotenv";

dotenv.config();

export const connectdb = async () => {
    try {
        
        await mongoose.connect(process.env.MONGODB_URL);
        console.log('MongoDB Atlas connected successfully');
    }
    catch (err) {
        console.log(`Error connecting to MongoDB: ${err}`);
        process.exit(1); 
    }
}