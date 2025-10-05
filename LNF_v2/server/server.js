import dotenv from 'dotenv'; 
dotenv.config(); 

import express from "express";
import cors from 'cors';
import cookieParser from "cookie-parser";
import http from 'http';
import { Server } from 'socket.io';
import path from 'path';
import { fileURLToPath } from 'url';

import loginrouter from "./routes/loginroute.js";
import object_query_router from "./routes/objectqueryroute.js";
import profile_router from "./routes/profileroute.js";
import my_items_router from "./routes/myitemsroute.js";
import chat_router from "./routes/chatroute.js";
import result_router from "./routes/resultroute.js";
import statistics_router from "./routes/statisticsroute.js";

import { connectdb } from "./config/database.js";
import { validate_token } from "./middlewares/validate_token.js"; 

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = http.createServer(app);
const port = process.env.PORT || 9000;

// --- CORS SETTINGS: CHANGE THIS BLOCK ---
const allowedOrigins = [
    'http://localhost:5173', 
    'https://lnf-v2.vercel.app'
];

const io = new Server(server, {
    cors: {
        origin: allowedOrigins,
        methods: ['GET', 'POST'],
        allowedHeaders: ['Content-Type', 'Authorization'],
        credentials: true,
        optionsSuccessStatus: 200
    }
});

app.set('socketio', io);

app.use(cors({
    origin: allowedOrigins,
    methods: ['GET', 'POST', 'OPTIONS', 'PATCH', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
}));
// --- END CORS SETTINGS ---

app.use(cookieParser());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Serve the Images folder as static files
// Images will be accessible at http://localhost:9000/images/<filename>
const imagesPath = path.join(__dirname, '../../Images');
app.use('/images', express.static(imagesPath));
console.log('📁 Serving static images from:', imagesPath);

connectdb();

app.use('/apis/lost-and-found/auth', loginrouter);
app.use('/apis/lost-and-found/object-query', validate_token, object_query_router);
app.use('/apis/lost-and-found/my-items', validate_token, my_items_router);
app.use('/apis/lost-and-found/MyAccount', validate_token, profile_router);
app.use('/apis/lost-and-found/chat', validate_token, chat_router);
app.use('/apis/lost-and-found/results', validate_token, result_router);
app.use('/apis/lost-and-found/statistics', statistics_router);


io.on('connection', (socket) => {
    console.log('A user connected:', socket.id);

    socket.on('joinRoom', (roomId) => {
        socket.join(roomId);
        console.log(`User ${socket.id} joined room ${roomId}`);
    });

    socket.on('disconnect', () => {
        console.log('User disconnected:', socket.id);
    });
});

app.get('/', (req, res) => {
  res.send('Lost and Found Backend is running ✅');
});

server.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});