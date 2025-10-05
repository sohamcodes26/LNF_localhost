import express from 'express';
import { getChatHistory, getOrCreateChatRoom, sendMessage} from '../controllers/chatcontroller.js';

const chat_router = express.Router();

chat_router.post('/room', getOrCreateChatRoom);

chat_router.post('/message', sendMessage);

chat_router.get('/history/:roomId', getChatHistory);

export default chat_router;
