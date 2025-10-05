import { ChatRoom, ChatMessage } from '../schema/chatschema.js';
import mongoose from 'mongoose';

// In your chat controller file

export const getOrCreateChatRoom = async (req, res) => {
    const currentUserId = req.id;
    const { otherUserId } = req.body;

    if (!otherUserId || currentUserId === otherUserId) {
        return res.status(400).json({ message: 'Valid otherUserId is required.' });
    }

    // Sort the user IDs to ensure consistency
    // The smaller ID will always be user1, the larger will be user2
    const user1 = currentUserId < otherUserId ? currentUserId : otherUserId;
    const user2 = currentUserId > otherUserId ? currentUserId : otherUserId;

    try {
        // Find a room where user1 and user2 match the sorted IDs
        let room = await ChatRoom.findOne({ user1, user2 });

        // If the room doesn't exist, create it
        if (!room) {
            room = new ChatRoom({ user1, user2 });
            await room.save();
            return res.status(201).json({ roomId: room._id });
        }

        // If the room already exists, return its ID
        return res.status(200).json({ roomId: room._id });

    } catch (error) {
        // This will now correctly catch race conditions thanks to the compound index
        if (error.code === 11000) {
           // In case of a race condition, re-fetch the room to be safe
           const existingRoom = await ChatRoom.findOne({ user1, user2 });
           return res.status(200).json({ roomId: existingRoom._id });
        }
        console.error('Error in getOrCreateChatRoom:', error);
        res.status(500).json({ message: 'Server error while getting or creating chat room.' });
    }
};

// In your chat controller file

export const sendMessage = async (req, res) => {
    const senderId = req.id;
    const { roomId, message } = req.body;

    if (!roomId || !message) {
        return res.status(400).json({ message: 'roomId and message are required.' });
    }

    try {
        const room = await ChatRoom.findById(roomId);
        if (!room) {
            return res.status(404).json({ message: 'Chat room not found.' });
        }

        // **CORRECTED LOGIC**: Check if senderId is either user1 or user2
        if (room.user1.toString() !== senderId && room.user2.toString() !== senderId) {
            return res.status(403).json({ message: 'Sender is not a member of this chat room.' });
        }

        // **CORRECTED LOGIC**: Find the receiverId
        const receiverId = senderId === room.user1.toString() ? room.user2 : room.user1;

        const chatMessage = new ChatMessage({
            chatRoom: roomId,
            sender: senderId,
            receiver: receiverId,
            message: message
        });
        await chatMessage.save();
        
        const populatedMessage = await chatMessage.populate('sender', 'email name');

        const io = req.app.get('socketio');
        io.to(roomId.toString()).emit('newMessage', populatedMessage); // Use roomId.toString() for socket.io rooms

        res.status(201).json(populatedMessage);

    } catch (error) {
        console.error('Error in sendMessage:', error);
        res.status(500).json({ message: 'Server error while sending message.' });
    }
};

// In your chat controller file

export const getChatHistory = async (req, res) => {
    const userId = req.id;
    const { roomId } = req.params;

    try {
        if (!mongoose.Types.ObjectId.isValid(roomId)) {
            return res.status(400).json({ message: 'Invalid roomId format.' });
        }

        const room = await ChatRoom.findById(roomId);
        if (!room) {
            return res.status(404).json({ message: 'Chat room not found.' });
        }
        
        // **CORRECTED LOGIC**: Check if userId is either user1 or user2
        if (room.user1.toString() !== userId && room.user2.toString() !== userId) {
            return res.status(403).json({ message: 'You are not a member of this chat room.' });
        }

        const messages = await ChatMessage.find({ chatRoom: roomId })
            .sort({ createdAt: 1 })
            .populate('sender', 'email name'); // Populating with name and email

        // Mark messages as read (your existing logic is fine)
        ChatMessage.updateMany(
            { chatRoom: roomId, receiver: userId, read: false },
            { $set: { read: true } }
        ).catch(err => console.error("Failed to mark messages as read:", err));

        res.status(200).json(messages);

    } catch (error) {
        console.error('Error in getChatHistory:', error);
        res.status(500).json({ message: 'Server error while fetching chat history.' });
    }
};