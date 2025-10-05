import mongoose from 'mongoose';
const { Schema } = mongoose;


const chatRoomSchema = new Schema({
    
    user1: {
        type: Schema.Types.ObjectId,
        ref: 'userlogin',
        required: true
    },
    user2: {
        type: Schema.Types.ObjectId,
        ref: 'userlogin',
        required: true
    }
}, {
    timestamps: true
});

chatRoomSchema.index({ user1: 1, user2: 1 }, { unique: true });


const chatMessageSchema = new Schema({
    chatRoom: {
        type: Schema.Types.ObjectId,
        ref: 'ChatRoom',
        required: true,
        index: true
    },
    sender: {
        type: Schema.Types.ObjectId,
        ref: 'userlogin',
        required: true
    },
    receiver: {
        type: Schema.Types.ObjectId,
        ref: 'userlogin',
        required: true
    },
    message: {
        type: String,
        required: true,
        trim: true
    },
    read: {
        type: Boolean,
        default: false
    }
}, {
    timestamps: true
});

const ChatRoom = mongoose.model('ChatRoom', chatRoomSchema);
const ChatMessage = mongoose.model('ChatMessage', chatMessageSchema);

export {
    ChatRoom,
    ChatMessage
};