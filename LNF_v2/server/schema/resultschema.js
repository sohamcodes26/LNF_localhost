import mongoose from 'mongoose';
const { Schema } = mongoose;


const resultSchema = new Schema({
    lostQuery: {
        type: Schema.Types.ObjectId,
        ref: 'LostItem', 
        required: true
    },
    foundQuery: {
        type: Schema.Types.ObjectId,
        ref: 'FoundItem', 
        required: true
    },
    lostItemOwner: {
        type: Schema.Types.ObjectId,
        ref: 'userlogin', 
        required: true
    },
    foundItemHolder: {
        type: Schema.Types.ObjectId,
        ref: 'userlogin', 
        required: true
    },
    status: {
        type: String,
        enum: ['pending', 'confirmed', 'transfer_complete', 'rejected'],
        default: 'pending'
    },
    transferCode: {
        type: String,
        default: null
    },
    matchConfidence: {
        type: Number,
        min: 0,
        max: 1
    }

}, {
    timestamps: true 
});
resultSchema.index({ lostQuery: 1, foundQuery: 1 }, { unique: true });

const Result = mongoose.model('Result', resultSchema);

export default Result;
