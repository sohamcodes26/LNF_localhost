import mongoose from 'mongoose';

// --- Updated AI Feature Fields for DINOv2 ---
const aiFeatureFields = {
    canonicalLabel: {
        type: String,
        trim: true,
        index: true 
    },
    // Text embeddings
    brand_embedding: { type: [Number] },
    material_embedding: { type: [Number] },
    markings_embedding: { type: [Number] }, // <-- ADD THIS LINE
    
    // Field for DINOv2 image embeddings
    image_embeddings: {
        type: [[Number]] // An array of arrays (embeddings)
    }
};

// --- Lost Item Schema ---
const lostItemSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'userlogin', required: true },
    objectName: { type: String, required: true, trim: true },
    brand: { type: String, trim: true },
    material: { type: String, trim: true },
    size: { type: String, trim: true },
    markings: { type: String, trim: true },
    colors: { type: [String] },
    images: { type: [String], required: false },
    locationLost: { type: String, required: true, trim: true },
    dateLost: { type: Date, required: true },
    status: { type: String, enum: ['not_resolved', 'resolved'], default: 'not_resolved' },
    ...aiFeatureFields 
},
{
    timestamps: true,
    collection: 'lost_items'
});

// --- Found Item Schema ---
const foundItemSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'userlogin', required: true },
    objectName: { type: String, required: true, trim: true },
    brand: { type: String, trim: true },
    material: { type: String, trim: true },
    size: { type: String, trim: true },
    markings: { type: String, trim: true },
    colors: { type: [String] },
    images: { type: [String], required: true },
    locationFound: { type: String, required: true, trim: true },
    dateFound: { type: Date, required: true },
    status: { type: String, enum: ['not_resolved', 'resolved'], default: 'not_resolved' },
    ...aiFeatureFields 
},
{
    timestamps: true,
    collection: 'found_items'
});

export const LostItem = mongoose.model('LostItem', lostItemSchema);
export const FoundItem = mongoose.model('FoundItem', foundItemSchema);