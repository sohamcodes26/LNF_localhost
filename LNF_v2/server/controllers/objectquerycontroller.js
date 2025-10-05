import { LostItem, FoundItem } from '../schema/objectqueryschema.js';
import Result from '../schema/resultschema.js';
import { processItemFeatures, findMatches } from '../services/ai_service.js';

export const reportLostItem = async (req, res) => {
    try {
        const { objectName, brand, material, size, markings, colors, locationLost, dateLost } = req.body;
        const userId = req.id;

        if (!objectName || !locationLost || !dateLost) {
            return res.status(400).json({ message: 'Missing required fields for lost item.' });
        }

        const imageUrls = req.files ? req.files.map(file => file.path) : [];

        const featuresPayload = { objectName, brand, material, size, markings, colors, images: imageUrls };
        const features = await processItemFeatures(featuresPayload);
        if (!features) {
            return res.status(500).json({ message: 'Could not process item features via AI service.' });
        }
        
        let newLostItem = new LostItem({
            userId, objectName, brand, material, size, markings, colors,
            images: imageUrls,
            locationLost,
            dateLost: new Date(dateLost),
            canonicalLabel: features.canonicalLabel,
            brand_embedding: features.brand_embedding,
            material_embedding: features.material_embedding,
            markings_embedding: features.markings_embedding, // <-- ADD THIS LINE
            image_embeddings: features.image_embeddings
        });
        
        await newLostItem.save();
        console.log(`[Node.js] Saved new lost item with ID: ${newLostItem._id}`);

        const itemsToSearch = await FoundItem.find({ status: 'not_resolved' }).lean();
        
        const oneWeekInMillis = 7 * 24 * 60 * 60 * 1000;
        const queryDate = newLostItem.dateLost;
        const queryLocation = newLostItem.locationLost;
        const queryLabel = newLostItem.canonicalLabel;

        const filteredItemsToSearch = itemsToSearch.filter(item => {
            if (item.canonicalLabel !== queryLabel) return false;
            const itemDate = new Date(item.dateFound);
            if (Math.abs(queryDate.getTime() - itemDate.getTime()) > oneWeekInMillis) return false;
            const itemLocation = item.locationFound;
            if (queryLocation !== 'Campus' && ![queryLocation, 'Campus'].includes(itemLocation)) return false;
            return true;
        });
        console.log(`[Node.js] Pre-filtered candidates from ${itemsToSearch.length} to ${filteredItemsToSearch.length} before sending to AI.`);

        if (filteredItemsToSearch.length > 0) {
            const matchResult = await findMatches(newLostItem.toObject(), filteredItemsToSearch);
            if (matchResult && matchResult.matches.length > 0) {
                const resultsToSave = matchResult.matches
                    .filter(match => newLostItem.userId.toString() !== filteredItemsToSearch.find(item => item._id.toString() === match._id).userId.toString())
                    .map(match => {
                        const foundItem = filteredItemsToSearch.find(item => item._id.toString() === match._id);
                        return {
                            lostQuery: newLostItem._id,
                            foundQuery: foundItem._id,
                            lostItemOwner: newLostItem.userId,
                            foundItemHolder: foundItem.userId,
                            matchConfidence: match.score
                        };
                    });
                if (resultsToSave.length > 0) {
                    await Result.insertMany(resultsToSave, { ordered: false }).catch(err => { if (err.code !== 11000) console.error("Error saving match results:", err); });
                }
            }
        }
        
        res.status(201).json({ message: 'Lost item reported successfully and search initiated.' });

    } catch (err) {
        console.error('Error reporting lost item:', err);
        res.status(500).json({ message: 'Internal Server Error.' });
    }
};

export const reportFoundItem = async (req, res) => {
    try {
        const { objectName, brand, material, size, markings, colors, locationFound, dateFound } = req.body;
        const userId = req.id;

        if (!objectName || !locationFound || !dateFound) {
            return res.status(400).json({ message: 'Missing required fields.' });
        }
        if (!req.files || req.files.length < 3) {
            return res.status(400).json({ message: 'A minimum of 3 images are required for found items.' });
        }

        const imageUrls = req.files.map(file => file.path);

        const featuresPayload = { objectName, brand, material, size, markings, colors, images: imageUrls };
        const features = await processItemFeatures(featuresPayload);
        if (!features) {
            return res.status(500).json({ message: 'Could not process item features via AI service.' });
        }

        let newFoundItem = new FoundItem({
            userId, objectName, brand, material, size, markings, colors,
            images: imageUrls,
            locationFound,
            dateFound: new Date(dateFound),
            canonicalLabel: features.canonicalLabel,
            brand_embedding: features.brand_embedding,
            material_embedding: features.material_embedding,
            markings_embedding: features.markings_embedding, // <-- ADD THIS LINE
            image_embeddings: features.image_embeddings
        });

        await newFoundItem.save();
        console.log(`[Node.js] Saved new found item with ID: ${newFoundItem._id}`);

        const itemsToSearch = await LostItem.find({ status: 'not_resolved' }).lean();

        const oneWeekInMillis = 7 * 24 * 60 * 60 * 1000;
        const queryDate = newFoundItem.dateFound;
        const queryLocation = newFoundItem.locationFound;
        const queryLabel = newFoundItem.canonicalLabel;

        const filteredItemsToSearch = itemsToSearch.filter(item => {
            if (item.canonicalLabel !== queryLabel) return false;
            const itemDate = new Date(item.dateLost);
            if (Math.abs(queryDate.getTime() - itemDate.getTime()) > oneWeekInMillis) return false;
            const itemLocation = item.locationLost;
            if (queryLocation !== 'Campus' && ![queryLocation, 'Campus'].includes(itemLocation)) return false;
            return true;
        });
        console.log(`[Node.js] Pre-filtered candidates from ${itemsToSearch.length} to ${filteredItemsToSearch.length} before sending to AI.`);
        
        if (filteredItemsToSearch.length > 0) {
            const matchResult = await findMatches(newFoundItem.toObject(), filteredItemsToSearch);
            if (matchResult && matchResult.matches.length > 0) {
                const resultsTo_save = matchResult.matches
                    .filter(match => newFoundItem.userId.toString() !== filteredItemsToSearch.find(item => item._id.toString() === match._id).userId.toString())
                    .map(match => {
                        const lostItem = filteredItemsToSearch.find(item => item._id.toString() === match._id);
                        return {
                            lostQuery: lostItem._id,
                            foundQuery: newFoundItem._id,
                            lostItemOwner: lostItem.userId,
                            foundItemHolder: newFoundItem.userId,
                            matchConfidence: match.score
                        };
                    });
                if (resultsTo_save.length > 0) {
                    await Result.insertMany(resultsTo_save, { ordered: false }).catch(err => { if (err.code !== 11000) console.error("Error saving match results:", err); });
                }
            }
        }

        res.status(201).json({ message: 'Found item reported successfully.' });

    } catch (err) {
        console.error('Error reporting found item:', err);
        res.status(500).json({ message: 'Internal Server Error.' });
    }
};