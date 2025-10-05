import { LostItem, FoundItem } from '../schema/objectqueryschema.js';

export const getMyLostItems = async (req, res) => {
    try {
        const userId = req.id;
        // No changes needed here as it just finds by userId
        const lostItems = await LostItem.find({ userId: userId }).sort({ createdAt: -1 });
        res.status(200).json({ lostItems: lostItems });

    } catch (err) {
        console.error('Error fetching lost items:', err);
        res.status(500).json({ message: 'Internal Server Error while fetching lost items.' });
    }
};

export const getMyFoundItems = async (req, res) => {
    try {
        const userId = req.id;
        // No changes needed here as it just finds by userId
        const foundItems = await FoundItem.find({ userId: userId }).sort({ createdAt: -1 });
        res.status(200).json({ foundItems: foundItems });
        
    } catch (err) {
        console.error('Error fetching found items:', err);
        res.status(500).json({ message: 'Internal Server Error while fetching found items.' });
    }
};
