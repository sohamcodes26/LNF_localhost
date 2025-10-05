import Result from '../schema/resultschema.js'; 
import mongoose from 'mongoose';
import { LostItem, FoundItem } from '../schema/objectqueryschema.js'; 

export const getMyMatches = async (req, res) => {
    const userId = req.id;

    try {
        // --- SELECT a string of all the new fields to populate ---
        const fieldsToPopulate = 'objectName brand material size markings colors images dateLost dateFound';

        const matches = await Result.find({
            $or: [{ lostItemOwner: userId }, { foundItemHolder: userId }]
        })
        // --- UPDATED .populate() calls ---
        .populate('lostQuery', fieldsToPopulate)
        .populate('foundQuery', fieldsToPopulate)
        .populate('lostItemOwner', 'name email')
        .populate('foundItemHolder', 'name email')
        .sort({ createdAt: -1 });

        res.status(200).json({userId, matches});
    } catch (error) {
        console.error('Error fetching matches:', error);
        res.status(500).json({ message: 'Server error while fetching matches.' });
    }
};

// --- No changes needed for the functions below as they don't populate item details ---

export const rejectMatch = async (req, res) => {
    const { resultId } = req.params;
    const userId = req.id;

    if (!mongoose.Types.ObjectId.isValid(resultId)) {
        return res.status(400).json({ message: 'Invalid result ID format.' });
    }

    try {
        const match = await Result.findById(resultId);
        if (!match) {
            return res.status(404).json({ message: 'Match result not found.' });
        }

        const isUserInvolved = match.lostItemOwner.toString() === userId || match.foundItemHolder.toString() === userId;
        if (!isUserInvolved) {
            return res.status(403).json({ message: 'You are not authorized to update this match.' });
        }

        match.status = 'rejected';
        await match.save();
        res.status(200).json(match);

    } catch (error) {
        console.error('Error rejecting match:', error);
        res.status(500).json({ message: 'Server error while rejecting match.' });
    }
};

export const confirmMatch = async (req, res) => {
    const { resultId } = req.params;
    const userId = req.id;

    if (!mongoose.Types.ObjectId.isValid(resultId)) {
        return res.status(400).json({ message: 'Invalid result ID format.' });
    }

    try {
        const primaryMatch = await Result.findById(resultId);
        if (!primaryMatch) {
            return res.status(404).json({ message: 'Match result not found.' });
        }

        const isUserInvolved = primaryMatch.lostItemOwner.toString() === userId || primaryMatch.foundItemHolder.toString() === userId;
        if (!isUserInvolved) {
            return res.status(403).json({ message: 'You are not authorized to update this match.' });
        }

        primaryMatch.status = 'confirmed';
        
        await Result.updateMany(
            {
                _id: { $ne: primaryMatch._id },
                status: 'pending',
                $or: [
                    { lostQuery: primaryMatch.lostQuery },
                    { foundQuery: primaryMatch.foundQuery }
                ]
            },
            { $set: { status: 'rejected' } }
        );

        await primaryMatch.save();
        const populatedMatch = await Result.findById(primaryMatch._id).populate('lostQuery', 'objectName').populate('foundQuery', 'objectName');
        res.status(200).json(populatedMatch);

    } catch (error) {
        console.error('Error confirming match:', error);
        res.status(500).json({ message: 'Server error while confirming match.' });
    }
};

export const generateTransferCode = async (req, res) => {
    const { resultId } = req.params;
    const userId = req.id;

    try {
        const match = await Result.findById(resultId);
        if (!match) {
            return res.status(404).json({ message: 'Match not found.' });
        }
        if (match.status !== 'confirmed') {
            return res.status(400).json({ message: 'Match must be confirmed to start a transfer.' });
        }
        if (match.foundItemHolder.toString() !== userId) {
            return res.status(403).json({ message: 'Only the item finder can generate a transfer code.' });
        }

        const code = Math.floor(100000 + Math.random() * 900000).toString();
        match.transferCode = code;
        await match.save();

        res.status(200).json({ transferCode: code });

    } catch (error) {
        console.error('Error generating transfer code:', error);
        res.status(500).json({ message: 'Server error.' });
    }
};

export const verifyTransferCode = async (req, res) => {
    const { resultId } = req.params;
    const userId = req.id;
    const { code } = req.body;

    if (!code || code.length !== 6) {
        return res.status(400).json({ message: 'A 6-digit transfer code is required.' });
    }

    try {
        const match = await Result.findById(resultId);
        if (!match) {
            return res.status(404).json({ message: 'Match not found.' });
        }
        if (match.status !== 'confirmed') {
            return res.status(400).json({ message: 'Match must be confirmed to verify a transfer.' });
        }
        if (match.lostItemOwner.toString() !== userId) {
            return res.status(403).json({ message: 'Only the item owner can verify a transfer.' });
        }
        if (match.transferCode !== code) {
            return res.status(400).json({ message: 'Invalid or expired transfer code.' });
        }
        
        match.status = 'transfer_complete';
        match.transferCode = null;
 
        await Promise.all([
            LostItem.findByIdAndUpdate(match.lostQuery, { status: 'resolved' }),
            FoundItem.findByIdAndUpdate(match.foundQuery, { status: 'resolved' }),
            match.save()
        ]);
        
        res.status(200).json({ message: 'Transfer complete! The items are now marked as resolved.', match });

    } catch (error) {
        console.error('Error verifying transfer code:', error);
        res.status(500).json({ message: 'Server error.' });
    }
};
