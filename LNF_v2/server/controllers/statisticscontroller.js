
import { FoundItem, LostItem } from '../schema/objectqueryschema.js';
import Result from '../schema/resultschema.js';

export const getDashboardStatistics = async (req, res) => {
    try {
        // --- 1. KPI Calculations ---
        const [
            totalLostItems,
            resolvedItems,
            totalFoundItems,
            successfulTransfers
        ] = await Promise.all([
            LostItem.countDocuments(),
            LostItem.countDocuments({ status: 'resolved' }),
            FoundItem.countDocuments(),
            Result.countDocuments({ status: 'transfer_complete' })
        ]);

        const resolutionRate = totalLostItems > 0 ? (resolvedItems / totalLostItems) * 100 : 0;

        // --- 2. Match Funnel Analysis ---
        const matchFunnelData = await Result.aggregate([
            { $group: { _id: '$status', count: { $sum: 1 } } }
        ]);

        const matchFunnel = matchFunnelData.reduce((acc, item) => {
            acc[item._id] = item.count;
            return acc;
        }, { pending: 0, confirmed: 0, transfer_complete: 0, rejected: 0 });


        // --- 3. Assemble the final JSON response ---
        // itemStats and locationStats have been removed.
        res.status(200).json({
            kpis: {
                totalLostItems,
                resolvedItems,
                totalFoundItems,
                successfulTransfers,
                resolutionRate: resolutionRate.toFixed(2) + '%'
            },
            matchFunnel
        });

    } catch (err) {
        console.error('Error fetching dashboard statistics:', err);
        res.status(500).json({ message: 'Internal Server Error.' });
    }
};