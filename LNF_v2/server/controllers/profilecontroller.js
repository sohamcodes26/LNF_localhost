import { userlogin } from '../schema/loginschema.js';
import { encrypt } from '../functions/encryption.js';
import { validatepass } from '../functions/validation.js';

export const getProfile = async (req, res) => {
    try {
        const userId = req.id;

        const user = await userlogin.findById(userId);

        if (!user) {
            return res.status(404).json({ message: 'User not found.' });
        }

        res.status(200).json({
            fullName: user.fullName,
            email: user.email
        });
    } catch (err) {
        console.error('Error fetching profile:', err);
        res.status(500).json({ message: 'Internal Server Error.' });
    }
};

export const changePassword = async (req, res) => {
    try {
        const userId = req.id;
        const { oldPassword, newPassword } = req.body;

        if (!oldPassword || !newPassword) {
            return res.status(400).json({ message: 'Old password and new password are required.' });
        }

        const user = await userlogin.findById(userId);

        if (!user) {
            return res.status(404).json({ message: 'User not found.' });
        }

        const detailsForValidation = {
            email: user.email, 
            password: oldPassword
        };

        const isValidOldPassword = await validatepass(detailsForValidation, user.user_password);

        if (!isValidOldPassword) {
            return res.status(401).json({ message: 'Incorrect old password.' });
        }

        const hashedNewPassword = await encrypt(user.email, newPassword);

        user.user_password = hashedNewPassword;
        await user.save();

        res.status(200).json({ message: 'Password changed successfully.' });
    } catch (err) {
        console.error('Error changing password:', err);
        res.status(500).json({ message: 'Internal Server Error.' });
    }
};
