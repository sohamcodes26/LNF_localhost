import jwt from 'jsonwebtoken'

export const validate_token = (req, res, next) => {
    try {
        const token = req.cookies.token;

        if (!token) {
            return res.status(401).json({ message: 'No token provided' });
        }

        const decoded = jwt.verify(token, process.env.SECRET_KEY);

        console.log('Token Validated');

        req.id = decoded.user;

        next();
    } catch (err) {
        console.error('Error validating token:', err);
        res.status(403).clearCookie('token').json({ message: 'Invalid or Expired Token' });
    }
}
