import jwt from 'jsonwebtoken'

export const generate_token = (result) => {

    const payload = {
        user: result.id,
        email: result.email
    };

    const token = jwt.sign(payload,process.env.SECRET_KEY,{expiresIn: '24h'})

    return token;

}