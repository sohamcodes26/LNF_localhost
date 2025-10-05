import bcrypt from "bcryptjs";


export const encrypt = async ( email, password) => {

    const len1 = email.length;
    const part = len1 << 5;

    password = password + part.toString();

    const first = await bcrypt.hash(password, 10);

    return first;
}
