import express from 'express';
import { getMyLostItems, getMyFoundItems } from '../controllers/myitemscontroller.js';
import { validate_token } from '../middlewares/validate_token.js';

const my_items_router = express.Router();

my_items_router.get('/my-lost-items', getMyLostItems);
my_items_router.get('/my-found-items', getMyFoundItems);

export default my_items_router;