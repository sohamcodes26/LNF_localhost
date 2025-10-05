import express from 'express';
import * as resultController from '../controllers/resultcontroller.js'; 

const router = express.Router();


router.get('/', resultController.getMyMatches);


router.patch('/:resultId/reject', resultController.rejectMatch);


router.patch('/:resultId/confirm', resultController.confirmMatch);


router.post('/:resultId/generate-code', resultController.generateTransferCode);


router.post('/:resultId/verify-code', resultController.verifyTransferCode);


export default router;
