import axios from 'axios';

// Local Flask API URL - Make sure the Flask app is running on port 5000
// Start the Flask server with: python app.py (from Ai_Pipeline directory)
const FLASK_API_URL = "http://localhost:5000";


export const processItemFeatures = async (itemData) => {
    try {
        console.log('Calling Flask API to process item features...');
        const response = await axios.post(`${FLASK_API_URL}/process`, itemData);
        console.log('Successfully received features from Flask API.');
          console.log(response.data);
        return response.data;
    } catch (error) {
        console.error('Error calling /process endpoint on Flask API:', error.response ? error.response.data : error.message);
        return null;
    }
};


export const findMatches = async (queryItem, searchList) => {
    try {
        console.log(`Calling Flask API to compare ${queryItem._id} against ${searchList.length} items...`);
        const payload = {
            queryItem,
            searchList,
        };
        const response = await axios.post(`${FLASK_API_URL}/compare`, payload);
        console.log('Successfully received comparison results from Flask API.');
        console.log(response.data);
        return response.data;
    } catch (error) {
        console.error('Error calling /compare endpoint on Flask API:', error.response ? error.response.data : error.message);
        return null;
    }
};
