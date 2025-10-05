// Mobile App Configuration for Production API
// src/config/api.js

const API_CONFIG = {
  // Production API URL (replace with your deployed backend URL)
  PRODUCTION_URL: 'https://your-deployed-api.railway.app', // Replace this!

  // Development URLs
  DEV_ANDROID_URL: 'http://10.0.2.2:8000',  // Android emulator
  DEV_IOS_URL: 'http://localhost:8000',      // iOS simulator
  DEV_DEVICE_URL: 'http://192.168.1.100:8000', // Your local IP

  // Automatically select URL based on environment
  getBaseURL: () => {
    if (__DEV__) {
      // Development mode
      if (Platform.OS === 'android') {
        return API_CONFIG.DEV_ANDROID_URL;
      } else {
        return API_CONFIG.DEV_IOS_URL;
      }
    } else {
      // Production mode
      return API_CONFIG.PRODUCTION_URL;
    }
  }
};

export default API_CONFIG;

// Usage in your api.js file:
// import API_CONFIG from '../config/api';
// 
// const baseURL = API_CONFIG.getBaseURL();
