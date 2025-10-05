// Updated src/services/api.js for production

import axios from 'axios';
import { Platform } from 'react-native';
import API_CONFIG from '../config/api';

class APIService {
  constructor() {
    const baseURL = API_CONFIG.getBaseURL();

    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
    console.log('🚀 API Client initialized with URL:', baseURL);
  }

  setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`📡 ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`✅ ${response.status} ${response.config.url}`);
        return response.data;
      },
      (error) => {
        console.error('❌ Response Error:', error.response?.status, error.response?.data);

        // Handle specific error cases
        if (error.response?.status === 401) {
          // Token expired or invalid - redirect to login
          // This would be handled by your auth context
        }

        return Promise.reject(error);
      }
    );
  }

  setAuthToken(token) {
    if (token) {
      this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete this.client.defaults.headers.common['Authorization'];
    }
  }

  // Test connection
  async testConnection() {
    try {
      const response = await this.client.get('/health');
      return { success: true, data: response };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || error.message 
      };
    }
  }

  // Authentication
  async signup(username, email, password) {
    return await this.client.post('/auth/signup', {
      username,
      email,
      password,
    });
  }

  async login(username, password) {
    return await this.client.post('/auth/login', {
      username,
      password,
    });
  }

  async getProfile() {
    return await this.client.get('/auth/profile');
  }

  // Messaging
  async sendMessage(recipientUsername, password, isViewOnce, timerSeconds, fileUri, fileType) {
    const formData = new FormData();

    formData.append('recipient_username', recipientUsername);
    formData.append('password', password);
    formData.append('is_view_once', isViewOnce.toString());

    if (timerSeconds) {
      formData.append('timer_seconds', timerSeconds.toString());
    }

    // Add file
    const filename = fileUri.split('/').pop();
    const match = /\.(\w+)$/.exec(filename);
    const fileExtension = match ? match[1] : 'jpg';

    formData.append('file', {
      uri: fileUri,
      type: fileType || `image/${fileExtension}`,
      name: filename,
    });

    return await this.client.post('/messages/send', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // Longer timeout for file upload
    });
  }

  async viewMessage(messageId, password, cameraImageUri) {
    const formData = new FormData();

    formData.append('password', password);

    // Add camera image for security check
    const filename = cameraImageUri.split('/').pop();
    formData.append('camera_image', {
      uri: cameraImageUri,
      type: 'image/jpeg',
      name: filename || 'security_check.jpg',
    });

    return await this.client.post(`/messages/${messageId}/view`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 30000,
    });
  }

  async getConversations() {
    return await this.client.get('/conversations');
  }

  async searchUsers(username) {
    return await this.client.get(`/users/search?username=${encodeURIComponent(username)}`);
  }
}

export const API = new APIService();
