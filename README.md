# 🔒 SecureVault - Production Ready Backend

**Privacy-first messaging API with advanced camera detection**

## 🚀 Quick Deploy

### 📱 **For Mobile Testing - Use Railway (Recommended)**

1. **Fork this repository to your GitHub account**

2. **Deploy to Railway (Free tier available):**
   ```bash
   # Install Railway CLI
   curl -fsSL https://railway.app/install.sh | sh

   # Login and deploy
   railway login
   railway new
   railway add
   railway deploy
   ```

3. **Set Environment Variables in Railway Dashboard:**
   - `MONGODB_URL`: Your MongoDB Atlas connection string (already configured)
   - `JWT_SECRET`: Random secret key (change from default!)
   - `DEBUG`: `False` for production

4. **Get your API URL:** `https://your-app-name.railway.app`

5. **Update mobile app configuration** with your Railway URL

---

## 📊 **API Endpoints**

### **Authentication**
- `POST /auth/signup` - Create new user account
- `POST /auth/login` - User login
- `GET /auth/profile` - Get user profile (requires auth)

### **Messaging**  
- `POST /messages/send` - Send encrypted message
- `POST /messages/{message_id}/view` - View message with camera detection
- `GET /conversations` - Get user conversations
- `GET /users/search` - Search users by username

### **System**
- `GET /health` - API health check
- `GET /` - API status

---

## 🔧 **Environment Variables**

```env
# Required
MONGODB_URL=mongodb+srv://ssawana:Ganesh%402024@cluster0.a2cfv2o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
DATABASE_NAME=SECURE_CHAT

# Security (CHANGE THESE!)
JWT_SECRET=your-super-secret-jwt-key-here
MASTER_KEY=your-encryption-master-key-here

# Optional
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

---

## 🔒 **Features**

### **🛡️ Advanced Security**
- **End-to-End Encryption**: AES-256 with PBKDF2
- **Camera Detection**: AI-powered threat detection
- **View-Once Messages**: Self-destructing content
- **Timer Messages**: Auto-delete after specified time
- **JWT Authentication**: Secure token-based auth

### **🤖 AI Detection Methods**
1. **Brightness Analysis** - Detects camera flash
2. **Edge Detection** - Finds rectangular objects (phones/cameras)  
3. **Face Detection** - Identifies multiple people
4. **Variance Analysis** - Detects motion/activity

---

## 📱 **Mobile App Integration**

### **Update your mobile app's API configuration:**

```javascript
// src/config/api.js
const API_CONFIG = {
  PRODUCTION_URL: 'https://your-railway-app.railway.app', // Your deployed URL
  // ... rest of config
};
```

### **Test API Connection:**

```javascript
// Test in your mobile app
const testResult = await API.testConnection();
console.log('API Connection:', testResult);
```

---

## 🧪 **Testing**

### **1. Health Check**
```bash
curl https://your-api-url.railway.app/health
```

### **2. Create Test User**
```bash
curl -X POST https://your-api-url.railway.app/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass123"}'
```

### **3. Login Test**
```bash
curl -X POST https://your-api-url.railway.app/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass123"}'
```

---

## 🚀 **Deployment Options**

### **Railway (Recommended)**
- ✅ Free tier available
- ✅ Automatic deployments from GitHub
- ✅ Built-in environment variables
- ✅ Custom domains
- ✅ Easy scaling

### **Heroku**
- ✅ Popular platform
- ✅ Git-based deployment
- ⚠️ Free tier limited

### **Docker**
- ✅ Run anywhere
- ✅ Consistent environment
- ✅ Easy scaling

---

## 📊 **Database Schema**

### **Collections in MongoDB (SECURE_CHAT)**

```javascript
// users
{
  _id: "uuid",
  username: "string",
  email: "string?",
  password_hash: "string",
  created_at: "datetime",
  last_active: "datetime",
  is_active: "boolean"
}

// messages  
{
  _id: "uuid",
  conversation_id: "uuid",
  sender_id: "uuid",
  recipient_id: "uuid", 
  encrypted_data: "binary",
  salt: "binary",
  message_type: "image|video",
  is_view_once: "boolean",
  timer_seconds: "number?",
  is_viewed: "boolean",
  is_burned: "boolean",
  created_at: "datetime",
  expires_at: "datetime",
  file_size: "number",
  file_extension: "string"
}

// conversations
{
  _id: "uuid",
  participants: ["uuid", "uuid"],
  created_at: "datetime", 
  last_message_at: "datetime"
}
```

---

## 🔧 **Development**

### **Local Setup**
```bash
# Clone repository
git clone https://github.com/Sahilprince/SECURE_CHAT.git
cd SECURE_CHAT

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.production .env
# Edit .env with your settings

# Run locally
python production_backend.py
```

### **API Documentation**
- Development: http://localhost:8000/docs
- Production: Disabled for security

---

## 🛡️ **Security Notes**

### **⚠️ Important for Production:**
1. **Change JWT_SECRET** to a random 32+ character string
2. **Change MASTER_KEY** to a new Fernet key
3. **Use HTTPS** in production (Railway provides this)
4. **Monitor logs** for security events
5. **Regular updates** for dependencies

### **🔒 Privacy Features:**
- **Zero-knowledge**: Server cannot decrypt messages
- **No persistent storage**: Images deleted after viewing/expiry
- **Camera detection**: Prevents unauthorized photography
- **View-once guarantee**: Messages permanently deleted

---

## 📞 **Support**

- **GitHub Issues**: Report bugs and feature requests
- **API Status**: Check `/health` endpoint
- **Database**: MongoDB Atlas cluster monitoring

---

## 📈 **Monitoring**

### **Health Checks**
- Database connectivity
- AI model status  
- Memory usage
- Response times

### **Security Events**
- Camera detection triggers
- Failed authentication attempts
- Message burn events

---

**🚀 Ready to deploy? Just push to GitHub and connect Railway!**
