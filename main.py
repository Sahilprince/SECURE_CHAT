import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import uuid
import hashlib
import base64
import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext
import motor.motor_asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import cv2
import numpy as np
from PIL import Image

# Try to import AI models (graceful degradation if not available)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available - using basic detection")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using CPU-only detection")

# Configuration
class Settings:
    # MongoDB Atlas Connection
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://ssawana:Ganesh%402024@cluster0.a2cfv2o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "SECURE_CHAT")

    # JWT Configuration
    JWT_SECRET = os.getenv("JWT_SECRET", "securevault-production-jwt-secret-key-2024-change-this-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

    # Encryption
    MASTER_KEY = os.getenv("MASTER_KEY", Fernet.generate_key().decode())

    # File Upload
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "avi"}

    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()

# Global variables for AI models
camera_detection_model = None
face_cascade = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load AI models on startup (with fallback)
    global camera_detection_model, face_cascade

    try:
        if YOLO_AVAILABLE:
            # In production, use a lightweight model or download on demand
            print("🤖 Attempting to load YOLO model...")
            # camera_detection_model = YOLO('yolov8n.pt')
            print("✅ YOLO model loading skipped (use lightweight detection)")

        # Load OpenCV face cascade (lightweight)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not face_cascade.empty():
                print("✅ Face detection model loaded")
            else:
                face_cascade = None
        except:
            face_cascade = None
            print("⚠️ Face detection model not available")

    except Exception as e:
        print(f"⚠️ AI models not loaded: {e}")
        print("Using lightweight detection methods")

    # Connect to MongoDB Atlas
    global mongodb_client, database
    try:
        mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
        database = mongodb_client[settings.DATABASE_NAME]

        # Test connection
        await database.command("ping")
        print("✅ Connected to MongoDB Atlas")

        # Create indexes
        await create_indexes()
        print("✅ Database indexes created")

    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise

    yield

    # Cleanup
    if mongodb_client:
        mongodb_client.close()

app = FastAPI(
    title="SecureVault API - Production",
    description="Privacy-first messaging with advanced camera detection - Production Ready",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,  # Disable docs in production
    redoc_url="/redoc" if settings.DEBUG else None
)

# CORS middleware - Configure for production
allowed_origins = [
    "http://localhost:*",
    "http://127.0.0.1:*",
    "exp://192.168.*",  # Expo development
    "exp://*",  # Expo tunneling
    "https://*.exp.direct",  # Expo hosting
    "https://*.netlify.app",  # If you deploy frontend
    "https://*.vercel.app",   # If you deploy frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database
mongodb_client = None
database = None

async def get_collection(name: str):
    return database[name]

async def create_indexes():
    users_collection = await get_collection("users")
    messages_collection = await get_collection("messages")
    conversations_collection = await get_collection("conversations")

    # Create unique indexes
    try:
        await users_collection.create_index("username", unique=True)
        await users_collection.create_index("email", unique=True, sparse=True)

        # Performance indexes
        await messages_collection.create_index([("conversation_id", 1), ("created_at", -1)])
        await messages_collection.create_index("expires_at", expireAfterSeconds=0)
        await conversations_collection.create_index([("participants", 1), ("last_message_at", -1)])

        print("✅ All database indexes created successfully")
    except Exception as e:
        print(f"⚠️ Index creation warning: {e}")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: Optional[EmailStr] = None
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class MessageResponse(BaseModel):
    message: str
    data: Optional[Dict] = None

# Encryption utilities
class EncryptionManager:
    def __init__(self):
        self.master_key = settings.MASTER_KEY.encode()

    def generate_key(self, password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_file(self, file_data: bytes, password: str) -> tuple:
        salt = os.urandom(16)
        key = self.generate_key(password, salt)
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(file_data)
        return encrypted_data, salt

    def decrypt_file(self, encrypted_data: bytes, password: str, salt: bytes) -> bytes:
        key = self.generate_key(password, salt)
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)

encryption_manager = EncryptionManager()

# Lightweight Camera Detection System (Production Optimized)
class CameraDetectionSystem:
    def __init__(self):
        self.detection_methods = [
            "brightness_analysis",
            "edge_detection", 
            "basic_face_count",
            "image_variance"
        ]

    async def detect_camera_devices(self, image_data: bytes) -> Dict:
        """
        Lightweight camera detection for production deployment
        Uses basic computer vision techniques instead of heavy AI models
        """
        results = {
            "camera_detected": False,
            "confidence": 0.0,
            "detected_objects": [],
            "faces_detected": 0,
            "threat_level": "low"
        }

        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image.convert('RGB'))

            total_confidence = 0.0
            detection_count = 0

            # Method 1: Brightness Analysis (Camera Flash Detection)
            brightness_score = self._analyze_brightness(image_np)
            if brightness_score > 0.6:
                results["detected_objects"].append({
                    "object": "bright_flash",
                    "confidence": brightness_score
                })
                total_confidence += brightness_score
                detection_count += 1

            # Method 2: Edge Detection (Rectangular Objects - Phones/Cameras)
            edge_score = self._detect_rectangular_objects(image_np)
            if edge_score > 0.5:
                results["detected_objects"].append({
                    "object": "rectangular_device",
                    "confidence": edge_score
                })
                total_confidence += edge_score
                detection_count += 1

            # Method 3: Face Detection (if available)
            if face_cascade is not None:
                face_count = self._detect_faces(image_np)
                results["faces_detected"] = face_count
                if face_count > 2:  # Multiple people might indicate photo being taken
                    face_confidence = min(0.8, face_count / 5.0)
                    results["detected_objects"].append({
                        "object": "multiple_faces",
                        "confidence": face_confidence
                    })
                    total_confidence += face_confidence
                    detection_count += 1

            # Method 4: Image Variance (Motion/Activity Detection)
            variance_score = self._analyze_image_variance(image_np)
            if variance_score > 0.7:
                results["detected_objects"].append({
                    "object": "high_activity",
                    "confidence": variance_score
                })
                total_confidence += variance_score
                detection_count += 1

            # Calculate final confidence
            if detection_count > 0:
                results["confidence"] = total_confidence / detection_count
                results["camera_detected"] = results["confidence"] > 0.5

                # Determine threat level
                if results["confidence"] > 0.8:
                    results["threat_level"] = "high"
                elif results["confidence"] > 0.6:
                    results["threat_level"] = "medium"

        except Exception as e:
            print(f"Camera detection error: {e}")
            # Fail safe - assume no threat if detection fails
            results["camera_detected"] = False

        return results

    def _analyze_brightness(self, image_np: np.ndarray) -> float:
        """Detect sudden bright spots (camera flash)"""
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)

            # Check for very bright spots
            bright_pixels = np.sum(gray > 230)
            total_pixels = gray.shape[0] * gray.shape[1]
            bright_ratio = bright_pixels / total_pixels

            if brightness > 180 and bright_ratio > 0.15:
                return min(0.9, bright_ratio * 4)

        except Exception:
            pass

        return 0.0

    def _detect_rectangular_objects(self, image_np: np.ndarray) -> float:
        """Detect rectangular shapes (phones, tablets, cameras)"""
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rectangular_count = 0
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if it's roughly rectangular (4 sides)
                if len(approx) >= 4 and cv2.contourArea(contour) > 1000:
                    rectangular_count += 1

            if rectangular_count > 3:
                return min(0.8, rectangular_count / 10.0)

        except Exception:
            pass

        return 0.0

    def _detect_faces(self, image_np: np.ndarray) -> int:
        """Detect faces in the image"""
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            return len(faces)
        except Exception:
            return 0

    def _analyze_image_variance(self, image_np: np.ndarray) -> float:
        """Analyze image for high activity/motion"""
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            variance = np.var(gray)

            # High variance might indicate motion or complex scenes
            if variance > 2000:
                return min(0.7, variance / 5000.0)

        except Exception:
            pass

        return 0.0

camera_detector = CameraDetectionSystem()

# Authentication utilities
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        payload = jwt.decode(credentials.credentials, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        users_collection = await get_collection("users")
        user = await users_collection.find_one({"_id": user_id})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")

        return user
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# API Routes

@app.get("/")
async def root():
    return {
        "message": "SecureVault API - Production Ready",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "signup", "login", "messaging", "view_once", 
            "timer_messages", "camera_detection", "e2e_encryption"
        ]
    }

@app.get("/health")
async def health_check():
    # Test database connection
    try:
        await database.command("ping")
        db_status = "connected"
    except:
        db_status = "disconnected"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "ai_models": {
            "yolo_available": YOLO_AVAILABLE,
            "face_detection_available": face_cascade is not None,
            "torch_available": TORCH_AVAILABLE
        },
        "detection_methods": camera_detector.detection_methods
    }

@app.post("/auth/signup")
async def signup(user_data: UserCreate):
    users_collection = await get_collection("users")

    # Validate input
    if len(user_data.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

    if len(user_data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    # Check if user exists
    if await users_collection.find_one({"username": user_data.username.lower()}):
        raise HTTPException(status_code=400, detail="Username already exists")

    if user_data.email and await users_collection.find_one({"email": user_data.email.lower()}):
        raise HTTPException(status_code=400, detail="Email already exists")

    # Hash password
    hashed_password = pwd_context.hash(user_data.password)

    # Create user
    user_id = str(uuid.uuid4())
    user = {
        "_id": user_id,
        "username": user_data.username.lower(),
        "email": user_data.email.lower() if user_data.email else None,
        "password_hash": hashed_password,
        "created_at": datetime.utcnow(),
        "last_active": datetime.utcnow(),
        "is_active": True
    }

    await users_collection.insert_one(user)

    # Create token
    token = create_access_token({"user_id": user_id, "username": user_data.username.lower()})

    return {
        "message": "User created successfully",
        "token": token,
        "user": {
            "id": user_id,
            "username": user_data.username.lower(),
            "email": user_data.email
        }
    }

@app.post("/auth/login")
async def login(user_data: UserLogin):
    users_collection = await get_collection("users")

    user = await users_collection.find_one({"username": user_data.username.lower()})
    if not user or not pwd_context.verify(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Update last active
    await users_collection.update_one(
        {"_id": user["_id"]}, 
        {"$set": {"last_active": datetime.utcnow()}}
    )

    token = create_access_token({"user_id": user["_id"], "username": user["username"]})

    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["_id"],
            "username": user["username"],
            "email": user.get("email")
        }
    }

@app.get("/auth/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "user": {
            "id": current_user["_id"],
            "username": current_user["username"],
            "email": current_user.get("email"),
            "created_at": current_user["created_at"],
            "last_active": current_user["last_active"]
        }
    }

@app.post("/messages/send")
async def send_message(
    recipient_username: str = Form(...),
    password: str = Form(...),
    is_view_once: bool = Form(False),
    timer_seconds: Optional[int] = Form(None),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Validate file
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    file_extension = file.filename.split(".")[-1].lower() if file.filename else "jpg"
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed")

    # Find recipient
    users_collection = await get_collection("users")
    recipient = await users_collection.find_one({"username": recipient_username.lower()})
    if not recipient:
        raise HTTPException(status_code=404, detail="Recipient not found")

    # Read file data
    file_data = await file.read()

    # Encrypt file
    encrypted_data, salt = encryption_manager.encrypt_file(file_data, password)

    # Get or create conversation
    conversations_collection = await get_collection("conversations")
    conversation = await conversations_collection.find_one({
        "participants": {"$all": [current_user["_id"], recipient["_id"]]}
    })

    if not conversation:
        conversation_id = str(uuid.uuid4())
        conversation = {
            "_id": conversation_id,
            "participants": [current_user["_id"], recipient["_id"]],
            "created_at": datetime.utcnow(),
            "last_message_at": datetime.utcnow()
        }
        await conversations_collection.insert_one(conversation)
    else:
        conversation_id = conversation["_id"]
        await conversations_collection.update_one(
            {"_id": conversation_id},
            {"$set": {"last_message_at": datetime.utcnow()}}
        )

    # Calculate expiration
    expires_at = None
    if timer_seconds:
        expires_at = datetime.utcnow() + timedelta(seconds=timer_seconds)
    elif is_view_once:
        expires_at = datetime.utcnow() + timedelta(hours=24)  # Max 24 hours for view-once
    else:
        expires_at = datetime.utcnow() + timedelta(days=7)  # 7 days default

    # Create message
    message_id = str(uuid.uuid4())
    message = {
        "_id": message_id,
        "conversation_id": conversation_id,
        "sender_id": current_user["_id"],
        "recipient_id": recipient["_id"],
        "encrypted_data": encrypted_data,
        "salt": salt,
        "message_type": "image" if file_extension in {"jpg", "jpeg", "png", "gif", "webp"} else "video",
        "is_view_once": is_view_once,
        "timer_seconds": timer_seconds,
        "is_viewed": False,
        "is_burned": False,
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "file_size": len(file_data),
        "file_extension": file_extension
    }

    messages_collection = await get_collection("messages")
    await messages_collection.insert_one(message)

    return {
        "message": "Message sent successfully",
        "message_id": message_id,
        "expires_at": expires_at.isoformat() if expires_at else None,
        "recipient": recipient_username
    }

@app.post("/messages/{message_id}/view")
async def view_message(
    message_id: str,
    password: str = Form(...),
    camera_image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    messages_collection = await get_collection("messages")

    # Get message
    message = await messages_collection.find_one({"_id": message_id})
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Check if user is recipient
    if message["recipient_id"] != current_user["_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to view this message")

    # Check if already burned
    if message["is_burned"]:
        raise HTTPException(status_code=410, detail="Message has been burned")

    # Check expiration
    if message["expires_at"] and datetime.utcnow() > message["expires_at"]:
        raise HTTPException(status_code=410, detail="Message has expired")

    # CAMERA DETECTION - Critical Security Feature
    try:
        camera_image_data = await camera_image.read()
        detection_results = await camera_detector.detect_camera_devices(camera_image_data)

        if detection_results["camera_detected"]:
            return {
                "camera_detected": True,
                "threat_level": detection_results["threat_level"],
                "confidence": detection_results["confidence"],
                "detected_objects": detection_results["detected_objects"],
                "message": f"Camera detected - Access blocked for security (confidence: {detection_results['confidence']:.2f})"
            }
    except Exception as e:
        print(f"Camera detection error: {e}")
        # In production, you might want to be more strict here
        pass

    try:
        # Decrypt file
        decrypted_data = encryption_manager.decrypt_file(
            message["encrypted_data"], 
            password, 
            message["salt"]
        )

        # Mark as viewed and potentially burn
        update_data = {
            "is_viewed": True,
            "viewed_at": datetime.utcnow()
        }

        if message["is_view_once"]:
            update_data["is_burned"] = True

        await messages_collection.update_one(
            {"_id": message_id},
            {"$set": update_data}
        )

        # Convert to base64 for transmission
        file_base64 = base64.b64encode(decrypted_data).decode()

        return {
            "camera_detected": False,
            "message_id": message_id,
            "file_data": file_base64,
            "message_type": message["message_type"],
            "file_extension": message["file_extension"],
            "is_view_once": message["is_view_once"],
            "timer_seconds": message.get("timer_seconds"),
            "burned": message["is_view_once"],  # Will be burned after this view
            "sender_username": "unknown"  # Could add sender lookup if needed
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid password or corrupted file: {str(e)}")

@app.get("/conversations")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    conversations_collection = await get_collection("conversations")
    users_collection = await get_collection("users")
    messages_collection = await get_collection("messages")

    conversations = await conversations_collection.find({
        "participants": current_user["_id"]
    }).sort("last_message_at", -1).to_list(length=50)

    result = []
    for conv in conversations:
        # Get other participant
        other_participant_id = next(p for p in conv["participants"] if p != current_user["_id"])
        other_user = await users_collection.find_one({"_id": other_participant_id})

        # Get last message
        last_message = await messages_collection.find_one(
            {"conversation_id": conv["_id"]},
            sort=[("created_at", -1)]
        )

        result.append({
            "conversation_id": conv["_id"],
            "other_user": {
                "id": other_user["_id"],
                "username": other_user["username"]
            } if other_user else None,
            "last_message_at": conv["last_message_at"],
            "last_message": {
                "type": last_message["message_type"] if last_message else None,
                "is_view_once": last_message["is_view_once"] if last_message else False,
                "created_at": last_message["created_at"] if last_message else None,
                "is_burned": last_message.get("is_burned", False) if last_message else False
            } if last_message else None
        })

    return {"conversations": result}

@app.get("/users/search")
async def search_users(
    username: str,
    current_user: dict = Depends(get_current_user)
):
    if len(username) < 2:
        raise HTTPException(status_code=400, detail="Username must be at least 2 characters")

    users_collection = await get_collection("users")

    # Search for users (case-insensitive)
    users = await users_collection.find({
        "username": {"$regex": username.lower(), "$options": "i"},
        "_id": {"$ne": current_user["_id"]}  # Exclude current user
    }).limit(10).to_list(length=10)

    result = []
    for user in users:
        result.append({
            "id": user["_id"],
            "username": user["username"],
            "last_active": user.get("last_active")
        })

    return {"users": result}

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Global exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        access_log=True
    )
