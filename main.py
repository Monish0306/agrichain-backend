import os
import json
import joblib
import pickle
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import bcrypt
from jose import jwt
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ─────────────────────────────────────────
# SECTION 1 — ENVIRONMENT & CONFIG
# ─────────────────────────────────────────

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_change_me_in_production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
MONITOR_USERNAME = "agrichain_monitor"
MONITOR_PASSWORD = "Monitor@AgriChain2026"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml_models")

# ─────────────────────────────────────────
# SECTION 2 — DATABASE SETUP
# ─────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agrichain.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )
else:
    # PostgreSQL — Supabase connection pooler (port 6543)
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
        connect_args={"sslmode": "require", "connect_timeout": 10},
    )

# ── Base and SessionLocal MUST be defined here, before any model classes ──
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _uuid_str() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────
# SECTION 3 — DATABASE MODELS
# ─────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=_uuid_str)
    phone = Column(String(32), unique=True, nullable=True, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    name = Column(String(120), nullable=False)
    role = Column(String(20), nullable=False)
    language = Column(String(40), nullable=False, server_default=text("'english'"))
    state = Column(String(80), nullable=True)
    district = Column(String(80), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    is_active = Column(Boolean, nullable=False, default=True)
    farms = relationship("Farm", back_populates="farmer", cascade="all, delete-orphan")
    listings = relationship("Listing", back_populates="farmer", cascade="all, delete-orphan")


class Farm(Base):
    __tablename__ = "farms"
    id = Column(String(36), primary_key=True, default=_uuid_str)
    farmer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    gps_lat = Column(Float, nullable=False)
    gps_lon = Column(Float, nullable=False)
    area_acres = Column(Float, nullable=False)
    soil_type = Column(String(80), nullable=False)
    district = Column(String(80), nullable=False)
    state = Column(String(80), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    farmer = relationship("User", back_populates="farms")
    crop_records = relationship("CropRecord", back_populates="farm", cascade="all, delete-orphan")


class Listing(Base):
    __tablename__ = "listings"
    id = Column(String(36), primary_key=True, default=_uuid_str)
    farmer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    crop_type = Column(String(80), nullable=False, index=True)
    quantity_kg = Column(Float, nullable=False)
    asking_price = Column(Float, nullable=False)
    quality_grade = Column(String(2), nullable=False)
    description = Column(String(500), nullable=True)
    location_lat = Column(Float, nullable=True)
    location_lon = Column(Float, nullable=True)
    district = Column(String(80), nullable=False, index=True)
    state = Column(String(80), nullable=False, index=True)
    status = Column(String(20), nullable=False, server_default=text("'active'"))
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    farmer = relationship("User", back_populates="listings")
    transactions = relationship("Transaction", back_populates="listing", cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(String(36), primary_key=True, default=_uuid_str)
    listing_id = Column(String(36), ForeignKey("listings.id"), nullable=False, index=True)
    farmer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    merchant_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    agreed_price = Column(Float, nullable=False)
    quantity_kg = Column(Float, nullable=False)
    status = Column(String(30), nullable=False, server_default=text("'pending'"))
    blockchain_tx_hash = Column(String(120), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
    listing = relationship("Listing", back_populates="transactions")


class CropRecord(Base):
    __tablename__ = "crop_records"
    id = Column(String(36), primary_key=True, default=_uuid_str)
    farm_id = Column(String(36), ForeignKey("farms.id"), nullable=False, index=True)
    crop_type = Column(String(80), nullable=False)
    planting_date = Column(DateTime(timezone=True), nullable=False)
    expected_harvest = Column(DateTime(timezone=True), nullable=True)
    notes = Column(String(500), nullable=True)
    farm = relationship("Farm", back_populates="crop_records")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────
# SECTION 4 — ML MODEL LOADING
# ─────────────────────────────────────────

# Global model variables
crop_model = None
crop_label_encoder = None
crop_scaler = None
fertilizer_model = None
fertilizer_label_encoder = None
soil_type_encoder = None
crop_type_encoder = None
fertilizer_scaler = None
price_models: dict = {}
available_commodities: list = []
schemes_data: list = []

# In-memory stores
MERCHANT_PASSWORD_HASHES: dict = {}
TRANSACTION_CONFIRMATIONS: dict = {}


def _safe_joblib_load(path: str, label: str):
    try:
        obj = joblib.load(path)
        print(f"  [OK] Loaded {label}")
        return obj
    except Exception as e:
        print(f"  [WARN] Could not load {label}: {e}")
        return None


def _safe_pickle_load(path: str, label: str):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  [OK] Loaded {label}")
        return obj
    except Exception as e:
        print(f"  [WARN] Could not load {label}: {e}")
        return None


def load_ml_artifacts():
    global crop_model, crop_label_encoder, crop_scaler
    global fertilizer_model, fertilizer_label_encoder
    global soil_type_encoder, crop_type_encoder, fertilizer_scaler
    global price_models, available_commodities

    print("\n[STARTUP] Loading ML models...")

    # ── Crop models ──
    crop_model = _safe_joblib_load(os.path.join(ML_DIR, "crop_model.pkl"), "crop_model")
    crop_label_encoder = _safe_joblib_load(
        os.path.join(ML_DIR, "crop_label_encoder.pkl"), "crop_label_encoder"
    )
    crop_scaler = _safe_joblib_load(os.path.join(ML_DIR, "crop_scaler.pkl"), "crop_scaler")

    # ── Fertilizer models ──
    fertilizer_model = _safe_joblib_load(
        os.path.join(ML_DIR, "fertilizer_model.pkl"), "fertilizer_model"
    )
    fertilizer_label_encoder = _safe_joblib_load(
        os.path.join(ML_DIR, "fertilizer_label_encoder.pkl"), "fertilizer_label_encoder"
    )
    soil_type_encoder = _safe_joblib_load(
        os.path.join(ML_DIR, "soil_type_encoder.pkl"), "soil_type_encoder"
    )
    crop_type_encoder = _safe_joblib_load(
        os.path.join(ML_DIR, "crop_type_encoder.pkl"), "crop_type_encoder"
    )
    fertilizer_scaler = _safe_joblib_load(
        os.path.join(ML_DIR, "fertilizer_scaler.pkl"), "fertilizer_scaler"
    )

    # ── Price models — scan ml_models/ AND ml_models/price_models/ ──
    price_models = {}
    scan_dirs = [ML_DIR, os.path.join(ML_DIR, "price_models")]

    print("\n[STARTUP] Loading price models...")
    for scan_dir in scan_dirs:
        if not os.path.isdir(scan_dir):
            continue
        for fn in os.listdir(scan_dir):
            if fn.startswith("price_") and fn.lower().endswith((".pkl", ".pickle")):
                commodity = os.path.splitext(fn)[0].replace("price_", "").strip().lower()
                if not commodity or commodity in price_models:
                    continue
                model = _safe_pickle_load(os.path.join(scan_dir, fn), f"price_{commodity}")
                if model is not None:
                    price_models[commodity] = model

    # ── available_commodities.pkl ──
    for ac_dir in [ML_DIR, os.path.join(ML_DIR, "price_models")]:
        ac_path = os.path.join(ac_dir, "available_commodities.pkl")
        if os.path.exists(ac_path):
            ac = _safe_pickle_load(ac_path, "available_commodities")
            if isinstance(ac, list):
                available_commodities = [str(x).strip() for x in ac if str(x).strip()]
            break

    # Fallback: derive from loaded price models
    if not available_commodities and price_models:
        available_commodities = list(price_models.keys())

    print(f"\n[STARTUP] Price models loaded: {len(price_models)} commodities")
    print(f"[STARTUP] Commodities: {sorted(price_models.keys())}")


def load_schemes():
    global schemes_data
    try:
        schemes_path = os.path.join(BASE_DIR, "schemes.json")
        with open(schemes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        schemes_data = data if isinstance(data, list) else []
        print(f"[STARTUP] Loaded {len(schemes_data)} government schemes")
    except Exception as e:
        schemes_data = []
        print(f"[STARTUP WARN] Failed to load schemes.json: {e}")


# ─────────────────────────────────────────
# SECTION 5 — AUTH HELPERS
# ─────────────────────────────────────────

security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, role: str) -> str:
    to_encode = dict(data)
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "role": role})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def _extract_token(credentials: HTTPAuthorizationCredentials) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )
    token = credentials.credentials
    if isinstance(token, str):
        token = token.strip().strip('"').strip("'")
        if token.lower().startswith("bearer "):
            token = token.split(" ", 1)[1].strip()
    return token


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    token = _extract_token(credentials)
    payload = verify_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    user = db.query(User).filter(User.id == str(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


def _role_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = _extract_token(credentials)
    payload = verify_token(token)
    role = payload.get("role")
    if not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    return str(role)


def require_farmer(role: str = Depends(_role_from_token)):
    if role != "farmer":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Farmer role required")


def require_merchant(role: str = Depends(_role_from_token)):
    if role != "merchant":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Merchant role required")


def require_monitor(role: str = Depends(_role_from_token)):
    if role != "monitor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Monitor role required")


# ─────────────────────────────────────────
# SECTION 6 — FASTAPI APP
# ─────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="AgriChain Intelligence Platform",
    version="1.0.0",
    description="AI-powered agriculture platform for Indian farmers — Crop Advisory, Marketplace, Price Intelligence, Finance",
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again in a minute."},
    )


@app.exception_handler(404)
def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not Found"})


@app.exception_handler(500)
def server_error_handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.on_event("startup")
def on_startup():
    # Create all DB tables
    Base.metadata.create_all(bind=engine)
    # Load ML artifacts
    load_ml_artifacts()
    # Load government schemes
    load_schemes()
    print("\n[STARTUP] ✅ AgriChain backend ready. Visit /docs to explore all endpoints.\n")


# ─────────────────────────────────────────
# SECTION 7 — HEALTH & DEBUG ENDPOINTS
# ─────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "name": "AgriChain Intelligence Platform",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
        "crop_model_loaded": crop_model is not None,
        "fertilizer_model_loaded": fertilizer_model is not None,
        "price_models_loaded": len(price_models),
        "schemes_loaded": len(schemes_data),
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "connected",
    }


@app.get("/api/debug/models", tags=["Debug"])
def debug_models():
    files_main = os.listdir(ML_DIR) if os.path.isdir(ML_DIR) else []
    price_subdir = os.path.join(ML_DIR, "price_models")
    files_sub = os.listdir(price_subdir) if os.path.isdir(price_subdir) else []
    return {
        "ml_dir": ML_DIR,
        "files_in_ml_models": sorted(files_main),
        "files_in_price_models_subfolder": sorted(files_sub),
        "loaded_price_models": sorted(price_models.keys()),
        "available_commodities": sorted(available_commodities),
        "crop_model_loaded": crop_model is not None,
        "fertilizer_model_loaded": fertilizer_model is not None,
        "schemes_loaded": len(schemes_data),
    }


# ─────────────────────────────────────────
# SECTION 8 — AUTH ENDPOINTS
# ─────────────────────────────────────────

class FarmerLoginRequest(BaseModel):
    phone: str = Field(..., example="9876543210")
    name: str = Field(..., example="Ravi Kumar")


class MerchantLoginRequest(BaseModel):
    email: str = Field(..., example="merchant@example.com")
    password: str = Field(..., example="Password@123")
    name: str = Field(..., example="Amit Traders")


class MonitorLoginRequest(BaseModel):
    username: str = Field(..., example="agrichain_monitor")
    password: str = Field(..., example="Monitor@AgriChain2026")


@app.post("/api/auth/farmer/login", tags=["Auth"])
def farmer_login(payload: FarmerLoginRequest, db: Session = Depends(get_db)):
    phone = payload.phone.strip()
    name = payload.name.strip()
    if not phone or not name:
        raise HTTPException(status_code=400, detail="phone and name are required")

    user = db.query(User).filter(User.phone == phone).first()
    if user is None:
        user = User(phone=phone, name=name, role="farmer")
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if user.role != "farmer":
            raise HTTPException(
                status_code=400,
                detail="Phone already registered with a different role",
            )
        if user.name != name:
            user.name = name
            db.commit()

    token = create_access_token({"sub": user.id, "name": user.name}, role="farmer")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "name": user.name,
        "role": user.role,
    }


@app.post("/api/auth/merchant/login", tags=["Auth"])
def merchant_login(payload: MerchantLoginRequest, db: Session = Depends(get_db)):
    email = payload.email.strip().lower()
    password = payload.password.strip()
    name = payload.name.strip()
    if not email or not password or not name:
        raise HTTPException(status_code=400, detail="email, password and name are required")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(email=email, name=name, role="merchant")
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if user.role != "merchant":
            raise HTTPException(
                status_code=400,
                detail="Email already registered with a different role",
            )
        if user.name != name:
            user.name = name
            db.commit()

    pw_hash = MERCHANT_PASSWORD_HASHES.get(email)
    if pw_hash is None:
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        MERCHANT_PASSWORD_HASHES[email] = pw_hash
    else:
        if not bcrypt.checkpw(password.encode("utf-8"), pw_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.id, "name": user.name}, role="merchant")
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "name": user.name,
        "role": user.role,
    }


@app.post("/api/auth/monitor/login", tags=["Auth"])
@limiter.limit("5/minute")
def monitor_login(request: Request, payload: MonitorLoginRequest):
    if payload.username.strip() != MONITOR_USERNAME or payload.password.strip() != MONITOR_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(
        {"sub": f"monitor:{payload.username}", "name": payload.username},
        role="monitor",
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": f"monitor:{payload.username}",
        "name": payload.username,
        "role": "monitor",
    }


@app.get("/api/auth/me", tags=["Auth"])
def me(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "phone": user.phone,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "language": user.language,
        "state": user.state,
        "district": user.district,
        "created_at": user.created_at,
        "is_active": user.is_active,
    }


# ─────────────────────────────────────────
# SECTION 9 — ADVISORY ENDPOINTS
# ─────────────────────────────────────────

VALID_SOIL_TYPES = {"Black", "Clayey", "Loamy", "Red", "Sandy"}

SOIL_TYPES_INFO = [
    {"name": "Black", "description": "Clay-rich soils with high moisture retention. Best for cotton, sugarcane, wheat."},
    {"name": "Clayey", "description": "Heavy soils with good water retention. Suitable for rice and pulses."},
    {"name": "Loamy", "description": "Best all-purpose soil. Suitable for almost all crops."},
    {"name": "Red", "description": "Iron-rich well-drained soils. Good for groundnut, millets, oilseeds."},
    {"name": "Sandy", "description": "Light fast-draining soils. Suitable for groundnut, sweet potato, millets."},
]

VALID_CROP_TYPES = {
    "Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
    "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat",
}

CROP_NAMES_22 = [
    "Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas", "Moth Beans",
    "Mung Bean", "Black Gram", "Lentil", "Pomegranate", "Banana", "Mango",
    "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya",
    "Coconut", "Cotton", "Jute", "Coffee",
]

WATER_REQUIREMENT_MM = {
    "Rice": 1200, "Maize": 600, "Chickpea": 400, "Kidney Beans": 500,
    "Pigeon Peas": 500, "Moth Beans": 350, "Mung Bean": 350, "Black Gram": 350,
    "Lentil": 400, "Pomegranate": 600, "Banana": 1200, "Mango": 700,
    "Grapes": 500, "Watermelon": 450, "Muskmelon": 450, "Apple": 800,
    "Orange": 800, "Papaya": 900, "Coconut": 1200, "Cotton": 700,
    "Jute": 800, "Coffee": 1200,
}


class AdvisoryRecommendRequest(BaseModel):
    nitrogen: float = Field(..., example=90, description="Nitrogen content in soil (kg/ha)")
    phosphorous: float = Field(..., example=42, description="Phosphorous content (kg/ha)")
    potassium: float = Field(..., example=43, description="Potassium content (kg/ha)")
    temperature: float = Field(..., example=25.0, description="Temperature in Celsius")
    humidity: float = Field(..., example=65.0, description="Humidity percentage (0-100)")
    ph: float = Field(..., example=6.5, description="Soil pH value (0-14)")
    rainfall: float = Field(..., example=200.0, description="Annual rainfall in mm")
    soil_type: str = Field(..., example="Loamy", description="One of: Black, Clayey, Loamy, Red, Sandy")
    crop_type: str = Field("Wheat", example="Wheat", description="One of: Barley, Cotton, Ground Nuts, Maize, Millets, Oil seeds, Paddy, Pulses, Sugarcane, Tobacco, Wheat")
    gps_lat: float = Field(12.97, example=12.97)
    gps_lon: float = Field(77.59, example=77.59)
    language: str = Field("english", example="english")


@app.post("/api/advisory/recommend", tags=["Advisory"], dependencies=[Depends(require_farmer)])
def recommend(payload: AdvisoryRecommendRequest, user: User = Depends(get_current_user)):
    if crop_model is None or crop_label_encoder is None or crop_scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Crop recommendation model not loaded. Check ml_models/ folder.",
        )
    if fertilizer_model is None or soil_type_encoder is None or crop_type_encoder is None or fertilizer_scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Fertilizer model not loaded. Check ml_models/ folder.",
        )

    soil_type = payload.soil_type.strip()
    crop_type = payload.crop_type.strip()

    if soil_type not in VALID_SOIL_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Invalid soil_type: '{soil_type}'",
                "valid_options": sorted(VALID_SOIL_TYPES),
                "hint": "Values are case-sensitive",
            },
        )

    crop_type_for_fertilizer = crop_type if crop_type in VALID_CROP_TYPES else "Wheat"

    # ── Crop recommendation ──
    try:
        features = np.array([[
            payload.nitrogen, payload.phosphorous, payload.potassium,
            payload.temperature, payload.humidity, payload.ph, payload.rainfall,
        ]], dtype=float)
        scaled = crop_scaler.transform(features)
        proba = crop_model.predict_proba(scaled)[0]
        top_idx = np.argsort(proba)[::-1][:3]
        top_crops = crop_label_encoder.inverse_transform(top_idx)
        recommended_crops = [
            {
                "name": str(c),
                "confidence": round(float(proba[top_idx[i]]), 4),
                "confidence_percent": round(float(proba[top_idx[i]]) * 100, 1),
                "water_requirement_mm": int(WATER_REQUIREMENT_MM.get(str(c), 600)),
            }
            for i, c in enumerate(top_crops)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop recommendation failed: {e}")

    # ── Fertilizer recommendation ──
    try:
        soil_enc = soil_type_encoder.transform([soil_type])[0]
        crop_enc = crop_type_encoder.transform([crop_type_for_fertilizer])[0]
        fert_features = np.array([[
            payload.temperature, payload.humidity, payload.ph,
            soil_enc, crop_enc,
            payload.nitrogen, payload.potassium, payload.phosphorous,
        ]], dtype=float)
        fert_scaled = fertilizer_scaler.transform(fert_features)
        fert_pred = fertilizer_model.predict(fert_scaled)
        fert_name = fertilizer_label_encoder.inverse_transform(fert_pred)[0]
        fertilizer_recommendation = {"name": str(fert_name), "dosage_kg_per_acre": 50}
    except Exception as e:
        fertilizer_recommendation = {
            "name": "DAP (Default)",
            "dosage_kg_per_acre": 50,
            "note": f"Model error: {e}",
        }

    # ── Weather from OpenWeatherMap ──
    weather_out = {"temp": None, "humidity": None, "description": None}
    try:
        if OPENWEATHER_API_KEY:
            url = (
                f"https://api.openweathermap.org/data/2.5/weather"
                f"?lat={payload.gps_lat}&lon={payload.gps_lon}"
                f"&appid={OPENWEATHER_API_KEY}&units=metric"
            )
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            w = r.json()
            weather_out = {
                "temp": float(w["main"]["temp"]),
                "humidity": float(w["main"]["humidity"]),
                "description": str(w["weather"][0]["description"]),
            }
    except Exception as e:
        weather_out = {"temp": None, "humidity": None, "description": f"Unavailable: {e}"}

    # ── Farming alerts ──
    farming_alerts = []
    h = weather_out.get("humidity") or payload.humidity
    t = weather_out.get("temp") or payload.temperature
    if h and float(h) > 80:
        farming_alerts.append({
            "message": "High humidity — elevated fungal disease risk. Consider protective spray.",
            "severity": "warning",
        })
    if t and float(t) > 40:
        farming_alerts.append({
            "message": "Extreme heat — risk of heat stress. Irrigate in early morning.",
            "severity": "critical",
        })
    if t and float(t) < 10:
        farming_alerts.append({
            "message": "Low temperature — possible frost risk. Protect young seedlings.",
            "severity": "warning",
        })

    best_crop = recommended_crops[0]["name"] if recommended_crops else "your crop"
    fert = fertilizer_recommendation["name"]
    dosage = fertilizer_recommendation["dosage_kg_per_acre"]

    return {
        "recommended_crops": recommended_crops,
        "fertilizer_recommendation": fertilizer_recommendation,
        "weather": weather_out,
        "farming_alerts": farming_alerts,
        "advice_summary": (
            f"Based on your soil and climate, {best_crop} is your best option. "
            f"Apply {fert} at {dosage} kg/acre for optimal yield."
        ),
    }


@app.get("/api/advisory/crops", tags=["Advisory"])
def advisory_crops():
    return {"crops": CROP_NAMES_22, "total": len(CROP_NAMES_22)}


@app.get("/api/advisory/soil-types", tags=["Advisory"])
def advisory_soil_types():
    return {
        "soil_types": SOIL_TYPES_INFO,
        "valid_values_for_api": sorted(VALID_SOIL_TYPES),
    }


@app.get("/api/advisory/crop-types-for-fertilizer", tags=["Advisory"])
def advisory_crop_types():
    return {
        "crop_types": sorted(VALID_CROP_TYPES),
        "note": "Use these exact values for crop_type in /recommend",
    }


# ─────────────────────────────────────────
# SECTION 10 — MARKETPLACE ENDPOINTS
# ─────────────────────────────────────────

class CreateListingRequest(BaseModel):
    crop_type: str = Field(..., example="Tomato")
    quantity_kg: float = Field(..., example=500.0)
    asking_price: float = Field(..., example=22.5, description="Price per kg in INR")
    quality_grade: str = Field(..., example="A", description="A, B, or C")
    description: Optional[str] = Field(None, example="Fresh tomatoes from Mysore farm")
    district: str = Field(..., example="Mysore")
    state: str = Field(..., example="Karnataka")
    location_lat: Optional[float] = Field(None, example=12.31)
    location_lon: Optional[float] = Field(None, example=76.65)


@app.post("/api/marketplace/listings", tags=["Marketplace"], dependencies=[Depends(require_farmer)])
def create_listing(
    payload: CreateListingRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    grade = payload.quality_grade.strip().upper()
    if grade not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="quality_grade must be A, B, or C")

    listing = Listing(
        farmer_id=user.id,
        crop_type=payload.crop_type.strip(),
        quantity_kg=payload.quantity_kg,
        asking_price=payload.asking_price,
        quality_grade=grade,
        description=payload.description,
        location_lat=payload.location_lat,
        location_lon=payload.location_lon,
        district=payload.district.strip(),
        state=payload.state.strip(),
        status="active",
    )
    db.add(listing)
    db.commit()
    db.refresh(listing)
    return {k: v for k, v in listing.__dict__.items() if not k.startswith("_")}


@app.get("/api/marketplace/listings", tags=["Marketplace"])
def list_listings(
    crop_type: Optional[str] = None,
    district: Optional[str] = None,
    state: Optional[str] = None,
    db: Session = Depends(get_db),
):
    q = (
        db.query(Listing, User)
        .join(User, Listing.farmer_id == User.id)
        .filter(Listing.status == "active")
    )
    if crop_type:
        q = q.filter(Listing.crop_type == crop_type)
    if district:
        q = q.filter(Listing.district == district)
    if state:
        q = q.filter(Listing.state == state)
    rows = q.order_by(Listing.created_at.desc()).all()
    out = []
    for listing, farmer in rows:
        d = {k: v for k, v in listing.__dict__.items() if not k.startswith("_")}
        d["farmer_name"] = farmer.name
        out.append(d)
    return {"listings": out, "count": len(out)}


@app.get("/api/marketplace/listings/{listing_id}", tags=["Marketplace"])
def get_listing(listing_id: str, db: Session = Depends(get_db)):
    row = (
        db.query(Listing, User)
        .join(User, Listing.farmer_id == User.id)
        .filter(Listing.id == listing_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Listing not found")
    listing, farmer = row
    d = {k: v for k, v in listing.__dict__.items() if not k.startswith("_")}
    d["farmer_name"] = farmer.name
    return d


class PlaceOrderRequest(BaseModel):
    quantity_kg: float = Field(..., example=100.0)
    offered_price: float = Field(..., example=20.0, description="Price per kg in INR")


@app.post("/api/marketplace/listings/{listing_id}/order", tags=["Marketplace"], dependencies=[Depends(require_merchant)])
def create_order(
    listing_id: str,
    payload: PlaceOrderRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    listing = db.query(Listing).filter(
        Listing.id == listing_id, Listing.status == "active"
    ).first()
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found or not active")
    if payload.quantity_kg <= 0 or payload.offered_price <= 0:
        raise HTTPException(
            status_code=400,
            detail="quantity_kg and offered_price must be positive",
        )

    tx = Transaction(
        listing_id=listing.id,
        farmer_id=listing.farmer_id,
        merchant_id=user.id,
        agreed_price=payload.offered_price,
        quantity_kg=payload.quantity_kg,
        status="pending",
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)
    TRANSACTION_CONFIRMATIONS[tx.id] = {"farmer": False, "merchant": False}
    return {k: v for k, v in tx.__dict__.items() if not k.startswith("_")}


@app.post("/api/marketplace/transactions/{transaction_id}/confirm", tags=["Marketplace"])
def confirm_transaction(
    transaction_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    tx = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")
    if user.role not in ("farmer", "merchant"):
        raise HTTPException(status_code=403, detail="Farmer or Merchant role required")

    confirmations = TRANSACTION_CONFIRMATIONS.setdefault(
        transaction_id, {"farmer": False, "merchant": False}
    )
    if user.role == "farmer":
        if user.id != tx.farmer_id:
            raise HTTPException(status_code=403, detail="Not your transaction")
        confirmations["farmer"] = True
    else:
        if user.id != tx.merchant_id:
            raise HTTPException(status_code=403, detail="Not your transaction")
        confirmations["merchant"] = True

    if confirmations["farmer"] and confirmations["merchant"]:
        tx.status = "completed"
        tx.completed_at = datetime.now(timezone.utc)
    else:
        tx.status = "confirmed"

    db.commit()
    db.refresh(tx)
    return {k: v for k, v in tx.__dict__.items() if not k.startswith("_")}


@app.get("/api/marketplace/my-listings", tags=["Marketplace"], dependencies=[Depends(require_farmer)])
def my_listings(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Listing)
        .filter(Listing.farmer_id == user.id)
        .order_by(Listing.created_at.desc())
        .all()
    )
    return {
        "listings": [
            {k: v for k, v in r.__dict__.items() if not k.startswith("_")} for r in rows
        ]
    }


@app.get("/api/marketplace/my-orders", tags=["Marketplace"], dependencies=[Depends(require_merchant)])
def my_orders(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Transaction)
        .filter(Transaction.merchant_id == user.id)
        .order_by(Transaction.created_at.desc())
        .all()
    )
    return {
        "transactions": [
            {k: v for k, v in r.__dict__.items() if not k.startswith("_")} for r in rows
        ]
    }


@app.get("/api/marketplace/route", tags=["Marketplace"])
def route(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float):
    try:
        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=false"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        route0 = (data.get("routes") or [{}])[0]
        return {
            "distance_km": round(float(route0.get("distance", 0)) / 1000, 2),
            "duration_minutes": round(float(route0.get("duration", 0)) / 60, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route lookup failed: {e}")


# ─────────────────────────────────────────
# SECTION 11 — PRICE ENDPOINTS
# ─────────────────────────────────────────

@app.get("/api/prices/commodities", tags=["Prices"])
def commodities():
    keys = sorted(price_models.keys())
    if not keys:
        keys = sorted({c.strip().lower() for c in available_commodities if str(c).strip()})
    return {"commodities": keys, "count": len(keys)}


@app.get("/api/prices/predict/{commodity}", tags=["Prices"])
def predict_price(commodity: str, days: int = 7):
    c = commodity.strip().lower()
    model = price_models.get(c)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Commodity '{c}' not found",
                "available_commodities": sorted(price_models.keys()),
            },
        )
    days = max(1, min(days, 365))

    try:
        future = model.make_future_dataframe(periods=int(days))
        forecast = model.predict(future).sort_values("ds")
        today = datetime.now(timezone.utc).date()

        today_rows = forecast[forecast["ds"].dt.date == today]
        if len(today_rows) > 0:
            current_price = float(today_rows.iloc[-1]["yhat"])
        else:
            current_price = (
                float(forecast.iloc[-(days + 1)]["yhat"])
                if len(forecast) > days
                else float(forecast.iloc[-1]["yhat"])
            )

        future_rows = forecast.tail(int(days))
        predictions = [
            {
                "date": row["ds"].date().isoformat(),
                "predicted_price": round(float(row["yhat"]), 2),
                "lower_bound": round(float(row["yhat_lower"]), 2) if "yhat_lower" in row else None,
                "upper_bound": round(float(row["yhat_upper"]), 2) if "yhat_upper" in row else None,
            }
            for _, row in future_rows.iterrows()
        ]

        max_row = future_rows.loc[future_rows["yhat"].idxmax()]
        max_price = float(max_row["yhat"])
        best_day = max_row["ds"].date().isoformat()
        price_increase_pct = (
            round(((max_price - current_price) / current_price) * 100, 1)
            if current_price > 0
            else 0
        )

        if max_price > current_price * 1.05:
            sell_recommendation = {
                "action": "WAIT",
                "reason": f"Price expected to rise {price_increase_pct}% to ₹{round(max_price, 2)}/kg",
                "best_day_to_sell": best_day,
                "expected_price_on_best_day": round(max_price, 2),
            }
        else:
            sell_recommendation = {
                "action": "SELL_NOW",
                "reason": "No significant price increase expected in next 7 days",
                "best_day_to_sell": best_day,
                "expected_price_on_best_day": round(max_price, 2),
            }

        return {
            "commodity": c,
            "current_price": round(current_price, 2),
            "currency": "INR/kg",
            "predictions": predictions,
            "sell_recommendation": sell_recommendation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Price prediction failed: {e}")


@app.get("/api/prices/current/{commodity}", tags=["Prices"])
def current_price(commodity: str):
    c = commodity.strip().lower()
    model = price_models.get(c)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Commodity '{c}' not found")
    try:
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future).sort_values("ds")
        last = forecast.iloc[-1]
        return {
            "commodity": c,
            "date": last["ds"].date().isoformat(),
            "price": round(float(last["yhat"]), 2),
            "currency": "INR/kg",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Current price lookup failed: {e}")


# ─────────────────────────────────────────
# SECTION 12 — FINANCE ENDPOINTS
# ─────────────────────────────────────────

PER_ACRE_COSTS = {
    "rice": 25000, "wheat": 20000, "tomato": 35000, "onion": 30000,
    "cotton": 40000, "maize": 18000, "sugarcane": 45000, "soybean": 22000,
    "groundnut": 28000, "potato": 32000,
}


def _scheme_matches(
    s: dict,
    state: Optional[str],
    crop_type: Optional[str],
    category: Optional[str],
) -> bool:
    try:
        st = (state or "").strip()
        ct = (crop_type or "").strip()
        cat = (category or "").strip()
        eligible_states = s.get("eligibility_states") or []
        eligible_crops = s.get("eligible_crops") or []
        eligible_categories = s.get("eligible_categories") or []
        state_ok = not st or "All India" in eligible_states or st in eligible_states
        crop_ok = not ct or "All Crops" in eligible_crops or ct in eligible_crops
        cat_ok = not cat or "All" in eligible_categories or cat in eligible_categories
        return bool(state_ok and crop_ok and cat_ok)
    except Exception:
        return False


class FinanceCalculateRequest(BaseModel):
    crop_type: str = Field(..., example="rice", description="Crop name e.g. rice, wheat, tomato")
    land_acres: float = Field(..., gt=0, example=2.0, description="Total land area in acres (> 0)")
    state: str = Field(..., example="Karnataka", description="Indian state name")
    category: str = Field("General", example="General", description="SC, ST, OBC, or General")


@app.post("/api/finance/calculate", tags=["Finance"])
def finance_calculate(payload: FinanceCalculateRequest):
    crop_type = payload.crop_type.strip().lower()
    land_acres = payload.land_acres
    state = payload.state.strip()
    category = payload.category.strip()

    cost_per_acre = float(PER_ACRE_COSTS.get(crop_type, 25000))
    total_investment = land_acres * cost_per_acre
    kcc_loan = total_investment * 0.80

    monthly_rate = 0.04 / 12
    n = 12
    emi = kcc_loan * monthly_rate * pow(1 + monthly_rate, n) / (pow(1 + monthly_rate, n) - 1)
    annual_interest = kcc_loan * 0.04

    matching = [
        s for s in schemes_data
        if _scheme_matches(s, state, payload.crop_type, category)
    ]

    total_subsidy = 0.0
    for s in matching:
        amt = s.get("subsidy_amount")
        pct = s.get("subsidy_percent")
        if isinstance(amt, (int, float)):
            total_subsidy += float(amt)
        if isinstance(pct, (int, float)):
            total_subsidy += total_investment * (float(pct) / 100.0)

    return {
        "crop_type": payload.crop_type,
        "land_acres": land_acres,
        "cost_per_acre": cost_per_acre,
        "total_investment": round(total_investment, 2),
        "kcc_loan_amount": round(kcc_loan, 2),
        "monthly_emi": round(float(emi), 2),
        "annual_interest": round(annual_interest, 2),
        "net_cost_after_subsidy": round(total_investment - total_subsidy, 2),
        "matching_schemes": matching,
        "schemes_found": len(matching),
        "total_subsidy_available": round(total_subsidy, 2),
    }


@app.get("/api/finance/schemes", tags=["Finance"])
def finance_schemes(
    state: Optional[str] = None,
    crop_type: Optional[str] = None,
    category: Optional[str] = None,
):
    matching = [s for s in schemes_data if _scheme_matches(s, state, crop_type, category)]
    return {"schemes": matching, "count": len(matching)}


@app.get("/api/finance/emi", tags=["Finance"])
def finance_emi(principal: float, annual_rate_percent: float, months: int):
    if principal <= 0 or annual_rate_percent < 0 or months <= 0:
        raise HTTPException(
            status_code=400,
            detail="principal must be > 0, annual_rate_percent >= 0, months > 0",
        )
    r = (annual_rate_percent / 100.0) / 12.0
    if r == 0:
        emi = principal / months
    else:
        emi = principal * r * pow(1 + r, months) / (pow(1 + r, months) - 1)
    total_payment = emi * months
    return {
        "principal": principal,
        "annual_rate_percent": annual_rate_percent,
        "months": months,
        "monthly_emi": round(float(emi), 2),
        "total_payment": round(float(total_payment), 2),
        "total_interest": round(float(total_payment - principal), 2),
    }


# ─────────────────────────────────────────
# SECTION 13 — MONITOR ENDPOINTS
# ─────────────────────────────────────────

@app.get("/api/monitor/overview", tags=["Monitor"], dependencies=[Depends(require_monitor)])
def monitor_overview(db: Session = Depends(get_db)):
    total_farmers = db.query(User).filter(User.role == "farmer").count()
    total_merchants = db.query(User).filter(User.role == "merchant").count()
    total_listings = db.query(Listing).count()
    total_transactions = db.query(Transaction).count()
    completed_transactions = db.query(Transaction).filter(Transaction.status == "completed").count()
    completed = db.query(Transaction).filter(Transaction.status == "completed").all()
    total_value = sum(float(tx.agreed_price) * float(tx.quantity_kg) for tx in completed)
    return {
        "total_farmers": total_farmers,
        "total_merchants": total_merchants,
        "total_listings": total_listings,
        "total_transactions": total_transactions,
        "completed_transactions": completed_transactions,
        "total_transaction_value": round(total_value, 2),
    }


@app.get("/api/monitor/transactions", tags=["Monitor"], dependencies=[Depends(require_monitor)])
def monitor_transactions(
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    txs = db.query(Transaction).all()
    out = []
    for tx in txs:
        if status_filter and tx.status != status_filter:
            continue
        listing = db.query(Listing).filter(Listing.id == tx.listing_id).first()
        farmer = db.query(User).filter(User.id == tx.farmer_id).first()
        merchant = db.query(User).filter(User.id == tx.merchant_id).first()
        out.append({
            "transaction_id": tx.id,
            "farmer_name": farmer.name if farmer else None,
            "merchant_name": merchant.name if merchant else None,
            "listing_crop_type": listing.crop_type if listing else None,
            "status": tx.status,
            "amount": round(float(tx.agreed_price) * float(tx.quantity_kg), 2),
            "created_at": tx.created_at,
        })
    out.sort(
        key=lambda x: x["created_at"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return {"transactions": out, "count": len(out)}


@app.get("/api/monitor/listings", tags=["Monitor"], dependencies=[Depends(require_monitor)])
def monitor_listings(db: Session = Depends(get_db)):
    rows = (
        db.query(Listing, User)
        .join(User, Listing.farmer_id == User.id)
        .order_by(Listing.created_at.desc())
        .all()
    )
    out = []
    for listing, farmer in rows:
        d = {k: v for k, v in listing.__dict__.items() if not k.startswith("_")}
        d["farmer_name"] = farmer.name
        out.append(d)
    return {"listings": out, "count": len(out)}


# ─────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )