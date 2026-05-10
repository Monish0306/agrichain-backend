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
DATA_DIR = os.path.join(BASE_DIR, "data")

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

    # ── SHAP explainer (optional) ──
    shap_path = os.path.join(ML_DIR, "crop_shap_explainer.pkl")
    if os.path.exists(shap_path):
        print(f"  [OK] Found crop_shap_explainer.pkl")
    else:
        print(f"  [INFO] crop_shap_explainer.pkl not found — will use rule-based SHAP approximation")

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
        schemes_path = os.path.join(DATA_DIR, "government_schemes.json")
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
    # Load lookup tables (groundwater + soil suitability)
    load_lookup_data()
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

    # ── SHAP feature importance (explainability) ──
    feature_names = ["Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"]
    feature_values = [
        payload.nitrogen, payload.phosphorous, payload.potassium,
        payload.temperature, payload.humidity, payload.ph, payload.rainfall,
    ]
    shap_explanation = []
    try:
        # Try real SHAP if explainer is loaded
        crop_shap_path = os.path.join(ML_DIR, "crop_shap_explainer.pkl")
        if os.path.exists(crop_shap_path):
            shap_explainer = joblib.load(crop_shap_path)
            shap_vals = shap_explainer.shap_values(scaled)
            # Use SHAP values for top crop class
            top_class_idx = int(top_idx[0])
            if isinstance(shap_vals, list):
                sv = shap_vals[top_class_idx][0] if top_class_idx < len(shap_vals) else shap_vals[0][0]
            else:
                sv = shap_vals[0]
            abs_vals = [abs(float(v)) for v in sv]
            total = sum(abs_vals) or 1.0
            shap_explanation = [
                {
                    "feature": feature_names[i],
                    "value": round(float(feature_values[i]), 2),
                    "importance_percent": round((abs_vals[i] / total) * 100, 1),
                    "impact": "positive" if float(sv[i]) > 0 else "negative",
                }
                for i in range(len(feature_names))
            ]
            shap_explanation.sort(key=lambda x: x["importance_percent"], reverse=True)
        else:
            # Rule-based SHAP approximation
            feature_weights = [0.20, 0.15, 0.15, 0.18, 0.12, 0.14, 0.06]
            total = sum(feature_weights)
            shap_explanation = [
                {
                    "feature": feature_names[i],
                    "value": round(float(feature_values[i]), 2),
                    "importance_percent": round((feature_weights[i] / total) * 100, 1),
                    "impact": "positive",
                }
                for i in range(len(feature_names))
            ]
            shap_explanation.sort(key=lambda x: x["importance_percent"], reverse=True)
    except Exception as shap_err:
        print(f"[WARN] SHAP failed: {shap_err}")
        shap_explanation = [
            {"feature": n, "value": round(float(v), 2), "importance_percent": round(100/7, 1)}
            for n, v in zip(feature_names, feature_values)
        ]

    # ── Soil suitability cross-check ──
    soil_compat = soil_suitability_data.get(soil_type, {})
    suitable_crops_for_soil = soil_compat.get("suitable_crops", [])
    for crop_rec in recommended_crops:
        crop_rec["soil_compatible"] = (
            crop_rec["name"].lower() in [c.lower() for c in suitable_crops_for_soil]
            if suitable_crops_for_soil else None
        )

    # ── Auto groundwater lookup by district ──
    groundwater_warning = None
    try:
        user_district = getattr(user, "district", None)
        if user_district:
            gw_match = next(
                (r for r in groundwater_data if r["district"].lower() == user_district.lower()),
                None
            )
            if gw_match and gw_match.get("category") in ("Critical", "Over-Exploited"):
                cat = gw_match["category"]
                groundwater_warning = {
                    "district": gw_match["district"],
                    "category": cat,
                    "message": GROUNDWATER_ADVICE.get(cat, {}).get("message", ""),
                    "icon": GROUNDWATER_ADVICE.get(cat, {}).get("icon", "⚠️"),
                }
    except Exception:
        pass

    # ── Groq translation if user prefers non-English ──
    user_language = getattr(user, "language", "english") or "english"
    advice_text = (
        f"Based on your soil and climate, {best_crop} is your best option. "
        f"Apply {fert} at {dosage} kg/acre for optimal yield."
    )
    if user_language != "english" and GROQ_API_KEY:
        translated_advice = _groq_translate(advice_text, user_language)
    else:
        translated_advice = advice_text

    return {
        "recommended_crops": recommended_crops,
        "fertilizer_recommendation": fertilizer_recommendation,
        "weather": weather_out,
        "farming_alerts": farming_alerts,
        "shap_explanation": shap_explanation,
        "soil_suitability": {
            "soil_type": soil_type,
            "suitable_crops": suitable_crops_for_soil[:8],
            "description": soil_compat.get("description", ""),
            "tip": soil_compat.get("tip", ""),
        },
        "advice_summary": advice_text,
        "advice_summary_translated": translated_advice,
        "user_language": user_language,
        "groundwater_warning": groundwater_warning,
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

# ─────────────────────────────────────────
# SECTION 14 — LOOKUP DATA LOADING
# ─────────────────────────────────────────

groundwater_data: list = []
soil_suitability_data: dict = {}


def load_lookup_data():
    global groundwater_data, soil_suitability_data

    # ── Groundwater ──
    gw_path = os.path.join(DATA_DIR, "groundwater_india.json")
    if os.path.exists(gw_path):
        with open(gw_path, "r", encoding="utf-8") as f:
            groundwater_data = json.load(f)
        print(f"[STARTUP] Loaded {len(groundwater_data)} groundwater district records")
    else:
        print(f"[STARTUP WARN] groundwater_india.json not found at {gw_path}")

    # ── Soil suitability ──
    ss_path = os.path.join(DATA_DIR, "soil_crop_suitability.json")
    if os.path.exists(ss_path):
        with open(ss_path, "r", encoding="utf-8") as f:
            soil_suitability_data = json.load(f)
        print(f"[STARTUP] Loaded {len(soil_suitability_data)} soil type records")
    else:
        print(f"[STARTUP WARN] soil_crop_suitability.json not found at {ss_path}")


# ─────────────────────────────────────────
# SECTION 15 — ADDITIONAL DB MODELS
# ─────────────────────────────────────────

class DiseaseReport(Base):
    __tablename__ = "disease_reports"
    id           = Column(String(36), primary_key=True, default=_uuid_str)
    farmer_id    = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    farm_id      = Column(String(36), ForeignKey("farms.id"), nullable=True)
    image_url    = Column(String(500), nullable=True)
    disease_detected = Column(String(200), nullable=True)
    confidence   = Column(Float, nullable=True)
    treatment_given  = Column(String(1000), nullable=True)
    severity     = Column(String(20), nullable=True)
    created_at   = Column(DateTime(timezone=True), nullable=False,
                          default=lambda: datetime.now(timezone.utc))


class WeatherAlert(Base):
    __tablename__ = "weather_alerts"
    id         = Column(String(36), primary_key=True, default=_uuid_str)
    district   = Column(String(80), nullable=False, index=True)
    state      = Column(String(80), nullable=True)
    alert_type = Column(String(80), nullable=False)
    severity   = Column(String(20), nullable=False)
    message    = Column(String(1000), nullable=False)
    sent_at    = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc))


class CropCalendar(Base):
    __tablename__ = "crop_calendar"
    id               = Column(String(36), primary_key=True, default=_uuid_str)
    crop_record_id   = Column(String(36), ForeignKey("crop_records.id"), nullable=False, index=True)
    farmer_id        = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    task_date        = Column(DateTime(timezone=True), nullable=False)
    task_type        = Column(String(80), nullable=False)
    task_description = Column(String(500), nullable=False)
    notified         = Column(Boolean, default=False)
    completed        = Column(Boolean, default=False)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id             = Column(String(36), primary_key=True, default=_uuid_str)
    user_id        = Column(String(36), nullable=False, index=True)
    action         = Column(String(200), nullable=False)
    ip_address     = Column(String(50), nullable=True)
    timestamp      = Column(DateTime(timezone=True), nullable=False,
                            default=lambda: datetime.now(timezone.utc))
    blockchain_tx  = Column(String(120), nullable=True)


# ─────────────────────────────────────────
# SECTION 16 — GROUNDWATER ENDPOINTS
# ─────────────────────────────────────────

GROUNDWATER_ADVICE = {
    "Safe":          {"color": "green",  "icon": "✅", "message": "Groundwater level is safe. Standard irrigation practices recommended."},
    "Semi-Critical": {"color": "yellow", "icon": "⚠️", "message": "Groundwater is semi-critical. Consider drip irrigation to reduce usage by 40%."},
    "Critical":      {"color": "orange", "icon": "🔴", "message": "Groundwater is CRITICAL. Switch to drought-tolerant crops. Avoid paddy cultivation."},
    "Over-Exploited":{"color": "red",    "icon": "🚨", "message": "Groundwater is OVER-EXPLOITED. Borewell drilling banned in this zone. Use only surface irrigation."},
}


@app.get("/api/groundwater/{district}", tags=["Groundwater"])
def get_groundwater(district: str, state: Optional[str] = None):
    """
    Returns groundwater level category for a district.
    Categories: Safe / Semi-Critical / Critical / Over-Exploited
    """
    district_clean = district.strip().title()
    match = None

    # Try exact match first
    for row in groundwater_data:
        if row["district"].lower() == district_clean.lower():
            if state is None or row["state"].lower() == state.strip().lower():
                match = row
                break

    # Fuzzy match if no exact
    if not match:
        for row in groundwater_data:
            if district_clean.lower() in row["district"].lower():
                match = row
                break

    if not match:
        return {
            "district": district_clean,
            "category": "Unknown",
            "level_mbgl": None,
            "advice": "District not found in database. Contact local groundwater board.",
            "color": "gray",
            "subsidy_tip": "Check PMKSY drip irrigation subsidy at pmksy.gov.in",
        }

    cat = match.get("category", "Safe")
    advice_info = GROUNDWATER_ADVICE.get(cat, GROUNDWATER_ADVICE["Safe"])

    return {
        "district": match["district"],
        "state": match["state"],
        "category": cat,
        "level_mbgl": match.get("level_mbgl"),
        "icon": advice_info["icon"],
        "color": advice_info["color"],
        "message": advice_info["message"],
        "subsidy_tip": (
            "You qualify for PMKSY drip irrigation subsidy (55% off). Apply at pmksy.gov.in"
            if cat in ("Semi-Critical", "Critical", "Over-Exploited")
            else "No immediate subsidy needed for irrigation."
        ),
        "recommended_crops": (
            ["Millet", "Groundnut", "Sorghum", "Cowpea", "Chickpea"]
            if cat in ("Critical", "Over-Exploited")
            else None
        ),
    }


@app.get("/api/groundwater", tags=["Groundwater"])
def list_groundwater(state: Optional[str] = None, category: Optional[str] = None):
    """List all groundwater districts, optionally filtered by state or category."""
    result = groundwater_data
    if state:
        result = [r for r in result if r.get("state", "").lower() == state.strip().lower()]
    if category:
        result = [r for r in result if r.get("category", "").lower() == category.strip().lower()]
    return {"districts": result, "count": len(result)}


# ─────────────────────────────────────────
# SECTION 17 — SOIL SUITABILITY ENDPOINTS
# ─────────────────────────────────────────

@app.get("/api/soil/suitability/{soil_type}", tags=["Soil"])
def get_soil_suitability(soil_type: str):
    """
    Returns crop suitability for a given soil type.
    Valid: Sandy, Clayey, Loamy, Black, Red
    """
    soil_clean = soil_type.strip().title()

    # Map common aliases
    aliases = {
        "Black Cotton": "Black",
        "Red Laterite": "Red",
        "Clay": "Clayey",
        "Sandy Loam": "Sandy",
        "Loam": "Loamy",
    }
    soil_clean = aliases.get(soil_clean, soil_clean)

    data = soil_suitability_data.get(soil_clean)
    if not data:
        return {
            "soil_type": soil_clean,
            "error": f"Soil type '{soil_clean}' not found",
            "valid_types": list(soil_suitability_data.keys()),
        }

    return {
        "soil_type": soil_clean,
        "description": data.get("description"),
        "suitable_crops": data.get("suitable_crops", []),
        "unsuitable_crops": data.get("unsuitable_crops", []),
        "ph_range": data.get("ph_range"),
        "irrigation_need": data.get("irrigation_need"),
        "tip": data.get("tip"),
    }


@app.get("/api/soil/all", tags=["Soil"])
def get_all_soil_types():
    """Returns all soil types with their crop suitability data."""
    return {
        "soil_types": [
            {
                "name": k,
                "description": v.get("description"),
                "suitable_crops": v.get("suitable_crops", []),
                "ph_range": v.get("ph_range"),
            }
            for k, v in soil_suitability_data.items()
        ],
        "count": len(soil_suitability_data),
    }


# ─────────────────────────────────────────
# SECTION 18 — WEATHER ENDPOINTS (ENHANCED)
# ─────────────────────────────────────────

FARMING_ALERT_RULES = [
    {"condition": "humidity_high",  "threshold": 80,  "severity": "warning",  "message": "High humidity — elevated fungal disease risk. Consider preventive fungicide spray."},
    {"condition": "humidity_vhigh", "threshold": 90,  "severity": "critical", "message": "Extreme humidity — do NOT spray pesticide. High chance of fungal disease outbreak."},
    {"condition": "temp_high",      "threshold": 40,  "severity": "critical", "message": "Extreme heat — irrigate in early morning only. Risk of heat stress on crops."},
    {"condition": "temp_low",       "threshold": 10,  "severity": "warning",  "message": "Low temperature — risk of frost. Cover young seedlings overnight."},
    {"condition": "rain_heavy",     "threshold": 50,  "severity": "warning",  "message": "Heavy rain forecast — do not spray pesticide. Wait until 2 days after rain stops."},
]


@app.get("/api/weather/forecast", tags=["Weather"])
def get_weather_forecast(lat: float, lon: float, district: Optional[str] = None):
    """
    Get 7-day weather forecast + groundwater status + farming alerts for a location.
    Uses OpenWeatherMap API if key is available, otherwise returns structured demo data.
    """
    weather_result = {}
    farming_alerts = []

    # ── Live weather from OpenWeatherMap ──
    if OPENWEATHER_API_KEY:
        try:
            url = (
                f"https://api.openweathermap.org/data/2.5/forecast"
                f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric&cnt=56"
            )
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            data = r.json()

            # Parse forecast list
            daily: dict = {}
            for item in data.get("list", []):
                date = item["dt_txt"].split(" ")[0]
                if date not in daily:
                    daily[date] = {
                        "date": date,
                        "temp_max": item["main"]["temp_max"],
                        "temp_min": item["main"]["temp_min"],
                        "humidity": item["main"]["humidity"],
                        "description": item["weather"][0]["description"],
                        "rain_mm": item.get("rain", {}).get("3h", 0),
                        "wind_kmh": round(item["wind"]["speed"] * 3.6, 1),
                    }
                else:
                    daily[date]["temp_max"] = max(daily[date]["temp_max"], item["main"]["temp_max"])
                    daily[date]["rain_mm"] += item.get("rain", {}).get("3h", 0)

            forecast_list = list(daily.values())[:7]

            # ── Generate farming alerts from forecast ──
            for day in forecast_list:
                h = day.get("humidity", 0)
                t = day.get("temp_max", 25)
                r_mm = day.get("rain_mm", 0)
                if h >= 90:
                    farming_alerts.append({"date": day["date"], "severity": "critical",
                        "message": f"Day {day['date']}: {FARMING_ALERT_RULES[1]['message']}"})
                elif h >= 80:
                    farming_alerts.append({"date": day["date"], "severity": "warning",
                        "message": f"Day {day['date']}: {FARMING_ALERT_RULES[0]['message']}"})
                if t >= 40:
                    farming_alerts.append({"date": day["date"], "severity": "critical",
                        "message": f"Day {day['date']}: {FARMING_ALERT_RULES[2]['message']}"})
                if t <= 10:
                    farming_alerts.append({"date": day["date"], "severity": "warning",
                        "message": f"Day {day['date']}: {FARMING_ALERT_RULES[4]['message']}"})
                if r_mm >= 50:
                    farming_alerts.append({"date": day["date"], "severity": "warning",
                        "message": f"Day {day['date']}: Heavy rain {round(r_mm,1)}mm — {FARMING_ALERT_RULES[3]['message']}"})

            weather_result = {
                "source": "OpenWeatherMap",
                "location": {"lat": lat, "lon": lon},
                "forecast": forecast_list,
                "farming_alerts": farming_alerts,
            }
        except Exception as e:
            weather_result = {"error": str(e), "source": "demo"}
    else:
        # Demo mode
        forecast_list = [
            {"date": f"Day {i+1}", "temp_max": 30 + i, "temp_min": 22,
             "humidity": 65 + i*2, "description": "partly cloudy", "rain_mm": 0, "wind_kmh": 12}
            for i in range(7)
        ]
        farming_alerts = [
            {"severity": "warning", "message": "Demo mode — add OPENWEATHER_API_KEY to Render env for live data"},
        ]
        weather_result = {"source": "demo", "forecast": forecast_list, "farming_alerts": farming_alerts}

    # ── Add groundwater info if district provided ──
    gw_info = None
    if district:
        gw_match = next(
            (r for r in groundwater_data if r["district"].lower() == district.strip().lower()), None
        )
        if gw_match:
            cat = gw_match.get("category", "Safe")
            gw_info = {
                "district": gw_match["district"],
                "category": cat,
                "level_mbgl": gw_match.get("level_mbgl"),
                "warning": GROUNDWATER_ADVICE.get(cat, {}).get("message"),
            }

    weather_result["groundwater"] = gw_info
    return weather_result


@app.get("/api/weather/current", tags=["Weather"])
def get_current_weather(lat: float, lon: float):
    """Get current weather conditions for a GPS location."""
    if not OPENWEATHER_API_KEY:
        return {
            "temp": 29.5, "humidity": 68, "description": "partly cloudy",
            "wind_kmh": 12, "pressure": 1012,
            "note": "Demo data — add OPENWEATHER_API_KEY env var for live data"
        }
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        w = r.json()
        return {
            "temp": round(w["main"]["temp"], 1),
            "feels_like": round(w["main"]["feels_like"], 1),
            "humidity": w["main"]["humidity"],
            "pressure": w["main"]["pressure"],
            "description": w["weather"][0]["description"],
            "wind_kmh": round(w["wind"]["speed"] * 3.6, 1),
            "visibility_km": round(w.get("visibility", 10000) / 1000, 1),
            "city": w.get("name", ""),
            "country": w.get("sys", {}).get("country", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather fetch failed: {e}")


# ─────────────────────────────────────────
# SECTION 19 — CROP CALENDAR ENDPOINTS
# ─────────────────────────────────────────

# Task templates per crop type (days from planting)
CROP_CALENDAR_TEMPLATES: dict = {
    "rice":      [
        (1,  "Soil Preparation", "Plough field to 15cm depth. Apply basal dose of NPK 120:60:60 kg/ha."),
        (3,  "Transplanting",    "Transplant 21-day-old seedlings at 20x15cm spacing."),
        (10, "Weed Control",     "Apply pre-emergence herbicide. Hand weed if needed."),
        (21, "Fertilizer",       "Apply top dressing — urea 60 kg/ha."),
        (35, "Pest Check",       "Check for stem borer and leaf folder. Apply chlorpyrifos if >10% infestation."),
        (50, "Water Management", "Maintain 5cm standing water. Drain 10 days before harvest."),
        (75, "Harvest Signal",   "Grain moisture at 20-25%. Golden colour indicates readiness."),
        (85, "Harvest",          "Harvest and thresh. Dry to 14% moisture before storage."),
    ],
    "wheat":     [
        (1,  "Soil Preparation", "Deep plough, apply FYM 10 tonnes/ha. Basal fertilizer NPK 120:60:40."),
        (7,  "Sowing",           "Sow at 100kg/ha seed rate. Depth: 5cm. Row spacing: 22cm."),
        (21, "First Irrigation", "Critical first irrigation (Crown Root Initiation stage)."),
        (35, "Fertilizer",       "Apply top dressing — split urea application 60 kg/ha."),
        (45, "Pest Monitoring",  "Watch for aphids and yellow rust. Spray propiconazole if rust appears."),
        (60, "Second Irrigation","Irrigation at heading stage is critical for grain filling."),
        (100,"Harvest",          "Harvest when grain moisture drops to 12-14%. Use combine if available."),
    ],
    "tomato":    [
        (1,  "Nursery",         "Prepare nursery bed. Sow seeds. Cover with fine soil + FYM."),
        (25, "Transplanting",   "Transplant 25-day seedlings at 60x45cm spacing. Water immediately."),
        (35, "Staking",         "Provide bamboo stakes. Tie plants. Encourage upward growth."),
        (42, "Fertilizer",      "Apply NPK 19:19:19 via drip or foliar spray."),
        (50, "Flower Check",    "Ensure pollination. Spray boron 0.3% for fruit set improvement."),
        (60, "Pest Control",    "Watch for leaf curl virus (whitefly vector). Apply imidacloprid."),
        (70, "First Harvest",   "Pick firm, pink-red fruits. Harvest every 4-5 days."),
        (90, "Final Harvest",   "Complete harvest. Remove crop residue to prevent disease carryover."),
    ],
    "cotton":    [
        (1,  "Soil Prep",       "Deep plough. Ridges at 90cm. Apply FYM + basal NPK 60:30:30 kg/ha."),
        (7,  "Sowing",          "Sow 2 seeds/hill at 90x60cm. Thin to 1 plant after germination."),
        (25, "Thinning",        "Remove weak plants. Apply 2% urea spray for early growth boost."),
        (40, "Fertilizer",      "Apply top dressing urea 30 kg/ha. Side-dress with K if deficient."),
        (55, "Pest Scout",      "Bollworm critical stage. Check 20 plants/acre. Spray if >5 eggs/plant."),
        (70, "Irrigation",      "Boll development stage — most critical. Maintain moisture."),
        (120,"Boll Opening",    "80% boll opening = harvest signal. Pick every 8-10 days."),
        (150,"Final Harvest",   "Complete picking. Separate grades A/B. Clean seeds for next season."),
    ],
    "groundnut": [
        (1,  "Soil Prep",       "Sandy loam preferred. Apply gypsum 400 kg/ha for calcium supplement."),
        (7,  "Sowing",          "Sow shelled seeds 5cm deep. 30x10cm spacing. Apply Rhizobium culture."),
        (20, "Earthing Up",     "Earth up soil around base. Supports pegging and pod formation."),
        (35, "Pest Check",      "Watch for leaf miner and thrips. Spray dimethoate 1ml/litre."),
        (50, "Peg Formation",   "Critical stage — no water stress. Irrigate if dry. Avoid waterlogging."),
        (80, "Maturity Check",  "Pull 3-4 plants. Inner pod surface should be dark. Shake — seeds rattle."),
        (100,"Harvest",         "Dig, shake off soil, dry in windrows 3-4 days before threshing."),
    ],
    "maize":     [
        (1,  "Soil Prep",       "Fine tilth. Apply FYM. Basal dose NPK 150:75:37.5 kg/ha."),
        (5,  "Sowing",          "Sow at 5cm depth. 60x20cm spacing. Two seeds/hill, thin later."),
        (15, "Gap Filling",     "Fill missing hills within 7 days of germination."),
        (30, "Top Dressing",    "Apply split urea — 75 kg/ha at knee-high stage."),
        (45, "Tasseling",       "Critical pollination stage — ensure adequate moisture."),
        (60, "Silking",         "Silk emerges — water stress at this stage cuts yield by 30%. Irrigate."),
        (80, "Maturity",        "Black layer formation on kernel base = physiological maturity."),
        (90, "Harvest",         "Dry to 15% moisture. Dehusk immediately after harvest."),
    ],
    "default":   [
        (1,  "Soil Preparation","Prepare field — plough, level, and apply organic matter."),
        (7,  "Planting",        "Plant seeds or seedlings at recommended spacing."),
        (21, "Fertilizer",      "Apply first dose of recommended fertilizer."),
        (35, "Weeding",         "Remove weeds. Conserve soil moisture with mulching."),
        (50, "Pest Check",      "Scout for pests and diseases. Apply control measures if needed."),
        (70, "Irrigation",      "Ensure adequate moisture during critical growth stages."),
        (90, "Harvest",         "Harvest at correct maturity. Follow post-harvest handling guidelines."),
    ],
}


class CreateCropRecordRequest(BaseModel):
    farm_id: str = Field(..., example="farm-uuid-here")
    crop_type: str = Field(..., example="tomato")
    planting_date: str = Field(..., example="2026-06-01", description="ISO date YYYY-MM-DD")
    notes: Optional[str] = Field(None, example="Planted in field A")


@app.post("/api/calendar/register-crop", tags=["Crop Calendar"],
          dependencies=[Depends(require_farmer)])
def register_crop_and_generate_calendar(
    payload: CreateCropRecordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Register a new crop and auto-generate a day-by-day task calendar from planting to harvest.
    """
    try:
        planting_dt = datetime.strptime(payload.planting_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="planting_date must be YYYY-MM-DD")

    # Create CropRecord
    record = CropRecord(
        farm_id=payload.farm_id,
        crop_type=payload.crop_type.strip().lower(),
        planting_date=planting_dt,
        notes=payload.notes,
    )
    db.add(record)
    db.flush()  # get record.id

    # Generate calendar tasks
    template = CROP_CALENDAR_TEMPLATES.get(
        payload.crop_type.strip().lower(),
        CROP_CALENDAR_TEMPLATES["default"],
    )

    tasks_created = []
    for day_offset, task_type, task_desc in template:
        task_date = planting_dt + timedelta(days=day_offset)
        task = CropCalendar(
            crop_record_id=record.id,
            farmer_id=user.id,
            task_date=task_date,
            task_type=task_type,
            task_description=task_desc,
            notified=False,
            completed=False,
        )
        db.add(task)
        tasks_created.append({
            "task_date": task_date.date().isoformat(),
            "task_type": task_type,
            "task_description": task_desc,
            "days_from_planting": day_offset,
        })

    db.commit()
    db.refresh(record)

    return {
        "crop_record_id": record.id,
        "crop_type": record.crop_type,
        "planting_date": payload.planting_date,
        "expected_harvest": tasks_created[-1]["task_date"] if tasks_created else None,
        "total_tasks": len(tasks_created),
        "calendar": tasks_created,
        "message": f"✅ Calendar generated with {len(tasks_created)} tasks for {payload.crop_type}",
    }


@app.get("/api/calendar/my-tasks", tags=["Crop Calendar"],
         dependencies=[Depends(require_farmer)])
def get_my_tasks(
    upcoming_days: int = 7,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns upcoming tasks for the farmer in the next N days."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=upcoming_days)

    tasks = (
        db.query(CropCalendar)
        .filter(
            CropCalendar.farmer_id == user.id,
            CropCalendar.task_date >= now,
            CropCalendar.task_date <= cutoff,
            CropCalendar.completed == False,
        )
        .order_by(CropCalendar.task_date)
        .all()
    )

    return {
        "upcoming_tasks": [
            {
                "id": t.id,
                "task_date": t.task_date.date().isoformat(),
                "task_type": t.task_type,
                "task_description": t.task_description,
                "days_away": (t.task_date.date() - now.date()).days,
                "completed": t.completed,
            }
            for t in tasks
        ],
        "count": len(tasks),
        "period_days": upcoming_days,
    }


@app.post("/api/calendar/complete-task/{task_id}", tags=["Crop Calendar"],
          dependencies=[Depends(require_farmer)])
def complete_task(
    task_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark a crop calendar task as completed."""
    task = db.query(CropCalendar).filter(
        CropCalendar.id == task_id,
        CropCalendar.farmer_id == user.id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.completed = True
    db.commit()
    return {"message": "Task marked as completed", "task_id": task_id}


@app.get("/api/calendar/full-calendar/{crop_record_id}", tags=["Crop Calendar"],
         dependencies=[Depends(require_farmer)])
def get_full_calendar(
    crop_record_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns the full calendar for a specific crop registration."""
    tasks = (
        db.query(CropCalendar)
        .filter(
            CropCalendar.crop_record_id == crop_record_id,
            CropCalendar.farmer_id == user.id,
        )
        .order_by(CropCalendar.task_date)
        .all()
    )
    if not tasks:
        raise HTTPException(status_code=404, detail="No calendar found for this crop record")

    now = datetime.now(timezone.utc).date()
    return {
        "crop_record_id": crop_record_id,
        "tasks": [
            {
                "id": t.id,
                "task_date": t.task_date.date().isoformat(),
                "task_type": t.task_type,
                "task_description": t.task_description,
                "completed": t.completed,
                "status": (
                    "completed" if t.completed
                    else "overdue" if t.task_date.date() < now
                    else "today" if t.task_date.date() == now
                    else "upcoming"
                ),
            }
            for t in tasks
        ],
        "total": len(tasks),
        "completed": sum(1 for t in tasks if t.completed),
        "overdue": sum(1 for t in tasks if not t.completed and t.task_date.date() < now),
    }


# ─────────────────────────────────────────
# SECTION 20 — DISEASE DETECTION ENDPOINT
# ─────────────────────────────────────────

# Disease treatment database
DISEASE_TREATMENTS: dict = {
    "Early Blight":        {"treatment": "Apply Mancozeb 75% WP at 2g/litre. Spray every 7 days. Remove infected leaves.", "severity": "moderate"},
    "Late Blight":         {"treatment": "Apply Metalaxyl + Mancozeb at 2.5g/litre. Critical — act within 24 hours.", "severity": "critical"},
    "Bacterial Spot":      {"treatment": "Spray copper oxychloride 3g/litre. Avoid overhead irrigation.", "severity": "moderate"},
    "Leaf Mold":           {"treatment": "Improve ventilation. Apply chlorothalonil 2g/litre in evening.", "severity": "low"},
    "Septoria Leaf Spot":  {"treatment": "Remove lower infected leaves. Apply difenoconazole 1ml/litre.", "severity": "moderate"},
    "Spider Mites":        {"treatment": "Apply abamectin 0.5ml/litre. Spray leaf undersides. Repeat after 5 days.", "severity": "moderate"},
    "Target Spot":         {"treatment": "Apply azoxystrobin 1ml/litre. Ensure good air circulation.", "severity": "low"},
    "Mosaic Virus":        {"treatment": "No cure — remove infected plants. Control aphid/whitefly vectors with imidacloprid.", "severity": "critical"},
    "Yellow Curl Virus":   {"treatment": "Remove infected plants. Spray imidacloprid 0.5ml/litre for whitefly control.", "severity": "critical"},
    "Healthy":             {"treatment": "No disease detected. Continue regular monitoring every 7 days.", "severity": "none"},
    "Powdery Mildew":      {"treatment": "Apply sulphur dust 25 kg/ha or wettable sulphur 3g/litre. Spray in cool evening.", "severity": "moderate"},
    "Downy Mildew":        {"treatment": "Apply metalaxyl 1g + mancozeb 2g per litre. Avoid leaf wetting.", "severity": "high"},
    "Rust":                {"treatment": "Apply propiconazole 1ml/litre or tebuconazole 1ml/litre. Act at first sign.", "severity": "high"},
    "Leaf Blight":         {"treatment": "Apply carbendazim 1g/litre. Remove crop debris. Improve drainage.", "severity": "moderate"},
    "Black Rot":           {"treatment": "Remove infected plant parts. Apply copper-based fungicide. Avoid wounding plants.", "severity": "high"},
}


@app.post("/api/advisory/disease-detect", tags=["Advisory"])
async def detect_disease(request: Request):
    """
    AI crop disease detection endpoint.
    If ONNX model is loaded — runs real inference.
    Otherwise — returns intelligent demo response based on filename/metadata.
    """
    from fastapi import UploadFile, File
    import io

    try:
        form = await request.form()
        file = form.get("file")

        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded. Send image as 'file' field.")

        img_bytes = await file.read()
        filename = getattr(file, "filename", "crop.jpg").lower()

        # Try ONNX inference if model available
        onnx_path = os.path.join(ML_DIR, "disease_model.onnx")
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                from PIL import Image as PILImage

                session = ort.InferenceSession(onnx_path)
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
                arr = np.array(img).astype(np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]

                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: arr})
                probs = outputs[0][0]
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])

                # Load class names
                classes_path = os.path.join(ML_DIR, "disease_classes.json")
                if os.path.exists(classes_path):
                    with open(classes_path) as cf:
                        classes = json.load(cf)
                    disease_name = classes[top_idx] if top_idx < len(classes) else f"Class_{top_idx}"
                else:
                    disease_name = f"Disease_Class_{top_idx}"

                disease_clean = disease_name.replace("_", " ").replace("  ", " ").strip().title()
                treatment_info = DISEASE_TREATMENTS.get(disease_clean, {
                    "treatment": "Consult your local KVK (Krishi Vigyan Kendra) for treatment advice.",
                    "severity": "unknown"
                })

                return {
                    "disease": disease_clean,
                    "confidence": round(confidence, 4),
                    "confidence_percent": round(confidence * 100, 1),
                    "treatment": treatment_info["treatment"],
                    "severity": treatment_info["severity"],
                    "model": "YOLOv8-ONNX",
                    "mode": "live_inference",
                }
            except Exception as onnx_err:
                print(f"[WARN] ONNX inference failed: {onnx_err}")

        # ── Demo mode — smart response based on image size ──
        img_size_kb = len(img_bytes) / 1024
        # Rotate through demo diseases for variety
        demo_diseases = [
            ("Early Blight", 0.94),
            ("Powdery Mildew", 0.87),
            ("Late Blight", 0.91),
            ("Healthy", 0.97),
            ("Rust", 0.83),
        ]
        # Pick based on file size modulo for deterministic demo
        idx = int(img_size_kb) % len(demo_diseases)
        demo_disease, demo_conf = demo_diseases[idx]
        treatment_info = DISEASE_TREATMENTS.get(demo_disease, {
            "treatment": "Consult local KVK for advice.", "severity": "moderate"
        })

        return {
            "disease": demo_disease,
            "confidence": demo_conf,
            "confidence_percent": round(demo_conf * 100, 1),
            "treatment": treatment_info["treatment"],
            "severity": treatment_info["severity"],
            "model": "Demo",
            "mode": "demo — train YOLOv8 in Colab and save disease_model.onnx to ml_models/ for live inference",
            "image_size_kb": round(img_size_kb, 1),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease detection failed: {str(e)}")


# ─────────────────────────────────────────
# SECTION 21 — AUDIT LOG ENDPOINT
# ─────────────────────────────────────────

@app.get("/api/monitor/audit-log", tags=["Monitor"],
         dependencies=[Depends(require_monitor)])
def get_audit_log(limit: int = 100, db: Session = Depends(get_db)):
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "logs": [
            {
                "id": l.id,
                "user_id": l.user_id,
                "action": l.action,
                "ip_address": l.ip_address,
                "timestamp": l.timestamp,
                "blockchain_tx": l.blockchain_tx,
            }
            for l in logs
        ],
        "count": len(logs),
    }


# ─────────────────────────────────────────
# ENTRYPOINT

# ─────────────────────────────────────────
# SECTION 22 — GROQ MULTILINGUAL AI
# ─────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

SUPPORTED_LANGUAGES = {
    "english":   "English",
    "tamil":     "Tamil (தமிழ்)",
    "hindi":     "Hindi (हिंदी)",
    "kannada":   "Kannada (ಕನ್ನಡ)",
    "telugu":    "Telugu (తెలుగు)",
    "marathi":   "Marathi (मराठी)",
    "gujarati":  "Gujarati (ગુજરાતી)",
    "punjabi":   "Punjabi (ਪੰਜਾਬੀ)",
    "bengali":   "Bengali (বাংলা)",
    "malayalam": "Malayalam (മലയാളം)",
    "odia":      "Odia (ଓଡ଼ିଆ)",
}


def _groq_translate(text: str, target_language: str) -> str:
    """Call Groq API to translate text to target language."""
    if not GROQ_API_KEY:
        return text  # Return original if no key

    lang_display = SUPPORTED_LANGUAGES.get(target_language.lower(), target_language.title())

    try:
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are an agricultural translator. Translate the following farming advice "
                        f"to {lang_display}. Keep numbers, crop names, and chemical names as-is. "
                        f"Use simple rural language that a farmer would understand. "
                        f"Return ONLY the translated text, nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "max_tokens": 1024,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        r = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[WARN] Groq translation failed: {e}")
        return text  # Fallback to original


class TranslateRequest(BaseModel):
    text: str = Field(..., example="Apply DAP fertilizer at 50 kg per acre for best yield.")
    target_language: str = Field(..., example="tamil",
        description="One of: english, tamil, hindi, kannada, telugu, marathi, gujarati, punjabi, bengali, malayalam, odia")


@app.post("/api/language/translate", tags=["Language"])
@limiter.limit("30/minute")
def translate_text(request: Request, payload: TranslateRequest):
    """
    Translate any farming advisory text to the farmer's regional language.
    Uses Groq Llama 3.3 70B — 6,000 tokens/minute free.
    """
    lang = payload.target_language.strip().lower()

    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Language '{lang}' not supported",
                "supported": list(SUPPORTED_LANGUAGES.keys()),
            },
        )

    if lang == "english":
        return {
            "original": payload.text,
            "translated": payload.text,
            "language": "english",
            "note": "Already in English",
        }

    if not GROQ_API_KEY:
        return {
            "original": payload.text,
            "translated": payload.text,
            "language": lang,
            "note": "GROQ_API_KEY not set — translation disabled. Add it to Render env vars.",
            "groq_status": "disabled",
        }

    translated = _groq_translate(payload.text, lang)
    return {
        "original": payload.text,
        "translated": translated,
        "language": lang,
        "language_display": SUPPORTED_LANGUAGES[lang],
        "model": GROQ_MODEL,
        "groq_status": "active",
    }


class AdvisoryTranslateRequest(BaseModel):
    advisory_result: dict = Field(..., description="Full JSON response from /api/advisory/recommend")
    target_language: str = Field(..., example="tamil")


@app.post("/api/language/translate-advisory", tags=["Language"])
def translate_advisory(payload: AdvisoryTranslateRequest):
    """
    Translate the full advisory recommendation output to farmer's language.
    Pass the entire response from /api/advisory/recommend and get it back translated.
    """
    lang = payload.target_language.strip().lower()
    if lang == "english" or lang not in SUPPORTED_LANGUAGES:
        return payload.advisory_result

    result = dict(payload.advisory_result)

    # Translate advice_summary
    if result.get("advice_summary"):
        result["advice_summary"] = _groq_translate(result["advice_summary"], lang)

    # Translate farming alerts
    if result.get("farming_alerts"):
        for alert in result["farming_alerts"]:
            if alert.get("message"):
                alert["message"] = _groq_translate(alert["message"], lang)

    # Translate fertilizer name
    if result.get("fertilizer_recommendation", {}).get("name"):
        fert = result["fertilizer_recommendation"]
        fert["name_translated"] = _groq_translate(fert["name"], lang)

    result["language"] = lang
    result["language_display"] = SUPPORTED_LANGUAGES.get(lang, lang)
    return result


@app.get("/api/language/supported", tags=["Language"])
def get_supported_languages():
    """Returns list of all supported languages for translation."""
    return {
        "languages": [
            {"code": k, "display": v}
            for k, v in SUPPORTED_LANGUAGES.items()
        ],
        "groq_active": bool(GROQ_API_KEY),
        "model": GROQ_MODEL if GROQ_API_KEY else "not configured",
    }


# ─────────────────────────────────────────
# SECTION 23 — USER PROFILE UPDATE
# ─────────────────────────────────────────

class UpdateProfileRequest(BaseModel):
    name:     Optional[str] = Field(None, example="Ravi Kumar")
    language: Optional[str] = Field(None, example="tamil")
    state:    Optional[str] = Field(None, example="Tamil Nadu")
    district: Optional[str] = Field(None, example="Cuddalore")


@app.patch("/api/auth/profile", tags=["Auth"])
def update_profile(
    payload: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update farmer/merchant profile — name, language, location."""
    if payload.name:
        user.name = payload.name.strip()
    if payload.language:
        lang = payload.language.strip().lower()
        if lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Language must be one of: {list(SUPPORTED_LANGUAGES.keys())}",
            )
        user.language = lang
    if payload.state:
        user.state = payload.state.strip()
    if payload.district:
        user.district = payload.district.strip()

    db.commit()
    db.refresh(user)
    return {
        "id": user.id,
        "name": user.name,
        "language": user.language,
        "state": user.state,
        "district": user.district,
        "message": "Profile updated successfully",
    }


# ─────────────────────────────────────────
# SECTION 24 — FARM MANAGEMENT ENDPOINTS
# ─────────────────────────────────────────

class CreateFarmRequest(BaseModel):
    gps_lat:    float = Field(..., example=11.7)
    gps_lon:    float = Field(..., example=79.7)
    area_acres: float = Field(..., gt=0, example=2.5)
    soil_type:  str   = Field(..., example="Loamy")
    district:   str   = Field(..., example="Cuddalore")
    state:      str   = Field(..., example="Tamil Nadu")


@app.post("/api/farms", tags=["Farms"], dependencies=[Depends(require_farmer)])
def create_farm(
    payload: CreateFarmRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Register a new farm for the logged-in farmer."""
    valid_soils = {"Sandy", "Clayey", "Loamy", "Black", "Red"}
    if payload.soil_type not in valid_soils:
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid soil_type", "valid_options": sorted(valid_soils)},
        )

    farm = Farm(
        farmer_id=user.id,
        gps_lat=payload.gps_lat,
        gps_lon=payload.gps_lon,
        area_acres=payload.area_acres,
        soil_type=payload.soil_type,
        district=payload.district.strip(),
        state=payload.state.strip(),
    )
    db.add(farm)
    db.commit()
    db.refresh(farm)

    # Auto-fetch groundwater info for this farm's district
    gw = next(
        (r for r in groundwater_data if r["district"].lower() == payload.district.strip().lower()),
        None,
    )
    result = {k: v for k, v in farm.__dict__.items() if not k.startswith("_")}
    if gw:
        result["groundwater_category"] = gw.get("category")
        result["groundwater_warning"]  = GROUNDWATER_ADVICE.get(
            gw["category"], {}
        ).get("message")
    return result


@app.get("/api/farms", tags=["Farms"], dependencies=[Depends(require_farmer)])
def get_my_farms(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all farms registered by the logged-in farmer."""
    farms = db.query(Farm).filter(Farm.farmer_id == user.id).all()
    result = []
    for farm in farms:
        d = {k: v for k, v in farm.__dict__.items() if not k.startswith("_")}
        # Attach groundwater info
        gw = next(
            (r for r in groundwater_data if r["district"].lower() == farm.district.lower()),
            None,
        )
        if gw:
            d["groundwater_category"] = gw.get("category")
        # Attach soil suitability
        ss = soil_suitability_data.get(farm.soil_type, {})
        d["suitable_crops"] = ss.get("suitable_crops", [])
        result.append(d)
    return {"farms": result, "count": len(result)}


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