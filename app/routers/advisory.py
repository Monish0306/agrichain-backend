from __future__ import annotations

from pathlib import Path

import httpx
import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings

router = APIRouter()


REPO_ROOT = Path(__file__).resolve().parents[3]
ML_DIR = REPO_ROOT / "agrichain_ml" / "models"


def _load_artifacts():
    model = joblib.load(ML_DIR / "crop_model.pkl")
    scaler = joblib.load(ML_DIR / "crop_scaler.pkl")
    label_encoder = joblib.load(ML_DIR / "crop_label_encoder.pkl")
    soil_encoder = joblib.load(ML_DIR / "soil_type_encoder.pkl")
    return model, scaler, label_encoder, soil_encoder


_artifacts = None


def _get_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()
    return _artifacts


class WeatherOut(BaseModel):
    source: str
    lat: float
    lng: float
    raw: dict


@router.get("/weather/{lat}/{lng}", response_model=WeatherOut)
async def weather(lat: float, lng: float):
    if not settings.openweather_api_key:
        raise HTTPException(status_code=500, detail="OPENWEATHER_API_KEY not configured")
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lng, "appid": settings.openweather_api_key, "units": "metric"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    return WeatherOut(source="openweathermap", lat=lat, lng=lng, raw=data)


class WeatherAlertOut(BaseModel):
    district: str
    active: bool
    severity: str | None = None
    message: str | None = None
    valid_until: str | None = None
    source: str = "demo"


@router.get("/weather/alerts/{district}", response_model=WeatherAlertOut)
def weather_alerts(district: str):
    # Demo stub: integrate IMD alerts here.
    return WeatherAlertOut(district=district, active=False)


class AdvisoryRecommendIn(BaseModel):
    lat: float
    lng: float
    soil_type: str
    season: str
    land_acres: float = Field(gt=0)
    water_source: str
    district: str | None = None
    state: str | None = None


class AdvisoryCropOut(BaseModel):
    crop: str
    expected_yield_per_acre_kg: float | None = None
    estimated_profit_at_msp_inr: float | None = None
    water_requirement: str | None = None
    risk_level: str | None = None
    growing_duration_days: int | None = None
    shap: list[dict]


class AdvisoryRecommendOut(BaseModel):
    recommended: list[AdvisoryCropOut]
    shap_top_features: list[dict]
    notes: list[str] = []


@router.post("/recommend", response_model=AdvisoryRecommendOut)
def recommend(body: AdvisoryRecommendIn):
    model, scaler, label_encoder, soil_encoder = _get_artifacts()

    # Best-effort feature vector: this depends on how your model was trained.
    # We encode soil_type; for missing features we use 0s.
    try:
        soil_val = soil_encoder.transform([body.soil_type])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unknown soil_type: {body.soil_type}") from e

    # Construct a small feature vector; adapt later once we inspect training pipeline.
    x = np.array([[soil_val, body.land_acres]], dtype=float)
    x_scaled = scaler.transform(x) if hasattr(scaler, "transform") else x

    proba = model.predict_proba(x_scaled)[0] if hasattr(model, "predict_proba") else None
    if proba is None:
        pred = model.predict(x_scaled)
        top_idx = [int(pred[0])]
        scores = [1.0]
    else:
        top_idx = list(np.argsort(proba)[::-1][:2].astype(int))
        scores = [float(proba[i]) for i in top_idx]

    crops = [str(label_encoder.inverse_transform([i])[0]) for i in top_idx]

    # SHAP is intentionally omitted on Windows/Python 3.13 demo environments where shap
    # frequently requires native build tools. We still return a SHAP-ready schema.
    shap_top = [
        {"feature": "soil_type", "contribution": 0.0},
        {"feature": "land_acres", "contribution": 0.0},
    ]

    recommended = []
    for crop, score in zip(crops, scores):
        recommended.append(
            AdvisoryCropOut(
                crop=crop,
                shap=[{"feature": f["feature"], "contribution": f["contribution"]} for f in shap_top],
            )
        )

    return AdvisoryRecommendOut(recommended=recommended, shap_top_features=shap_top, notes=["demo-grade output"])


class PesticideIn(BaseModel):
    crop_type: str
    humidity: float | None = None
    temperature_c: float | None = None
    rainfall_mm: float | None = None


@router.post("/pesticide")
def pesticide(_: PesticideIn):
    # Demo stub: integrate CIBRC verification and rules model here.
    return {
        "pesticide": "Mancozeb 75% WP",
        "dilution_ratio": "2g per liter",
        "best_time": "Evening",
        "re_entry_interval_hours": 24,
        "approx_price_inr": 350,
        "verified_cibrc": False,
        "source": "demo",
    }

