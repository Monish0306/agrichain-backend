from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter

router = APIRouter()


@router.get("/current/{crop}/{mandi}")
def current_price(crop: str, mandi: str):
    # Demo stub: integrate AGMARKNET live.
    return {"crop": crop, "mandi": mandi, "price_per_kg": None, "source": "demo"}


@router.get("/history/{crop}/{mandi}")
def history(crop: str, mandi: str):
    # Demo stub: return empty series
    return {"crop": crop, "mandi": mandi, "days": 30, "items": []}


@router.get("/predict/{crop}/{mandi}")
def predict(crop: str, mandi: str):
    # Demo stub: return 7 days of fake predictions
    today = date.today()
    items = []
    for i in range(7):
        d = today + timedelta(days=i)
        items.append(
            {
                "date": d.isoformat(),
                "predicted_price": None,
                "confidence_lower": None,
                "confidence_upper": None,
            }
        )
    return {"crop": crop, "mandi": mandi, "items": items, "model_version": "demo"}


@router.get("/nearby/{crop}/{lat}/{lng}")
def nearby(crop: str, lat: float, lng: float):
    # Demo stub: integrate mandi proximity lookup
    return {"crop": crop, "lat": lat, "lng": lng, "items": []}

