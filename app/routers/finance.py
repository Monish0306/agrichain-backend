from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class FinanceCalculateIn(BaseModel):
    crop: str
    land_acres: float = Field(gt=0)
    state: str
    category: str
    ownership_type: str | None = None


@router.post("/calculate")
def calculate(body: FinanceCalculateIn):
    # Demo-grade calculator (replace with real crop cost tables + MSP + mandi prices).
    seed = 1200 * body.land_acres
    fertilizer = 1800 * body.land_acres
    pesticide = 900 * body.land_acres
    labor = 2500 * body.land_acres
    irrigation = 800 * body.land_acres
    transport = 600 * body.land_acres
    total = seed + fertilizer + pesticide + labor + irrigation + transport

    # Rough yield and prices (demo placeholders)
    expected_yield_per_acre = 450
    total_kg = expected_yield_per_acre * body.land_acres
    msp = 10
    market = 12
    revenue_msp = total_kg * msp
    revenue_market = total_kg * market

    # Simple KCC EMI approximation (annual 4% -> monthly)
    annual_rate = 0.04
    months = 6
    r = annual_rate / 12
    emi = (total * r * (1 + r) ** months) / ((1 + r) ** months - 1) if r > 0 else total / months

    return {
        "inputs": body.model_dump(),
        "investment_breakdown": {
            "seed": round(seed, 2),
            "fertilizer": round(fertilizer, 2),
            "pesticide": round(pesticide, 2),
            "labor": round(labor, 2),
            "irrigation": round(irrigation, 2),
            "transport": round(transport, 2),
            "total": round(total, 2),
        },
        "revenue_projection": {
            "expected_yield_per_acre_kg": expected_yield_per_acre,
            "total_kg": round(total_kg, 2),
            "msp_price_per_kg": msp,
            "market_price_per_kg": market,
            "revenue_at_msp": round(revenue_msp, 2),
            "revenue_at_market": round(revenue_market, 2),
        },
        "loan": {
            "principal": round(total, 2),
            "annual_interest_rate": annual_rate,
            "months": months,
            "monthly_emi": round(float(emi), 2),
        },
        "schemes": [],
    }


@router.get("/schemes/{state}/{crop}/{category}")
def schemes(state: str, crop: str, category: str):
    # Demo stub: replace with DB-backed scheme finder
    return {"state": state, "crop": crop, "category": category, "items": []}

