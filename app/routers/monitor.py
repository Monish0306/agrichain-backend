from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/overview")
def overview():
    return {
        "transactions_today": 0,
        "transactions_week": 0,
        "transactions_month": 0,
        "active_farmers": 0,
        "active_merchants": 0,
        "trade_volume_inr": 0,
    }


@router.get("/transactions")
def transactions(page: int = 1, page_size: int = 50):
    return {"page": page, "page_size": page_size, "items": []}


@router.get("/fraud-flags")
def fraud_flags():
    return {"items": []}


@router.get("/geographic-data")
def geographic_data():
    return {"items": []}

