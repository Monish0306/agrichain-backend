from __future__ import annotations

from datetime import date
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


ListingStatus = Literal["active", "sold", "expired"]


class ListingCreateIn(BaseModel):
    crop_type: str
    quantity_kg: float = Field(gt=0)
    quality_grade: Literal["A", "B", "C"]
    asking_price_per_kg: float = Field(gt=0)
    harvest_date: date
    gps_latitude: float
    gps_longitude: float
    photos: list[str] = []  # TODO: replace with upload/IPFS
    farmer_wallet_address: str | None = None


@router.post("/create")
def create_listing(body: ListingCreateIn):
    # Demo stub: persist to DB and deploy CropListing.sol
    listing_id = str(uuid4())
    return {
        "id": listing_id,
        "status": "active",
        "blockchain_contract_address": None,
        "blockchain_listing_hash": None,
        "qr_payload": {"type": "listing", "id": listing_id},
        "listing": body.model_dump(),
    }


@router.get("")
def list_listings(
    crop_type: str | None = None,
    quality_grade: str | None = None,
    district: str | None = None,
    min_qty: float | None = None,
    max_qty: float | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
):
    # Demo stub: return empty list
    return {"items": [], "filters": {"crop_type": crop_type, "quality_grade": quality_grade, "district": district}}


@router.get("/{listing_id}")
def get_listing(listing_id: str):
    return {"id": listing_id, "found": False}


class PlaceOrderIn(BaseModel):
    agreed_price_per_kg: float = Field(gt=0)
    merchant_wallet_address: str | None = None


@router.post("/{listing_id}/order")
def place_order(listing_id: str, body: PlaceOrderIn):
    # Demo stub: create transaction + deploy TradeEscrow.sol + hold funds
    tx_id = str(uuid4())
    return {"transaction_id": tx_id, "listing_id": listing_id, "escrow_contract_address": None, "status": "pending"}

