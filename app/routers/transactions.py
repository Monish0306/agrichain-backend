from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ConfirmDeliveryIn(BaseModel):
    actor: str  # "farmer" or "merchant"
    confirmed: bool = True


@router.post("/{transaction_id}/confirm-delivery")
def confirm_delivery(transaction_id: str, body: ConfirmDeliveryIn):
    # Demo stub: update confirmations; if both true release escrow
    return {"transaction_id": transaction_id, "confirmed_by": body.actor, "payment_released": False}

