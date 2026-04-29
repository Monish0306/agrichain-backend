from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from jose import jwt
from pydantic import BaseModel, EmailStr, Field

from app.core.config import settings

router = APIRouter()


# Demo-grade OTP store (in-memory). Swap to Redis in production.
_otp_store: dict[str, dict[str, str | float]] = {}


Role = Literal["farmer", "merchant", "monitor"]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    role: Role


def _issue_token(subject: str, role: Role, expires_in: timedelta) -> TokenResponse:
    now = datetime.now(timezone.utc)
    exp = now + expires_in
    payload = {"sub": subject, "role": role, "iat": int(now.timestamp()), "exp": int(exp.timestamp())}
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
    return TokenResponse(access_token=token, expires_at=exp, role=role)


class FarmerRequestOtpIn(BaseModel):
    phone: str = Field(min_length=8, max_length=20)


@router.post("/farmer/request-otp")
def request_farmer_otp(body: FarmerRequestOtpIn):
    otp = f"{secrets.randbelow(1_000_000):06d}"
    _otp_store[body.phone] = {"otp": otp, "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()}
    # For demo we return OTP; in production, send SMS.
    return {"sent": True, "demo_otp": otp, "expires_in_seconds": 300}


class FarmerVerifyOtpIn(BaseModel):
    phone: str = Field(min_length=8, max_length=20)
    otp: str = Field(min_length=4, max_length=10)
    language_preference: str | None = None


@router.post("/farmer/verify-otp", response_model=TokenResponse)
def verify_farmer_otp(body: FarmerVerifyOtpIn):
    entry = _otp_store.get(body.phone)
    if not entry:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OTP not requested")
    if float(entry["expires_at"]) < datetime.now(timezone.utc).timestamp():
        _otp_store.pop(body.phone, None)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OTP expired")
    if str(entry["otp"]) != body.otp:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OTP")

    _otp_store.pop(body.phone, None)
    return _issue_token(
        subject=f"farmer:{body.phone}",
        role="farmer",
        expires_in=timedelta(hours=settings.farmer_access_token_hours),
    )


class MerchantRegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    gst_or_fssai: str | None = None


@router.post("/merchant/register")
def merchant_register(body: MerchantRegisterIn):
    # Stub for demo. Persist to DB in next iteration.
    return {"registered": True, "email": body.email, "verification": "pending"}


class MerchantLoginIn(BaseModel):
    email: EmailStr
    password: str


@router.post("/merchant/login", response_model=TokenResponse)
def merchant_login(body: MerchantLoginIn):
    # Stub for demo. Validate against DB in next iteration.
    return _issue_token(
        subject=f"merchant:{body.email}",
        role="merchant",
        expires_in=timedelta(hours=settings.merchant_access_token_hours),
    )


class MonitorLoginIn(BaseModel):
    username: str
    password: str


def _monitor_rate_limit_key(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    return f"monitor:{ip}"


_monitor_failures: dict[str, dict[str, float]] = {}


@router.post("/monitor/login", response_model=TokenResponse)
def monitor_login(body: MonitorLoginIn, request: Request):
    # Demo rate-limit: 5 failed attempts => 24h lock by IP.
    key = _monitor_rate_limit_key(request)
    now = datetime.now(timezone.utc).timestamp()
    st = _monitor_failures.get(key)
    if st and st.get("locked_until", 0) > now:
        raise HTTPException(status_code=429, detail="Too many failed attempts; try later")

    ok = (settings.monitor_username and settings.monitor_password) and (
        body.username == settings.monitor_username and body.password == settings.monitor_password
    )
    if not ok:
        if not st or st.get("reset_at", 0) <= now:
            st = {"count": 0, "reset_at": now + 60 * 10, "locked_until": 0}
        st["count"] = st.get("count", 0) + 1
        if st["count"] >= 5:
            st["locked_until"] = now + 60 * 60 * 24
        _monitor_failures[key] = st
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    _monitor_failures.pop(key, None)
    return _issue_token(
        subject=f"monitor:{body.username}",
        role="monitor",
        expires_in=timedelta(hours=settings.monitor_access_token_hours),
    )


class RefreshTokenIn(BaseModel):
    refresh_token: str


@router.post("/refresh-token")
def refresh_token(_: RefreshTokenIn):
    raise HTTPException(status_code=501, detail="Not implemented (demo)")


@router.post("/logout")
def logout():
    return {"logged_out": True}

