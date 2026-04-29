from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/{farm_lat}/{farm_lng}/{destination_lat}/{destination_lng}")
def route(
    farm_lat: float,
    farm_lng: float,
    destination_lat: float,
    destination_lng: float,
):
    # Demo stub: integrate OSRM here.
    return {
        "distance_km": None,
        "duration_minutes": None,
        "polyline": None,
        "source": "demo",
        "from": {"lat": farm_lat, "lng": farm_lng},
        "to": {"lat": destination_lat, "lng": destination_lng},
    }

