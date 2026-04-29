from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import advisory, auth, finance, listings, monitor, prices, route, transactions


app = FastAPI(title="AgriChain Intelligence Platform API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(advisory.router, prefix="/api/advisory", tags=["advisory"])
app.include_router(listings.router, prefix="/api/listings", tags=["marketplace"])
app.include_router(transactions.router, prefix="/api/transactions", tags=["transactions"])
app.include_router(route.router, prefix="/api/route", tags=["route"])
app.include_router(prices.router, prefix="/api/prices", tags=["prices"])
app.include_router(finance.router, prefix="/api/finance", tags=["finance"])
app.include_router(monitor.router, prefix="/api/monitor", tags=["monitor"])

