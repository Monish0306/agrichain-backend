## AgriChain Backend (FastAPI)

### Setup

```powershell
cd d:\Projects\AgriChain\agrichain-backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Create `.env` (already present in your repo locally) with at least:

- `DATABASE_URL`
- `SECRET_KEY`
- `OPENWEATHER_API_KEY` (optional for `/api/advisory/weather/*`)

### Run

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Notes

- `.env` is ignored via `.gitignore` to avoid committing secrets.
- SHAP is **not installed** by default because on Windows + Python 3.13 it commonly requires MSVC build tools. The advisory endpoint returns a SHAP-ready schema with placeholder contributions for now.

