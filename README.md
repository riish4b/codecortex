# CodeCortex Backend

AI-powered document verification and identity authentication system.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Frontend (React)                │
│   Upload Docs → WebSocket → View Results        │
└──────────────────────┬──────────────────────────┘
                       │ HTTP + WebSocket
┌──────────────────────▼──────────────────────────┐
│              Backend (FastAPI)                   │
│                                                  │
│  ┌──────────────┐  ┌─────────────────────────┐  │
│  │ ML Pipeline  │  │    AI Service (OpenRouter)│  │
│  │ • ELA        │  │    • Risk Assessment      │  │
│  │ • MRZ OCR    │  │    • Document Analysis    │  │
│  │ • Face Match │  │    • Report Generation    │  │
│  │ • Fingerprint│  │    • Visa/NID Verify      │  │
│  │ • Iris Scan  │  │                           │  │
│  │ • Hologram   │  │    Model: Gemini 2.5 Flash│  │
│  │ • Tamper Det │  │                           │  │
│  └──────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Tech Stack

- **Runtime:** Python 3.13
- **Framework:** FastAPI + Uvicorn
- **Communication:** WebSocket (real-time)
- **ML:** OpenCV, Pillow, Tesseract, scikit-learn
- **AI:** OpenRouter (Google Gemini 2.5 Flash)

## Setup

### Prerequisites
- Python 3.10+
- Tesseract OCR installed on system

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Server starts at `http://0.0.0.0:8000`

### Test AI Integration

```bash
curl http://localhost:8000/api/test-ai
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/verify` | Upload documents for verification |
| WS | `/ws/verify/{tracking_id}` | Real-time verification pipeline |
| GET | `/api/test-ai` | Test OpenRouter AI connection |

## Verification Pipeline

1. **ELA Forensics** — JPEG compression analysis for manipulation detection
2. **MRZ Extraction** — Tesseract OCR for passport machine-readable zone
3. **Facial Biometrics** — Face similarity via OpenCV DNN / histogram comparison
4. **Fingerprint Matching** — ORB minutiae feature extraction
5. **Iris Scan** — Gabor filter texture pattern matching
6. **NFC Digital Footprint** — eMRTD chip verification
7. **Hologram Detection** — HSV color variance OVD analysis
8. **INTERPOL Database** — Criminal record & watchlist check
9. **AI Visa Verification** — Gemini 2.5 Flash authenticity analysis
10. **AI National ID Verification** — Gemini 2.5 Flash document check
11. **AI Risk Aggregation** — AI-powered risk score with reasoning
12. **AI Report Generation** — Full verification report with recommendations

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Hardcoded (dev) |
| `REDIS_URL` | Redis broker for Celery | `redis://localhost:6379/0` |

## Project Structure

```
backend/
├── main.py              # FastAPI app + WebSocket pipeline
├── requirements.txt     # Python dependencies
├── worker.py            # Celery task queue (optional)
├── scaffold.py          # Project scaffolding
└── services/
    ├── ai_service.py    # OpenRouter AI integration
    └── ml_pipeline.py   # Computer vision models
```
