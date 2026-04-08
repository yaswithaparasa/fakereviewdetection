# 🛡 ReviewScan — Fake Review Detection System

94.5% accurate fake review classifier trained on 40,000 Amazon reviews (balanced: 20k genuine / 20k fake).

## Architecture

```
fake_reviews_dataset.csv  ──▶  train_model.py  ──▶  model.pkl
                                                        │
                              FastAPI (main.py) ◀───────┘
                                    │
                              React (App.jsx) ──▶  Browser
```

**Model**: TF-IDF (trigrams, 60k features) + Logistic Regression
- Accuracy: **94.5%** on 8,087-review test set
- Genuine F1: 0.94 | Fake F1: 0.95
- Latency: ~5ms per prediction

---

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

API docs: `http://localhost:8000/docs`

### 2. Frontend

```bash
# scaffold with Vite (if starting fresh)
npm create vite@latest frontend -- --template react
cd frontend
npm install

# copy App.jsx into src/
cp ../App.jsx src/App.jsx

npm run dev
```

Open `http://localhost:5173`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status + model info |
| POST | `/predict` | Classify single review |
| POST | `/batch-predict` | Classify up to 30 reviews |
| POST | `/analyze-url` | Scrape + classify Amazon product page |

### POST /predict

```json
// Request
{ "review": "This product is amazing! Best purchase ever!!!" }

// Response
{
  "prediction": "Fake",
  "confidence": 0.8821,
  "fake_probability": 0.8821,
  "genuine_probability": 0.1179,
  "risk_level": "High",
  "signals": [
    "Excessive exclamation marks detected",
    "Repetitive generic praise language"
  ],
  "description": "The model flags this review as likely fake (88% confidence) because...",
  "review": "This product is amazing!...",
  "latency_ms": 4.2
}
```

### POST /batch-predict

```json
// Request
[
  { "review": "Great product, works perfectly!" },
  { "review": "After 3 months of daily use, the battery still holds charge well. One minor issue is the button placement." }
]

// Response
{
  "total": 2,
  "fake": 1,
  "genuine": 1,
  "fake_rate": 0.5,
  "latency_ms": 6.1,
  "results": [ ... ]
}
```

### POST /analyze-url (requires requests + beautifulsoup4)

```json
// Request
{ "url": "https://www.amazon.com/dp/B0XXXXXXXX" }

// Response
{
  "site": "Amazon",
  "total": 8,
  "fake": 3,
  "genuine": 5,
  "fake_rate": 0.375,
  "overall_verdict": "Mostly Trustworthy",
  "results": [ ... ]
}
```

---

## Dataset

- **Source**: `fake_reviews_dataset.csv`
- **Size**: 40,432 reviews
- **Labels**: `OR` = fake/computer-generated, `CG` = genuine/human-written
- **Balance**: 50/50 (20,216 each)
- **Categories**: Electronics, Books, Home & Kitchen, Clothing, Toys, Movies, Tools, Sports, Pets, Kindle

## Model Training

To retrain from scratch:

```bash
python train_model.py
```

This runs `train_model.py` in the backend folder and outputs `model.pkl`.

---

## Project Structure

```
project/
├── backend/
│   ├── main.py              ← FastAPI app + all endpoints
│   ├── train_model.py       ← Model training script
│   ├── model.pkl            ← Trained model (generated)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx          ← Main React app
│   │   └── main.jsx         ← Entry point
│   ├── package.json
│   └── index.html
└── README.md
```