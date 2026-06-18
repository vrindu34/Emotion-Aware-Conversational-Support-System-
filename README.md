# Emotion-Aware Conversational Support System
### Neuro-Fuzzy Inference · Adaptive Responses · Spotify Integration

---

## Overview

This system detects emotions from natural language text using a custom
**ANFIS (Adaptive Neuro-Fuzzy Inference System)**, then generates
empathetic, context-aware responses with:

- Multi-label emotion detection with intensity scores
- Adaptive conversational responses tuned to emotion severity
- Spotify track recommendations matched to emotional state
- Personalised wellness suggestions per emotion
- Short-term emotional memory and trend tracking
- Full explainability via fuzzy rule firing logs

---

## Architecture

```
User Text
    │
    ▼
NLP Preprocessing (tokenise · clean · TF-IDF features)
    │
    ▼
ANFIS Engine  ← 5-layer neuro-fuzzy inference
    ├── Layer 1: Gaussian membership functions
    ├── Layer 2: Rule antecedent product T-norm
    ├── Layer 3: Normalised firing strengths
    ├── Layer 4: Takagi-Sugeno linear consequents
    └── Layer 5: Defuzzification (ANFIS × lexicon blend)
    │
    ├──► Response Strategy Engine  → adaptive text response
    ├──► Spotify Module            → emotion-matched tracks
    ├──► Suggestion Engine         → wellness actions
    └──► Emotion Memory            → trend tracking
    │
    ▼
FastAPI  →  Streamlit UI
```

---

## Project Structure

```
emotion_support/
├── backend/
│   ├── engines/
│   │   ├── anfis_engine.py      # Core ANFIS + fuzzy rules
│   │   └── response_engine.py   # Response templates
│   ├── modules/
│   │   ├── spotify_module.py    # Spotify API integration
│   │   ├── emotion_memory.py    # Sliding window memory
│   │   └── suggestion_engine.py # Wellness suggestions
│   └── main.py                  # FastAPI app
├── frontend/
│   └── app.py                   # Streamlit UI
├── .env.example                 # Copy → .env and fill credentials
├── requirements.txt
├── run.py                       # Launch script
└── README.md
```

---

## Setup

### 1. Clone / place this folder

```bash
cd emotion_support
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate.bat      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
SPOTIFY_CLIENT_ID=your_id_here
SPOTIFY_CLIENT_SECRET=your_secret_here
```

**Getting Spotify credentials:**
1. Go to https://developer.spotify.com/dashboard
2. Create a new app (any name/description)
3. Copy Client ID and Client Secret into `.env`

> The system works without Spotify credentials — it falls back to
> curated playlist links automatically.

### 5. Run the system

```bash
python run.py
```

Or separately:
```bash
# Terminal 1 — backend
python -m uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

### 6. Open in browser

| Service | URL |
|---------|-----|
| Chat UI | http://localhost:8501 |
| API docs (Swagger) | http://localhost:8000/docs |
| API (ReDoc) | http://localhost:8000/redoc |

---

## API Reference

### `POST /api/chat`
Main endpoint. Full pipeline — emotion detection, response, music, suggestions.

```json
{
  "text": "I feel really overwhelmed and anxious",
  "session_id": "optional-uuid-for-continuity",
  "include_spotify": true,
  "include_suggestions": true,
  "include_explanation": false
}
```

**Response:**
```json
{
  "session_id": "...",
  "response": "That sounds like a lot to deal with...",
  "emotions": [
    {"emotion": "anxiety", "score": 0.78, "level": "high"},
    {"emotion": "sadness", "score": 0.42, "level": "medium"}
  ],
  "dominant_emotion": "anxiety",
  "dominant_score": 0.78,
  "severity": "severe",
  "spotify": {
    "tracks": [...],
    "message": "Some calming music to help ease the tension:"
  },
  "wellness_suggestions": [
    "Try box breathing: inhale 4, hold 4, exhale 4..."
  ]
}
```

### `POST /api/analyze`
Emotion analysis only (no response generation, no memory write).

### `GET /api/history/{session_id}`
Retrieve all recorded turns for a session.

### `GET /api/trends/{session_id}`
Get emotion trend report (rising/falling/stable per emotion).

### `DELETE /api/session/{session_id}`
Clear session memory.

---

## ANFIS Technical Details

The engine uses a **5-layer adaptive neuro-fuzzy architecture**:

| Layer | Operation | Implementation |
|-------|-----------|----------------|
| L1 | Fuzzification | Gaussian MFs (3 levels: low/medium/high per emotion) |
| L2 | Rule antecedents | Product T-norm across rule inputs |
| L3 | Normalisation | Strength / sum of all strengths |
| L4 | Consequents | Takagi-Sugeno linear: w × feature_value |
| L5 | Defuzzification | ANFIS output blended with lexicon scores (α=0.55) |

**15 fuzzy rules** cover major emotion combinations:
- `R01`: High anxiety → calm supportive response
- `R03`: High anxiety + medium sadness → supportive + severe
- `R05`: High anger → de-escalation
- `R07`: High happiness → celebratory engagement
- ...and 11 more

**Lexicon layer** provides cold-start interpretability:
- 8 emotion categories × ~15–20 seed words each
- Negation detection (not, never, don't, etc.)
- Intensifier amplification (very, extremely, absolutely, etc.)

---

## Spotify Integration Details

The `SpotifyModule` uses:
- **Client Credentials flow** (no user login required)
- **Recommendations endpoint** with audio feature targets:
  - Valence, energy, tempo, acousticness per emotion
- **Genre seeds** per emotion (acoustic, ambient, pop, rock, etc.)
- **Feature blending** based on secondary emotion scores
- **Graceful fallback** to curated playlist links when credentials absent

Emotion → audio profile examples:
- Anxiety → low energy (0.30), low tempo (68 BPM), high acousticness — calming, not hype
- Happiness → high valence (0.88), high energy (0.80), danceable
- Sadness → low valence (0.20), slow tempo (72 BPM), acoustic

---

## Extending the System

**Add new emotions:** Edit `EMOTIONS` in `anfis_engine.py`, add lexicon entries,
and add new fuzzy rules.

**Add new response templates:** Edit `RESPONSE_TEMPLATES` in `response_engine.py`.

**Add new suggestions:** Edit `SUGGESTIONS` dict in `suggestion_engine.py`.

**Train the ANFIS weights:** Add a `train(X, y)` method to `ANFISEngine` using
scikit-learn's `LinearRegression` for the consequent layer and gradient descent
for the MF parameters.
