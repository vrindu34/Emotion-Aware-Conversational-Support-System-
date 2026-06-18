"""
main.py
───────
FastAPI application entry point.

Endpoints:
  POST /api/chat          – Main chat endpoint
  POST /api/analyze       – Emotion analysis only (no response)
  GET  /api/history/{sid} – Session emotion history
  GET  /api/trends/{sid}  – Emotional trend report
  DELETE /api/session/{sid} – Clear session memory
  GET  /health            – Health check
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engines.anfis_engine import get_engine
from .engines.response_engine import get_response_engine
from .modules.emotion_memory import get_memory
from .modules.spotify_module import get_spotify_module
from .modules.suggestion_engine import get_suggestion_engine

# ──────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emotion-Aware Conversational Support System",
    description="Neuro-fuzzy emotion detection with adaptive empathetic responses.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="User message text")
    session_id: Optional[str] = Field(None, description="Session ID for memory continuity")
    include_spotify: bool = Field(True, description="Include Spotify track suggestions")
    include_suggestions: bool = Field(True, description="Include wellness suggestions")
    include_explanation: bool = Field(False, description="Include fuzzy rule explanation")


class EmotionScore(BaseModel):
    emotion: str
    score: float
    level: str   # 'low' | 'medium' | 'high'


class FiredRule(BaseModel):
    rule_id: str
    strength: float
    description: str


class SpotifyTrack(BaseModel):
    name: str
    artist: str
    album: str
    preview_url: Optional[str]
    spotify_url: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    emotions: List[EmotionScore]
    dominant_emotion: str
    dominant_score: float
    severity: str
    response_label: str
    spotify: Optional[Dict[str, Any]]
    wellness_suggestions: Optional[List[str]]
    crisis_note: Optional[str]
    explanation: Optional[str]
    trend_note: Optional[str]
    fired_rules: Optional[List[FiredRule]]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _emotion_level(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _scores_to_list(scores: Dict[str, float]) -> List[EmotionScore]:
    return sorted(
        [
            EmotionScore(
                emotion=em,
                score=round(score, 3),
                level=_emotion_level(score),
            )
            for em, score in scores.items()
        ],
        key=lambda x: -x.score,
    )


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main endpoint. Processes user text through the full pipeline:
    ANFIS → Response → Spotify → Suggestions → Memory → Response.
    """
    session_id = req.session_id or str(uuid.uuid4())

    # ── Retrieve conversation history ──
    memory = get_memory()
    history = memory.get_history(session_id)

    # ── ANFIS inference ──
    engine = get_engine()
    result = engine.infer(req.text)

    # ── Trend report (before recording this turn) ──
    trend = memory.get_trend_report(session_id)
    trend_note = trend.context_note or None

    # ── Generate response — pass user text for context-aware matching ──
    response_engine = get_response_engine()
    response_text = response_engine.generate(result, user_text=req.text, conversation_history=history)

    # ── Spotify ──
    spotify_data = None
    if req.include_spotify:
        spotify_mod = get_spotify_module()
        spotify_data = spotify_mod.get_tracks_for_emotion(
            result.dominant_emotion,
            result.emotion_scores,
        )

    # ── Wellness suggestions ──
    wellness_data = None
    crisis_note = None
    if req.include_suggestions:
        suggestion_eng = get_suggestion_engine()
        sugg = suggestion_eng.get_suggestions(result)
        wellness_data = sugg.get("wellness_actions")
        crisis_note = sugg.get("crisis_note")

    # ── Record in memory ──
    memory.record(
        session_id=session_id,
        user_text=req.text,
        emotion_scores=result.emotion_scores,
        dominant_emotion=result.dominant_emotion,
        dominant_score=result.dominant_score,
        response_label=result.response_label,
        severity=result.severity,
    )

    # ── Build fired rules for response ──
    fired_rules_out = None
    if req.include_explanation:
        from .engines.anfis_engine import FUZZY_RULES
        fired_rules_out = [
            FiredRule(
                rule_id=rid,
                strength=round(strength, 4),
                description=next((r.description for r in FUZZY_RULES if r.rule_id == rid), ""),
            )
            for rid, strength in result.fired_rules
        ]

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        emotions=_scores_to_list(result.emotion_scores),
        dominant_emotion=result.dominant_emotion,
        dominant_score=round(result.dominant_score, 3),
        severity=result.severity,
        response_label=result.response_label,
        spotify=spotify_data,
        wellness_suggestions=wellness_data,
        crisis_note=crisis_note,
        explanation=result.explanation if req.include_explanation else None,
        trend_note=trend_note,
        fired_rules=fired_rules_out,
    )


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Emotion analysis only — no response generation, no memory write."""
    result = get_engine().infer(req.text)
    return {
        "emotions": _scores_to_list(result.emotion_scores),
        "dominant_emotion": result.dominant_emotion,
        "dominant_score": round(result.dominant_score, 3),
        "severity": result.severity,
        "response_label": result.response_label,
        "explanation": result.explanation,
    }


@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    memory = get_memory()
    history = memory.get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or empty.")
    return {"session_id": session_id, "history": history}


@app.get("/api/trends/{session_id}")
def get_trends(session_id: str):
    memory = get_memory()
    report = memory.get_trend_report(session_id)
    return {
        "session_id": session_id,
        "trends": report.trends,
        "persistent_emotions": report.persistent_emotions,
        "session_dominant": report.session_dominant,
        "context_note": report.context_note,
    }


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    get_memory().clear(session_id)
    return {"status": "cleared", "session_id": session_id}