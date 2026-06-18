"""
app.py  Streamlit Frontend (Standalone, no FastAPI needed)
Directly imports backend modules for Streamlit Cloud deployment.
"""

from __future__ import annotations

import sys
import os
import uuid
import random
from typing import Any, Dict, List, Optional

# ── Path setup so backend imports work ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotly.graph_objects as go
import streamlit as st

# ── Direct backend imports (no FastAPI needed) ──────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engines.anfis_engine import get_engine, FUZZY_RULES
from backend.engines.response_engine import get_response_engine
from backend.modules.memory import get_memory
from backend.modules.suggestions import get_suggestion_engine

# ──────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emotion Support",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: Georgia, "Times New Roman", serif; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8e4df; }
  .stApp { background: #fafaf8; }

  .user-bubble {
    background: #f0f4ff; border-left: 3px solid #4a6fa5;
    padding: 10px 16px; border-radius: 0 10px 10px 0;
    max-width: 78%; margin: 8px 0;
    font-family: sans-serif; font-size: 14px; line-height: 1.6;
  }
  .bot-bubble {
    background: #ffffff; border: 1px solid #e8efe8; border-left: 3px solid #6aaa82;
    padding: 10px 16px; border-radius: 0 10px 10px 0;
    max-width: 78%; margin: 8px 0 8px auto;
    font-family: sans-serif; font-size: 14px; line-height: 1.6;
  }
  .bubble-label { font-size: 11px; color: #888; margin-bottom: 4px; font-family: sans-serif; }

  .panel-label {
    font-size: 11px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.7px;
    font-family: sans-serif; margin-bottom: 8px;
  }

  .emotion-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600; font-family: sans-serif; text-transform: capitalize;
  }
  .badge-sadness   { background:#d4e3f7; color:#1a3a5c; }
  .badge-anxiety   { background:#fff3cd; color:#7a5000; }
  .badge-anger     { background:#f8d7da; color:#721c24; }
  .badge-happiness { background:#d4f7d4; color:#1e5e1e; }
  .badge-disgust   { background:#e8d5f7; color:#3d1c6e; }
  .badge-fear      { background:#ede0d4; color:#3e2a1a; }
  .badge-surprise  { background:#cff4fc; color:#005f67; }
  .badge-neutral   { background:#f1f1f1; color:#444444; }
  .badge-mild      { background:#d4f7d4; color:#1e5e1e; }
  .badge-moderate  { background:#fff3cd; color:#7a5000; }
  .badge-severe    { background:#f8d7da; color:#721c24; }

  .crisis-box {
    background: #fff8e1; border: 1px solid #ffc107; border-radius: 8px;
    padding: 10px 14px; font-size: 13px; font-family: sans-serif; color: #5f4300; margin: 8px 0;
  }
  .empty-state { text-align: center; margin-top: 5rem; color: #999; font-family: sans-serif; }

  .stTextArea textarea {
    font-family: sans-serif !important; font-size: 14px !important;
    background: #fafaf8 !important; border-radius: 10px !important;
  }
  div[data-testid="stFormSubmitButton"] button {
    background: #4a6fa5 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-family: sans-serif !important; font-weight: 600 !important;
    font-size: 14px !important;
  }
  div[data-testid="stFormSubmitButton"] button:hover { background: #3a5f95 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "messages": [],
        "emotion_history": [],
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ──────────────────────────────────────────────────────────────
# Core inference (replaces API call)
# ──────────────────────────────────────────────────────────────

def _emotion_level(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def process_message(text: str, include_explanation: bool = False) -> Optional[Dict[str, Any]]:
    try:
        session_id = st.session_state.session_id
        memory = get_memory()
        history = memory.get_history(session_id)

        # ANFIS inference
        engine = get_engine()
        result = engine.infer(text)

        # Trend note
        trend = memory.get_trend_report(session_id)
        trend_note = trend.context_note or None

        # Response
        response_engine = get_response_engine()
        response_text = response_engine.generate(result, user_text=text, conversation_history=history)

        # Wellness suggestions
        suggestion_eng = get_suggestion_engine()
        sugg = suggestion_eng.get_suggestions(result)
        wellness_data = sugg.get("wellness_actions")
        crisis_note = sugg.get("crisis_note")

        # Record memory
        memory.record(
            session_id=session_id,
            user_text=text,
            emotion_scores=result.emotion_scores,
            dominant_emotion=result.dominant_emotion,
            dominant_score=result.dominant_score,
            response_label=result.response_label,
            severity=result.severity,
        )

        # Build emotions list
        emotions = sorted(
            [{"emotion": em, "score": round(sc, 3), "level": _emotion_level(sc)}
             for em, sc in result.emotion_scores.items()],
            key=lambda x: -x["score"]
        )

        # Explanation
        explanation = None
        fired_rules_out = None
        if include_explanation:
            explanation = result.explanation
            fired_rules_out = [
                {
                    "rule_id": rid,
                    "strength": round(strength, 4),
                    "description": next((r.description for r in FUZZY_RULES if r.rule_id == rid), ""),
                }
                for rid, strength in result.fired_rules
            ]

        return {
            "session_id": session_id,
            "response": response_text,
            "emotions": emotions,
            "dominant_emotion": result.dominant_emotion,
            "dominant_score": round(result.dominant_score, 3),
            "severity": result.severity,
            "response_label": result.response_label,
            "wellness_suggestions": wellness_data,
            "crisis_note": crisis_note,
            "explanation": explanation,
            "trend_note": trend_note,
            "fired_rules": fired_rules_out,
        }

    except Exception as e:
        st.error(f"Error processing message: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Emotion colors & charts
# ──────────────────────────────────────────────────────────────

EMOTION_COLORS = {
    "sadness": "#6c8ebf", "anxiety": "#d79b00", "anger": "#ae4132",
    "happiness": "#4a9e5c", "disgust": "#7b5ea7", "fear": "#6f4e37",
    "surprise": "#0097a7", "neutral": "#888888",
}


def render_emotion_bars(emotions: List[Dict]) -> None:
    if not emotions:
        return
    sorted_em = sorted(emotions, key=lambda x: -x["score"])[:6]
    fig = go.Figure(go.Bar(
        y=[e["emotion"].capitalize() for e in sorted_em],
        x=[e["score"] for e in sorted_em],
        orientation="h",
        marker_color=[EMOTION_COLORS.get(e["emotion"], "#888") for e in sorted_em],
        text=[f"{e['score']:.0%}" for e in sorted_em],
        textposition="outside",
    ))
    fig.update_layout(
        height=220, margin=dict(l=10, r=50, t=6, b=6),
        xaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#f0eee8", tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(family="sans-serif", size=12, color="#444"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_trend_chart(emotion_history: List[Dict]) -> None:
    if len(emotion_history) < 2:
        st.caption("Send a few messages to see emotion trends.")
        return
    turns = list(range(1, len(emotion_history) + 1))
    fig = go.Figure()
    for em in ["sadness", "anxiety", "anger", "happiness"]:
        scores = [h.get("scores", {}).get(em, 0) for h in emotion_history]
        if max(scores) > 0.05:
            fig.add_trace(go.Scatter(
                x=turns, y=scores, mode="lines+markers", name=em.capitalize(),
                line=dict(color=EMOTION_COLORS.get(em, "#888"), width=2),
                marker=dict(size=5),
            ))
    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(title="Turn", tickmode="linear", tickfont=dict(size=10), title_font=dict(size=10)),
        yaxis=dict(range=[0, 1], tickformat=".0%", tickfont=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        font=dict(family="sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🤝 Emotion Support")
    st.caption("Neuro-fuzzy empathetic system")
    st.markdown("---")

    with st.expander("⚙️ Settings", expanded=True):
        show_explanation = st.toggle("Fuzzy rule explanation", value=False)
        show_wellness    = st.toggle("Wellness tips", value=True)

    st.markdown("---")
    st.markdown('<p class="panel-label">Emotion Trend</p>', unsafe_allow_html=True)
    render_trend_chart(st.session_state.emotion_history)
    st.markdown("---")

    st.caption(f"Session: `{st.session_state.session_id[:12]}…`")
    if st.button("🔄 New Session", use_container_width=True):
        for key in ["session_id", "messages", "emotion_history", "last_result"]:
            st.session_state.pop(key, None)
        st.rerun()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

st.markdown("## Emotion-Aware Conversational Support")
st.caption("Talk about how you're feeling. The system detects emotions and responds with empathy.")
st.markdown("---")

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
      <div style="font-size:42px;margin-bottom:12px">🤝</div>
      <p style="font-size:16px;color:#555;font-family:Georgia,serif;">
        Share how you're feeling.<br>The system will listen and respond with empathy.
      </p>
      <p style="font-size:13px;color:#aaa;">
        e.g. "I've been really anxious lately and can't seem to relax"
      </p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble"><div class="bubble-label">You</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-bubble"><div class="bubble-label">Support</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

# ── Emotion panel ──────────────────────────────────────────────
if st.session_state.last_result:
    result = st.session_state.last_result
    st.markdown("---")

    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.markdown('<p class="panel-label">Detected Emotions</p>', unsafe_allow_html=True)
        render_emotion_bars(result.get("emotions", []))

    with col2:
        st.markdown('<p class="panel-label">Emotion State</p>', unsafe_allow_html=True)
        dom = result.get("dominant_emotion", "neutral")
        sev = result.get("severity", "mild")
        st.markdown(
            f'<span class="emotion-badge badge-{dom}">{dom.capitalize()}</span>'
            f'&nbsp;<span class="emotion-badge badge-{sev}">{sev}</span>',
            unsafe_allow_html=True,
        )
        st.metric("Score", f"{int(result.get('dominant_score', 0) * 100)}%")
        if result.get("trend_note"):
            st.info(result["trend_note"])

    if show_wellness and result.get("wellness_suggestions"):
        st.markdown('<p class="panel-label" style="margin-top:1rem">💡 Wellness Tips</p>', unsafe_allow_html=True)
        tips = result["wellness_suggestions"]
        tip_cols = st.columns(len(tips))
        for i, tip in enumerate(tips):
            with tip_cols[i]:
                st.markdown(
                    f'<div style="background:#f7faf7;border:1px solid #d4eeda;border-radius:10px;'
                    f'padding:12px 14px;font-family:sans-serif;font-size:13px;color:#2d5a3d;'
                    f'line-height:1.5;">'
                    f'<span style="color:#6aaa82;font-weight:700;margin-right:6px">✦</span>{tip}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if result.get("crisis_note"):
        st.markdown(
            f'<div class="crisis-box">⚠️ {result["crisis_note"]}</div>',
            unsafe_allow_html=True,
        )

    if show_explanation and result.get("explanation"):
        with st.expander("🔍 Fuzzy Rule Explanation"):
            st.code(result["explanation"], language="text")
            if result.get("fired_rules"):
                for rule in result["fired_rules"]:
                    st.markdown(
                        f"**{rule['rule_id']}** (strength: `{rule['strength']:.4f}`): {rule['description']}"
                    )

# ── Input ──────────────────────────────────────────────────────
st.markdown("---")
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Message",
        placeholder="Type anything… e.g. 'I've been really anxious lately and can't seem to relax'",
        height=80,
        label_visibility="collapsed",
    )
    col_a, col_b = st.columns([5, 1])
    with col_b:
        submitted = st.form_submit_button("Send", use_container_width=True, type="primary")

st.caption("Press Send to submit.")

if submitted and user_input.strip():
    with st.spinner("Processing…"):
        result = process_message(
            text=user_input.strip(),
            include_explanation=show_explanation,
        )
    if result:
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.session_state.messages.append({"role": "bot",  "content": result["response"]})
        st.session_state.last_result = result
        st.session_state.emotion_history.append({
            "dominant": result.get("dominant_emotion"),
            "scores":   {e["emotion"]: e["score"] for e in result.get("emotions", [])},
        })
        st.rerun()
