"""
app.py  –  Streamlit Frontend (Redesigned)
──────────────────────────────────────────
Chat-based interface for the Emotion-Aware Support System.

Features:
  • Conversational chat UI with styled message bubbles
  • Emotion intensity bar chart (Plotly)
  • Spotify track suggestions with links
  • Wellness action suggestions
  • Emotion trend line over session
  • Fuzzy rule explainability toggle
  • Session management
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"

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
  .track-card {
    background: #fafaf8; border: 1px solid #e8e4df; border-radius: 8px;
    padding: 8px 12px; margin: 4px 0; font-family: sans-serif; font-size: 13px;
  }
  .spotify-link { color: #1DB954 !important; font-weight: 600; text-decoration: none; }
  .wellness-tip { font-size: 13px; font-family: sans-serif; color: #444; padding: 3px 0; }
  .tip-dot { color: #6aaa82; font-weight: 700; }
  .section-divider { border: none; border-top: 1px solid #ece8e2; margin: 1rem 0; }
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
# API helpers
# ──────────────────────────────────────────────────────────────

def call_chat(text: str, include_explanation: bool = False, include_spotify: bool = True) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={
                "text": text,
                "session_id": st.session_state.session_id,
                "include_spotify": include_spotify,
                "include_suggestions": True,
                "include_explanation": include_explanation,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(
            "⚠️ Cannot connect to backend. Start it first:\n\n"
            "```\ncd emotion_support\nuvicorn backend.main:app --reload --port 8000\n```"
        )
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def call_trends() -> Optional[Dict]:
    try:
        resp = requests.get(f"{API_BASE}/trends/{st.session_state.session_id}", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Emotion colors
# ──────────────────────────────────────────────────────────────

EMOTION_COLORS = {
    "sadness": "#6c8ebf", "anxiety": "#d79b00", "anger": "#ae4132",
    "happiness": "#4a9e5c", "disgust": "#7b5ea7", "fear": "#6f4e37",
    "surprise": "#0097a7", "neutral": "#888888",
}


# ──────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────

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
        show_spotify     = st.toggle("Music suggestions", value=True)
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

    trends = call_trends()
    if trends and trends.get("context_note"):
        st.info(f"💡 {trends['context_note']}")


# ──────────────────────────────────────────────────────────────
# Main — Header
# ──────────────────────────────────────────────────────────────

st.markdown("## Emotion-Aware Conversational Support")
st.caption("Talk about how you're feeling. The system detects emotions and responds with empathy.")
st.markdown("---")

# ──────────────────────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────
# Emotion panel
# ──────────────────────────────────────────────────────────────

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

    # Wellness tips — full width row so they always show
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

    if show_spotify and result.get("spotify"):
        spotify = result["spotify"]
        st.markdown("---")
        st.markdown('<p class="panel-label">🎵 Music Suggestions</p>', unsafe_allow_html=True)
        st.caption(spotify.get("message", ""))
        if spotify.get("source") == "fallback_playlist":
            st.caption("_Using curated playlists — add Spotify credentials in `.env` for live recommendations_")

        tracks = spotify.get("tracks", [])
        if tracks:
            cols = st.columns(min(len(tracks), 3))
            for i, track in enumerate(tracks[:3]):
                with cols[i]:
                    name   = track.get("name", "Unknown")
                    artist = track.get("artist", "")
                    url    = track.get("spotify_url", "#")
                    st.markdown(
                        f'<div class="track-card">'
                        f'🎧 <b>{name}</b><br>'
                        f'<span style="color:#888;font-size:12px">{artist}</span><br>'
                        f'<a class="spotify-link" href="{url}" target="_blank">▶ Open in Spotify</a>'
                        f'</div>',
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


# ──────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────

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
        result = call_chat(
            text=user_input.strip(),
            include_explanation=show_explanation,
            include_spotify=show_spotify,
        )
    if result:
        st.session_state.messages.append({"role": "user",  "content": user_input.strip()})
        st.session_state.messages.append({"role": "bot",   "content": result["response"]})
        st.session_state.last_result = result
        st.session_state.emotion_history.append({
            "dominant": result.get("dominant_emotion"),
            "scores":   {e["emotion"]: e["score"] for e in result.get("emotions", [])},
        })
        st.rerun()