"""
emotion_memory.py
─────────────────
Short-term emotion memory with sliding window trend analysis.

Stores per-session interaction records (last N turns) and computes:
  - Emotion trend vectors (are emotions rising or falling?)
  - Persistent emotional state detection
  - Context-aware prompts for the response engine

Memory is in-process (per session). For production, swap _store for Redis/DB.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..engines.anfis_engine import EMOTIONS

# How many turns to keep in memory
MEMORY_WINDOW = 10
# How many consecutive turns with high score constitutes "persistent"
PERSISTENCE_THRESHOLD = 3
# Score above which an emotion is considered "active"
ACTIVE_THRESHOLD = 0.45


@dataclass
class MemoryEntry:
    timestamp: float
    user_text: str
    emotion_scores: Dict[str, float]
    dominant_emotion: str
    dominant_score: float
    response_label: str
    severity: str


@dataclass
class TrendReport:
    trends: Dict[str, str]          # {emotion: 'rising'|'falling'|'stable'}
    persistent_emotions: List[str]  # emotions active for >= PERSISTENCE_THRESHOLD turns
    session_dominant: str           # most frequent dominant emotion in window
    context_note: str               # natural-language note for response engine


class EmotionMemory:
    """
    Maintains a per-session sliding window of emotion records
    and derives trend information for context-aware responses.
    """

    def __init__(self) -> None:
        # session_id → deque of MemoryEntry
        self._store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MEMORY_WINDOW))

    # ── record ────────────────────────────────────────────────

    def record(
        self,
        session_id: str,
        user_text: str,
        emotion_scores: Dict[str, float],
        dominant_emotion: str,
        dominant_score: float,
        response_label: str,
        severity: str,
    ) -> None:
        entry = MemoryEntry(
            timestamp=time.time(),
            user_text=user_text,
            emotion_scores=emotion_scores,
            dominant_emotion=dominant_emotion,
            dominant_score=dominant_score,
            response_label=response_label,
            severity=severity,
        )
        self._store[session_id].append(entry)

    # ── retrieve ──────────────────────────────────────────────

    def get_history(self, session_id: str) -> List[Dict]:
        return [
            {
                "timestamp": e.timestamp,
                "user_text": e.user_text,
                "emotion_scores": e.emotion_scores,
                "dominant_emotion": e.dominant_emotion,
                "dominant_score": e.dominant_score,
            }
            for e in self._store.get(session_id, [])
        ]

    def get_trend_report(self, session_id: str) -> TrendReport:
        entries = list(self._store.get(session_id, []))
        if len(entries) < 2:
            return TrendReport(
                trends={em: "stable" for em in EMOTIONS},
                persistent_emotions=[],
                session_dominant="neutral",
                context_note="",
            )

        trends = self._compute_trends(entries)
        persistent = self._find_persistent_emotions(entries)
        session_dominant = self._session_dominant(entries)
        note = self._build_context_note(persistent, trends, session_dominant)

        return TrendReport(
            trends=trends,
            persistent_emotions=persistent,
            session_dominant=session_dominant,
            context_note=note,
        )

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def session_length(self, session_id: str) -> int:
        return len(self._store.get(session_id, []))

    # ── analysis helpers ──────────────────────────────────────

    @staticmethod
    def _compute_trends(entries: List[MemoryEntry]) -> Dict[str, str]:
        """
        For each emotion, compare average of the first half vs second half
        of the memory window to determine direction.
        """
        n = len(entries)
        mid = max(1, n // 2)
        first_half = entries[:mid]
        second_half = entries[mid:]

        trends: Dict[str, str] = {}
        for em in EMOTIONS:
            avg_first = sum(e.emotion_scores.get(em, 0) for e in first_half) / len(first_half)
            avg_second = sum(e.emotion_scores.get(em, 0) for e in second_half) / len(second_half)
            delta = avg_second - avg_first
            if delta > 0.08:
                trends[em] = "rising"
            elif delta < -0.08:
                trends[em] = "falling"
            else:
                trends[em] = "stable"

        return trends

    @staticmethod
    def _find_persistent_emotions(entries: List[MemoryEntry]) -> List[str]:
        """Return emotions that have been above ACTIVE_THRESHOLD for the last N turns."""
        if len(entries) < PERSISTENCE_THRESHOLD:
            return []

        recent = entries[-PERSISTENCE_THRESHOLD:]
        persistent = []
        for em in EMOTIONS:
            if all(e.emotion_scores.get(em, 0) >= ACTIVE_THRESHOLD for e in recent):
                persistent.append(em)
        return persistent

    @staticmethod
    def _session_dominant(entries: List[MemoryEntry]) -> str:
        counts: Dict[str, int] = defaultdict(int)
        for e in entries:
            counts[e.dominant_emotion] += 1
        return max(counts, key=counts.get) if counts else "neutral"

    @staticmethod
    def _build_context_note(
        persistent: List[str],
        trends: Dict[str, str],
        session_dominant: str,
    ) -> str:
        if not persistent and not any(v == "rising" for v in trends.values()):
            return ""

        lines = []
        if "anxiety" in persistent:
            lines.append("You've seemed quite anxious throughout our conversation.")
        if "sadness" in persistent:
            lines.append("There's been a persistent sadness in what you've been sharing.")
        if "anger" in persistent:
            lines.append("I've noticed ongoing frustration in your messages.")

        rising = [em for em, trend in trends.items() if trend == "rising" and em not in persistent]
        for em in rising:
            if em in ("anxiety", "sadness", "anger", "fear"):
                lines.append(f"I'm noticing your {em} seems to be increasing.")

        return " ".join(lines)


# Module singleton
_memory: EmotionMemory | None = None


def get_memory() -> EmotionMemory:
    global _memory
    if _memory is None:
        _memory = EmotionMemory()
    return _memory
