"""
suggestion_engine.py
─────────────────────
Generates actionable, personalised wellness suggestions based on
detected emotions and their intensities.

Suggestion categories:
  breathing   – grounding / anxiety reduction
  mindfulness – presence-focused activities
  movement    – physical activity for mood regulation
  social      – connection-based suggestions
  creative    – expressive activities
  cognitive   – reframing / journaling
  rest        – sleep/recovery for exhaustion
"""

from __future__ import annotations

import random
from typing import Dict, List

from ..engines.anfis_engine import ANFISResult


# ──────────────────────────────────────────────────────────────
# Suggestion pools
# ──────────────────────────────────────────────────────────────

SUGGESTIONS: Dict[str, Dict[str, List[str]]] = {
    "anxiety": {
        "breathing": [
            "Try box breathing: inhale for 4 counts, hold 4, exhale 4, hold 4. Repeat 4 times.",
            "4-7-8 breathing: inhale for 4 seconds, hold for 7, exhale slowly for 8.",
            "Take 5 slow, deep breaths. Focus entirely on the sensation of each one.",
        ],
        "mindfulness": [
            "Try the 5-4-3-2-1 grounding technique: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
            "Place both feet flat on the floor and feel their weight. Stay with that sensation for 60 seconds.",
            "Body scan: starting from your toes, slowly notice each part of your body without judgment.",
        ],
        "movement": [
            "A 10-minute walk, even inside, can significantly reduce cortisol.",
            "Try progressive muscle relaxation: tense each muscle group for 5 seconds, then release.",
            "Gentle stretching for 5 minutes — focus on your shoulders and neck where tension builds.",
        ],
        "cognitive": [
            "Write down your three biggest worries right now. Then, next to each one, write whether it's in your control.",
            "Ask yourself: will this matter in 5 years? In 5 months? In 5 days?",
        ],
    },
    "sadness": {
        "social": [
            "Reach out to one person you trust — even just a short message to say you're thinking of them.",
            "Sometimes being around people (even in a café, library, or park) can help without needing to talk.",
        ],
        "movement": [
            "Even a 15-minute walk in natural light can help lift mood by boosting serotonin.",
            "Gentle yoga or stretching can release stored tension from the body.",
        ],
        "creative": [
            "Try journaling: write without any goal — just let your thoughts flow for 10 minutes.",
            "Draw, doodle, or colour something — it doesn't need to be good. Just expressive.",
            "Make a playlist of songs that feel honest about how you're feeling right now.",
        ],
        "rest": [
            "If you're feeling depleted, rest is productive — give yourself permission to do less today.",
            "A warm shower or bath can signal safety to the nervous system.",
        ],
    },
    "anger": {
        "movement": [
            "Intense physical exercise — a run, a workout, even vigorous cleaning — can metabolise cortisol and adrenaline.",
            "Try punching a pillow or tearing up paper. Physical release in a safe way.",
        ],
        "breathing": [
            "Long exhale breathing: breathe in for 4 counts, out for 8. The extended exhale activates your parasympathetic system.",
            "Count backwards from 20 slowly before responding to the thing that triggered you.",
        ],
        "cognitive": [
            "Write a letter you won't send — say everything you want to say, uncensored.",
            "Try to name the specific unmet need behind the anger. Often anger is hurt in disguise.",
        ],
    },
    "happiness": {
        "social": [
            "Share this good energy with someone who might need a lift — good moods are contagious.",
            "Write a note of appreciation to someone who's positively impacted you recently.",
        ],
        "creative": [
            "Channel this energy into something creative — music, writing, cooking, building.",
            "Document this moment: write, photograph, or voice-note what's making you feel this way.",
        ],
        "mindfulness": [
            "Take a moment to really savour this feeling — let yourself fully feel it without rushing past.",
        ],
    },
    "fear": {
        "breathing": [
            "Slow your breathing immediately: in for 4, out for 6. Repeat until your heart rate steadies.",
            "Physiological sigh: two short inhales through the nose, then one long exhale through the mouth.",
        ],
        "mindfulness": [
            "Name your fear out loud or write it down. Externalising fear reduces its power.",
            "5-4-3-2-1 grounding: name 5 things you can see right now. Bring yourself into the present.",
        ],
        "social": [
            "If the fear is about something real, tell one trusted person. You don't have to carry it alone.",
        ],
    },
    "disgust": {
        "mindfulness": [
            "Create some physical distance from whatever triggered this — move to a different space if possible.",
            "Cleansing ritual: wash your hands, change your clothes, open a window. Signal a reset to your body.",
        ],
        "cognitive": [
            "Journaling: write about exactly what triggered this response and why it hit you so strongly.",
        ],
    },
    "neutral": {
        "mindfulness": [
            "Use this calm moment to check in: how are you really doing today?",
            "A good time to do something you've been putting off, while your mind is clear.",
        ],
        "movement": [
            "A walk, some stretching, or light activity can keep this calm energy flowing.",
        ],
    },
}

# Crisis-level suggestions (shown when severity is 'severe' for sadness/fear/anxiety)
CRISIS_NOTE = (
    "If you're feeling overwhelmed or unsafe, please reach out to a mental health professional "
    "or a support line. You deserve real support, and help is available."
)


class SuggestionEngine:
    """
    Generates a ranked, personalised list of wellness suggestions
    based on detected emotions and intensities.
    """

    def get_suggestions(
        self,
        result: ANFISResult,
        n_per_category: int = 1,
        max_total: int = 4,
    ) -> Dict:
        """
        Returns a structured suggestion response with:
          - wellness_actions: List[str]
          - categories_used: List[str]
          - crisis_note: Optional[str]
        """
        dominant = result.dominant_emotion
        severity = result.severity
        scores = result.emotion_scores

        # Build weighted emotion list (dominant first, then by score)
        ordered_emotions = sorted(
            [(em, sc) for em, sc in scores.items() if sc > 0.25],
            key=lambda x: -x[1],
        )[:3]

        suggestions: List[str] = []
        categories_used: List[str] = []

        for emotion, _ in ordered_emotions:
            if emotion not in SUGGESTIONS:
                continue
            pool = SUGGESTIONS[emotion]
            # Pick the most relevant category for this emotion
            preferred_categories = self._preferred_categories(emotion, severity)
            for cat in preferred_categories:
                if cat in pool and len(suggestions) < max_total:
                    item = random.choice(pool[cat])
                    if item not in suggestions:
                        suggestions.append(item)
                        categories_used.append(f"{emotion}/{cat}")
                        break

        # Deduplicate
        seen: set = set()
        unique = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        # Crisis note for severe negative emotions
        show_crisis = (
            severity == "severe"
            and dominant in ("sadness", "fear", "anxiety")
            and result.dominant_score > 0.70
        )

        return {
            "wellness_actions": unique[:max_total],
            "categories_used": categories_used[:max_total],
            "crisis_note": CRISIS_NOTE if show_crisis else None,
        }

    @staticmethod
    def _preferred_categories(emotion: str, severity: str) -> List[str]:
        """Order of preferred categories per emotion × severity."""
        mapping: Dict[str, Dict[str, List[str]]] = {
            "anxiety": {
                "mild":     ["breathing", "mindfulness"],
                "moderate": ["breathing", "mindfulness", "movement"],
                "severe":   ["breathing", "mindfulness", "movement", "cognitive"],
            },
            "sadness": {
                "mild":     ["creative", "social"],
                "moderate": ["social", "movement", "creative"],
                "severe":   ["social", "rest", "creative", "movement"],
            },
            "anger": {
                "mild":     ["breathing", "movement"],
                "moderate": ["movement", "breathing", "cognitive"],
                "severe":   ["movement", "breathing", "cognitive"],
            },
            "happiness": {
                "mild":     ["mindfulness", "social"],
                "moderate": ["social", "creative"],
                "severe":   ["creative", "social"],
            },
            "fear": {
                "mild":     ["breathing", "mindfulness"],
                "moderate": ["breathing", "mindfulness", "social"],
                "severe":   ["breathing", "social", "mindfulness"],
            },
            "disgust":  {"mild": ["mindfulness"], "moderate": ["mindfulness", "cognitive"], "severe": ["cognitive"]},
            "neutral":  {"mild": ["mindfulness"], "moderate": ["movement"], "severe": ["mindfulness"]},
            "surprise": {"mild": ["mindfulness"], "moderate": ["social"], "severe": ["social"]},
        }
        em_map = mapping.get(emotion, {"mild": ["mindfulness"], "moderate": ["mindfulness"], "severe": ["breathing"]})
        return em_map.get(severity, em_map.get("moderate", ["mindfulness"]))


_suggestion_engine: SuggestionEngine | None = None


def get_suggestion_engine() -> SuggestionEngine:
    global _suggestion_engine
    if _suggestion_engine is None:
        _suggestion_engine = SuggestionEngine()
    return _suggestion_engine
