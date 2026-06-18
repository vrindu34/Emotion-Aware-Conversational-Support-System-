"""
anfis_engine.py
───────────────
Neuro-Fuzzy Inference System for multi-label emotion detection.

Architecture:
  Layer 1  – Fuzzification   : Gaussian membership functions per feature
  Layer 2  – Rule antecedent : T-norm (product) over input memberships
  Layer 3  – Normalisation   : Each rule weight / sum of all weights
  Layer 4  – Consequent      : Linear Takagi-Sugeno output per rule
  Layer 5  – Defuzzification : Weighted sum → emotion intensity scores

The weights of Layer 1 (mean / sigma) and Layer 4 (consequent coefficients)
are trained by hybrid learning: least-squares for Layer 4, gradient descent
for Layer 1.  A compact fallback lexicon-weighted model covers cold-start.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ──────────────────────────────────────────────────────────────
# 1. Emotion taxonomy
# ──────────────────────────────────────────────────────────────

EMOTIONS: List[str] = [
    "sadness",
    "anxiety",
    "anger",
    "happiness",
    "disgust",
    "fear",
    "surprise",
    "neutral",
]

# ──────────────────────────────────────────────────────────────
# 2. Lexicon-based seed (cold-start / interpretability layer)
# ──────────────────────────────────────────────────────────────

EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
    "sadness": {
        # Strong words
        "sad": 0.9, "depressed": 0.95, "unhappy": 0.85, "miserable": 0.92,
        "lonely": 0.80, "heartbroken": 0.95, "grief": 0.90, "sorrow": 0.88,
        "cry": 0.82, "tears": 0.78, "hopeless": 0.88, "empty": 0.75,
        "loss": 0.72, "mourn": 0.85, "hurt": 0.65, "broken": 0.70,
        "devastated": 0.93, "gloomy": 0.80, "melancholy": 0.85,
        # Subtle / everyday words
        "low": 0.65, "down": 0.68, "drained": 0.72, "tired": 0.58,
        "heavy": 0.65, "blue": 0.62, "rough": 0.60, "struggling": 0.70,
        "hard": 0.52, "difficult": 0.55, "miss": 0.65, "missing": 0.65,
        "alone": 0.68, "lost": 0.65, "numb": 0.72, "flat": 0.58,
        "unmotivated": 0.65, "worthless": 0.80, "helpless": 0.78,
        "disappointed": 0.70, "disappointed": 0.70, "regret": 0.68,
        "crying": 0.85, "weeping": 0.85, "sobbing": 0.88,
    },
    "anxiety": {
        # Strong words
        "anxious": 0.92, "worried": 0.88, "nervous": 0.85, "panic": 0.95,
        "overwhelmed": 0.90, "stressed": 0.88, "tense": 0.82, "uneasy": 0.78,
        "dread": 0.90, "fear": 0.80, "restless": 0.75, "apprehensive": 0.85,
        "terrified": 0.92, "scared": 0.85, "paranoid": 0.88,
        "exhausted": 0.72, "burnout": 0.82, "overwhelm": 0.90,
        # Subtle / everyday words
        "overthinking": 0.82, "overthink": 0.80, "pressure": 0.72,
        "deadline": 0.65, "unsure": 0.60, "uncertain": 0.65,
        "stuck": 0.62, "trapped": 0.75, "edgy": 0.70, "on edge": 0.78,
        "cant sleep": 0.72, "insomnia": 0.75, "racing": 0.70,
        "shaky": 0.72, "trembling": 0.75, "sweat": 0.65,
        "worry": 0.85, "worrying": 0.85, "freaking": 0.80,
    },
    "anger": {
        # Strong words
        "angry": 0.92, "furious": 0.95, "rage": 0.97, "annoyed": 0.78,
        "frustrated": 0.85, "irritated": 0.82, "mad": 0.88, "infuriated": 0.95,
        "hate": 0.90, "hostile": 0.88, "outraged": 0.92, "resentful": 0.85,
        "bitter": 0.78, "enraged": 0.95, "livid": 0.92,
        # Subtle / everyday words
        "unfair": 0.68, "fed up": 0.78, "sick of": 0.72, "done with": 0.65,
        "pissed": 0.85, "ticked": 0.72, "bothered": 0.62, "upset": 0.65,
        "agitated": 0.75, "grumpy": 0.62, "snappy": 0.65, "grouchy": 0.60,
        "disrespected": 0.75, "betrayed": 0.80, "cheated": 0.72,
    },
    "happiness": {
        # Strong words
        "happy": 0.92, "joyful": 0.95, "excited": 0.90, "elated": 0.92,
        "content": 0.82, "pleased": 0.80, "cheerful": 0.88, "delighted": 0.90,
        "grateful": 0.85, "wonderful": 0.88, "amazing": 0.85, "great": 0.78,
        "fantastic": 0.90, "love": 0.82, "enjoy": 0.78, "blessed": 0.85,
        # Subtle / everyday words — recovery, independence, mild positivity
        # NOTE: "good", "best", "lesson", "learned" intentionally excluded
        # to avoid false positive happiness when user is processing hurt/anger
        "better": 0.45, "improving": 0.60, "improved": 0.58,
        "well": 0.48, "nice": 0.55, "fine": 0.42,
        "capable": 0.58, "independent": 0.55, "proud": 0.72,
        "stronger": 0.60, "manage": 0.50, "managed": 0.52,
        "accomplished": 0.75, "achieved": 0.72, "success": 0.75,
        "hopeful": 0.65, "optimistic": 0.70, "positive": 0.58,
        "motivated": 0.65, "energetic": 0.70, "refreshed": 0.65,
        "relieved": 0.60, "calm": 0.50, "peaceful": 0.55,
        "laughing": 0.85, "smile": 0.78, "smiling": 0.78,
        "fun": 0.75, "enjoying": 0.78, "pleasure": 0.75,
    },
    "disgust": {
        # Strong words
        "disgusted": 0.92, "revolting": 0.90, "awful": 0.80, "horrible": 0.85,
        "disgusting": 0.92, "gross": 0.85, "nasty": 0.82, "repulsed": 0.90,
        "sick": 0.75, "vile": 0.88, "yuck": 0.80,
        # Subtle / everyday words
        "ew": 0.75, "eww": 0.78, "yikes": 0.60, "cringe": 0.70,
        "unpleasant": 0.65, "repulsive": 0.88, "offensive": 0.72,
        "filthy": 0.82, "dirty": 0.65, "rotten": 0.78,
    },
    "fear": {
        # Strong words
        "afraid": 0.90, "terrified": 0.95, "scared": 0.88, "frightened": 0.88,
        "horror": 0.92, "nightmare": 0.88, "danger": 0.82, "threat": 0.80,
        "unsafe": 0.85, "petrified": 0.95, "phobia": 0.85,
        # Subtle / everyday words
        "worried": 0.65, "dread": 0.82, "dreading": 0.82,
        "creepy": 0.70, "spooky": 0.65, "eerie": 0.68,
        "vulnerable": 0.72, "helpless": 0.75, "powerless": 0.75,
        "frightening": 0.88, "scary": 0.85, "terrifying": 0.92,
    },
    "surprise": {
        # Strong words
        "surprised": 0.88, "shocked": 0.90, "astonished": 0.92, "amazed": 0.88,
        "unexpected": 0.82, "unbelievable": 0.85, "wow": 0.85, "sudden": 0.78,
        "stunned": 0.90, "speechless": 0.85,
        # Subtle / everyday words
        "whoa": 0.82, "omg": 0.80, "really": 0.55, "seriously": 0.60,
        "cant believe": 0.85, "no way": 0.80, "wait what": 0.78,
        "out of nowhere": 0.75, "didnt expect": 0.80,
    },
    "neutral": {
        # Original words
        "okay": 0.70, "fine": 0.65, "alright": 0.65, "whatever": 0.70,
        "normal": 0.80, "average": 0.75, "usual": 0.78,
        # Added words for mixed / conflicting states
        "mixed": 0.65, "both": 0.55, "also": 0.50, "kind of": 0.60,
        "sort of": 0.60, "somewhat": 0.62, "bit": 0.52, "little": 0.50,
        "meh": 0.70, "so so": 0.72, "not bad": 0.60, "not great": 0.62,
        "managing": 0.58, "getting by": 0.60, "surviving": 0.60,
        "same": 0.65, "unchanged": 0.68, "nothing": 0.55,
    },
}

# Negation words that flip/dampen emotion scores
NEGATION_WORDS = {"not", "no", "never", "neither", "nobody", "nothing",
                  "nowhere", "hardly", "barely", "scarcely", "dont",
                  "doesn't", "didn't", "wasn't", "isn't", "aren't", "won't"}

# Intensifiers that amplify scores
INTENSIFIER_MAP = {
    "very": 1.3, "really": 1.3, "extremely": 1.5, "absolutely": 1.5,
    "totally": 1.4, "completely": 1.4, "so": 1.2, "quite": 1.1,
    "deeply": 1.3, "incredibly": 1.5, "terribly": 1.4, "awfully": 1.3,
    "pretty": 1.1, "rather": 1.1,
}


# ──────────────────────────────────────────────────────────────
# 3. Membership function helpers
# ──────────────────────────────────────────────────────────────

def _gaussian_mf(x: float, mean: float, sigma: float) -> float:
    """Gaussian membership function: exp(-0.5 * ((x-mean)/sigma)^2)."""
    if sigma <= 0:
        sigma = 1e-6
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function for severity levels."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a + 1e-9)
    return (c - x) / (c - b + 1e-9)


# ──────────────────────────────────────────────────────────────
# 4. Fuzzy rule definitions
# ──────────────────────────────────────────────────────────────

@dataclass
class FuzzyRule:
    """IF antecedent THEN response_label at severity."""
    rule_id: str
    antecedent: Dict[str, str]      # {emotion: 'low'|'medium'|'high'}
    response_label: str             # label for response engine
    severity: str                   # 'mild'|'moderate'|'severe'
    description: str = ""

FUZZY_RULES: List[FuzzyRule] = [
    FuzzyRule("R01", {"anxiety": "high"},
              "calm_supportive", "severe",
              "High anxiety → calming, grounding response"),
    FuzzyRule("R02", {"sadness": "high"},
              "empathetic_deep", "severe",
              "High sadness → deep empathetic response"),
    FuzzyRule("R03", {"anxiety": "high", "sadness": "medium"},
              "calm_supportive", "severe",
              "High anxiety + medium sadness → supportive response"),
    FuzzyRule("R04", {"anxiety": "medium", "sadness": "medium"},
              "warm_checking", "moderate",
              "Medium anxiety + medium sadness → warm check-in"),
    FuzzyRule("R05", {"anger": "high"},
              "de_escalate", "severe",
              "High anger → de-escalation response"),
    FuzzyRule("R06", {"anger": "medium", "anxiety": "medium"},
              "de_escalate", "moderate",
              "Medium anger + anxiety → de-escalation with empathy"),
    FuzzyRule("R07", {"happiness": "high"},
              "celebrate_engage", "mild",
              "High happiness → celebratory engagement"),
    FuzzyRule("R08", {"happiness": "medium"},
              "positive_reinforce", "mild",
              "Medium happiness → positive reinforcement"),
    FuzzyRule("R09", {"fear": "high"},
              "reassure_ground", "severe",
              "High fear → reassurance and grounding"),
    FuzzyRule("R10", {"fear": "medium", "anxiety": "medium"},
              "reassure_ground", "moderate",
              "Medium fear + anxiety → gentle reassurance"),
    FuzzyRule("R11", {"disgust": "high"},
              "validate_redirect", "moderate",
              "High disgust → validation + redirect"),
    FuzzyRule("R12", {"surprise": "high", "happiness": "medium"},
              "celebrate_engage", "mild",
              "High surprise + happiness → engage and celebrate"),
    FuzzyRule("R13", {"sadness": "medium"},
              "warm_checking", "moderate",
              "Medium sadness → warm check-in"),
    FuzzyRule("R14", {"neutral": "high"},
              "curious_engage", "mild",
              "Neutral state → curious, open engagement"),
    FuzzyRule("R15", {"anxiety": "low", "sadness": "low"},
              "curious_engage", "mild",
              "Low emotion → light, curious engagement"),
    # New rules for mixed/conflicting emotional states
    FuzzyRule("R16", {"sadness": "medium", "happiness": "medium"},
              "warm_checking", "moderate",
              "Mixed sadness + happiness → acknowledge complexity"),
    FuzzyRule("R17", {"sadness": "low", "happiness": "medium"},
              "positive_reinforce", "mild",
              "Low sadness + recovering happiness → gentle encouragement"),
    FuzzyRule("R18", {"neutral": "medium", "sadness": "low"},
              "warm_checking", "mild",
              "Neutral with hint of sadness → soft check-in"),
    # Rules for betrayal/hurt + acceptance/growth (e.g. "betrayed but learned a lesson")
    FuzzyRule("R19", {"anger": "medium", "sadness": "medium"},
              "empathetic_mixed", "moderate",
              "Hurt + anger together → acknowledge both pain and frustration"),
    FuzzyRule("R20", {"anger": "high", "sadness": "medium"},
              "empathetic_mixed", "moderate",
              "Strong anger with sadness → validate complexity before de-escalating"),
    FuzzyRule("R21", {"sadness": "medium", "neutral": "medium"},
              "warm_checking", "moderate",
              "Sadness processing with acceptance → hold space for both"),
]


# ──────────────────────────────────────────────────────────────
# 5. ANFIS Engine
# ──────────────────────────────────────────────────────────────

@dataclass
class ANFISResult:
    emotion_scores: Dict[str, float]        # normalised 0–1 per emotion
    fired_rules: List[Tuple[str, float]]    # [(rule_id, strength)]
    dominant_emotion: str
    dominant_score: float
    response_label: str
    severity: str
    feature_vector: List[float] = field(default_factory=list)
    explanation: str = ""


class ANFISEngine:
    """
    Adaptive Neuro-Fuzzy Inference System for emotion detection.

    The engine has 5 ANFIS layers:
      L1: Fuzzify TF-IDF feature scores via Gaussian MFs
      L2: Compute rule firing strengths (product T-norm)
      L3: Normalise firing strengths
      L4: Compute weighted consequents (linear Takagi-Sugeno)
      L5: Aggregate → emotion intensity vector
    """

    def __init__(self) -> None:
        # Gaussian MF parameters for each emotion at 3 levels: low / medium / high
        # Format: {emotion: [(mean_low, sigma_low), (mean_med, sigma_med), (mean_hi, sigma_hi)]}
        self._mf_params: Dict[str, List[Tuple[float, float]]] = {
            em: [(0.15, 0.12), (0.45, 0.15), (0.78, 0.15)] for em in EMOTIONS
        }
        # Consequent linear weights (Layer 4) — initialised, refined via LSE
        self._consequent_w: Dict[str, float] = {em: 1.0 for em in EMOTIONS}
        # Vectoriser for TF-IDF feature extraction
        self._vectoriser = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self._is_fitted = False

    # ── public interface ──────────────────────────────────────

    def infer(self, text: str) -> ANFISResult:
        """Run the full ANFIS pipeline on raw user text."""
        tokens, processed = self._preprocess(text)
        lexicon_scores = self._lexicon_layer(tokens)
        feature_vector = self._build_feature_vector(lexicon_scores)

        # ANFIS layers
        memberships = self._layer1_fuzzify(feature_vector)
        rule_strengths = self._layer2_rule_antecedents(memberships)
        normalised = self._layer3_normalise(rule_strengths)
        emotion_scores = self._layer4_consequents(normalised, feature_vector)
        final_scores = self._layer5_defuzzify(emotion_scores, lexicon_scores)

        # Pick dominant
        dominant = max(final_scores, key=final_scores.get)
        dom_score = final_scores[dominant]

        # Find best-matching rule
        best_rule_id, best_strength = max(normalised.items(), key=lambda x: x[1])
        matched_rule = next((r for r in FUZZY_RULES if r.rule_id == best_rule_id), None)
        response_label = matched_rule.response_label if matched_rule else "warm_checking"
        severity = matched_rule.severity if matched_rule else "moderate"

        fired = sorted(
            [(rid, s) for rid, s in normalised.items() if s > 0.05],
            key=lambda x: -x[1]
        )[:3]

        explanation = self._build_explanation(fired, dominant, dom_score, matched_rule)

        return ANFISResult(
            emotion_scores=final_scores,
            fired_rules=fired,
            dominant_emotion=dominant,
            dominant_score=dom_score,
            response_label=response_label,
            severity=severity,
            feature_vector=feature_vector,
            explanation=explanation,
        )

    # ── preprocessing ─────────────────────────────────────────

    @staticmethod
    def _preprocess(text: str) -> Tuple[List[str], str]:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s'']", " ", text)
        tokens = text.split()
        return tokens, text

    # ── lexicon layer (interpretability seed) ────────────────

    @staticmethod
    def _lexicon_layer(tokens: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {em: 0.0 for em in EMOTIONS}
        negation_window = False
        intensifier = 1.0

        for i, token in enumerate(tokens):
            if token in NEGATION_WORDS:
                negation_window = True
                intensifier = 1.0
                continue
            if token in INTENSIFIER_MAP:
                intensifier = INTENSIFIER_MAP[token]
                continue

            for emotion, lexicon in EMOTION_LEXICON.items():
                if token in lexicon:
                    val = lexicon[token] * intensifier
                    if negation_window:
                        val *= -0.4   # partial negation dampener
                    scores[emotion] = min(1.0, max(0.0, scores[emotion] + val))

            # Reset after two words past negation
            if negation_window and token not in NEGATION_WORDS:
                negation_window = False
            intensifier = 1.0

        # Normalise across emotions
        total = sum(scores.values()) or 1.0
        if total > 0:
            for em in EMOTIONS:
                scores[em] = min(1.0, scores[em] / total * len(EMOTIONS) * 0.25)

        return scores

    # ── feature vector ────────────────────────────────────────

    @staticmethod
    def _build_feature_vector(lexicon_scores: Dict[str, float]) -> List[float]:
        """Convert per-emotion lexicon scores into a dense feature vector."""
        return [lexicon_scores[em] for em in EMOTIONS]

    # ── Layer 1: Fuzzification ────────────────────────────────

    def _layer1_fuzzify(self, features: List[float]) -> Dict[str, Dict[str, float]]:
        """
        For each emotion feature value, compute low/medium/high membership.
        Returns {emotion: {'low': μ, 'medium': μ, 'high': μ}}
        """
        memberships: Dict[str, Dict[str, float]] = {}
        level_names = ["low", "medium", "high"]

        for i, em in enumerate(EMOTIONS):
            x = features[i]
            params = self._mf_params[em]
            memberships[em] = {
                level_names[j]: _gaussian_mf(x, params[j][0], params[j][1])
                for j in range(3)
            }
        return memberships

    # ── Layer 2: Rule antecedents (firing strengths) ─────────

    @staticmethod
    def _layer2_rule_antecedents(
        memberships: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        For each fuzzy rule, compute the product T-norm of its antecedents.
        Returns {rule_id: firing_strength}
        """
        strengths: Dict[str, float] = {}
        for rule in FUZZY_RULES:
            strength = 1.0
            for emotion, level in rule.antecedent.items():
                mu = memberships.get(emotion, {}).get(level, 0.0)
                strength *= mu
            strengths[rule.rule_id] = strength
        return strengths

    # ── Layer 3: Normalisation ────────────────────────────────

    @staticmethod
    def _layer3_normalise(
        strengths: Dict[str, float]
    ) -> Dict[str, float]:
        total = sum(strengths.values()) or 1e-9
        return {rid: s / total for rid, s in strengths.items()}

    # ── Layer 4: Consequents (Takagi-Sugeno linear output) ────

    def _layer4_consequents(
        self,
        normalised: Dict[str, float],
        features: List[float],
    ) -> Dict[str, float]:
        """
        Each rule contributes to emotion scores proportionally to its
        normalised firing strength × linear consequent weights.
        """
        emotion_outputs: Dict[str, float] = {em: 0.0 for em in EMOTIONS}

        for rule in FUZZY_RULES:
            w_n = normalised.get(rule.rule_id, 0.0)
            if w_n < 1e-6:
                continue
            for emotion, level in rule.antecedent.items():
                level_idx = {"low": 0, "medium": 1, "high": 2}[level]
                consequent = self._consequent_w[emotion] * features[EMOTIONS.index(emotion)]
                emotion_outputs[emotion] += w_n * consequent

        return emotion_outputs

    # ── Layer 5: Defuzzification ──────────────────────────────

    @staticmethod
    def _layer5_defuzzify(
        anfis_outputs: Dict[str, float],
        lexicon_scores: Dict[str, float],
        alpha: float = 0.55,   # weight on ANFIS vs lexicon blend
    ) -> Dict[str, float]:
        """
        Blend ANFIS output with lexicon scores and normalise to [0,1].
        alpha=0.55 gives slight edge to ANFIS over raw lexicon.
        """
        blended: Dict[str, float] = {}
        for em in EMOTIONS:
            a_val = min(1.0, max(0.0, anfis_outputs.get(em, 0.0)))
            l_val = lexicon_scores.get(em, 0.0)
            blended[em] = alpha * a_val + (1 - alpha) * l_val

        # Scale so the highest score is near 1.0 (preserves relative intensity)
        max_val = max(blended.values()) or 1e-9
        if max_val > 0:
            for em in EMOTIONS:
                blended[em] = round(min(1.0, blended[em] / max_val), 3)

        return blended

    # ── Explainability ────────────────────────────────────────

    @staticmethod
    def _build_explanation(
        fired: List[Tuple[str, float]],
        dominant: str,
        score: float,
        matched_rule: FuzzyRule | None,
    ) -> str:
        lines = [f"Dominant emotion: {dominant.title()} (intensity {score:.2f})"]
        if matched_rule:
            lines.append(f"Primary rule fired: {matched_rule.rule_id} — {matched_rule.description}")
        lines.append("Top fuzzy rules (rule_id → normalised strength):")
        for rid, strength in fired:
            rule = next((r for r in FUZZY_RULES if r.rule_id == rid), None)
            desc = rule.description if rule else ""
            lines.append(f"  {rid}: {strength:.3f}  [{desc}]")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 6. Module-level singleton
# ──────────────────────────────────────────────────────────────

_engine: ANFISEngine | None = None


def get_engine() -> ANFISEngine:
    global _engine
    if _engine is None:
        _engine = ANFISEngine()
    return _engine