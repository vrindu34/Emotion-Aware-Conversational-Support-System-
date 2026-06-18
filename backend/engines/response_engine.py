"""
response_engine.py
──────────────────
Maps emotion intensities + ANFIS response labels to empathetic, context-aware
conversational responses.

ARCHITECTURE (priority order in generate()):
  1. Context signal override  – phrase-pattern match on raw user text
  2. Mixed-emotion composer   – detects 2+ significant emotions and builds a
                               response that acknowledges ALL of them, not just dominant
  3. Single-emotion template  – fallback when emotion is genuinely singular
  4. Trend-aware prefix       – prepended when persistent pattern detected in history

Response labels (from fuzzy rules):
  calm_supportive    – high anxiety
  empathetic_deep    – high sadness
  warm_checking      – medium mixed
  de_escalate        – anger
  celebrate_engage   – happiness / positive surprise
  positive_reinforce – mild happiness
  reassure_ground    – fear
  validate_redirect  – disgust
  curious_engage     – neutral / low emotion
  empathetic_mixed   – anger + sadness together
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

from .anfis_engine import ANFISResult, EMOTIONS


# ──────────────────────────────────────────────────────────────
# Context signal detection
# ──────────────────────────────────────────────────────────────

CONTEXT_SIGNALS: Dict[str, List[str]] = {
    "solitude_positive": [
        "alone", "own", "myself", "independent", "solitude", "by myself",
        "on my own", "need anyone", "don't need", "not need", "without",
        "self sufficient", "self-sufficient", "learning to be",
    ],
    "recovery": [
        "better", "improving", "getting there", "slowly", "step by step",
        "day by day", "healing", "working on", "trying", "progress",
        "used to be", "compared to", "than before", "than yesterday",
        "than last", "slowly getting",
    ],
    "mixed_emotions": [
        "but also", "at the same time", "both", "kind of", "sort of",
        "mixed", "confusing", "complicated", "weird feeling", "strange",
        "not sure how", "don't know how", "hard to explain",
    ],
    "loneliness": [
        "lonely", "alone", "no one", "nobody", "isolated", "by myself",
        "no friends", "no one to talk", "no one around",
    ],
    "existential": [
        "is it good", "is it bad", "is that normal", "should i", "is this okay",
        "is that okay", "what does it mean", "does it mean", "why do i",
        "is it healthy", "is it wrong", "is it right",
    ],
    "overthinking": [
        "overthink", "overthinking", "can't stop thinking", "keeps coming back",
        "in my head", "my head", "mind won't", "racing thoughts",
    ],
    "exhaustion": [
        "tired", "exhausted", "drained", "no energy", "can't do anything",
        "don't want to", "don't feel like", "unmotivated",
    ],
    "asking_for_opinion": [
        "is that good", "is that bad", "is this good", "is this bad",
        "good or bad", "bad or good", "what do you think", "do you think",
        "is it okay", "is it fine", "is it normal", "should i be",
        "is that healthy", "is it healthy",
    ],
    "betrayal": [
        "betrayed", "betrayal", "stabbed in the back", "lied to", "used",
        "let down", "let me down", "hurt by", "trusted", "trust broken",
        "trust issues", "cant trust", "disappointed by", "took advantage",
    ],
    "acceptance_growth": [
        "learned a lesson", "learned from", "lesson learned", "good for me",
        "for the best", "it was for the best", "i know it happened for",
        "needed to happen", "made me stronger", "grew from", "growth",
        "moving on", "moved on", "acceptance", "accept it", "at peace",
        "i understand now", "makes sense now", "i get it now",
    ],
    "bittersweet": [
        "weird", "strange feeling", "feels weird", "feels strange",
        "dont know how to feel", "don't know how to feel",
        "not sure how to feel", "hard to feel", "complicated feelings",
        "bittersweet", "mixed feelings",
    ],
}


def detect_context(text: str) -> List[str]:
    """Return list of detected context signal keys from user text."""
    text_lower = text.lower()
    detected = []
    for signal, phrases in CONTEXT_SIGNALS.items():
        for phrase in phrases:
            if phrase in text_lower:
                detected.append(signal)
                break
    return detected


# ──────────────────────────────────────────────────────────────
# Context-aware response overrides (phrase-pattern based)
# ──────────────────────────────────────────────────────────────

CONTEXT_RESPONSES: Dict[str, List[str]] = {
    "solitude_positive+asking_for_opinion": [
        "That's a really thoughtful question. Learning to be comfortable with yourself — not out of pain, but out of genuine independence — is one of the most grounding things a person can do. It's not about not needing people, it's about not *depending* on them for your sense of okayness. That's actually quite healthy. How does it feel from the inside?",
        "There's a difference between loneliness and solitude, and it sounds like you might be moving toward the latter. Solitude — choosing to be okay with yourself — is something a lot of people never find. Whether it's 'good' or 'bad' depends on how it feels to you. Does it feel like peace, or like something you're forcing yourself to accept?",
        "Honestly? Learning to be okay on your own is a real strength. It doesn't mean you'll never need anyone — it means you stop being afraid of being with yourself. That's different from shutting people out. How does it feel day to day?",
    ],
    "loneliness+solitude_positive": [
        "That's a really complex feeling to sit with — lonely, but also discovering something in the solitude. Those two things can both be true at once. How long have you been feeling this way?",
        "It sounds like you're going through a kind of quiet transition — learning what it means to be okay on your own, even when part of you misses connection. That's not easy, but it's meaningful. What's been making you feel lonely?",
        "Lonely but also growing — that's not a contradiction, that's just being human. Some of the most important things we learn about ourselves happen in the quiet. What does 'doing everything on your own' feel like for you?",
    ],
    "mixed_emotions": [
        "It sounds like you're holding two things at once — and that's actually really honest. Emotions aren't always clean or simple. Can you tell me more about the part that feels complicated?",
        "Mixed feelings like that are really common, even if they're hard to explain. You don't have to pick one. What does the heavier side of it feel like?",
        "That kind of 'both at the same time' feeling can be confusing to sit with. I'd rather understand both sides than just the obvious one — what's the part that feels harder to name?",
    ],
    "recovery": [
        "It sounds like you've come a real distance from where you were. That matters, even if it doesn't always feel dramatic. What's been helping you get here?",
        "Progress like that — even quiet, slow progress — is worth acknowledging. How does it feel compared to where you were?",
        "It's good to hear you're doing better than before. Sometimes that's all we can ask for. What's shifted for you?",
    ],
    "existential": [
        "That's a genuinely good question to sit with. There's no universal answer, but I'm curious what *you* think — does it feel right to you?",
        "I think the fact that you're asking that question means you're paying attention to yourself, which is already something. What's making you wonder about it?",
        "Rather than me giving you a verdict — what does your gut say? Does it feel healthy or does something feel off about it?",
    ],
    "exhaustion": [
        "That kind of tiredness that goes beyond just sleep — it's real and it's heavy. How long have you been feeling this way?",
        "Running on empty is its own kind of hard. What's been draining you most?",
        "Sometimes the body and mind just hit a wall. Is this a new feeling or has it been building for a while?",
    ],
    "betrayal+acceptance_growth": [
        "Feeling betrayed and also knowing it led to growth — those two things can coexist, and it takes a lot of self-awareness to hold both at once. What's the part that still stings the most?",
        "Learning something important from pain doesn't make the hurt disappear. It sounds like you're carrying both the wound and the wisdom from it. How are you doing with that?",
        "That's a really honest and complex place to be — you know it happened for a reason, but it still hurts. Which side of that feels heavier right now?",
    ],
    "betrayal+bittersweet": [
        "Feeling betrayed but also not quite knowing how to process it — that's a really honest place to be. The hurt is real even when part of you understands. What's been going through your mind most?",
        "Feeling betrayed but also at peace with it is a strange in-between place. You don't have to rush past either feeling. What's the part that still feels unresolved?",
        "It makes sense that it feels weird — you're processing something that's both painful and clarifying at the same time. What does the hurt part feel like?",
    ],
    "acceptance_growth+bittersweet": [
        "Knowing something was for the best doesn't always make the feelings clean or simple. That 'weird' feeling is real — you can be at peace with the outcome and still feel the loss. How long have you been carrying this?",
        "There's something honest about saying 'it's good for me, but it still feels weird.' That's not confusion — that's you being real with yourself. What's the weird feeling about, exactly?",
    ],
    "betrayal": [
        "Being betrayed by someone you trusted — that's one of the sharper kinds of hurt. How are you doing with it?",
        "That kind of betrayal can shake your sense of trust in ways that go beyond just the person involved. How long have you been sitting with this?",
        "I'm really sorry that happened. Betrayal by someone close to you is its own category of painful. Do you want to talk about what happened?",
    ],
    "bittersweet": [
        "That 'don't know how to feel' space is real — sometimes emotions don't arrive as one clean thing. What's the part that's hardest to name?",
        "When something is both hard and okay at the same time, it can be genuinely disorienting. What feels most unresolved about it?",
    ],
}


def get_context_response(signals: List[str]) -> Optional[str]:
    """
    Check if any combination of signals matches a context override.
    Returns a response string if matched, None otherwise.
    """
    # Check compound keys first (more specific)
    for key in CONTEXT_RESPONSES:
        if "+" in key:
            parts = key.split("+")
            if all(p in signals for p in parts):
                return random.choice(CONTEXT_RESPONSES[key])

    # Then single signal keys
    for signal in signals:
        if signal in CONTEXT_RESPONSES:
            return random.choice(CONTEXT_RESPONSES[signal])

    return None


# ──────────────────────────────────────────────────────────────
# Mixed-emotion composer
# Builds responses that name and hold MULTIPLE emotions at once
# ──────────────────────────────────────────────────────────────

# Threshold: an emotion must score above this to count as "present"
SECONDARY_THRESHOLD = 0.28

# Emotion pair → response templates that name BOTH emotions explicitly.
# Keys are frozensets so order doesn't matter.
# Each template uses {em1} and {em2} placeholders filled at runtime
# with human-readable emotion labels.
MIXED_PAIR_TEMPLATES: Dict[Tuple[str, str], Dict[str, List[str]]] = {
    # sadness + anger  (hurt + frustration, betrayal, injustice)
    ("sadness", "anger"): {
        "moderate": [
            "I'm hearing both real hurt and real frustration in what you're sharing — and those two things make complete sense together. Neither cancels the other out. Which part do you most want to sit with right now?",
            "That sounds like it's both painful and infuriating at the same time. You don't have to pick one feeling to lead with. What happened?",
            "Hurt and angry at the same time — that's one of the harder combinations to carry. I want to hear both sides of it. Tell me more.",
        ],
        "severe": [
            "I can hear real pain and real anger in what you're saying. Both of those are completely valid, and I'm not going to try to simplify this into one thing. Let's go through it together — what's the full picture?",
            "That sounds like it cut deep and made you furious at the same time. You're allowed to feel both. What happened?",
        ],
        "mild": [
            "It sounds like there's a mix of hurt and frustration underneath this. I'd rather hear all of it than just one part — where do you want to start?",
        ],
    },
    # anxiety + sadness  (overwhelmed + low, burnout, grief with worry)
    ("anxiety", "sadness"): {
        "moderate": [
            "I'm picking up on both some heaviness and some stress in what you're sharing. That's a draining combination — feeling down and also unsettled at the same time. What's been going on?",
            "It sounds like you're carrying both worry and sadness at once — and that's a lot. I'm here for all of it. What feels most pressing?",
            "There's something both heavy and anxious in what you're describing. You don't have to untangle it before we talk — just tell me what's been happening.",
        ],
        "severe": [
            "That sounds exhausting — being both deeply sad and really overwhelmed at the same time. I want you to know you don't have to hold all of this alone. What's been weighing on you most?",
            "I hear a lot going on — both real pain and real pressure. Let's slow down. You don't have to have it figured out. Just tell me what's sitting heaviest.",
        ],
        "mild": [
            "Something feels both a little sad and a little unsettled in what you're sharing. I'm here — what's been on your mind?",
        ],
    },
    # fear + sadness  (grief with dread, loss with uncertainty)
    ("fear", "sadness"): {
        "moderate": [
            "I'm sensing both some real sadness and some fear in what you're sharing — and those often go together. It's okay to feel both. What's been happening?",
            "That sounds both painful and frightening to sit with. You don't have to manage either feeling on your own. What's going on?",
        ],
        "severe": [
            "What you're describing sounds both deeply sad and genuinely scary. That's a really heavy place to be. I'm here — take your time and tell me what's happening.",
            "I hear both real grief and real fear in what you're sharing. Both of those deserve space. Can you tell me more about what's going on?",
        ],
        "mild": [
            "It sounds like there's something both a little sad and a little worrying sitting with you. I'm here — what's on your mind?",
        ],
    },
    # anxiety + anger  (stress + frustration, injustice + overwhelm)
    ("anxiety", "anger"): {
        "moderate": [
            "It sounds like you're both really stressed and pretty frustrated right now — and those two feed into each other. What's been going on?",
            "I'm hearing both pressure and some real anger in what you're sharing. That's a lot to sit with at once. What's the main thing that's been getting to you?",
        ],
        "severe": [
            "That sounds both incredibly stressful and genuinely infuriating. I don't want to simplify it — both sides are real. What's been happening?",
            "Being overwhelmed and furious at the same time is exhausting. I hear both. Let's slow down — what's at the centre of this?",
        ],
        "mild": [
            "Something feels both a little stressed and a little frustrated in what you're sharing. What's been going on?",
        ],
    },
    # happiness + sadness  (bittersweet, mixed relief, transitions)
    ("happiness", "sadness"): {
        "moderate": [
            "It sounds like you're feeling two things at once — something that's good, and something that still hurts. That's not a contradiction. Which side feels more present right now?",
            "Feeling both happy and sad at the same time is one of the more honest and complex emotions there is. I'd rather sit with both than pick one. What's going on?",
            "That bittersweet feeling — something good alongside something painful — can be genuinely hard to process. Tell me more about both sides.",
        ],
        "mild": [
            "It sounds like there's something good and something heavy mixed together. Both are worth talking about — what's the fuller picture?",
            "That mix of feeling okay and also feeling sad is real and valid. What's been happening?",
        ],
        "severe": [
            "It sounds like you're holding something that's both meaningful and painful at the same time — and that's genuinely hard to sit with. I'm here for both sides. What's going on?",
        ],
    },
    # happiness + anxiety  (excited but scared, good news with nerves)
    ("happiness", "anxiety"): {
        "moderate": [
            "It sounds like something exciting is mixed with some nerves — and that combination makes a lot of sense. What's going on?",
            "Feeling happy but also a little on edge at the same time — that's really common when something big is happening. Tell me more.",
        ],
        "mild": [
            "Something good seems to be happening, but there's also some tension in it. What's the full picture?",
        ],
        "severe": [
            "I'm picking up on something that's both really exciting and also genuinely overwhelming. Both of those are worth talking about. What's going on?",
        ],
    },
    # happiness + fear  (good things with underlying dread, hope mixed with worry)
    ("happiness", "fear"): {
        "moderate": [
            "It sounds like something positive is happening, but there's also something that feels a little frightening underneath it. That combination is real — what's going on?",
            "Feeling happy but also a bit scared at the same time — I'd like to understand both sides. Tell me more.",
        ],
        "mild": [
            "Something good seems mixed with something that feels a bit unsettling. I'm here for both — what's been happening?",
        ],
        "severe": [
            "I'm hearing something that's both hopeful and also genuinely scary at the same time. That's a lot to hold. What's going on?",
        ],
    },
    # anger + fear  (rage with vulnerability, threat response)
    ("anger", "fear"): {
        "moderate": [
            "I'm hearing both real anger and something that sounds a bit frightening underneath it — and those often go together when we feel threatened or wronged. What's been happening?",
            "Anger and fear at the same time is one of the hardest combinations to sit with. Both make sense. Can you tell me more about what's going on?",
        ],
        "severe": [
            "That sounds both genuinely frightening and absolutely maddening. I'm not going to ask you to pick one — both are real. What happened?",
            "I hear both real fear and real anger in what you're sharing. You're not alone in this. Let's go through it — what's happening?",
        ],
        "mild": [
            "Something feels both a little unsettling and a bit frustrating in what you're sharing. What's been going on?",
        ],
    },
    # disgust + anger  (moral outrage, violation)
    ("disgust", "anger"): {
        "moderate": [
            "That sounds genuinely revolting and also completely infuriating. Both of those reactions make complete sense. What happened?",
            "I'm hearing both disgust and real anger — and that combination usually means something important was violated. Tell me more.",
        ],
        "severe": [
            "That sounds absolutely maddening and deeply wrong at the same time. Your reaction makes complete sense. What happened?",
        ],
        "mild": [
            "Something feels both unsettling and frustrating in what you're sharing. What's been going on?",
        ],
    },
    # sadness + disgust  (shame, loss with revulsion)
    ("sadness", "disgust"): {
        "moderate": [
            "That sounds both painful and really unpleasant to sit with. Both of those reactions make sense. Tell me what's been happening.",
            "I'm picking up on both some sadness and some real discomfort in what you're sharing. You don't have to hold that alone — what's going on?",
        ],
        "mild": [
            "Something feels both sad and a bit off-putting in what you're sharing. I'd like to hear more — what's been happening?",
        ],
        "severe": [
            "That sounds genuinely painful and deeply unsettling at the same time. Both of those feelings deserve to be heard. What's going on?",
        ],
    },
}

# Human-readable names for emotion labels used in fallback blended responses
EMOTION_DISPLAY: Dict[str, str] = {
    "sadness": "sadness",
    "anxiety": "anxiety",
    "anger": "anger",
    "happiness": "something positive",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "a kind of calm",
}

# Generic multi-emotion fallback (when pair not in MIXED_PAIR_TEMPLATES)
GENERIC_MIXED_TEMPLATES = [
    "It sounds like there's more than one thing going on emotionally — and I'd rather hear all of it than just one part. What feels most present right now?",
    "I'm picking up on a few different feelings in what you're sharing. You don't have to simplify it for me. What's the part that's sitting heaviest?",
    "That sounds like a mix of things to be feeling at once — and that's completely valid. Where do you want to start?",
    "Emotions don't always come one at a time, and it sounds like you're holding a few things together right now. Tell me more about what's going on.",
]


def get_mixed_emotion_response(result: ANFISResult) -> Optional[str]:
    """
    If 2+ emotions are above SECONDARY_THRESHOLD, compose a response that
    names and holds all of them. Returns None if emotion is essentially singular.

    Priority:
      1. Specific pair template (MIXED_PAIR_TEMPLATES) for the top 2 emotions
      2. Generic multi-emotion template
    """
    scores = result.emotion_scores
    severity = result.severity

    # Get all emotions above threshold, sorted by score desc
    significant = sorted(
        [(em, sc) for em, sc in scores.items()
         if sc >= SECONDARY_THRESHOLD and em != "neutral"],
        key=lambda x: -x[1],
    )

    # Only proceed if 2+ significant emotions present
    if len(significant) < 2:
        return None

    top1, top2 = significant[0][0], significant[1][0]

    # Look up pair template (order-independent)
    pair_key = tuple(sorted([top1, top2]))
    pair_templates = MIXED_PAIR_TEMPLATES.get(pair_key)  # type: ignore[call-overload]

    if pair_templates:
        pool = pair_templates.get(severity, pair_templates.get("moderate", []))
        if pool:
            return random.choice(pool)

    # Fallback: generic mixed template
    return random.choice(GENERIC_MIXED_TEMPLATES)


# ──────────────────────────────────────────────────────────────
# Single-emotion label correction
# Prevents positive templates firing when dominant emotion is negative
# ──────────────────────────────────────────────────────────────

def correct_label(result: ANFISResult) -> str:
    """
    If the fuzzy rule picked a positive label (celebrate_engage, positive_reinforce)
    but the user's dominant emotion is actually negative, override to the correct label.
    """
    label = result.response_label
    dominant = result.dominant_emotion

    negative_emotions = {"sadness", "anger", "fear", "anxiety", "disgust"}
    positive_labels   = {"celebrate_engage", "positive_reinforce"}

    if label in positive_labels and dominant in negative_emotions:
        overrides = {
            "anger":   "de_escalate",
            "sadness": "empathetic_deep",
            "fear":    "reassure_ground",
            "anxiety": "calm_supportive",
            "disgust": "validate_redirect",
        }
        return overrides.get(dominant, "warm_checking")

    return label


# ──────────────────────────────────────────────────────────────
# Generic template pool (single-emotion fallback)
# ──────────────────────────────────────────────────────────────

RESPONSE_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "calm_supportive": {
        "mild": [
            "It sounds like things feel a bit much right now. Take a breath—I'm here. What's on your mind?",
            "I can sense some tension in what you're sharing. Want to talk through what's weighing on you?",
        ],
        "moderate": [
            "That sounds really stressful. You don't have to carry this alone—can you tell me more about what's overwhelming you?",
            "I hear you. When everything piles up like that, it can feel impossible. Let's slow down together. What feels most pressing?",
            "It makes sense to feel this way given what you're dealing with. I'm here, and we can take it one piece at a time.",
        ],
        "severe": [
            "That sounds like a lot to hold at once. I'm really glad you're talking about it. Can you take one slow breath with me? Tell me what's making things feel so overwhelming.",
            "I can hear how much pressure you're under right now. You don't have to have it all figured out—just talking can help. What's the heaviest thing on your mind?",
            "It sounds absolutely exhausting. Please know you don't have to push through this alone. What would feel most helpful right now—talking it through, or just being heard?",
        ],
    },
    "empathetic_deep": {
        "mild": [
            "I'm sorry you're feeling this way. Sometimes sadness just settles in without a clear reason. I'm here if you want to talk.",
            "That sounds heavy. It's okay to sit with difficult feelings — you don't have to rush past them.",
        ],
        "moderate": [
            "What you're feeling sounds really painful. I want you to know that makes sense, and I'm not going anywhere. Would you like to share more?",
            "I'm really sorry you're going through this. Sadness like that deserves to be heard, not pushed away. What's been the hardest part?",
            "It takes courage to name how you're feeling. I'm here, and I genuinely want to understand what you're going through.",
        ],
        "severe": [
            "That sounds genuinely painful, and I don't want to rush past it. You matter, and what you're feeling matters. Can you tell me more about what's going on?",
            "I hear real pain in what you're sharing. You're not alone in this, and there's no pressure to feel better right away. I'm here with you.",
            "I'm sorry you're carrying that. Would it help to talk through what's been happening, or would you just like some company right now?",
        ],
    },
    "warm_checking": {
        "mild": [
            "How are you really doing? I'm here and happy to listen.",
            "Something seems to be sitting with you. Want to share what's on your mind?",
        ],
        "moderate": [
            "It sounds like things have been a bit rough. I'm here — no judgment, just listening. What's been going on?",
            "I'm picking up on some heaviness in what you're saying. Would it help to talk it through?",
            "You've been carrying a few things at once. How are you holding up honestly?",
        ],
        "severe": [
            "It sounds like you've been dealing with a lot lately. I want to check in properly — how are you really, underneath it all?",
            "I hear a mix of things in what you're sharing. I'm not going anywhere — tell me what's been happening for you.",
        ],
    },
    "de_escalate": {
        "mild": [
            "That would frustrate anyone. What's been going on?",
            "I can hear that something's been getting under your skin. Want to vent about it?",
        ],
        "moderate": [
            "That sounds genuinely infuriating. Your feelings are completely valid. Do you want to talk through what happened?",
            "I can understand why you'd feel that way — that sounds really unfair. Sometimes saying it out loud helps. What's the full story?",
            "That kind of frustration can be draining. I'm here to listen without judgment. Tell me more.",
        ],
        "severe": [
            "That sounds absolutely maddening, and I don't blame you for feeling this way. Before anything else — are you okay? Do you want to talk through what happened, step by step?",
            "I hear real anger in what you're sharing, and it sounds justified. Let's slow down and go through it together. I'm on your side.",
        ],
    },
    "celebrate_engage": {
        "mild": [
            "That's good to hear. What's been making things feel better?",
            "It sounds like something positive is happening for you. Tell me more.",
        ],
        "moderate": [
            "That genuinely makes me smile! What's been going so well? I want to hear about it.",
            "That's wonderful! It sounds like things are really clicking for you. What's been the highlight?",
        ],
        "severe": [
            "I can feel your excitement! This is great — what happened? Tell me everything.",
            "That's amazing! You sound really lit up. What's the big news?",
        ],
    },
    "positive_reinforce": {
        "mild": [
            "That's a good sign. It sounds like things are moving in a better direction. How does it feel?",
            "I'm glad to hear that. Even small good feelings are worth acknowledging.",
        ],
        "moderate": [
            "It's great that you're feeling this way. What do you think is behind it?",
            "That positivity comes through clearly. What's been contributing to that for you?",
        ],
        "severe": [
            "You sound genuinely happy right now, and that's worth sitting with. What's been happening?",
        ],
    },
    "reassure_ground": {
        "mild": [
            "That sounds unsettling. You're safe here — what's worrying you?",
            "It makes sense to feel that way. Would you like to talk through what's making you uneasy?",
        ],
        "moderate": [
            "I hear you — that sounds genuinely scary. You're not alone in this. Can you tell me more about what's happening?",
            "Fear like that is real and valid. Let's slow down together. What's the biggest concern on your mind right now?",
            "You're safe here. Take your time. What's been making you feel afraid?",
        ],
        "severe": [
            "That sounds terrifying, and I want you to know you don't have to face it alone. Take a breath — I'm right here. Can you tell me what's happening?",
            "I can hear how scared you are, and that matters. You're not alone. Let's take this one moment at a time — what feels most frightening right now?",
        ],
    },
    "validate_redirect": {
        "mild": [
            "That does sound unpleasant. It's okay to feel that way. Want to talk about it?",
            "I can understand why that would feel off-putting. What's been going on?",
        ],
        "moderate": [
            "That sounds genuinely awful, and your reaction makes complete sense. Do you want to work through it together?",
            "That does sound terrible. I don't blame you for reacting this way. What happened?",
        ],
        "severe": [
            "That sounds deeply unsettling and I completely understand why you feel the way you do. Can you tell me more so I can better understand what you're going through?",
        ],
    },
    "curious_engage": {
        "mild": [
            "What's been on your mind lately?",
            "I'm here and happy to chat — what's going on with you today?",
            "Tell me what's up. I'm all ears.",
        ],
        "moderate": [
            "I'm here and curious — what's on your mind?",
            "Sounds like you've got something to share. I'd love to hear it.",
        ],
        "severe": [
            "Something seems to be brewing. I'm here and happy to listen — what's going on?",
        ],
    },
    "empathetic_mixed": {
        "mild": [
            "It sounds like there's more than one feeling in there — and they don't all have to make sense together. What's sitting heaviest right now?",
            "That sounds like a complicated mix of things to feel at once. I'm here for all of it — where do you want to start?",
        ],
        "moderate": [
            "I hear something that's both hurt and angry at once — and that makes a lot of sense together. Which part of this do you most want to talk through?",
            "Hurt and frustrated at the same time is a real and valid place to be. Neither feeling cancels the other out. What happened?",
            "It sounds like there are a few layers to this. I don't want to simplify it — tell me more about what you're feeling.",
        ],
        "severe": [
            "That sounds like it cut deep — and it makes sense that you'd feel both wounded and furious about it. You don't have to pick one emotion to lead with. I'm here for the full picture.",
            "I hear real pain and real anger in what you're sharing. Both of those make complete sense. Let's go through this together — what happened?",
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# Response engine
# ──────────────────────────────────────────────────────────────

class ResponseEngine:
    """
    Selects an empathetic response based on ANFIS output + context signals.

    Priority order:
      1. Context-aware override (phrase-pattern match on raw user text)
      2. Mixed-emotion composer (2+ emotions above threshold → names them all)
      3. Single-emotion template (label + severity, with label correction)
      4. Trend-aware prefix added if applicable
    """

    def generate(
        self,
        result: ANFISResult,
        user_text: str = "",
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:

        # ── Step 1: Context signal override ──────────────────────
        signals = detect_context(user_text) if user_text else []
        context_response = get_context_response(signals)
        if context_response:
            return context_response

        # ── Step 2: Mixed-emotion composer ───────────────────────
        # If 2+ emotions are significantly present, build a response
        # that acknowledges ALL of them rather than collapsing to one.
        mixed_response = get_mixed_emotion_response(result)
        if mixed_response:
            # Add trend prefix if applicable even to mixed responses
            prefix = self._trend_prefix(result, conversation_history)
            return f"{prefix}{mixed_response}" if prefix else mixed_response

        # ── Step 3: Single-emotion template ──────────────────────
        label = correct_label(result)
        severity = result.severity

        pool = RESPONSE_TEMPLATES.get(label, RESPONSE_TEMPLATES["warm_checking"])
        severity_pool = pool.get(severity, pool.get("moderate", []))
        if not severity_pool:
            severity_pool = pool[list(pool.keys())[0]]

        prefix = self._trend_prefix(result, conversation_history)
        base = random.choice(severity_pool)
        return f"{prefix}{base}" if prefix else base

    # ── trend-aware prefix ─────────────────────────────────────

    @staticmethod
    def _trend_prefix(
        result: ANFISResult,
        history: Optional[List[Dict]],
    ) -> str:
        if not history or len(history) < 3:
            return ""

        anxiety_count = sum(
            1 for h in history[-4:]
            if h.get("emotion_scores", {}).get("anxiety", 0) > 0.5
        )
        sadness_count = sum(
            1 for h in history[-4:]
            if h.get("emotion_scores", {}).get("sadness", 0) > 0.5
        )

        if anxiety_count >= 3:
            return "You've seemed quite stressed across our recent conversation. "
        if sadness_count >= 3:
            return "You've been carrying a lot of sadness lately, and I want to acknowledge that. "

        return ""


# Module singleton
_response_engine: ResponseEngine | None = None


def get_response_engine() -> ResponseEngine:
    global _response_engine
    if _response_engine is None:
        _response_engine = ResponseEngine()
    return _response_engine