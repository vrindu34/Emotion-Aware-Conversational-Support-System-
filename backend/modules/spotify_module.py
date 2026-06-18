"""
spotify_module.py
─────────────────
Authenticates with Spotify via Client Credentials flow and retrieves
emotion-appropriate track recommendations.

Emotion → Spotify audio-feature targets:
  sadness    → low valence, low energy, slow tempo
  anxiety    → medium valence, high energy, fast tempo
  anger      → low valence, high energy, loud
  happiness  → high valence, high energy
  fear       → low valence, low energy
  disgust    → low valence, medium energy
  surprise   → high valence, medium-high energy
  neutral    → medium valence, medium energy

The module uses Spotify's Search API with genre-based queries
to find emotion-appropriate tracks.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_RECOMMEND_URL = "https://api.spotify.com/v1/recommendations"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

# ──────────────────────────────────────────────────────────────
# Emotion → audio feature targets
# ──────────────────────────────────────────────────────────────

EMOTION_AUDIO_FEATURES: Dict[str, Dict[str, float]] = {
    "sadness": {
        "target_valence": 0.20,
        "target_energy": 0.25,
        "target_tempo": 72.0,
        "target_acousticness": 0.80,
        "target_instrumentalness": 0.40,
    },
    "anxiety": {
        "target_valence": 0.35,
        "target_energy": 0.30,   # deliberately LOWER — calm, not hype
        "target_tempo": 68.0,
        "target_acousticness": 0.75,
        "target_instrumentalness": 0.50,
    },
    "anger": {
        "target_valence": 0.30,
        "target_energy": 0.80,
        "target_tempo": 140.0,
        "target_loudness": -4.0,
        "target_acousticness": 0.10,
    },
    "happiness": {
        "target_valence": 0.88,
        "target_energy": 0.80,
        "target_tempo": 120.0,
        "target_danceability": 0.75,
        "target_acousticness": 0.20,
    },
    "fear": {
        "target_valence": 0.20,
        "target_energy": 0.20,
        "target_tempo": 65.0,
        "target_acousticness": 0.85,
        "target_instrumentalness": 0.60,
    },
    "disgust": {
        "target_valence": 0.28,
        "target_energy": 0.45,
        "target_tempo": 90.0,
        "target_acousticness": 0.50,
    },
    "surprise": {
        "target_valence": 0.70,
        "target_energy": 0.72,
        "target_tempo": 115.0,
        "target_danceability": 0.65,
    },
    "neutral": {
        "target_valence": 0.50,
        "target_energy": 0.50,
        "target_tempo": 100.0,
        "target_acousticness": 0.40,
    },
}

# Emotion → Spotify genre seeds (max 5 combined seeds)
EMOTION_GENRES: Dict[str, List[str]] = {
    "sadness":   ["acoustic", "sad", "piano", "indie", "singer-songwriter"],
    "anxiety":   ["ambient", "chill", "meditation", "piano", "new-age"],
    "anger":     ["rock", "metal", "hard-rock", "punk", "alternative"],
    "happiness": ["pop", "happy", "dance", "funk", "disco"],
    "fear":      ["ambient", "classical", "sleep", "new-age", "piano"],
    "disgust":   ["alternative", "indie", "post-punk", "grunge", "emo"],
    "surprise":  ["pop", "electro", "dance", "indie-pop", "synth-pop"],
    "neutral":   ["pop", "indie", "folk", "acoustic", "chill"],
}

# Search queries per emotion — used by the Search API fallback
EMOTION_SEARCH_QUERIES: Dict[str, List[str]] = {
    "sadness":   ["sad acoustic songs", "melancholy indie", "emotional piano"],
    "anxiety":   ["calm ambient music", "relaxing meditation", "peaceful piano"],
    "anger":     ["hard rock energy", "metal intense", "punk rock"],
    "happiness": ["happy pop hits", "feel good dance", "upbeat funk"],
    "fear":      ["peaceful classical", "calming sleep music", "soft ambient"],
    "disgust":   ["alternative indie", "grunge rock", "post punk"],
    "surprise":  ["upbeat pop", "electronic dance", "indie pop energetic"],
    "neutral":   ["chill indie pop", "easy listening folk", "acoustic pop"],
}

# Fallback curated playlists per emotion (used when API is unavailable)
FALLBACK_PLAYLISTS: Dict[str, List[Dict[str, str]]] = {
    "sadness": [
        {"name": "Sad Songs", "artist": "Adele", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1"},
        {"name": "Sad Indie", "artist": "Various Artists", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DWZUAyxnpnPiZ"},
    ],
    "anxiety": [
        {"name": "Calm Vibes", "artist": "Various Artists", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DX3Ogo9pFvBkY"},
        {"name": "Deep Focus", "artist": "Various Artists", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ"},
    ],
    "anger": [
        {"name": "Anger Management", "artist": "Various Artists", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DWTggY0yqBxES"},
    ],
    "happiness": [
        {"name": "Happy Hits!", "artist": "Various Artists", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC"},
        {"name": "Good Vibes", "artist": "Various Artists", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DX0UrRvztWcAU"},
    ],
    "fear": [
        {"name": "Peaceful Piano", "artist": "Various Artists", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO"},
    ],
    "neutral": [
        {"name": "Chill Hits", "artist": "Various Artists", "spotify_url": "https://open.spotify.com/playlist/37i9dQZF1DX0MLFaUdXnjA"},
    ],
}


@dataclass
class SpotifyTrack:
    name: str
    artist: str
    album: str
    preview_url: Optional[str]
    spotify_url: str
    valence: Optional[float] = None
    energy: Optional[float] = None
    tempo: Optional[float] = None


class SpotifyModule:
    """
    Handles Spotify Client Credentials auth and emotion-based recommendations.
    Uses Search API for track discovery (Recommendations API deprecated for new apps).
    Gracefully degrades to fallback playlists when credentials are absent.
    """

    def __init__(self) -> None:
        self._client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
        self._client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._available = bool(self._client_id and self._client_secret
                               and self._client_id != "your_spotify_client_id_here")

    # ── public interface ──────────────────────────────────────

    def get_tracks_for_emotion(
        self,
        dominant_emotion: str,
        emotion_scores: Dict[str, float],
        n_tracks: int = 3,
    ) -> Dict:
        """
        Return track recommendations for the given emotion state.
        Falls back to curated playlists if Spotify API is unavailable.
        """
        emotion = dominant_emotion if dominant_emotion in EMOTION_AUDIO_FEATURES else "neutral"

        if not self._available:
            return self._fallback_response(emotion)

        try:
            token = self._get_access_token()
            if not token:
                return self._fallback_response(emotion)
            tracks = self._fetch_recommendations(emotion, emotion_scores, token, n_tracks)
            if not tracks:
                return self._fallback_response(emotion)
            return {
                "source": "spotify_api",
                "emotion": emotion,
                "tracks": [self._track_to_dict(t) for t in tracks],
                "message": self._suggestion_message(emotion),
            }
        except Exception as exc:
            return {
                "source": "fallback",
                "emotion": emotion,
                "tracks": FALLBACK_PLAYLISTS.get(emotion, FALLBACK_PLAYLISTS["neutral"]),
                "message": self._suggestion_message(emotion),
                "error": str(exc),
            }

    # ── auth ──────────────────────────────────────────────────

    def _get_access_token(self) -> Optional[str]:
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        resp = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(self._client_id, self._client_secret),
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        return self._access_token

    # ── recommendations (Search API) ──────────────────────────

    def _fetch_recommendations(
        self,
        emotion: str,
        scores: Dict[str, float],
        token: str,
        n: int,
    ) -> List[SpotifyTrack]:
        """
        Uses Spotify Search API to find emotion-appropriate tracks.
        The Recommendations endpoint was deprecated for new apps in 2024.
        """
        queries = EMOTION_SEARCH_QUERIES.get(emotion, ["chill music"])
        headers = {"Authorization": f"Bearer {token}"}
        tracks = []

        for query in queries:
            if len(tracks) >= n:
                break
            params = {
                "q": query,
                "type": "track",
                "limit": 2,
                "market": "US",
            }
            resp = requests.get(
                SPOTIFY_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            items = resp.json().get("tracks", {}).get("items", [])
            for t in items:
                if len(tracks) < n:
                    tracks.append(self._parse_track(t))

        return tracks

    @staticmethod
    def _blend_features(
        base: Dict[str, float],
        scores: Dict[str, float],
        primary: str,
    ) -> Dict[str, float]:
        """
        Slightly nudge valence/energy based on secondary emotion scores.
        E.g. high happiness + anxiety → slightly raise energy target.
        """
        blended = dict(base)
        secondary_happiness = scores.get("happiness", 0)
        secondary_anxiety = scores.get("anxiety", 0)

        if primary in ("sadness", "fear") and secondary_happiness > 0.3:
            # A bit of hope — raise valence slightly
            blended["target_valence"] = min(0.6, blended.get("target_valence", 0.2) + 0.10)

        if primary == "anxiety" and secondary_anxiety > 0.6:
            # Very anxious — go even calmer
            blended["target_energy"] = max(0.10, blended.get("target_energy", 0.30) - 0.10)
            blended["target_tempo"] = max(55.0, blended.get("target_tempo", 68.0) - 8.0)

        return blended

    @staticmethod
    def _parse_track(t: Dict) -> SpotifyTrack:
        artists = ", ".join(a["name"] for a in t.get("artists", []))
        return SpotifyTrack(
            name=t.get("name", "Unknown"),
            artist=artists,
            album=t.get("album", {}).get("name", ""),
            preview_url=t.get("preview_url"),
            spotify_url=t.get("external_urls", {}).get("spotify", ""),
        )

    @staticmethod
    def _track_to_dict(track: SpotifyTrack) -> Dict:
        return {
            "name": track.name,
            "artist": track.artist,
            "album": track.album,
            "preview_url": track.preview_url,
            "spotify_url": track.spotify_url,
        }

    @staticmethod
    def _fallback_response(emotion: str) -> Dict:
        return {
            "source": "fallback_playlist",
            "emotion": emotion,
            "tracks": FALLBACK_PLAYLISTS.get(emotion, FALLBACK_PLAYLISTS["neutral"]),
            "message": SpotifyModule._suggestion_message(emotion),
        }

    @staticmethod
    def _suggestion_message(emotion: str) -> str:
        messages = {
            "sadness":  "Some gentle music that might hold space with you:",
            "anxiety":  "Here's some calming music to help ease the tension:",
            "anger":    "Music that matches and channels that energy:",
            "happiness":"A soundtrack for your good mood:",
            "fear":     "Some soothing sounds to help you feel safer:",
            "disgust":  "Something to shift the atmosphere:",
            "surprise": "Music to ride that wave of feeling:",
            "neutral":  "Some easygoing music to accompany you:",
        }
        return messages.get(emotion, "Some music that might help:")


# Module singleton
_spotify: SpotifyModule | None = None


def get_spotify_module() -> SpotifyModule:
    global _spotify
    if _spotify is None:
        _spotify = SpotifyModule()
    return _spotify