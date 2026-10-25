"""
Stat Tracker — pipeline entry point for per-event stat tracking.

Accepts raw events from the vision pipeline, enriches each with:
  - court zone classification
  - learning insight label + coaching detail (for misses and turnovers)
  - clip extraction window (2s lead, 5s trail)

Then forwards enriched events to BoxScoreManager for counting.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Optional

from src.logic.box_score_manager import BoxScoreManager

# ---------------------------------------------------------------------------
# Court geometry (NBA dimensions in metres)
# ---------------------------------------------------------------------------

COURT_WIDTH = 28.0       # baseline to baseline
COURT_HEIGHT = 15.0      # sideline to sideline
THREE_POINT_RADIUS = 7.24
PAINT_X_DEPTH = 5.8      # paint depth from baseline
PAINT_Y_HALF = 2.45      # half-width of paint


def classify_court_zone(x: float, y: float, attacking_hoop: str = "auto") -> str:
    """Return a zone label for a coordinate pair.

    Court model: hoop A at (0, 7.5), hoop B at (28, 7.5).
    attacking_hoop="auto" (default) infers the attacking direction from the
    nearest hoop — correct for shots and rebounds which happen near one basket.
    Pass "A" or "B" explicitly when the attacking team is known.
    """
    if attacking_hoop == "auto":
        dist_a = (x ** 2 + (y - COURT_HEIGHT / 2) ** 2) ** 0.5
        dist_b = ((x - COURT_WIDTH) ** 2 + (y - COURT_HEIGHT / 2) ** 2) ** 0.5
        attacking_hoop = "A" if dist_a <= dist_b else "B"

    hoop_x = 0.0 if attacking_hoop == "A" else COURT_WIDTH
    dist = ((x - hoop_x) ** 2 + (y - COURT_HEIGHT / 2) ** 2) ** 0.5

    if dist <= 1.5:
        return "restricted_area"
    if abs(x - hoop_x) < PAINT_X_DEPTH and abs(y - COURT_HEIGHT / 2) < PAINT_Y_HALF:
        return "paint"
    if dist <= THREE_POINT_RADIUS:
        return "mid_range"
    # Beyond the arc — front court or back court?
    in_front_court = (attacking_hoop == "A" and x < COURT_WIDTH / 2) or \
                     (attacking_hoop == "B" and x > COURT_WIDTH / 2)
    return "three_point" if in_front_court else "backcourt"


# ---------------------------------------------------------------------------
# Learning insight classification tables
# ---------------------------------------------------------------------------

# (zone, contested) → (insight_label, coaching_detail)
_MISS_INSIGHT: dict[tuple[str, bool], tuple[str, str]] = {
    ("restricted_area", True):  ("Shot Selection",  "Contested layup — consider a pass or pull-up earlier"),
    ("restricted_area", False): ("Mechanics",        "Open look at the rim — focus on touch and angle"),
    ("paint", True):            ("Shot Selection",  "Contested mid-post — explore post move or kick-out"),
    ("paint", False):           ("Footwork",         "Open paint touch — check pivot and release point"),
    ("mid_range", True):        ("Shot Selection",  "Contested mid-range — create space or find the open man"),
    ("mid_range", False):       ("Timing",           "Open mid-range miss — rhythm and release consistency"),
    ("three_point", True):      ("Shot Selection",  "Heavily contested three — look for cutter or drive"),
    ("three_point", False):     ("Mechanics",        "Open three miss — maintain consistent shooting form"),
    ("backcourt", True):        ("Decision Making",  "Backcourt shot under pressure — reset the offense"),
    ("backcourt", False):       ("Decision Making",  "Backcourt shot — advance the ball before shooting"),
}

_TURNOVER_INSIGHT: dict[str, dict[str, tuple[str, str]]] = {
    "turnover_steal": {
        "restricted_area": ("Ball Security",   "Lost ball at the rim — go up strong or kick out earlier"),
        "paint":           ("Ball Security",   "Ball stripped in traffic — use the off-arm as a shield"),
        "mid_range":       ("Ball Security",   "Dribble stolen — keep the dribble alive on drives"),
        "three_point":     ("Decision Making", "Stripped on the perimeter — see the help defender sooner"),
        "backcourt":       ("Spacing",         "Turnover in backcourt — advance with purpose"),
    },
    "turnover_unforced": {
        "restricted_area": ("Footwork",        "Travel or charge at rim — gather earlier before contact"),
        "paint":           ("Spacing",         "Out-of-bounds or bad pass in paint — read the defense"),
        "mid_range":       ("Decision Making", "Unforced error in mid-range — slow down, value possession"),
        "three_point":     ("Decision Making", "Errant pass on perimeter — trust the offense, don't force"),
        "backcourt":       ("Positioning",     "Backcourt violation or bad outlet — improve transition reads"),
    },
}

_POSITIVE_INSIGHT: dict[str, str] = {
    "two_pointer_made":   "Great execution",
    "three_pointer_made": "Excellent shot selection",
    "free_throw_made":    "Consistent free-throw routine",
    "assist":             "High basketball IQ — found the open man",
    "steal":              "Active hands and great anticipation",
    "block":              "Disciplined shot-blocking — stayed vertical",
    "rebound_offensive":  "Outstanding offensive positioning and effort",
    "rebound_defensive":  "Strong defensive rebounding instinct",
}

LEARNING_MOMENT_EVENTS = frozenset({
    "two_pointer_missed", "three_pointer_missed", "free_throw_missed",
    "turnover_steal", "turnover_unforced",
})

_LEARNING_CATEGORY = "Efficiency Improvement"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

REVIEW_THRESHOLD = 0.60  # events below this confidence get needs_review=True


@dataclass
class EnrichedEvent:
    event_id: str
    event_type: str
    player_id: str
    player_name: str
    team: str
    timestamp_ms: int
    frame_start: int
    coordinates: dict           # {"x": float, "y": float}
    court_zone: str
    sub_type: Optional[str]
    contested: bool
    mismatch: bool
    reason: str
    is_learning_moment: bool
    learning_category: Optional[str]
    learning_insight: Optional[str]   # "Ball Security" | "Shot Selection" | etc.
    learning_detail: Optional[str]    # full coaching sentence
    clip_window_ms: tuple[int, int]   # (start_ms, end_ms) for clip extraction
    confidence: float = 1.0           # 0.0–1.0; detection certainty
    game_state: str = "live"          # "live" | "dead_ball" | "free_throw"
    needs_review: bool = False        # True if confidence < REVIEW_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team": self.team,
            "timestamp_ms": self.timestamp_ms,
            "frame_start": self.frame_start,
            "coordinates": self.coordinates,
            "court_zone": self.court_zone,
            "sub_type": self.sub_type,
            "contested": self.contested,
            "mismatch": self.mismatch,
            "reason": self.reason,
            "is_learning_moment": self.is_learning_moment,
            "learning_category": self.learning_category,
            "learning_insight": self.learning_insight,
            "learning_detail": self.learning_detail,
            "clip_window_ms": list(self.clip_window_ms),
            "confidence": self.confidence,
            "game_state": self.game_state,
            "needs_review": self.needs_review,
        }


# ---------------------------------------------------------------------------
# Stat Tracker
# ---------------------------------------------------------------------------

class StatTracker:
    """
    Enriches raw vision-pipeline events with learning context, then forwards
    them to BoxScoreManager for counting.

    Raw event dict shape:
    {
        "event_type":    str,          # e.g. "two_pointer_missed"
        "player_id":     str,
        "player_name":   str,
        "team":          str,
        "timestamp_ms":  int,
        "frame_start":   int,
        "coordinates":   {"x": float, "y": float},
        "sub_type":      str | None,   # "swish" | "bank_shot" | "rim_in"
        "contested":     bool,
        "mismatch":      bool,
        "reason":        str,
        "event_id":      str,          # optional; generated if absent
    }
    """

    CLIP_LEAD_MS = 2_000    # 2 seconds before the event
    CLIP_TRAIL_MS = 5_000   # 5 seconds after → ~7s total clip

    def __init__(self, attacking_hoop: str = "A") -> None:
        self.box_score = BoxScoreManager()
        self.attacking_hoop = attacking_hoop
        self._events: list[EnrichedEvent] = []
        self._last_seen: dict = {}    # player_id → last sampled frame index seen
        self._seen_count: dict = {}   # player_id → total sampled frames appeared in

    # ------------------------------------------------------------------
    # On-court tracking
    # ------------------------------------------------------------------

    def mark_seen(self, player_id, sampled_frame_idx: int) -> None:
        """Record that player_id was visible in this sampled frame."""
        key = str(player_id)
        self._last_seen[key] = sampled_frame_idx
        self._seen_count[key] = self._seen_count.get(key, 0) + 1

    def apply_seen_counts(self) -> None:
        """Push seen-frame counts to PlayerStats so get_box_score can filter."""
        for pid, count in self._seen_count.items():
            ps = self.box_score._players.get(str(pid))
            if ps is not None:
                ps.frames_seen = count

    def compute_on_court(self, total_sampled: int, window_frac: float = 0.15) -> set:
        """
        Return the set of player_ids that were visible in the last
        window_frac fraction of sampled frames — these are 'on court'
        at the end of the clip. Everyone else was subbed out earlier.
        """
        if not self._last_seen or total_sampled == 0:
            return set()
        threshold = total_sampled * (1.0 - window_frac)
        return {pid for pid, f in self._last_seen.items() if f >= threshold}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(self, raw: dict) -> EnrichedEvent:
        """Enrich one raw event dict, update the box score, and store it."""
        enriched = self._enrich(raw)
        self.box_score.ingest_event(self._to_bsm_event(enriched))
        self._events.append(enriched)
        return enriched

    def track_all(self, raws: list[dict]) -> list[EnrichedEvent]:
        return [self.track(r) for r in raws]

    @property
    def events(self) -> list[EnrichedEvent]:
        return list(self._events)

    @property
    def learning_moments(self) -> list[EnrichedEvent]:
        return [e for e in self._events if e.is_learning_moment]

    def get_player_events(self, player_id: str) -> list[EnrichedEvent]:
        return [e for e in self._events if e.player_id == player_id]

    def export_events_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _enrich(self, raw: dict) -> EnrichedEvent:
        event_type = raw["event_type"]
        coords = raw.get("coordinates", {"x": 0.0, "y": 0.0})
        timestamp_ms = raw.get("timestamp_ms", 0)
        contested = bool(raw.get("contested", False))
        zone = classify_court_zone(
            coords.get("x", 0.0), coords.get("y", 0.0)
        )
        is_learning = event_type in LEARNING_MOMENT_EVENTS
        insight, detail = self._classify_insight(event_type, zone, contested)

        confidence = float(raw.get("confidence", 1.0))

        return EnrichedEvent(
            event_id=raw.get("event_id", str(uuid.uuid4())),
            event_type=event_type,
            player_id=str(raw.get("player_id", "unknown")),
            player_name=raw.get("player_name", "Unknown"),
            team=raw.get("team", "unknown"),
            timestamp_ms=timestamp_ms,
            frame_start=raw.get("frame_start", 0),
            coordinates=coords,
            court_zone=zone,
            sub_type=raw.get("sub_type"),
            contested=contested,
            mismatch=bool(raw.get("mismatch", False)),
            reason=raw.get("reason", ""),
            is_learning_moment=is_learning,
            learning_category=_LEARNING_CATEGORY if is_learning else None,
            learning_insight=insight,
            learning_detail=detail,
            clip_window_ms=(
                max(0, timestamp_ms - self.CLIP_LEAD_MS),
                timestamp_ms + self.CLIP_TRAIL_MS,
            ),
            confidence=confidence,
            game_state=raw.get("game_state", "live"),
            needs_review=confidence < REVIEW_THRESHOLD,
        )

    def _classify_insight(
        self, event_type: str, zone: str, contested: bool
    ) -> tuple[Optional[str], Optional[str]]:
        if event_type in {"two_pointer_missed", "three_pointer_missed"}:
            key = (zone, contested)
            return _MISS_INSIGHT.get(key, ("Mechanics", "Review shot mechanics for this zone"))

        if event_type == "free_throw_missed":
            return "Mechanics", "Maintain a consistent free-throw routine and follow-through"

        if event_type in _TURNOVER_INSIGHT:
            zone_map = _TURNOVER_INSIGHT[event_type]
            return zone_map.get(zone, ("Decision Making", "Review possession decision in this situation"))

        label = _POSITIVE_INSIGHT.get(event_type)
        return (label, None) if label else (None, None)

    @staticmethod
    def _to_bsm_event(e: EnrichedEvent) -> dict:
        """Translate EnrichedEvent to the dict shape BoxScoreManager expects."""
        return {
            "event": e.event_type,
            "player_id": e.player_id,
            "player_name": e.player_name,
            "team": e.team,
            "sub_type": e.sub_type,
            "contested": e.contested,
            "mismatch": e.mismatch,
            "reason": e.reason,
        }
