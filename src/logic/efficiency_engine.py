"""
Efficiency Engine — identifies recurring patterns in Learning Moments
to surface actionable coaching insights per player.

Four analysis passes:
  zone_weakness   — court zone where learning moments cluster
  pattern         — most repeated learning_insight category
  streak          — consecutive learning moments (fatigue / adjustment)
  tendency        — left/right court bias in turnovers
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.logic.stat_tracker import EnrichedEvent, StatTracker

# ---------------------------------------------------------------------------
# Drill suggestions
# ---------------------------------------------------------------------------

DRILL_MAP: dict[str, str] = {
    "Ball Security":   "Ball-handling circuit: chair drills, spin moves, weak-hand dribble combos",
    "Shot Selection":  "Film review: mark 'bad shot' possessions; set shot-quality thresholds with coach",
    "Mechanics":       "Form shooting: 10-feet wall routine, slow-motion release drill",
    "Footwork":        "Footwork ladder: jab-step, pivot, and gather drills at speed",
    "Spacing":         "5-on-0 motion offense: emphasise spacing principles with cut reads",
    "Timing":          "Catch-and-shoot reps: partner passes into rhythm shots from multiple spots",
    "Decision Making": "3-on-3 live reads: force two-pass rule before shot; coach confirms reads",
    "Positioning":     "Defensive shell drill: help-side positioning and weak-side awareness cues",
}

MIN_EVENTS_FOR_PATTERN = 2   # minimum supporting events to emit an insight
STREAK_THRESHOLD = 3          # consecutive learning moments that flag a streak
ZONE_RATIO_THRESHOLD = 0.35   # zone must account for ≥35% of moments to be notable
DIRECTIONAL_RATIO_THRESHOLD = 0.65  # side-of-court bias threshold for tendency insights

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class PlayerInsight:
    player_id: str
    player_name: str
    insight_type: str           # "zone_weakness" | "pattern" | "streak" | "tendency"
    description: str
    learning_category: str
    supporting_event_ids: list[str]
    confidence: float           # 0.0 – 1.0
    suggested_drill: str
    # Populated for streak insights
    streak_length: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "insight_type": self.insight_type,
            "description": self.description,
            "learning_category": self.learning_category,
            "supporting_event_ids": self.supporting_event_ids,
            "confidence": round(self.confidence, 3),
            "suggested_drill": self.suggested_drill,
            "streak_length": self.streak_length,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EfficiencyEngine:
    """
    Analyzes the enriched event log from a StatTracker to find per-player
    patterns in Learning Moments. Returns a list of PlayerInsight objects
    sorted by confidence (highest first).
    """

    def __init__(self, tracker: StatTracker) -> None:
        self._tracker = tracker

    def analyze(self) -> list[PlayerInsight]:
        """Run all analysis passes across all players."""
        insights: list[PlayerInsight] = []
        for pid in {e.player_id for e in self._tracker.events}:
            insights.extend(self.analyze_player(pid))
        insights.sort(key=lambda i: i.confidence, reverse=True)
        return insights

    def analyze_player(self, player_id: str) -> list[PlayerInsight]:
        events = self._tracker.get_player_events(player_id)
        moments = [e for e in events if e.is_learning_moment]
        if not moments:
            return []

        insights: list[PlayerInsight] = []
        insights.extend(self._zone_weakness(moments))
        insights.extend(self._category_pattern(moments))
        insights.extend(self._streak_detection(events))
        insights.extend(self._directional_tendency(moments))
        insights.sort(key=lambda i: i.confidence, reverse=True)
        return insights

    # ------------------------------------------------------------------
    # Analysis passes
    # ------------------------------------------------------------------

    def _zone_weakness(self, moments: list[EnrichedEvent]) -> list[PlayerInsight]:
        """Flag court zones where a player generates disproportionate learning moments."""
        zone_counts: Counter[str] = Counter(e.court_zone for e in moments)
        total = len(moments)
        insights = []

        for zone, count in zone_counts.items():
            if count < MIN_EVENTS_FOR_PATTERN:
                continue
            ratio = count / total
            if ratio < ZONE_RATIO_THRESHOLD:
                continue

            zone_events = [e for e in moments if e.court_zone == zone]
            top = Counter(
                e.learning_insight for e in zone_events if e.learning_insight
            ).most_common(1)
            category = top[0][0] if top else "Decision Making"
            name = moments[0].player_name

            insights.append(PlayerInsight(
                player_id=moments[0].player_id,
                player_name=name,
                insight_type="zone_weakness",
                description=(
                    f"{name} produces {count} of {total} learning moments "
                    f"({ratio:.0%}) in the {zone.replace('_', ' ')} zone. "
                    f"Primary issue: {category}."
                ),
                learning_category=category,
                supporting_event_ids=[e.event_id for e in zone_events],
                confidence=min(0.95, 0.4 + ratio * 0.8),
                suggested_drill=DRILL_MAP.get(category, "Targeted zone practice"),
            ))

        return insights

    def _category_pattern(self, moments: list[EnrichedEvent]) -> list[PlayerInsight]:
        """Surface the most repeated learning_insight category across all moments."""
        cat_counts: Counter[str] = Counter(
            e.learning_insight for e in moments if e.learning_insight
        )
        total = len(moments)
        insights = []

        for category, count in cat_counts.most_common(2):
            if count < MIN_EVENTS_FOR_PATTERN:
                continue
            ratio = count / total
            cat_events = [e for e in moments if e.learning_insight == category]
            name = moments[0].player_name
            example = next(
                (e.learning_detail for e in reversed(cat_events) if e.learning_detail), ""
            )
            insights.append(PlayerInsight(
                player_id=moments[0].player_id,
                player_name=name,
                insight_type="pattern",
                description=(
                    f"{name} shows a recurring '{category}' issue "
                    f"({count}/{total} learning moments). Example: {example}"
                ),
                learning_category=category,
                supporting_event_ids=[e.event_id for e in cat_events],
                confidence=min(0.95, 0.3 + ratio),
                suggested_drill=DRILL_MAP.get(category, "Targeted repetition"),
            ))

        return insights

    def _streak_detection(self, events: list[EnrichedEvent]) -> list[PlayerInsight]:
        """Flag consecutive runs of learning moments (possible fatigue or defensive adjustment)."""
        insights = []
        streak: list[EnrichedEvent] = []

        for event in events:
            if event.is_learning_moment:
                streak.append(event)
            else:
                if len(streak) >= STREAK_THRESHOLD:
                    insights.append(self._build_streak_insight(streak))
                streak = []

        if len(streak) >= STREAK_THRESHOLD:
            insights.append(self._build_streak_insight(streak))

        return insights

    def _directional_tendency(self, moments: list[EnrichedEvent]) -> list[PlayerInsight]:
        """Detect left/right court bias in turnovers (y < 7.5 = left, y >= 7.5 = right)."""
        to_events = [
            e for e in moments
            if e.event_type in {"turnover_steal", "turnover_unforced"}
        ]
        if len(to_events) < MIN_EVENTS_FOR_PATTERN:
            return []

        total = len(to_events)
        left = [e for e in to_events if e.coordinates.get("y", 7.5) < 7.5]
        right = [e for e in to_events if e.coordinates.get("y", 7.5) >= 7.5]
        name = moments[0].player_name
        insights = []

        for side_events, side_name in [(left, "left"), (right, "right")]:
            count = len(side_events)
            if count < MIN_EVENTS_FOR_PATTERN:
                continue
            ratio = count / total
            if ratio < DIRECTIONAL_RATIO_THRESHOLD:
                continue
            insights.append(PlayerInsight(
                player_id=moments[0].player_id,
                player_name=name,
                insight_type="tendency",
                description=(
                    f"{name} loses possession predominantly on the {side_name} side "
                    f"({count}/{total} turnovers, {ratio:.0%}). "
                    f"Defenders may be overplaying the {side_name}."
                ),
                learning_category="Ball Security",
                supporting_event_ids=[e.event_id for e in side_events],
                confidence=min(0.90, 0.4 + ratio * 0.6),
                suggested_drill=DRILL_MAP["Ball Security"],
            ))

        return insights

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_streak_insight(streak: list[EnrichedEvent]) -> PlayerInsight:
        name = streak[0].player_name
        types_str = ", ".join(e.event_type.replace("_", " ") for e in streak)
        top = Counter(e.learning_insight for e in streak if e.learning_insight).most_common(1)
        category = top[0][0] if top else "Decision Making"

        return PlayerInsight(
            player_id=streak[0].player_id,
            player_name=name,
            insight_type="streak",
            description=(
                f"{name} had {len(streak)} consecutive learning moments: {types_str}. "
                f"May indicate fatigue or a defensive adjustment. Primary issue: {category}."
            ),
            learning_category=category,
            supporting_event_ids=[e.event_id for e in streak],
            confidence=min(0.95, 0.5 + len(streak) * 0.1),
            suggested_drill=DRILL_MAP.get(
                category, "Slow the game down — focus on one possession at a time"
            ),
            streak_length=len(streak),
        )
