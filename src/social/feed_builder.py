"""
Feed Builder — constructs three social/dashboard feeds for CourtLog.ai.

  Game Feed   : Box Score header card + full chronological Play-by-Play.
  Plays Feed  : Every event as a vertical clip card; highlights first, then
                learning moments as "Optimization Opportunity" cards.
  Learning Feed: Only learning moments + EfficiencyEngine pattern insights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from src.logic.stat_tracker import EnrichedEvent, StatTracker
from src.logic.efficiency_engine import EfficiencyEngine, PlayerInsight

# ---------------------------------------------------------------------------
# PbP text templates
# ---------------------------------------------------------------------------

_MADE = {
    "two_pointer_made":   "{player} scores 2 — {zone} bucket{sub}.",
    "three_pointer_made": "{player} hits from three! {zone}{sub}.",
    "free_throw_made":    "{player} converts the free throw.",
}

_MISS = {
    "two_pointer_missed":   "{player} misses the {zone} attempt{sub}. {tag}",
    "three_pointer_missed": "{player} misses from three{sub}. {tag}",
    "free_throw_missed":    "{player} misses the free throw. {tag}",
}

_POSSESSION = {
    "assist":             "{player} dishes the assist.",
    "rebound_offensive":  "{player} pulls down the offensive board.",
    "rebound_defensive":  "{player} secures the defensive rebound.",
    "steal":              "{player} comes up with the steal!",
    "block":              "{player} blocks the shot!",
    "turnover_steal":     "{player} loses the ball — stolen. {tag}",
    "turnover_unforced":  "{player} turns it over. {tag}",
}

_HIGHLIGHT_LABELS = {
    "three_pointer_made": "SPLASH",
    "two_pointer_made":   "BUCKET",
    "steal":              "TAKE",
    "block":              "REJECTION",
    "assist":             "DIME",
    "rebound_offensive":  "O-BOARD",
    "rebound_defensive":  "D-BOARD",
}


def _zone(z: str) -> str:
    return z.replace("_", " ").title()


def _sub(s: Optional[str]) -> str:
    if s == "swish":
        return " — pure swish"
    if s == "bank_shot":
        return " off the glass"
    if s == "rim_in":
        return " — rattles in"
    return ""


def _tag(insight: Optional[str]) -> str:
    return f"[{insight}]" if insight else ""


def pbp_text(event: EnrichedEvent) -> str:
    """Generate a single Play-by-Play sentence for an event."""
    p = event.player_name
    z = _zone(event.court_zone)
    s = _sub(event.sub_type)
    t = _tag(event.learning_insight)

    if event.event_type in _MADE:
        return _MADE[event.event_type].format(player=p, zone=z, sub=s)
    if event.event_type in _MISS:
        return _MISS[event.event_type].format(player=p, zone=z, sub=s, tag=t)
    if event.event_type in _POSSESSION:
        return _POSSESSION[event.event_type].format(player=p, tag=t)
    return f"{p} — {event.event_type.replace('_', ' ')}"


def _ms_to_clock(ms: int) -> str:
    total_s = ms // 1000
    return f"{total_s // 60}:{total_s % 60:02d}"


# ---------------------------------------------------------------------------
# Feed card model
# ---------------------------------------------------------------------------

@dataclass
class FeedCard:
    card_id: str
    card_type: str           # "box_score" | "pbp" | "clip" | "learning" | "insight"
    player_id: Optional[str]
    player_name: Optional[str]
    team: Optional[str]
    timestamp_ms: int
    title: str
    body: str
    clip_window_ms: Optional[tuple[int, int]]
    is_learning_moment: bool
    learning_category: Optional[str]
    learning_insight: Optional[str]
    court_zone: Optional[str]
    event_type: Optional[str]

    def to_dict(self) -> dict:
        return {
            "card_id": self.card_id,
            "card_type": self.card_type,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team": self.team,
            "timestamp_ms": self.timestamp_ms,
            "title": self.title,
            "body": self.body,
            "clip_window_ms": list(self.clip_window_ms) if self.clip_window_ms else None,
            "is_learning_moment": self.is_learning_moment,
            "learning_category": self.learning_category,
            "learning_insight": self.learning_insight,
            "court_zone": self.court_zone,
            "event_type": self.event_type,
        }


@dataclass
class Feed:
    feed_type: str
    game_id: str
    cards: list[FeedCard] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feed_type": self.feed_type,
            "game_id": self.game_id,
            "total_cards": len(self.cards),
            "cards": [c.to_dict() for c in self.cards],
        }

    def export_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Feed Builder
# ---------------------------------------------------------------------------

class FeedBuilder:
    """
    Constructs the three CourtLog.ai feeds from a StatTracker session.

    Usage:
        tracker = StatTracker()
        tracker.track_all(raw_events)
        builder = FeedBuilder(tracker, game_id="game_001")

        game_feed    = builder.build_game_feed()
        plays_feed   = builder.build_plays_feed()
        learning_feed = builder.build_learning_feed()
    """

    def __init__(self, tracker: StatTracker, game_id: str) -> None:
        self._tracker = tracker
        self._game_id = game_id
        self._efficiency = EfficiencyEngine(tracker)

    # ------------------------------------------------------------------
    # Public: feed constructors
    # ------------------------------------------------------------------

    def build_game_feed(self) -> Feed:
        """Box Score header card + full chronological PbP."""
        feed = Feed(feed_type="game", game_id=self._game_id)

        feed.cards.append(FeedCard(
            card_id=f"{self._game_id}_bscore",
            card_type="box_score",
            player_id=None, player_name=None, team=None,
            timestamp_ms=0,
            title="Box Score",
            body="Full box score — see game summary.",
            clip_window_ms=None,
            is_learning_moment=False,
            learning_category=None, learning_insight=None,
            court_zone=None, event_type=None,
        ))

        for event in sorted(self._tracker.events, key=lambda e: e.timestamp_ms):
            feed.cards.append(FeedCard(
                card_id=f"pbp_{event.event_id}",
                card_type="pbp",
                player_id=event.player_id,
                player_name=event.player_name,
                team=event.team,
                timestamp_ms=event.timestamp_ms,
                title=_ms_to_clock(event.timestamp_ms),
                body=pbp_text(event),
                clip_window_ms=event.clip_window_ms,
                is_learning_moment=event.is_learning_moment,
                learning_category=event.learning_category,
                learning_insight=event.learning_insight,
                court_zone=event.court_zone,
                event_type=event.event_type,
            ))

        return feed

    def build_plays_feed(self) -> Feed:
        """
        Every event as a vertical clip card.
        Highlight plays (made shots, steals, blocks) come first;
        learning moments follow as Optimization Opportunity cards.
        """
        feed = Feed(feed_type="plays", game_id=self._game_id)
        highlights: list[FeedCard] = []
        learning: list[FeedCard] = []

        for event in sorted(self._tracker.events, key=lambda e: e.timestamp_ms):
            card = FeedCard(
                card_id=f"clip_{event.event_id}",
                card_type="clip",
                player_id=event.player_id,
                player_name=event.player_name,
                team=event.team,
                timestamp_ms=event.timestamp_ms,
                title=self._clip_title(event),
                body=pbp_text(event),
                clip_window_ms=event.clip_window_ms,
                is_learning_moment=event.is_learning_moment,
                learning_category=event.learning_category,
                learning_insight=event.learning_insight,
                court_zone=event.court_zone,
                event_type=event.event_type,
            )
            if event.is_learning_moment:
                learning.append(card)
            else:
                highlights.append(card)

        feed.cards = highlights + learning
        return feed

    def build_learning_feed(self) -> Feed:
        """
        Learning moments as Optimization Opportunity cards, followed by
        pattern-level PlayerInsight cards from the EfficiencyEngine.
        """
        feed = Feed(feed_type="learning", game_id=self._game_id)

        for event in sorted(self._tracker.learning_moments, key=lambda e: e.timestamp_ms):
            detail = event.learning_detail or event.learning_insight or ""
            feed.cards.append(FeedCard(
                card_id=f"learn_{event.event_id}",
                card_type="learning",
                player_id=event.player_id,
                player_name=event.player_name,
                team=event.team,
                timestamp_ms=event.timestamp_ms,
                title=f"Optimization Opportunity — {event.learning_insight or 'Review'}",
                body=f"{pbp_text(event)}\nFocus: {detail}",
                clip_window_ms=event.clip_window_ms,
                is_learning_moment=True,
                learning_category=event.learning_category,
                learning_insight=event.learning_insight,
                court_zone=event.court_zone,
                event_type=event.event_type,
            ))

        for insight in self._efficiency.analyze():
            feed.cards.append(self._insight_card(insight))

        return feed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_title(event: EnrichedEvent) -> str:
        if event.event_type in _HIGHLIGHT_LABELS:
            return f"{_HIGHLIGHT_LABELS[event.event_type]} — {event.player_name}"
        if event.is_learning_moment:
            return f"Optimization Opportunity — {event.player_name}"
        return f"{event.event_type.replace('_', ' ').title()} — {event.player_name}"

    @staticmethod
    def _insight_card(insight: PlayerInsight) -> FeedCard:
        suffix = insight.learning_category[:4]
        return FeedCard(
            card_id=f"insight_{insight.player_id}_{insight.insight_type}_{suffix}",
            card_type="insight",
            player_id=insight.player_id,
            player_name=insight.player_name,
            team=None,
            timestamp_ms=0,
            title=f"Pattern Detected — {insight.learning_category}",
            body=(
                f"{insight.description}\n"
                f"Drill: {insight.suggested_drill}\n"
                f"Confidence: {insight.confidence:.0%}"
            ),
            clip_window_ms=None,
            is_learning_moment=True,
            learning_category=insight.learning_category,
            learning_insight=insight.insight_type,
            court_zone=None,
            event_type=None,
        )
