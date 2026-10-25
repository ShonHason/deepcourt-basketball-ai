"""
Narrator — converts an EnrichedEvent log into a structured game narrative.

Produces:
  - Full prose play-by-play with live score context
  - Quarter/period section headers (12-minute NBA quarters)
  - Scoring run callouts (≥5 consecutive points by one team)
  - Lead change annotations
  - Inline coaching notes for every learning moment

Output is a NarrativeOutput dataclass that can be written to .txt or
consumed by pdf_exporter.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from src.logic.stat_tracker import EnrichedEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUARTER_DURATION_MS = 10 * 60 * 1000   # 10-minute FIBA quarters
SCORING_RUN_THRESHOLD = 5               # points in a row to flag a run

# ---------------------------------------------------------------------------
# Game state tracker
# ---------------------------------------------------------------------------

@dataclass
class _TeamScore:
    team_id: str
    pts: int = 0


@dataclass
class _ScoringRun:
    team: str
    points: int
    start_ms: int
    end_ms: int
    events: list[str] = field(default_factory=list)  # event_ids


class _GameState:
    def __init__(self, home_team: str, away_team: str) -> None:
        self.home = _TeamScore(home_team)
        self.away = _TeamScore(away_team)
        self._run_team: Optional[str] = None
        self._run_pts: int = 0
        self._run_start_ms: int = 0
        self._run_events: list[str] = []
        self.completed_runs: list[_ScoringRun] = []
        self.lead_changes: int = 0
        self._leading_team: Optional[str] = None

    def score(self, team: str, pts: int, ts: int, event_id: str) -> tuple[str, Optional[str]]:
        """
        Register a score. Returns (score_context_str, run_callout_str | None).
        score_context_str describes the new game state after this score.
        run_callout_str is non-None when a scoring run just reached the threshold.
        """
        if team == self.home.team_id:
            self.home.pts += pts
        else:
            self.away.pts += pts

        # Lead change detection
        old_leader = self._leading_team
        if self.home.pts > self.away.pts:
            new_leader = self.home.team_id
        elif self.away.pts > self.home.pts:
            new_leader = self.away.team_id
        else:
            new_leader = None   # tied

        if new_leader != old_leader and old_leader is not None:
            self.lead_changes += 1
        self._leading_team = new_leader

        # Scoring run tracking
        run_callout: Optional[str] = None
        if self._run_team == team:
            self._run_pts += pts
            self._run_events.append(event_id)
        else:
            # Previous run ended
            if self._run_team and self._run_pts >= SCORING_RUN_THRESHOLD:
                self.completed_runs.append(_ScoringRun(
                    team=self._run_team,
                    points=self._run_pts,
                    start_ms=self._run_start_ms,
                    end_ms=ts,
                    events=list(self._run_events),
                ))
            self._run_team = team
            self._run_pts = pts
            self._run_start_ms = ts
            self._run_events = [event_id]

        # Emit callout exactly when run crosses the threshold
        if self._run_pts >= SCORING_RUN_THRESHOLD and self._run_pts - pts < SCORING_RUN_THRESHOLD:
            run_callout = (
                f"*** {team.upper()} ON A {self._run_pts}-0 RUN "
                f"({_ms_clock(self._run_start_ms)} to {_ms_clock(ts)}) ***"
            )

        return self._score_context(team, pts), run_callout

    def _score_context(self, scoring_team: str, pts: int) -> str:
        h, a = self.home.pts, self.away.pts
        score_str = f"{self.home.team_id} {h}, {self.away.team_id} {a}"

        if h == a:
            return f"ties the game at {h}. {score_str}"

        leading = self.home.team_id if h > a else self.away.team_id
        trailing = self.away.team_id if h > a else self.home.team_id
        diff = abs(h - a)

        if scoring_team == leading:
            verb = "extends the lead" if diff > pts else "takes the lead"
            return f"{verb} to {diff}. {score_str}"
        else:
            if diff == 0:
                return f"ties the game at {h}. {score_str}"
            return f"cuts the deficit to {diff}. {score_str}"

    @property
    def current_score(self) -> str:
        return f"{self.home.team_id} {self.home.pts}, {self.away.team_id} {self.away.pts}"

    def flush_run(self, final_ts: int) -> None:
        if self._run_team and self._run_pts >= SCORING_RUN_THRESHOLD:
            self.completed_runs.append(_ScoringRun(
                team=self._run_team,
                points=self._run_pts,
                start_ms=self._run_start_ms,
                end_ms=final_ts,
                events=list(self._run_events),
            ))
        self._run_team = None
        self._run_pts = 0


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class PbPLine:
    clock: str
    quarter: int
    player_name: str
    team: str
    text: str
    is_learning_moment: bool
    learning_insight: Optional[str]
    event_type: str
    timestamp_ms: int

    def to_dict(self) -> dict:
        return {
            "clock": self.clock,
            "quarter": self.quarter,
            "player_name": self.player_name,
            "team": self.team,
            "text": self.text,
            "is_learning_moment": self.is_learning_moment,
            "learning_insight": self.learning_insight,
            "event_type": self.event_type,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class KeyMoment:
    timestamp_ms: int
    clock: str
    quarter: int
    kind: str        # "lead_change" | "scoring_run" | "clutch_score" | "streak_broken"
    description: str


@dataclass
class NarrativeOutput:
    full_text: str
    pbp_lines: list[PbPLine]
    key_moments: list[KeyMoment]
    scoring_runs: list[_ScoringRun]
    final_score: str
    total_lead_changes: int
    total_learning_moments: int

    def export_txt(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.full_text)

    def export_json(self, path: str) -> None:
        out = {
            "final_score": self.final_score,
            "total_lead_changes": self.total_lead_changes,
            "total_learning_moments": self.total_learning_moments,
            "key_moments": [
                {"clock": km.clock, "quarter": km.quarter,
                 "kind": km.kind, "description": km.description}
                for km in self.key_moments
            ],
            "scoring_runs": [
                {"team": r.team, "points": r.points,
                 "start_ms": r.start_ms, "end_ms": r.end_ms}
                for r in self.scoring_runs
            ],
            "pbp": [line.to_dict() for line in self.pbp_lines],
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------

_SCORING_EVENTS = frozenset({
    "two_pointer_made", "three_pointer_made", "free_throw_made",
})

_POINT_VALUES = {
    "two_pointer_made": 2,
    "three_pointer_made": 3,
    "free_throw_made": 1,
}


class Narrator:
    """
    Produces a full game narrative from a sorted list of EnrichedEvents.

    Usage:
        narrator = Narrator(home_team="home", away_team="away")
        output = narrator.narrate(tracker.events)
        output.export_txt("game_pbp.txt")
    """

    def __init__(
        self,
        home_team: str = "home",
        away_team: str = "away",
        game_start_ms: int = 0,
        period_timestamps: Optional[dict[int, int]] = None,
        quarter_duration_ms: Optional[int] = None,
    ) -> None:
        self.home_team = home_team
        self.away_team = away_team
        self._game_start_ms = game_start_ms
        # Allow caller to override the period length (FIBA=10min, NBA=12min,
        # NCAA halves=20min, HS=8min).  Falls back to the module constant.
        self._quarter_ms = quarter_duration_ms if quarter_duration_ms else QUARTER_DURATION_MS
        # Sorted list of (video_ts_ms, period) from scoreboard OCR.
        # When present, overrides the fixed quarter-duration calculation.
        self._period_ts: list[tuple[int, int]] = []
        if period_timestamps:
            self._period_ts = sorted(
                [(ts, p) for p, ts in period_timestamps.items()],
                key=lambda x: x[0],
            )

    def _adjusted_ms(self, ms: int) -> int:
        """Subtract the game-start offset so clock counts from tip-off, not video start."""
        return max(0, ms - self._game_start_ms)

    def _period_for_ts(self, video_ts_ms: int) -> int:
        """Return the period number for a given video timestamp."""
        if not self._period_ts:
            return _quarter_number(self._adjusted_ms(video_ts_ms), self._quarter_ms)
        # Walk backward through period boundaries to find which period this ts falls in
        period = 1
        for boundary_ts, p in self._period_ts:
            if video_ts_ms >= boundary_ts:
                period = p
        return period

    def narrate(self, events: list[EnrichedEvent]) -> NarrativeOutput:
        sorted_events = sorted(events, key=lambda e: e.timestamp_ms)
        state = _GameState(self.home_team, self.away_team)

        pbp_lines: list[PbPLine] = []
        key_moments: list[KeyMoment] = []
        prose_blocks: list[str] = []
        current_quarter = 0

        for event in sorted_events:
            adj_ms  = self._adjusted_ms(event.timestamp_ms)
            quarter = self._period_for_ts(event.timestamp_ms)

            # Quarter header
            if quarter != current_quarter:
                current_quarter = quarter
                header = f"\n{'=' * 40}\n  {_quarter_label(quarter)}\n{'=' * 40}"
                prose_blocks.append(header)

            clock = _ms_clock(adj_ms)
            base_text = _event_sentence(event)
            context_suffix = ""
            run_callout: Optional[str] = None

            # Score events get live game context
            if event.event_type in _SCORING_EVENTS:
                pts = _POINT_VALUES[event.event_type]
                context, run_callout = state.score(
                    event.team, pts, adj_ms, event.event_id
                )
                context_suffix = f" {context}"

                # Lead change key moment
                if "takes the lead" in context or "ties the game" in context:
                    key_moments.append(KeyMoment(
                        timestamp_ms=adj_ms,
                        clock=clock,
                        quarter=quarter,
                        kind="lead_change",
                        description=f"{event.player_name} — {base_text} {context}",
                    ))

            # Learning moment inline coaching note
            coaching = ""
            if event.is_learning_moment and event.learning_detail:
                coaching = f"\n  > Coach: {event.learning_detail}"

            line_text = f"{clock}  {base_text}{context_suffix}{coaching}"
            prose_blocks.append(line_text)

            # Scoring run callout
            if run_callout:
                prose_blocks.append(f"\n  {run_callout}\n")
                key_moments.append(KeyMoment(
                    timestamp_ms=adj_ms,
                    clock=clock,
                    quarter=quarter,
                    kind="scoring_run",
                    description=run_callout.strip("* "),
                ))

            pbp_lines.append(PbPLine(
                clock=clock,
                quarter=quarter,
                player_name=event.player_name,
                team=event.team,
                text=f"{base_text}{context_suffix}",
                is_learning_moment=event.is_learning_moment,
                learning_insight=event.learning_insight,
                event_type=event.event_type,
                timestamp_ms=adj_ms,
            ))

        last_adj = self._adjusted_ms(sorted_events[-1].timestamp_ms) if sorted_events else 0
        state.flush_run(last_adj)

        # Final summary block
        total_lm = sum(1 for e in sorted_events if e.is_learning_moment)
        prose_blocks.append(
            f"\n{'=' * 40}\n"
            f"  FINAL: {state.current_score}\n"
            f"  Lead changes: {state.lead_changes}  |  "
            f"Learning moments: {total_lm}\n"
            f"{'=' * 40}\n"
        )

        return NarrativeOutput(
            full_text="\n".join(prose_blocks),
            pbp_lines=pbp_lines,
            key_moments=key_moments,
            scoring_runs=state.completed_runs,
            final_score=state.current_score,
            total_lead_changes=state.lead_changes,
            total_learning_moments=total_lm,
        )


# ---------------------------------------------------------------------------
# Sentence templates (parallel to feed_builder but richer)
# ---------------------------------------------------------------------------

_SUB_PHRASES = {
    "swish": " — pure swish",
    "bank_shot": " off the glass",
    "rim_in": " — rattles in",
}

_ZONE_LABELS = {
    "restricted_area": "at the rim",
    "paint": "in the paint",
    "mid_range": "from mid-range",
    "three_point": "from three",
    "backcourt": "from deep in the backcourt",
}


def _event_sentence(event: EnrichedEvent) -> str:
    p = event.player_name
    zone = _ZONE_LABELS.get(event.court_zone, f"({event.court_zone})")
    sub = _SUB_PHRASES.get(event.sub_type or "", "")
    tag = f" [{event.learning_insight}]" if event.learning_insight else ""

    templates = {
        "two_pointer_made":      f"{p} scores 2 {zone}{sub}.",
        "two_pointer_missed":    f"{p} misses {zone}{sub}.{tag}",
        "three_pointer_made":    f"{p} connects from three{sub}!",
        "three_pointer_missed":  f"{p} misses the three-point attempt.{tag}",
        "free_throw_made":       f"{p} makes the free throw.",
        "free_throw_missed":     f"{p} misses the free throw.{tag}",
        "assist":                f"{p} with the assist.",
        "rebound_offensive":     f"{p} grabs the offensive board.",
        "rebound_defensive":     f"{p} with the defensive rebound.",
        "steal":                 f"{p} comes up with the steal!",
        "block":                 f"{p} rejects the shot!",
        "turnover_steal":        f"{p} loses the ball — stripped.{tag}",
        "turnover_unforced":     f"{p} commits the turnover.{tag}",
    }
    return templates.get(event.event_type, f"{p} — {event.event_type.replace('_', ' ')}")


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def _ms_clock(ms: int) -> str:
    total_s = ms // 1000
    return f"{total_s // 60}:{total_s % 60:02d}"


def _quarter_number(ms: int, quarter_ms: int = QUARTER_DURATION_MS) -> int:
    return ms // quarter_ms + 1


def _quarter_label(n: int) -> str:
    labels = {1: "1st Quarter", 2: "2nd Quarter", 3: "3rd Quarter", 4: "4th Quarter"}
    return labels.get(n, f"OT {n - 4}")
