"""
GameStateMachine — tracks live/dead-ball/free-throw game context.

Sits between EventValidator and StatTracker in the pipeline frame loop.
Filters events that should not count during dead ball (inbound turnovers),
reclassifies shots near the FT line when in FREE_THROW state, and annotates
all events with the current game_state string.

Usage in pipeline frame loop:
    game_sm = GameStateMachine()

    for prim in prim_gen.drain():
        raw_events = validator.push(prim)
        for ev in raw_events:
            ev = game_sm.on_event(ev, ball_court)
            if ev is not None:
                stat_tracker.track(ev)

    game_sm.on_frame(fd, ball_court)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class GameState(Enum):
    LIVE       = "live"
    DEAD_BALL  = "dead_ball"
    FREE_THROW = "free_throw"


# Free throw lines are 4.57 m from each baseline (court length = 28 m)
_FT_LINE_X = (4.57, 23.43)
_FT_LINE_Y = 7.5          # court centre-line (width = 15 m)
_FT_TOL_X  = 1.5          # ±metres on x to count as "near FT line"
_FT_TOL_Y  = 2.0          # ±metres from centre-line

DEAD_BALL_TIMEOUT_MS      = 12_000   # safety reset: DEAD_BALL → LIVE
BALL_DISAPPEAR_THRESHOLD  = 5        # consecutive frames with no ball → DEAD_BALL
FT_STILL_FRAMES           = 5        # ball frames near FT line before → FREE_THROW
FREE_THROW_TIMEOUT_MS     = 30_000   # safety reset: FREE_THROW → LIVE


def _near_ft_line(x: float, y: float) -> bool:
    """Return True if court coordinates are close to either FT line."""
    for ft_x in _FT_LINE_X:
        if abs(x - ft_x) <= _FT_TOL_X and abs(y - _FT_LINE_Y) <= _FT_TOL_Y:
            return True
    return False


class GameStateMachine:
    def __init__(self) -> None:
        self.state: GameState = GameState.LIVE
        self._dead_ball_start_ms: int = 0
        self._ft_start_ms: int = 0
        self._consecutive_no_ball: int = 0
        self._ft_candidate_frames: int = 0
        self._pending_dead_ball_ts: int = 0   # deferred transition after made basket

    # ------------------------------------------------------------------
    # Per-event hook (call after validator.push(), before stat_tracker.track())
    # ------------------------------------------------------------------

    def on_event(self, event: dict, ball_court: Optional[dict]) -> Optional[dict]:
        """
        Annotate event with current game_state.
        Returns None to suppress the event entirely, or the (possibly modified)
        event dict to forward to stat_tracker.
        """
        etype = event.get("event_type", "")

        # Made basket → schedule DEAD_BALL for next frame, not now.
        # Deferring prevents the assist (emitted in the same tick) from being
        # tagged dead_ball — the basket and its assist are both live plays.
        if etype in ("two_pointer_made", "three_pointer_made"):
            self._pending_dead_ball_ts = event.get("timestamp_ms", self._pending_dead_ball_ts)

        # Annotate with current live/dead/ft state
        event["game_state"] = self.state.value

        # Suppress dead-ball inbound turnovers — these are not real turnovers.
        # Shots and rebounds are NOT suppressed: the dead_ball state machine has
        # false positives (ball off-screen, scoreboard flicker) that would drop
        # real scoring plays.  Only suppress the stat that is definitively wrong
        # during dead ball: unforced turnovers from inbound plays.
        if self.state == GameState.DEAD_BALL and etype in ("turnover_unforced",):
            return None

        # In FREE_THROW state: reclassify 2PT shots near the FT line
        if self.state == GameState.FREE_THROW and etype in (
            "two_pointer_missed", "two_pointer_made"
        ):
            coords = event.get("coordinates", {})
            if _near_ft_line(coords.get("x", 0.0), coords.get("y", _FT_LINE_Y)):
                suffix = "made" if etype.endswith("_made") else "missed"
                event["event_type"] = f"free_throw_{suffix}"
                event["reason"] = event.get("reason", "") + " [reclassified from 2PT — FT state]"

        return event

    # ------------------------------------------------------------------
    # Per-frame hook (call once per sampled frame, after prim_gen.push())
    # ------------------------------------------------------------------

    def on_frame(self, fd, ball_court: Optional[dict]) -> None:
        """Update game state from vision signals."""
        ts_ms: int = fd.timestamp_ms

        # Apply deferred DEAD_BALL transition (scheduled by a made-basket event)
        if self._pending_dead_ball_ts > 0 and self.state == GameState.LIVE:
            self.state = GameState.DEAD_BALL
            self._dead_ball_start_ms = self._pending_dead_ball_ts
            self._pending_dead_ball_ts = 0
            self._ft_candidate_frames = 0
            return

        # ── Safety resets ────────────────────────────────────────────────
        if self.state == GameState.DEAD_BALL:
            if ts_ms - self._dead_ball_start_ms > DEAD_BALL_TIMEOUT_MS:
                self.state = GameState.LIVE
                self._consecutive_no_ball = 0
            return

        if self.state == GameState.FREE_THROW:
            if ts_ms - self._ft_start_ms > FREE_THROW_TIMEOUT_MS:
                self.state = GameState.LIVE
            return

        # ── LIVE state: watch for triggers ───────────────────────────────
        if ball_court is not None:
            self._consecutive_no_ball = 0
            x, y = ball_court.get("x", 14.0), ball_court.get("y", 7.5)

            # Ball out of court bounds → dead ball
            if x < -1.0 or x > 29.0 or y < -1.0 or y > 16.0:
                self.state = GameState.DEAD_BALL
                self._dead_ball_start_ms = ts_ms
                self._ft_candidate_frames = 0
                return

            # Ball near FT line for several consecutive frames → FREE_THROW
            if _near_ft_line(x, y):
                self._ft_candidate_frames += 1
                if self._ft_candidate_frames >= FT_STILL_FRAMES:
                    self.state = GameState.FREE_THROW
                    self._ft_start_ms = ts_ms
                    self._ft_candidate_frames = 0
            else:
                self._ft_candidate_frames = 0
        else:
            self._ft_candidate_frames = 0
            self._consecutive_no_ball += 1
            if self._consecutive_no_ball >= BALL_DISAPPEAR_THRESHOLD:
                self.state = GameState.DEAD_BALL
                self._dead_ball_start_ms = ts_ms
                self._consecutive_no_ball = 0

    # ------------------------------------------------------------------
    # Period boundary (from ScoreboardReader)
    # ------------------------------------------------------------------

    def notify_period_boundary(self, period: int, ts_ms: int) -> None:
        """Called when ScoreboardReader detects a period transition."""
        self.state = GameState.DEAD_BALL
        self._dead_ball_start_ms = ts_ms
        self._consecutive_no_ball = 0
        self._ft_candidate_frames = 0
