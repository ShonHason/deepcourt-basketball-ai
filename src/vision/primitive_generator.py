"""
PrimitiveGenerator — frame sequence → CV primitives.

Watches a stream of FrameDetections (with track_ids assigned by Tracker)
and emits raw basketball observations that EventValidator turns into
scored events (assists, steals, rebounds, shots made/missed, etc.).

Primitive dict schema
─────────────────────
Every primitive has:
    primitive     str    possession_change | hand_contact
    player_id     int    track_id of the main player
    player_team   str    "unknown" until a team classifier is wired in
    timestamp_ms  int
    frame_idx     int
    coordinates   dict   {"x": float, "y": float}  court metres

Type-specific extra keys:
    hand_contact  → distance_px: float
"""

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Optional

from src.vision.detector import (
    FrameDetections,
    PlayerDetection,
    pixel_distance,
    is_contested,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

POSSESSION_DIST_PX         = 200  # foot-to-ball distance to claim possession
CONTACT_THRESHOLD_PX       = 20   # wrist-to-ball for hand_contact (~0.88m real at 640px);
                                    # tighter than 35px to reduce false blocks from pose noise
CONTACT_COOLDOWN_FRAMES    = 5    # min sampled frames between hand_contact emissions per player
POSSESSION_COOLDOWN_FRAMES = 45   # min frames between possession_change emissions (~15s at 3 FPS = one possession)
POSSESSION_CONFIRM_FRAMES  = 2    # ball must stay near same player N frames before change emitted
# Active roster window: a player is "on court" if they appeared in at least
# ROSTER_MIN_SEEN of the last ROSTER_WINDOW_FRAMES sampled frames.
# This prevents coaches/refs/spectators and ghost detections from receiving stats.
ROSTER_WINDOW_FRAMES = 20   # ~4 seconds at 5 fps
ROSTER_MIN_SEEN      = 3    # visible in ≥3 of the last 20 frames (~15% presence)
BALL_HISTORY_SIZE          = 6    # sampled frames of ball position to keep

# Shot detection thresholds
SHOT_TIMEOUT_FRAMES        = 25   # sampled frames (~5s) before an in-flight shot is ruled a miss
RIM_PROXIMITY_FACTOR       = 2.0  # ball within N * rim_width of rim center = "near rim"
THREE_POINT_X_RATIO        = 0.28 # court_x < this ratio OR > (1-ratio) = three-point territory
SHOT_COOLDOWN_FRAMES       = 15   # min sampled frames between shot_taken emissions

# Rim-approach detection (primary method)
# Ball must travel from FAR → NEAR the rim in one continuous arc.
SHOT_RIM_FAR_PX            = 350  # ball must have been this far from rim to count as a shot
SHOT_RIM_NEAR_PX           = 90   # ball is "at the rim" within this distance
SHOT_APPROACH_MIN_FRAMES   = 3    # ball must have been far for at least N frames before approaching

# Release detection (secondary / fallback)
SHOT_UPWARD_VY_PX          = 20   # min upward px/frame
SHOT_RELEASE_DIST_PX       = 220  # max px from player foot to ball at release
SHOT_UPWARD_CONFIRM_FRAMES = 3    # ball must move upward this many consecutive frames
SHOT_MIN_SEPARATION_PX     = 30   # ball must separate from player per frame


# ---------------------------------------------------------------------------
# PrimitiveGenerator
# ---------------------------------------------------------------------------

class PrimitiveGenerator:
    """
    Stateful; call push(fd) for each frame, then drain() to collect primitives.

    Args:
        court_length_m: real court length in metres (default 28 m)
        court_width_m:  real court width in metres (default 15 m)
    """

    def __init__(
        self,
        court_length_m:  float = 28.0,
        court_width_m:   float = 15.0,
        hoop_a_x_ratio:  float = 0.10,   # left hoop at ~10% of frame width
        hoop_b_x_ratio:  float = 0.90,   # right hoop at ~90% of frame width
    ) -> None:
        self._court_l = court_length_m
        self._court_w = court_width_m
        # Court spans from hoop_a to hoop_b in pixel space; everything outside
        # that range (sideline staff, bench area) maps to the closest court edge.
        self._hoop_a_x = hoop_a_x_ratio
        self._hoop_b_x = hoop_b_x_ratio
        self._possessor_id: Optional[int] = None
        self._prev_possessor_id: Optional[int] = None   # for assist detection
        self._prev_possessor_ts: int = 0                # timestamp of last possession change
        self._possessor_start_ts: int = 0               # when current possessor received ball
        self._last_possessor_id: Optional[int] = None   # most recent non-None possessor (fallback)
        self._last_possessor_ts: int = 0                # when _last_possessor_id last held ball
        self._buffer: list[dict] = []
        self._player_names: dict[int, str] = {}      # track_id → display name
        self._possession_cooldown: int = 0            # frames until next emission allowed
        self._candidate_possessor: Optional[int] = None   # player being confirmed
        self._candidate_frames: int = 0                   # consecutive frames candidate held ball
        # Ball trajectory history: each entry is (bx, by, frame_idx) or None when ball absent
        self._ball_history: deque = deque(maxlen=BALL_HISTORY_SIZE)
        # Per-player hand contact cooldown: track_id → sampled frames until next emission allowed
        self._contact_cooldown: dict[int, int] = {}
        # Active roster: rolling window of player sets; only players seen frequently
        # enough are considered valid attribution targets for possession/rebounds.
        self._recent_players: deque = deque(maxlen=ROSTER_WINDOW_FRAMES)
        self._active_roster: set[int] = set()
        # Possessor history — kept for potential future use / assist detection
        self._possessor_history: deque = deque(maxlen=60)   # ~12s at 5fps

        # Shot tracking state
        self._shot_in_flight: bool = False
        self._shot_player_id: Optional[int] = None
        self._shot_player_name: str = ""
        self._shot_ts: int = 0
        self._shot_frame: int = 0
        self._shot_coords: dict = {}
        self._shot_is_three: bool = False
        self._shot_frames_elapsed: int = 0         # frames since shot_taken
        self._shot_cooldown: int = 0               # frames until next shot_taken allowed
        # Rim-approach shot detection state
        # Tracks how long ball has been far from each rim
        self._rim_far_frames: dict[str, int] = {}    # side → consecutive frames ball was far
        self._rim_approach_fired: bool = False       # prevent double-fire per approach

        # Consecutive upward movement counter (fallback release detection)
        self._upward_streak: int = 0
        self._prev_ball_dist: dict[int, float] = {}  # track_id → prev dist to ball
        # Per-rim: was ball above rim in previous frame? key = "left"|"right"
        self._ball_above_rim: dict[str, bool] = {}
        # Locked rim positions once both hoops seen (pixel coords)
        self._rim_positions: dict[str, tuple[float, float, float]] = {}  # side → (cx, cy, width)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def push(self, fd: FrameDetections) -> None:
        # Update active roster from this frame's detections
        frame_ids = frozenset(
            p.track_id for p in fd.players if p.track_id is not None
        )
        self._recent_players.append(frame_ids)
        counts = Counter(tid for ids in self._recent_players for tid in ids)
        self._active_roster = {tid for tid, cnt in counts.items() if cnt >= ROSTER_MIN_SEEN}

        # Record ball position before other tracking so trajectory is current
        if fd.ball is not None:
            bx, by = fd.ball.center
            self._ball_history.append((bx, by, fd.frame_idx))
        self._track_possession(fd)
        self._track_hand_contact(fd)
        self._track_shot(fd)

    def drain(self) -> list[dict]:
        out, self._buffer = self._buffer, []
        return out

    def reset(self) -> None:
        self._possessor_id = None
        self._prev_possessor_id = None
        self._prev_possessor_ts = 0
        self._possessor_start_ts = 0
        self._buffer.clear()
        self._player_names.clear()
        self._possession_cooldown = 0
        self._candidate_possessor = None
        self._candidate_frames = 0
        self._ball_history.clear()
        self._contact_cooldown.clear()
        self._recent_players.clear()
        self._active_roster.clear()
        self._shot_in_flight = False
        self._shot_player_id = None
        self._shot_frames_elapsed = 0
        self._shot_cooldown = 0
        self._upward_streak = 0
        self._prev_ball_dist.clear()
        self._rim_far_frames.clear()
        self._rim_approach_fired = False
        self._ball_above_rim.clear()
        self._rim_positions.clear()

    # ------------------------------------------------------------------
    # Possession
    # ------------------------------------------------------------------

    def _track_possession(self, fd: FrameDetections) -> None:
        if fd.ball is None or not fd.players:
            # No ball visible — reset candidate to avoid stale confirmation
            self._candidate_possessor = None
            self._candidate_frames = 0
            if self._possession_cooldown > 0:
                self._possession_cooldown -= 1
            return

        bx, by = fd.ball.center
        nearest: Optional[PlayerDetection] = None
        nearest_dist = POSSESSION_DIST_PX

        # Update the name cache for all visible players regardless of roster status.
        for p in fd.players:
            if p.track_id is None:
                continue
            jersey = getattr(p, "jersey_number", None)
            if jersey:
                self._player_names[p.track_id] = f"#{jersey}"
            elif p.track_id not in self._player_names:
                self._player_names[p.track_id] = f"#{p.track_id}"

        # Prefer active-roster players for possession attribution — this filters
        # out coaches, refs, and ghost detections.  The filter is SOFT: if no
        # roster player is close enough (e.g., a player just subbed in and hasn't
        # built up their ROSTER_MIN_SEEN count yet), we retry with all players so
        # a real basket is never silently dropped.
        use_roster = len(self._active_roster) >= 4
        for pass_num in range(2):  # pass 0 = roster-only, pass 1 = fallback all
            for p in fd.players:
                if p.track_id is None:
                    continue
                if pass_num == 0 and use_roster and p.track_id not in self._active_roster:
                    continue
                d = pixel_distance(p.foot_center, (bx, by))
                if d < nearest_dist:
                    nearest_dist = d
                    nearest = p
            if nearest is not None:
                break  # found someone in the preferred pass — no need for fallback

        if nearest is None:
            self._candidate_possessor = None
            self._candidate_frames = 0
            return

        # Accumulate consecutive frames where same player is nearest to ball
        if nearest.track_id == self._candidate_possessor:
            self._candidate_frames += 1
        else:
            self._candidate_possessor = nearest.track_id
            self._candidate_frames = 1

        # Need N consecutive frames before claiming this player has possession
        if self._candidate_frames < POSSESSION_CONFIRM_FRAMES:
            return

        # Confirmed possessor — check if it changed
        if nearest.track_id == self._possessor_id:
            if self._possession_cooldown > 0:
                self._possession_cooldown -= 1
            return

        # Possession has changed to a new player.
        # Always update tracking state immediately (no cooldown on tracking,
        # only on primitive emission) so prev_possessor_id stays fresh for assists.
        old_id   = self._possessor_id
        old_name = self._player_names.get(old_id, f"#{old_id}") if old_id is not None else "unknown"
        new_name = self._player_names.get(nearest.track_id, f"#{nearest.track_id}")

        # held_ms = how long the outgoing possessor had the ball.
        # Must be computed BEFORE updating _possessor_start_ts.
        held_ms = fd.timestamp_ms - self._possessor_start_ts

        self._prev_possessor_id   = old_id
        self._prev_possessor_ts   = fd.timestamp_ms
        self._possessor_id        = nearest.track_id
        self._possessor_start_ts  = fd.timestamp_ms
        self._last_possessor_id   = nearest.track_id   # keep fallback fresh
        self._last_possessor_ts   = fd.timestamp_ms
        self._candidate_frames    = 0
        # Log to possessor history for retroactive shot attribution
        self._possessor_history.append((fd.frame_idx, fd.timestamp_ms, nearest.track_id))

        # Very first ball pickup is not a turnover
        if old_id is None:
            return

        # Rate-limit turnover primitive emission
        if self._possession_cooldown > 0:
            self._possession_cooldown -= 1
            return

        self._possession_cooldown = POSSESSION_COOLDOWN_FRAMES

        from src.logic.event_validator import MIN_POSSESSION_MS as _MIN_POSS_MS
        poss_conf = min(1.0, held_ms / _MIN_POSS_MS) if held_ms > 0 else 0.5
        self._emit({
            "primitive":        "possession_change",
            "player_id":        nearest.track_id,
            "player_name":      new_name,
            "player_team":      "unknown",
            "from_player_id":   old_id,
            "from_player_name": old_name,
            "from_team":        "unknown",
            "new_holder_id":    nearest.track_id,
            "to_team":          "unknown",
            "timestamp_ms":     fd.timestamp_ms,
            "frame_idx":        fd.frame_idx,
            "coordinates":      self._to_court(nearest.foot_center, fd),
            "held_ms":          held_ms,
            "confidence":       round(poss_conf, 3),
        })

    # ------------------------------------------------------------------
    # Hand contact
    # ------------------------------------------------------------------

    def _track_hand_contact(self, fd: FrameDetections) -> None:
        if fd.ball is None:
            return
        bx, by = fd.ball.center
        # Tick down all per-player cooldowns
        for tid in list(self._contact_cooldown):
            self._contact_cooldown[tid] -= 1
            if self._contact_cooldown[tid] <= 0:
                del self._contact_cooldown[tid]
        for p in fd.players:
            if p.track_id is None:
                continue
            if self._contact_cooldown.get(p.track_id, 0) > 0:
                continue
            wp = p.wrist_position()
            if wp is None:
                continue
            dist = pixel_distance(wp, (bx, by))
            if dist < CONTACT_THRESHOLD_PX:
                p_name = self._player_names.get(p.track_id, f"#{p.track_id}")
                contact_conf = max(0.0, 1.0 - dist / CONTACT_THRESHOLD_PX)
                self._emit({
                    "primitive":      "hand_contact",
                    "player_id":      p.track_id,
                    "player_name":    p_name,
                    "player_team":    "unknown",
                    "defender_id":    p.track_id,
                    "defender_name":  p_name,
                    "defender_team":  "unknown",
                    "ball_holder_id": self._possessor_id if self._possessor_id is not None else "unknown",
                    "timestamp_ms":   fd.timestamp_ms,
                    "frame_idx":      fd.frame_idx,
                    "coordinates":    self._to_court(p.foot_center, fd),
                    "distance_px":    dist,
                    "confidence":     round(contact_conf, 3),
                })
                self._contact_cooldown[p.track_id] = CONTACT_COOLDOWN_FRAMES

    # ------------------------------------------------------------------
    # Shot detection
    # ------------------------------------------------------------------

    def _track_shot(self, fd: FrameDetections) -> None:
        """
        Emit shot_taken and shot_outcome primitives.

        Primary method — rim approach:
          Ball tracked far from rim (>SHOT_RIM_FAR_PX) for N frames, then
          suddenly close (<SHOT_RIM_NEAR_PX). That arc = a shot attempt.
          Shooter = last known possessor before ball left possession range.
          This works regardless of camera angle and doesn't require seeing the
          release moment.

        Fallback — upward release:
          Ball moves strongly upward × 3 consecutive frames while separating
          from nearest player. Used when ball doesn't get close to the rim
          (e.g., missed shot that doesn't reach the rim visible).

        Shot made:   ball crosses from above to below the rim plane.
        Shot missed: timeout after SHOT_TIMEOUT_FRAMES without a rim crossing.
        """
        # --- Update locked rim positions ---
        for hoop in fd.hoops:
            if hoop.confidence < 0.35:
                continue
            cx = (hoop.bbox[0] + hoop.bbox[2]) / 2
            cy = (hoop.bbox[1] + hoop.bbox[3]) / 2
            w  = hoop.bbox[2] - hoop.bbox[0]
            side = "left" if cx < fd.frame_width * 0.5 else "right"
            self._rim_positions[side] = (cx, cy, w)

        # Tick shot cooldown
        if self._shot_cooldown > 0:
            self._shot_cooldown -= 1

        # --- Shot in flight: advance timer and check outcome ---
        if self._shot_in_flight:
            self._shot_frames_elapsed += 1

            ball = fd.ball
            made = False
            missed = False

            for side, (rim_cx, rim_cy, rim_w) in self._rim_positions.items():
                if ball is not None:
                    bx, by = ball.center
                    dist = ((bx - rim_cx) ** 2 + (by - rim_cy) ** 2) ** 0.5

                    if dist < rim_w * RIM_PROXIMITY_FACTOR:
                        was_above = self._ball_above_rim.get(side)
                        is_above  = by < rim_cy

                        # Ball crossed from above to below rim = MADE
                        if was_above is True and not is_above:
                            # Only credit as made if ball is within rim width horizontally
                            if abs(bx - rim_cx) < rim_w * 0.8:
                                made = True

                        self._ball_above_rim[side] = is_above

            # Timeout = miss
            if self._shot_frames_elapsed >= SHOT_TIMEOUT_FRAMES:
                missed = True

            if made or missed:
                player_name = self._shot_player_name or f"#{self._shot_player_id}"
                self._emit({
                    "primitive":   "shot_outcome",
                    "player_id":   self._shot_player_id,
                    "player_name": player_name,
                    "player_team": "unknown",
                    "made":        made,
                    "sub_type":    "rim_in" if made else None,
                    "timestamp_ms": fd.timestamp_ms,
                    "frame_idx":    fd.frame_idx,
                    "frame_start":  fd.frame_idx,
                    "coordinates":  self._shot_coords,
                    "confidence":   0.75 if made else 0.55,
                    "reason": (
                        "Ball crossed rim plane (above→below)" if made
                        else f"Shot timed out after {self._shot_frames_elapsed} frames without rim crossing"
                    ),
                })
                self._shot_in_flight = False
                self._shot_cooldown  = SHOT_COOLDOWN_FRAMES
            return

        # ── Primary: rim-approach detection ─────────────────────────────────
        ball = fd.ball

        if ball is not None and not self._shot_in_flight and self._shot_cooldown == 0:
            bx, by = ball.center

            for side, (rim_cx, rim_cy, rim_w) in self._rim_positions.items():
                dist_to_rim = ((bx - rim_cx) ** 2 + (by - rim_cy) ** 2) ** 0.5

                if dist_to_rim > SHOT_RIM_FAR_PX:
                    # Ball is far — accumulate "was far" counter
                    self._rim_far_frames[side] = self._rim_far_frames.get(side, 0) + 1
                    self._rim_approach_fired = False
                elif (dist_to_rim < SHOT_RIM_NEAR_PX
                      and self._rim_far_frames.get(side, 0) >= SHOT_APPROACH_MIN_FRAMES
                      and not self._rim_approach_fired):
                    # Ball arrived at rim after being far — shot detected!
                    self._rim_far_frames[side] = 0
                    self._rim_approach_fired = True

                    # Shooter = last possessor before ball got near rim
                    shooter_id   = self._last_possessor_id or self._possessor_id
                    shooter_name = self._player_names.get(shooter_id, f"#{shooter_id}") if shooter_id else "unknown"

                    if shooter_id is None:
                        continue

                    # Use last possessor's court position for 2pt/3pt
                    # (not the rim position — we want where the shot was TAKEN from)
                    shooter_coords = self._shot_coords if self._shot_coords else self._to_court((bx, by), fd)
                    # Find shooter's actual position from player detections
                    for p in fd.players:
                        if p.track_id == shooter_id:
                            shooter_coords = self._to_court(p.foot_center, fd)
                            break

                    is_three = (
                        shooter_coords["x"] < THREE_POINT_X_RATIO * self._court_l
                        or shooter_coords["x"] > (1 - THREE_POINT_X_RATIO) * self._court_l
                    )

                    self._shot_in_flight      = True
                    self._shot_player_id      = shooter_id
                    self._shot_player_name    = shooter_name
                    self._shot_ts             = fd.timestamp_ms
                    self._shot_frame          = fd.frame_idx
                    self._shot_coords         = shooter_coords
                    self._shot_is_three       = is_three
                    self._shot_frames_elapsed = 0
                    self._ball_above_rim.clear()

                    self._emit({
                        "primitive":    "shot_taken",
                        "player_id":    shooter_id,
                        "player_name":  shooter_name,
                        "player_team":  "unknown",
                        "is_three":     is_three,
                        "timestamp_ms": fd.timestamp_ms,
                        "frame_idx":    fd.frame_idx,
                        "coordinates":  shooter_coords,
                        "prev_possessor_id":   self._prev_possessor_id,
                        "prev_possessor_name": self._player_names.get(self._prev_possessor_id, "") if self._prev_possessor_id else "",
                        "prev_possessor_ts":   self._prev_possessor_ts,
                        "confidence":   0.82,
                        "reason": (
                            f"Ball approached {side} rim from {dist_to_rim:.0f}px "
                            f"(was far {self._rim_far_frames.get(side, 0)+SHOT_APPROACH_MIN_FRAMES} frames), "
                            f"last possessor #{shooter_id}"
                        ),
                    })
                    break  # one shot per frame

        # ── Fallback: upward-release detection ───────────────────────────────
        # Used when ball misses the rim entirely (airball) so rim-approach
        # never fires but the upward arc is still visible.
        if not self._shot_in_flight and self._shot_cooldown == 0 and ball is not None:
            bx, by = ball.center
            if len(self._ball_history) >= 2:
                prev_bx, prev_by, _ = self._ball_history[-2]
                vy = by - prev_by

                nearest_id: Optional[int] = None
                nearest_dist = float("inf")
                for p in fd.players:
                    if p.track_id is None:
                        continue
                    d = ((p.foot_center[0] - bx) ** 2 + (p.foot_center[1] - by) ** 2) ** 0.5
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_id = p.track_id

                separating = False
                if nearest_id is not None:
                    prev_dist = self._prev_ball_dist.get(nearest_id, nearest_dist)
                    separating = (nearest_dist - prev_dist) > SHOT_MIN_SEPARATION_PX
                    self._prev_ball_dist[nearest_id] = nearest_dist

                if vy < -SHOT_UPWARD_VY_PX and separating:
                    self._upward_streak += 1
                else:
                    self._upward_streak = 0

                if self._upward_streak >= SHOT_UPWARD_CONFIRM_FRAMES and nearest_dist <= SHOT_RELEASE_DIST_PX:
                    self._upward_streak = 0
                    shooter_id   = nearest_id or self._possessor_id or self._last_possessor_id
                    shooter_name = self._player_names.get(shooter_id, f"#{shooter_id}") if shooter_id else "unknown"

                    if shooter_id is not None:
                        coords   = self._to_court((bx, by), fd)
                        is_three = (
                            coords["x"] < THREE_POINT_X_RATIO * self._court_l
                            or coords["x"] > (1 - THREE_POINT_X_RATIO) * self._court_l
                        )
                        self._shot_in_flight      = True
                        self._shot_player_id      = shooter_id
                        self._shot_player_name    = shooter_name
                        self._shot_ts             = fd.timestamp_ms
                        self._shot_frame          = fd.frame_idx
                        self._shot_coords         = coords
                        self._shot_is_three       = is_three
                        self._shot_frames_elapsed = 0
                        self._ball_above_rim.clear()
                        self._emit({
                            "primitive":    "shot_taken",
                            "player_id":    shooter_id,
                            "player_name":  shooter_name,
                            "player_team":  "unknown",
                            "is_three":     is_three,
                            "timestamp_ms": fd.timestamp_ms,
                            "frame_idx":    fd.frame_idx,
                            "coordinates":  coords,
                            "prev_possessor_id":   self._prev_possessor_id,
                            "prev_possessor_name": self._player_names.get(self._prev_possessor_id, "") if self._prev_possessor_id else "",
                            "prev_possessor_ts":   self._prev_possessor_ts,
                            "confidence":   0.60,
                            "reason": (
                                f"Airball fallback: upward {abs(vy):.0f}px/frame × "
                                f"{SHOT_UPWARD_CONFIRM_FRAMES} frames from #{shooter_id}"
                            ),
                        })
        elif ball is None:
            self._upward_streak = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    def _to_court(
        self, pixel_pos: tuple[float, float], fd: FrameDetections
    ) -> dict:
        px, py = pixel_pos
        fw, fh = fd.frame_width, fd.frame_height
        # Map pixel x relative to the court span (hoop A → hoop B).
        # A simple full-width linear map places court_x=0 at frame edge,
        # but both hoops are inset (default 10%/90%) so court edges would be
        # off by ~10% — enough to push inside-the-arc shots past the 3pt line.
        court_span_px = (self._hoop_b_x - self._hoop_a_x) * fw
        raw_x = (px - self._hoop_a_x * fw) / court_span_px * self._court_l
        court_x = max(0.0, min(self._court_l, raw_x))
        court_y = (py / fh) * self._court_w
        return {
            "x": round(court_x, 2),
            "y": round(court_y, 2),
        }

    def _emit(self, primitive: dict) -> None:
        self._buffer.append(primitive)
