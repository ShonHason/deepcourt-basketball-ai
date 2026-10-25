"""
RimTracker — velocity-limited hoop position tracker for panning broadcast cameras.

Treats the hoop as a moving object in pixel space (it moves because the camera pans,
not because the rim physically moves). Distinguishes between:

  • Camera pan  — hoop moves smoothly across the frame at plausible velocity
                  → follow it, update velocity estimate
  • False positive — detected object jumps impossibly far in one frame
                  → reject, keep last known position
  • Hoop off-screen — camera has panned past the basket
                  → enter OFF_SCREEN state, extrapolate from velocity

State machine:
  CALIBRATING → TRACKING → OFF_SCREEN → TRACKING
                    ↑__________________________|

Stage 1 of the court-calibration roadmap.
Stage 2 will add court keypoint detection.
Stage 3 will replace pixel tracking with world-coordinate homography.
"""

from __future__ import annotations
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Maximum pixel displacement per sampled frame that is consistent with a
# camera pan. At 5 fps, a fast pan moves the rim ~150-250 px per frame.
# A false positive (scoreboard, fan sign) appears 400+ px away instantly.
MAX_PAN_VELOCITY_PX  = 500   # px/frame — above this = likely false positive
MAX_JUMP_PX          = 600   # hard cap — if new detection is farther, always reject

# How many frames we accept a detection being outside the predicted window
# before we update the velocity model (handles rapid zoom changes)
VELOCITY_GRACE       = 3

# After this many consecutive frames with no in-range detection, declare OFF_SCREEN
OFF_SCREEN_FRAMES    = 20    # ~4 s at 5 fps — broadcast camera pans take time

# Minimum detections before we trust the position enough to fire shots
CALIBRATION_MIN      = 4

# EMA weight for position smoothing (0 = never update, 1 = always jump to new)
SMOOTH_ALPHA         = 0.35

# EMA weight for velocity estimation
VEL_ALPHA            = 0.40


class RimTracker:
    """
    Tracks one basketball rim across a panning/zooming broadcast camera.

    Usage::
        tracker = RimTracker(side="left")
        # per sampled frame:
        tracker.update(cx, cy)        # when rim detected at (cx, cy)
        tracker.update(None, None)    # when rim not detected this frame

        pos = tracker.pos             # current best estimate (or None if OFF_SCREEN)
        tracker.ready                 # True once calibration complete
        tracker.state                 # "calibrating" | "tracking" | "off_screen"
    """

    def __init__(self, side: str) -> None:
        self.side = side
        self._state      = "calibrating"
        self._pos: Optional[tuple[float, float]] = None   # (cx, cy) smoothed
        self._vx: float  = 0.0    # velocity estimate (px/frame)
        self._vy: float  = 0.0
        self._lost: int  = 0      # consecutive frames with no valid detection
        self._n_detections: int = 0
        self._grace: int = 0      # frames since last large-but-accepted jump
        self._history: deque = deque(maxlen=6)   # recent raw detections for debugging

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, cx: Optional[float], cy: Optional[float]) -> None:
        """Call once per sampled frame with the raw hoop detection (or None)."""
        if cx is None or cy is None:
            self._on_miss()
            return

        self._history.append((cx, cy))

        if self._pos is None:
            # First detection — initialise
            self._pos = (cx, cy)
            self._n_detections += 1
            if self._n_detections >= CALIBRATION_MIN:
                self._state = "tracking"
            return

        predicted_cx = self._pos[0] + self._vx
        predicted_cy = self._pos[1] + self._vy
        dist_to_predicted = _dist((cx, cy), (predicted_cx, predicted_cy))
        dist_to_current   = _dist((cx, cy), self._pos)

        # Hard cap — impossible jump, always reject
        if dist_to_current > MAX_JUMP_PX:
            self._on_miss()
            return

        # Check against velocity-predicted position
        if dist_to_predicted > MAX_PAN_VELOCITY_PX:
            if self._grace < VELOCITY_GRACE:
                # Give a few frames of grace for rapid zoom / cut
                self._grace += 1
                self._on_miss()
                return
            # Too many grace frames — this is a real reposition (zoom/cut)
            # Reset velocity and follow the detection
            self._vx, self._vy = 0.0, 0.0
            self._grace = 0

        # Accepted detection — update velocity then position
        new_vx = cx - self._pos[0]
        new_vy = cy - self._pos[1]
        self._vx = VEL_ALPHA * new_vx + (1 - VEL_ALPHA) * self._vx
        self._vy = VEL_ALPHA * new_vy + (1 - VEL_ALPHA) * self._vy

        self._pos = (
            SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * self._pos[0],
            SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * self._pos[1],
        )
        self._lost  = 0
        self._grace = 0
        self._n_detections += 1

        if self._state == "calibrating" and self._n_detections >= CALIBRATION_MIN:
            self._state = "tracking"
            print(f"  [RimTracker] {self.side} hoop TRACKING at "
                  f"({self._pos[0]:.0f}, {self._pos[1]:.0f})")
        elif self._state == "off_screen":
            self._state = "tracking"
            print(f"  [RimTracker] {self.side} hoop RE-ACQUIRED at "
                  f"({self._pos[0]:.0f}, {self._pos[1]:.0f})")

    def _on_miss(self) -> None:
        """No valid detection this frame — extrapolate or go OFF_SCREEN."""
        self._lost += 1
        if self._pos is not None and self._state == "tracking":
            # Extrapolate position along last known velocity (decays quickly)
            self._pos = (
                self._pos[0] + self._vx * 0.6,
                self._pos[1] + self._vy * 0.6,
            )
            # Decay velocity so we don't drift forever
            self._vx *= 0.7
            self._vy *= 0.7

        if self._lost >= OFF_SCREEN_FRAMES and self._state != "calibrating":
            self._state = "off_screen"
            # Stop extrapolating — position is stale
            self._vx, self._vy = 0.0, 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pos(self) -> Optional[tuple[float, float]]:
        """Best current position estimate. None if off-screen or not yet calibrated."""
        if self._state == "calibrating":
            return self._pos   # return partial estimate (may be None)
        if self._state == "off_screen":
            return None        # explicitly unknown — don't fire shots
        return self._pos

    @property
    def ready(self) -> bool:
        """True once calibration complete and hoop is on screen."""
        return self._state == "tracking"

    @property
    def state(self) -> str:
        return self._state

    @property
    def velocity(self) -> tuple[float, float]:
        return (self._vx, self._vy)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dist(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
