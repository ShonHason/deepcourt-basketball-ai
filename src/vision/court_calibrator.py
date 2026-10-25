"""
CourtCalibrator — auto-detect hoop pixel positions from early game frames.

Why this exists:
  The court coordinate mapping (pixel → real metres) requires knowing where
  the two hoops are in the frame.  Default values (10% / 90%) work for
  standard full-court side cameras, but fail for end-court phones, tablets,
  or high-angle ceiling cameras common in practice gyms.

How it works:
  Prefer explicit hoop detections from the basketball/Roboflow detector.
  Collect left/right hoop x-positions over early sampled frames, then average
  the stable detections.  Ball positions are intentionally not used: youth and
  high-school clips contain passes, warmups, and side baskets, so ball clusters
  are a weak proxy for the actual game hoops.

  The result is a (hoop_a_x_ratio, hoop_b_x_ratio) tuple in [0, 1] that
  can be passed directly to PrimitiveGenerator and Detector.

  If fewer than MIN_HOOP_SAMPLES positions are collected, the calibrator
  returns the DEFAULT values.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_HOOP_A_X = 0.10   # left hoop, standard wide-angle full-court camera
DEFAULT_HOOP_B_X = 0.90   # right hoop

CALIB_FRAMES    = 200     # sample the first N frames for calibration
MIN_HOOP_SAMPLES = 5      # minimum hoop detections per side before trusting result
MIN_HOOP_CONF    = 0.40   # synthetic anchors use lower confidence and are ignored


class CourtCalibrator:
    """
    Feed frames via update(); call result() after CALIB_FRAMES to get
    calibrated (hoop_a_x_ratio, hoop_b_x_ratio).
    """

    def __init__(self) -> None:
        self._left_hoop_xs: list[float] = []   # normalised [0,1]
        self._right_hoop_xs: list[float] = []  # normalised [0,1]
        self._frames_seen: int = 0
        self._locked: bool = False
        self._hoop_a: float = DEFAULT_HOOP_A_X
        self._hoop_b: float = DEFAULT_HOOP_B_X

    @property
    def ready(self) -> bool:
        return self._locked

    def update(self, fd) -> None:
        """
        Feed one FrameDetections.  Stop collecting after CALIB_FRAMES.
        """
        if self._locked:
            return
        self._frames_seen += 1

        for hoop in getattr(fd, "hoops", []) or []:
            if getattr(hoop, "confidence", 0.0) < MIN_HOOP_CONF:
                continue
            hx, _ = hoop.center
            x_ratio = hx / fd.frame_width if fd.frame_width else 0.0
            if hoop.side == "left" or x_ratio < 0.5:
                self._left_hoop_xs.append(x_ratio)
            elif hoop.side == "right" or x_ratio >= 0.5:
                self._right_hoop_xs.append(x_ratio)

        if self._frames_seen >= CALIB_FRAMES:
            self._compute()
            self._locked = True
        elif (
            len(self._left_hoop_xs) >= MIN_HOOP_SAMPLES
            and len(self._right_hoop_xs) >= MIN_HOOP_SAMPLES
        ):
            self._compute()
            self._locked = True

    def result(self) -> tuple[float, float]:
        """Return (hoop_a_x_ratio, hoop_b_x_ratio)."""
        if (
            not self._locked
            and len(self._left_hoop_xs) >= MIN_HOOP_SAMPLES
            and len(self._right_hoop_xs) >= MIN_HOOP_SAMPLES
        ):
            self._compute()
        return (self._hoop_a, self._hoop_b)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute(self) -> None:
        if (
            len(self._left_hoop_xs) < MIN_HOOP_SAMPLES
            or len(self._right_hoop_xs) < MIN_HOOP_SAMPLES
        ):
            return  # keep defaults

        left_mean = _trimmed_mean(self._left_hoop_xs)
        right_mean = _trimmed_mean(self._right_hoop_xs)

        # Sanity check: the two hoops should be on opposite halves of the court
        # and at least 30% of frame width apart.
        if right_mean - left_mean < 0.30:
            return  # looks like a halfcourt video or bad data — keep defaults

        # Clamp to [0.05, 0.45] and [0.55, 0.95] so we never place a hoop
        # at the very edge or in the wrong half.
        self._hoop_a = max(0.05, min(0.45, left_mean))
        self._hoop_b = max(0.55, min(0.95, right_mean))


def _trimmed_mean(values: list[float]) -> float:
    values = sorted(values)
    if len(values) >= 5:
        trim = max(1, len(values) // 10)
        values = values[trim:-trim] or values
    return sum(values) / len(values)
