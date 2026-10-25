"""
CourtMask — finds the playing court boundary from painted court lines.

Strategy:
  1. Sample the hardwood floor color from the lower-centre of the frame.
  2. Build a floor-color mask (anything that looks like hardwood).
  3. Within that mask, detect the bright white/yellow court lines using
     HoughLinesP.
  4. Find the near sideline (the lowest horizontal line) — everything above
     it is on-court; the bench/spectator area is below it.
  5. Also find left/right boundary lines to handle end-zone bench players.
  6. Cache the boundary — rebuild every UPDATE_EVERY frames.

foot_on_court(cx, foot_y) returns False for bench players, coaches, and
spectators whose feet land outside the detected court boundary.
"""

from __future__ import annotations
from typing import Optional


UPDATE_EVERY    = 60    # rebuild mask every N sampled frames
MIN_FLOOR_FRAC  = 0.15  # at least 15% of frame must look like hardwood


class CourtMask:
    def __init__(self) -> None:
        self._frame_count    = 0
        self._valid          = False
        # Court boundary in pixel coords (set after detection)
        self._near_y: Optional[float] = None   # near sideline y (players must be above)
        self._far_y:  Optional[float] = None   # far sideline y  (players must be below)
        self._left_x: Optional[float] = None   # left boundary
        self._right_x: Optional[float] = None  # right boundary
        self._frame_w = 0
        self._frame_h = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame) -> None:
        self._frame_count += 1
        if self._frame_count % UPDATE_EVERY != 1:
            return
        try:
            self._build(frame)
        except Exception:
            pass

    def foot_on_court(self, cx: float, foot_y: float) -> bool:
        """
        Return True if the foot pixel is inside the detected court boundary.
        Fails open (returns True) when the mask is not yet calibrated.
        """
        if not self._valid:
            return True

        # Near sideline: bench is between camera and this line, so bench
        # players' feet appear BELOW it (higher y-value in image coords).
        if self._near_y is not None and foot_y > self._near_y + 12:
            return False

        # Far sideline: far wall / stands appear above this line.
        if self._far_y is not None and foot_y < self._far_y - 12:
            return False

        # Left/right boundaries (handles end-zone bench players).
        if self._left_x is not None and cx < self._left_x - 20:
            return False
        if self._right_x is not None and cx > self._right_x + 20:
            return False

        return True

    def sideline_ys(self) -> tuple[Optional[float], Optional[float]]:
        """Return (near_y, far_y) pixel Y coordinates of the two sidelines. None if not detected."""
        return (self._near_y, self._far_y)

    @property
    def valid(self) -> bool:
        return self._valid

    def debug_overlay(self, frame):
        """Draw detected court boundary on a copy of frame (for visualisation)."""
        import cv2, numpy as np
        out = frame.copy()
        if not self._valid:
            return out
        h, w = out.shape[:2]
        if self._near_y is not None:
            cv2.line(out, (0, int(self._near_y)), (w, int(self._near_y)), (0,255,0), 2)
        if self._far_y is not None:
            cv2.line(out, (0, int(self._far_y)),  (w, int(self._far_y)),  (0,200,0), 1)
        if self._left_x is not None:
            cv2.line(out, (int(self._left_x), 0), (int(self._left_x), h), (0,255,0), 2)
        if self._right_x is not None:
            cv2.line(out, (int(self._right_x),0), (int(self._right_x),h), (0,255,0), 2)
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self, frame) -> None:
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        self._frame_h, self._frame_w = h, w
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── Step 1: floor-color mask ──────────────────────────────────────
        # Sample hardwood color from bottom-centre strip (almost always floor)
        sy1, sy2 = int(h * 0.72), int(h * 0.92)
        sx1, sx2 = int(w * 0.25), int(w * 0.75)
        sample = hsv[sy1:sy2, sx1:sx2]
        med_h = float(np.median(sample[:, :, 0]))
        med_s = float(np.median(sample[:, :, 1]))
        med_v = float(np.median(sample[:, :, 2]))

        lo = np.array([max(0,   med_h - 20), max(0,   med_s - 60), max(0,   med_v - 70)], np.uint8)
        hi = np.array([min(179, med_h + 20), min(255, med_s + 60), min(255, med_v + 70)], np.uint8)
        floor_mask = cv2.inRange(hsv, lo, hi)

        # Sanity: floor must cover reasonable area
        if float(np.mean(floor_mask > 0)) < MIN_FLOOR_FRAC:
            self._valid = False
            return

        # ── Step 2: detect court line pixels (white/bright on floor) ─────
        # Court lines: high value, low saturation — brighter than the wood
        line_mask = ((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 160)).astype(np.uint8) * 255
        # Only keep line pixels that sit on the floor region
        line_on_floor = cv2.bitwise_and(line_mask, floor_mask)

        # Morphological close to connect line segments
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        line_on_floor = cv2.dilate(line_on_floor, k, iterations=1)

        # ── Step 3: Hough line detection ──────────────────────────────────
        edges = cv2.Canny(line_on_floor, 30, 120)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=int(w * 0.10),      # line must span 10% of width
            minLineLength=int(w * 0.10),
            maxLineGap=int(w * 0.05),
        )

        if lines is None:
            # No court lines detected — fall back to floor-only boundary
            # Use bottom of the detected floor region as near sideline
            rows = np.where(floor_mask > 0)
            if len(rows[0]) == 0:
                self._valid = False
                return
            self._near_y  = float(np.percentile(rows[0], 92))
            self._far_y   = float(np.percentile(rows[0], 8))
            self._left_x  = None
            self._right_x = None
            self._valid   = True
            return

        # ── Step 4: classify lines as horizontal / vertical ───────────────
        h_lines: list[tuple] = []
        v_lines: list[tuple] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx == 0 and dy == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 20:          # nearly horizontal
                mid_y = (y1 + y2) / 2
                h_lines.append((mid_y, x1, y1, x2, y2))
            elif angle > 70:        # nearly vertical
                mid_x = (x1 + x2) / 2
                v_lines.append((mid_x, x1, y1, x2, y2))

        # ── Step 5: find boundary lines ───────────────────────────────────
        # Near sideline = lowest horizontal line (highest y in image coords)
        # Require it to be in the lower 60% of the frame
        h_below = [l for l in h_lines if l[0] > h * 0.40]
        if h_below:
            self._near_y = float(max(l[0] for l in h_below))
        else:
            # Fallback: floor region bottom percentile
            rows = np.where(floor_mask > 0)
            self._near_y = float(np.percentile(rows[0], 90)) if len(rows[0]) else None

        # Far sideline = highest horizontal line in the upper half
        h_above = [l for l in h_lines if l[0] < h * 0.55]
        self._far_y = float(min(l[0] for l in h_above)) if h_above else None

        # ── Left/right boundaries ─────────────────────────────────────────
        # Primary: vertical Hough lines at the frame edges.
        v_left  = [l for l in v_lines if l[0] < w * 0.30]
        v_right = [l for l in v_lines if l[0] > w * 0.70]
        if v_left:
            self._left_x = float(min(l[0] for l in v_left))
        else:
            self._left_x = None

        if v_right:
            self._right_x = float(max(l[0] for l in v_right))
        else:
            self._right_x = None

        # Fallback: use the horizontal x-extent of detected sideline segments.
        # The near sideline (and far sideline) span the court width.  Taking the
        # min/max x of those segments gives us the court left/right edge without
        # needing painted vertical lines, which are often absent or faint.
        all_h_segs = h_below + (h_above if h_above else [])
        if all_h_segs:
            seg_left_xs  = [min(l[1], l[3]) for l in all_h_segs]
            seg_right_xs = [max(l[1], l[3]) for l in all_h_segs]
            if self._left_x is None:
                # 20th percentile — conservative (don't cut into the court)
                self._left_x = float(np.percentile(seg_left_xs, 20))
            if self._right_x is None:
                self._right_x = float(np.percentile(seg_right_xs, 80))

        self._valid = True
