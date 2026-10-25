"""
BallDetectorCV — OpenCV fallback ball detector.

Used when the COCO YOLO model fails to detect the sports ball class,
which is common with general-purpose weights on broadcast basketball footage.

Approach: HSV colour segmentation isolates the orange basketball, then
blob analysis filters by size, circularity, and proximity to last position.
No background model is needed, so it works at any frame sampling rate.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import cv2
    import numpy as np
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False

# Basketball orange-brown HSV range (OpenCV hue 0–179, half of 360°)
_BALL_HSV_LOW  = (5,  90, 70)    # (H_min, S_min, V_min)
_BALL_HSV_HIGH = (32, 255, 255)  # (H_max, S_max, V_max)

# Expected ball radius range in pixels (wide-angle full-court camera)
_MIN_RADIUS_PX = 7
_MAX_RADIUS_PX = 55

_MIN_CIRCULARITY = 0.55      # floor reflections and jerseys score < 0.45; real ball ≥ 0.55
_MIN_ASPECT      = 0.55      # tighter — rejects elongated reflections and limb blobs
_PLAYER_OVERLAP_THRESH = 0.85  # discard if >85% of blob inside any player bbox
# Only the upper 55% of a player bbox counts as "jersey" exclusion zone —
# a ball at their feet (lower 45%) should not be rejected.
_PLAYER_JERSEY_FRAC = 0.55
_PROX_NEAR_PX   = 180        # within this distance → proximity bonus
_PROX_FAR_PX    = 450        # beyond this → proximity penalty


class BallDetectorCV:
    """
    Single-frame basketball detector using HSV colour + blob filtering.

    Returns an (x1, y1, x2, y2) bbox tuple (float) or None.
    The caller (Detector) wraps the result in a BallDetection object so
    this module has no import dependency on detector.py.
    """

    def __init__(
        self,
        min_radius_px: int = _MIN_RADIUS_PX,
        max_radius_px: int = _MAX_RADIUS_PX,
        min_circularity: float = _MIN_CIRCULARITY,
    ) -> None:
        self._min_area = int(math.pi * min_radius_px ** 2)
        self._max_area = int(math.pi * max_radius_px ** 2)
        self._min_circ = min_circularity
        self._last_pos: Optional[tuple[float, float]] = None

    def detect(
        self,
        frame,
        player_bboxes: Optional[list[tuple[float, float, float, float]]] = None,
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Detect the basketball in a single BGR frame.

        Returns (x1, y1, x2, y2) in pixels, or None.

        Args:
            frame:         BGR numpy array (H×W×3)
            player_bboxes: (x1,y1,x2,y2) boxes of detected players —
                           candidates that are mostly inside a player box
                           are discarded to avoid picking up orange jerseys.
        """
        if not _CV_AVAILABLE:
            return None

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo   = np.array(_BALL_HSV_LOW,  dtype=np.uint8)
        hi   = np.array(_BALL_HSV_HIGH, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox: Optional[tuple[float, float, float, float]] = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self._min_area <= area <= self._max_area):
                continue

            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue

            circ = 4.0 * math.pi * area / (perim * perim)
            if circ < self._min_circ:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw == 0 or bh == 0:
                continue
            aspect = min(bw, bh) / max(bw, bh)
            if aspect < _MIN_ASPECT:
                continue

            cx, cy = bx + bw / 2.0, by + bh / 2.0

            # Discard blobs at frame edges (court markings, sideline paint)
            fh, fw = frame.shape[:2]
            if cx < fw * 0.03 or cx > fw * 0.97 or cy > fh * 0.92 or cy < fh * 0.03:
                continue

            # Discard blobs mostly inside a player bounding box
            if player_bboxes and _inside_player(bx, by, bw, bh, player_bboxes, _PLAYER_OVERLAP_THRESH):
                continue

            # Proximity bonus: prefer candidates near last known position
            prox = 1.0
            if self._last_pos is not None:
                d = math.dist((cx, cy), self._last_pos)
                if d < _PROX_NEAR_PX:
                    prox = 1.6
                elif d > _PROX_FAR_PX:
                    prox = 0.6

            score = circ * aspect * prox
            if score > best_score:
                best_score = score
                best_bbox  = (float(bx), float(by), float(bx + bw), float(by + bh))

        if best_bbox is None:
            return None

        x1, y1, x2, y2 = best_bbox
        self._last_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
        return best_bbox

    def reset(self) -> None:
        """Clear tracking state between clips."""
        self._last_pos = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inside_player(
    bx: int, by: int, bw: int, bh: int,
    player_bboxes: list[tuple[float, float, float, float]],
    threshold: float,
) -> bool:
    blob_area = bw * bh
    if blob_area == 0:
        return False
    for px1, py1, px2, py2 in player_bboxes:
        # Only use the upper jersey/torso portion of each player bbox.
        # A ball sitting on the floor at a player's feet should not be
        # rejected because it overlaps with the feet region.
        jersey_y2 = py1 + (py2 - py1) * _PLAYER_JERSEY_FRAC
        ix1 = max(float(bx), px1);  ix2 = min(float(bx + bw), px2)
        iy1 = max(float(by), py1);  iy2 = min(float(by + bh), jersey_y2)
        if ix2 > ix1 and iy2 > iy1:
            overlap = (ix2 - ix1) * (iy2 - iy1) / blob_area
            if overlap > threshold:
                return True
    return False
