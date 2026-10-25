"""
CourtHomography — pixel coordinates → real-world court metres for panning cameras.

How it works
────────────
Stage 3 of the court-calibration roadmap (see rim_tracker.py).

The two basketball hoops are fixed landmarks at known world positions:
  left hoop  → (1.575 m, 7.5 m)   [FIBA: 1.575 m from baseline, court centre]
  right hoop → (26.425 m, 7.5 m)

Their pixel X-positions (from RimTracker) define where each baseline sits in the
image. The near/far sideline pixel Ys (from CourtMask) define the Y-span. Together
these four quantities let us construct four pixel↔world corner correspondences and
solve for a 3×3 homography with cv2.findHomography.

The homography is recomputed every UPDATE_INTERVAL frames so it stays accurate as
the camera pans across the court.

Court coordinate system (FIBA, metres)
  x: 0 (left baseline)  →  28 (right baseline)
  y: 0 (far sideline)   →  15 (near sideline — camera side)

Usage
─────
    hom = CourtHomography()

    # per sampled frame:
    hom.update(
        left_hoop_px  = tracker_l.pos,   # (cx, cy) or None
        right_hoop_px = tracker_r.pos,   # (cx, cy) or None
        near_y        = court_mask.sideline_ys()[0],
        far_y         = court_mask.sideline_ys()[1],
        frame_w=W, frame_h=H,
    )

    court_pos = hom.pixel_to_court(px, py)  # (x_m, y_m) or None
    minimap   = hom.draw_minimap(players, ball_court)
"""

from __future__ import annotations
from typing import Optional


# ── FIBA court dimensions ─────────────────────────────────────────────────────
COURT_LEN = 28.0    # metres, x-axis
COURT_WID = 15.0    # metres, y-axis

LEFT_HOOP_WORLD  = (1.575, 7.5)
RIGHT_HOOP_WORLD = (26.425, 7.5)

# How often to recompute H (every N sampled frames)
UPDATE_INTERVAL = 8


class CourtHomography:
    """
    Maintains a per-frame homography between the video frame and a standard
    FIBA basketball court coordinate system.
    """

    def __init__(self) -> None:
        self._H: Optional[object]     = None   # pixel → world (3×3 numpy float32)
        self._H_inv: Optional[object] = None   # world → pixel
        self._frame_count: int        = 0
        self._last_anchors: Optional[tuple] = None   # (lx, rx, near_y, far_y)
        self._minimap_base: Optional[object] = None  # cached blank court image

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        left_hoop_px:  Optional[tuple[float, float]],
        right_hoop_px: Optional[tuple[float, float]],
        near_y: Optional[float],
        far_y:  Optional[float],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        Feed latest anchor estimates. Call once per sampled frame.
        Missing values fall back to the last known good values, then to defaults.
        """
        self._frame_count += 1

        # Resolve hoop X positions
        lx = left_hoop_px[0]  if left_hoop_px  is not None else (
             self._last_anchors[0] if self._last_anchors else None)
        rx = right_hoop_px[0] if right_hoop_px is not None else (
             self._last_anchors[1] if self._last_anchors else None)

        if lx is None or rx is None:
            return   # not enough anchors yet

        # Resolve sideline Ys
        if near_y is None:
            near_y = self._last_anchors[2] if self._last_anchors else frame_h * 0.85
        if far_y is None:
            far_y  = self._last_anchors[3] if self._last_anchors else frame_h * 0.15

        self._last_anchors = (lx, rx, near_y, far_y)

        if self._H is None or self._frame_count % UPDATE_INTERVAL == 1:
            self._compute(lx, rx, near_y, far_y)

    def pixel_to_court(
        self, px: float, py: float
    ) -> Optional[tuple[float, float]]:
        """
        Map a single pixel position to court metres.
        Returns None if the homography is not yet calibrated.
        """
        if self._H is None:
            return None
        import cv2
        import numpy as np

        pt = np.float32([[[px, py]]])
        res = cv2.perspectiveTransform(pt, self._H)
        x_m = float(np.clip(res[0][0][0], -3.0, COURT_LEN + 3.0))
        y_m = float(np.clip(res[0][0][1], -3.0, COURT_WID + 3.0))
        return (x_m, y_m)

    def court_to_pixel(
        self, x_m: float, y_m: float
    ) -> Optional[tuple[int, int]]:
        """Inverse: court metres → pixel position."""
        if self._H_inv is None:
            return None
        import cv2
        import numpy as np

        pt = np.float32([[[x_m, y_m]]])
        res = cv2.perspectiveTransform(pt, self._H_inv)
        return (int(res[0][0][0]), int(res[0][0][1]))

    @property
    def ready(self) -> bool:
        return self._H is not None

    def draw_minimap(
        self,
        players: list[tuple[float, float, str]],   # (x_m, y_m, team)
        ball_court: Optional[tuple[float, float]] = None,
        scale: int = 10,
        pad: int   = 8,
    ):
        """
        Build and return a BGR minimap image showing player/ball positions.

        players: list of (x_m, y_m, team_label)  — team_label: 'a', 'b', or 'unknown'
        ball_court: (x_m, y_m) or None
        scale: pixels per metre
        pad: padding in pixels around the court
        """
        import cv2
        import numpy as np

        W = int(COURT_LEN * scale) + 2 * pad
        H = int(COURT_WID * scale) + 2 * pad

        img = self._blank_court(scale, pad).copy()

        def to_px(xm: float, ym: float) -> tuple[int, int]:
            return (int(xm * scale + pad), int(ym * scale + pad))

        # Players
        TEAM_COLORS = {
            "a":       (220,  80,  80),   # blue-ish
            "b":       ( 80,  80, 220),   # red-ish
            "unknown": (200, 200, 200),   # grey
        }
        for x_m, y_m, team in players:
            if not (0 <= x_m <= COURT_LEN and 0 <= y_m <= COURT_WID):
                continue
            color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])
            px, py = to_px(x_m, y_m)
            cv2.circle(img, (px, py), 5, color, -1)
            cv2.circle(img, (px, py), 5, (255, 255, 255), 1)

        # Ball
        if ball_court is not None:
            bx, by = ball_court
            if 0 <= bx <= COURT_LEN and 0 <= by <= COURT_WID:
                bpx, bpy = to_px(bx, by)
                cv2.circle(img, (bpx, bpy), 4, (0, 165, 255), -1)
                cv2.circle(img, (bpx, bpy), 4, (255, 255, 255), 1)

        # Hoops
        lhx, lhy = to_px(*LEFT_HOOP_WORLD)
        rhx, rhy = to_px(*RIGHT_HOOP_WORLD)
        cv2.circle(img, (lhx, lhy), 3, (0, 200, 255), -1)
        cv2.circle(img, (rhx, rhy), 3, (0, 200, 255), -1)

        return img

    # ── Internal ─────────────────────────────────────────────────────────────

    def _compute(
        self, lx: float, rx: float, near_y: float, far_y: float
    ) -> None:
        import cv2
        import numpy as np

        # Sanity: hoops should be separated by at least 30% of expected distance
        if rx - lx < 50:
            return

        # Sanity: near sideline should be below far sideline
        if near_y <= far_y:
            return

        # Four pixel corners derived from hoop X positions and sideline Ys
        src = np.float32([
            [lx, near_y],            # near-left court corner
            [rx, near_y],            # near-right court corner
            [lx, far_y],             # far-left court corner
            [rx, far_y],             # far-right court corner
        ])

        # Corresponding FIBA world corners (metres)
        dst = np.float32([
            [0.0,       COURT_WID],  # near-left  (0, 15)
            [COURT_LEN, COURT_WID],  # near-right (28, 15)
            [0.0,       0.0],        # far-left   (0, 0)
            [COURT_LEN, 0.0],        # far-right  (28, 0)
        ])

        H, _ = cv2.findHomography(src, dst)
        if H is not None:
            self._H     = H.astype(np.float32)
            self._H_inv = np.linalg.inv(H).astype(np.float32)

    def _blank_court(self, scale: int, pad: int):
        """Return a cached blank court drawing (hardwood + white lines)."""
        import cv2
        import numpy as np

        key = (scale, pad)
        if self._minimap_base is not None:
            return self._minimap_base

        W = int(COURT_LEN * scale) + 2 * pad
        H = int(COURT_WID * scale) + 2 * pad
        img = np.zeros((H, W, 3), dtype=np.uint8)

        def to_px(xm: float, ym: float) -> tuple[int, int]:
            return (int(xm * scale + pad), int(ym * scale + pad))

        # Hardwood floor
        img[:] = (50, 100, 160)   # dark blue fallback
        cv2.rectangle(img, to_px(0, 0), to_px(COURT_LEN, COURT_WID),
                      (120, 170, 210), -1)

        LW = 1   # line width
        WHITE = (255, 255, 255)

        # Court boundary
        cv2.rectangle(img, to_px(0, 0), to_px(COURT_LEN, COURT_WID), WHITE, LW)

        # Half-court line
        cv2.line(img, to_px(14, 0), to_px(14, COURT_WID), WHITE, LW)

        # Centre circle (r=1.8m)
        cc = to_px(14, 7.5)
        cv2.circle(img, cc, int(1.8 * scale), WHITE, LW)

        # Left paint (5.8m deep, 4.9m wide centred at y=7.5)
        cv2.rectangle(img, to_px(0, 5.05), to_px(5.8, 9.95), WHITE, LW)
        # Left free-throw circle
        fc_l = to_px(5.8, 7.5)
        cv2.circle(img, fc_l, int(1.8 * scale), WHITE, LW)

        # Right paint
        cv2.rectangle(img, to_px(22.2, 5.05), to_px(COURT_LEN, 9.95), WHITE, LW)
        # Right free-throw circle
        fc_r = to_px(22.2, 7.5)
        cv2.circle(img, fc_r, int(1.8 * scale), WHITE, LW)

        # Left three-point arc
        lh = to_px(*LEFT_HOOP_WORLD)
        r3 = int(6.75 * scale)
        # Full circle then mask out right half with hardwood colour
        cv2.circle(img, lh, r3, WHITE, LW)
        # Erase right portion (x > lh[0]) so only arc facing right side shows
        cv2.rectangle(img, (lh[0], 0), (W, H), (120, 170, 210), -1)
        # Redraw boundary and paint on top
        cv2.rectangle(img, to_px(0, 0), to_px(COURT_LEN, COURT_WID), WHITE, LW)
        cv2.rectangle(img, to_px(0, 5.05), to_px(5.8, 9.95), WHITE, LW)
        cv2.circle(img, lh, r3, WHITE, LW)
        # Erase again to clip
        cv2.rectangle(img, (lh[0], 0), (W, H), (120, 170, 210), -1)
        cv2.rectangle(img, to_px(0, 0), to_px(COURT_LEN, COURT_WID), WHITE, LW)

        # Better: just draw the arc directly
        img2 = np.zeros((H, W, 3), dtype=np.uint8)
        img2[:] = (120, 170, 210)
        # Court boundary
        cv2.rectangle(img2, to_px(0, 0), to_px(COURT_LEN, COURT_WID), WHITE, LW)
        # Half-court
        cv2.line(img2, to_px(14, 0), to_px(14, COURT_WID), WHITE, LW)
        # Centre circle
        cv2.circle(img2, cc, int(1.8 * scale), WHITE, LW)
        # Paints
        cv2.rectangle(img2, to_px(0, 5.05), to_px(5.8, 9.95), WHITE, LW)
        cv2.circle(img2, fc_l, int(1.8 * scale), WHITE, LW)
        cv2.rectangle(img2, to_px(22.2, 5.05), to_px(COURT_LEN, 9.95), WHITE, LW)
        cv2.circle(img2, fc_r, int(1.8 * scale), WHITE, LW)
        # Left three-point arc (angle: pointing away from left baseline = right)
        cv2.ellipse(img2, lh, (r3, r3), 0, -90, 90, WHITE, LW)
        # Left 3pt straight sections (near + far sideline edges, ~0.9m from sideline)
        cv2.line(img2, to_px(0, 0.9),  to_px(1.575 + 0.4, 0.9),  WHITE, LW)
        cv2.line(img2, to_px(0, 14.1), to_px(1.575 + 0.4, 14.1), WHITE, LW)
        # Right three-point arc
        rh = to_px(*RIGHT_HOOP_WORLD)
        cv2.ellipse(img2, rh, (r3, r3), 0, 90, 270, WHITE, LW)
        cv2.line(img2, to_px(COURT_LEN, 0.9),  to_px(26.425 - 0.4, 0.9),  WHITE, LW)
        cv2.line(img2, to_px(COURT_LEN, 14.1), to_px(26.425 - 0.4, 14.1), WHITE, LW)

        self._minimap_base = img2
        return img2
