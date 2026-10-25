"""
Detector — frame-level basketball object detection and pose estimation.

Wraps YOLOv11 (object detection) and YOLO-Pose (keypoint estimation).
Outputs structured FrameDetections that tracker.py will consume for
persistent player IDs and jersey OCR.

Interest Zone optimisation:
  Full-resolution inference runs only within ~4m of each hoop.
  Centre-court frames are downscaled 50% before inference, then
  coordinates are scaled back up — halves GPU load without missing
  possession events near the hoops.

Processing target: 5-10 FPS (caller decides frame sampling rate).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

# ---------------------------------------------------------------------------
# COCO class IDs used by YOLOv11 base weights
# ---------------------------------------------------------------------------
COCO_PERSON = 0
COCO_SPORTS_BALL = 32
COCO_TV    = 62   # scoreboard / display screen
COCO_CLOCK = 74   # scoreboard clock face

# YOLO-Pose COCO keypoint indices
KP_NOSE          = 0
KP_LEFT_EYE      = 1
KP_RIGHT_EYE     = 2
KP_LEFT_EAR      = 3
KP_RIGHT_EAR     = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW    = 7
KP_RIGHT_ELBOW   = 8
KP_LEFT_WRIST    = 9
KP_RIGHT_WRIST   = 10
KP_LEFT_HIP      = 11
KP_RIGHT_HIP     = 12
KP_LEFT_KNEE     = 13
KP_RIGHT_KNEE    = 14
KP_LEFT_ANKLE    = 15
KP_RIGHT_ANKLE   = 16

# ---------------------------------------------------------------------------
# Court geometry for interest zones (pixels depend on camera; use relative %)
# Default assumes a wide-angle full-court camera, hoops at ~10% and ~90% width
# ---------------------------------------------------------------------------
DEFAULT_HOOP_A_X_RATIO = 0.10   # hoop A at 10% of frame width
DEFAULT_HOOP_B_X_RATIO = 0.90   # hoop B at 90% of frame width
INTEREST_ZONE_RADIUS_RATIO = 0.20   # 20% of frame width around each hoop
DOWNSCALE_FACTOR = 0.5          # centre-court processed at 50% resolution

# Confidence thresholds
BALL_CONF = 0.35
PLAYER_CONF = 0.45              # lowered: court mask handles spatial filtering now
HOOP_CONF = 0.40

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _default_bball_model_path() -> str:
    """
    Pick the best available basketball detector in a predictable order.

    Priority:
      1. Explicit env override (DEEPCOURT_BBALL_MODEL)
      2. Locally-trained best checkpoint (v10 final, then v8)
      3. Known fallbacks
    """
    env_path = os.getenv("DEEPCOURT_BBALL_MODEL")
    if env_path:
        return env_path

    candidates = [
        PROJECT_ROOT / "models/deepcourt_v10_final.pt",
        PROJECT_ROOT / "models/deepcourt_v8_best.pt",
        PROJECT_ROOT / "models/deepcourt_ball_rim_best.pt",
        PROJECT_ROOT / "models/deepcourt_ball_rim_v2_epoch1_best.pt",
        PROJECT_ROOT / "models/deepcourt_ball_rim_v2_yolo11s_epoch1_best.pt",
        PROJECT_ROOT / "models/basketball_best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "models/basketball_best.pt"


# Basketball-specific model (detects Basketball + Basketball Hoop)
BBALL_MODEL_PATH = _default_bball_model_path()
BBALL_BALL_CONF  = 0.25   # trusted immediately when context is weak
BBALL_BALL_LOW_CONF = 0.01  # keep weak candidates for temporal/context scoring
BBALL_HOOP_CONF  = 0.40
# Ignore hoop detections at frame edges (side/practice baskets)
BBALL_HOOP_X_MIN = 0.08
BBALL_HOOP_X_MAX = 0.92
# Game hoops are usually in the upper half of a fixed youth/high-school camera.
# Keep this loose: some gyms place the far rim around 35-42% of frame height.
BBALL_HOOP_Y_MAX = 0.48
# Hoop position smoothing: rolling window of recent detections
BBALL_HOOP_SMOOTH = 10   # frames to average over
BBALL_HOOP_LOCK_MIN_SAMPLES = 4
BBALL_HOOP_LOCK_X_TOL = 0.12
BBALL_HOOP_LOCK_Y_TOL = 0.16
KEYPOINT_CONF = 0.50            # minimum confidence to use a keypoint

# Player size/position filters — CourtMask is now the primary spatial gate.
# These thresholds are intentionally loose: they only block single-pixel noise
# and very low-quality blobs. The court mask (foot_on_court) rejects bench/crowd
# geometrically regardless of pixel size.
PLAYER_MIN_HEIGHT_PX = 35       # was 55 — wide-shot players can be 40–50px tall
PLAYER_MIN_AREA_PX   = 800      # was 1800
PLAYER_MAX_Y_RATIO   = 0.88     # reject detections above this fraction of frame height

# Ball selection is intentionally context-aware. Basketballs are tiny in wide
# footage, so a low-confidence model candidate can still be right, but only if
# it fits player proximity, expected ball scale, and the recent ball path.
BALL_TRACK_HISTORY = 8
BALL_STRONG_CONF = 0.35
BALL_CONTEXT_CONF = 0.08
BALL_MIN_ACCEPT_SCORE = 0.20
BALL_FLOOR_Y_RATIO = 0.78
BALL_APPEARANCE_LEARN_CONF = 0.50
BALL_APPEARANCE_MATCH_BONUS = 0.18
BALL_APPEARANCE_SEARCH_CONF = 0.19
BALL_TRACK_MAX_MISSES = 5
BALL_APPEARANCE_MAX_FRAME_GAP = 15

# Active-rim calibration. Raw rim models see every basket in the gym; production
# logic should expose only the two stable game rims after a short warm-up.
RIM_CLUSTER_RADIUS_RATIO = 0.045
RIM_ACTIVE_MIN_OBSERVATIONS = 3
RIM_ACTIVE_MATCH_RADIUS_RATIO = 0.075


# ---------------------------------------------------------------------------
# Output data models
# ---------------------------------------------------------------------------

@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float

    @property
    def visible(self) -> bool:
        return self.confidence >= KEYPOINT_CONF


@dataclass
class PlayerDetection:
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2  (pixels)
    confidence: float
    keypoints: list[Keypoint]                 # 17 COCO keypoints
    in_interest_zone: bool = False
    # Filled by tracker.py after ByteTrack assignment
    track_id: Optional[int] = None
    jersey_number: Optional[str] = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot_center(self) -> tuple[float, float]:
        """Bottom-centre of the bounding box — more accurate floor position."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    def kp(self, idx: int) -> Keypoint:
        return self.keypoints[idx] if idx < len(self.keypoints) else Keypoint(0, 0, 0)

    # ------------------------------------------------------------------
    # Pose helpers used by EventValidator / action recognition
    # ------------------------------------------------------------------

    def hands_up(self) -> bool:
        """Both wrists above both shoulders — defender contesting a shot."""
        lw = self.kp(KP_LEFT_WRIST)
        rw = self.kp(KP_RIGHT_WRIST)
        ls = self.kp(KP_LEFT_SHOULDER)
        rs = self.kp(KP_RIGHT_SHOULDER)
        if not (lw.visible and rw.visible and ls.visible and rs.visible):
            return False
        # In image coords y increases downward, so "above" means smaller y
        return lw.y < ls.y and rw.y < rs.y

    def shooting_pose(self) -> bool:
        """
        Dominant wrist above the nose (or shoulder as fallback) AND elbow bent.
        Using the shoulder as a fallback reference improves detection for small
        player boxes in wide-angle cameras where nose keypoints are unreliable.
        """
        nose = self.kp(KP_NOSE)
        ls   = self.kp(KP_LEFT_SHOULDER)
        rs   = self.kp(KP_RIGHT_SHOULDER)
        lw   = self.kp(KP_LEFT_WRIST)
        rw   = self.kp(KP_RIGHT_WRIST)
        le   = self.kp(KP_LEFT_ELBOW)
        re   = self.kp(KP_RIGHT_ELBOW)

        for wrist, elbow, shoulder in [(lw, le, ls), (rw, re, rs)]:
            if not wrist.visible:
                continue
            # Use nose as reference; fall back to shoulder if nose not detected
            ref = nose if nose.visible else (shoulder if shoulder.visible else None)
            if ref is None:
                continue
            if wrist.y < ref.y:          # wrist above reference point
                if elbow.visible:
                    elbow_angle = _angle_deg(wrist, elbow)
                    if 50 <= elbow_angle <= 170:
                        return True
                else:
                    # No elbow keypoint but wrist is clearly raised — count it
                    if wrist.y < ref.y - 15:
                        return True
        return False

    def driving_direction(self) -> Optional[str]:
        """
        Estimate 'left' or 'right' drive from hip tilt.
        Returns None if keypoints not confident enough.
        """
        lh = self.kp(KP_LEFT_HIP)
        rh = self.kp(KP_RIGHT_HIP)
        if not (lh.visible and rh.visible):
            return None
        return "left" if lh.x < rh.x else "right"

    def wrist_position(self) -> Optional[tuple[float, float]]:
        """Return (x, y) of the higher wrist (useful for contesting distance)."""
        lw = self.kp(KP_LEFT_WRIST)
        rw = self.kp(KP_RIGHT_WRIST)
        candidates = [w for w in (lw, rw) if w.visible]
        if not candidates:
            return None
        # Higher wrist = smaller y in image coords
        best = min(candidates, key=lambda k: k.y)
        return best.x, best.y


@dataclass
class BallDetection:
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def radius_px(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return ((x2 - x1) + (y2 - y1)) / 4


@dataclass
class HoopDetection:
    bbox: tuple[float, float, float, float]
    confidence: float
    side: str = "unknown"   # "left" | "right" | "unknown"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class FrameDetections:
    frame_idx: int
    timestamp_ms: int
    frame_width: int
    frame_height: int
    players: list[PlayerDetection] = field(default_factory=list)
    ball: Optional[BallDetection] = None
    hoops: list[HoopDetection] = field(default_factory=list)
    raw_ball_candidates: int = 0
    raw_hoop_candidates: int = 0
    ball_source: str = ""
    ball_score: float = 0.0
    active_rims_calibrated: bool = False

    def players_in_zone(self, zone: str) -> list[PlayerDetection]:
        if zone == "interest":
            return [p for p in self.players if p.in_interest_zone]
        return self.players

    def nearest_player_to_ball(self) -> Optional[PlayerDetection]:
        if self.ball is None or not self.players:
            return None
        bx, by = self.ball.center
        return min(
            self.players,
            key=lambda p: math.dist(p.center, (bx, by)),
        )

    def defender_proximity_to_ball(self, defenders: list[PlayerDetection]) -> float:
        """Return distance in pixels from nearest defender wrist to ball centre."""
        if self.ball is None or not defenders:
            return float("inf")
        bx, by = self.ball.center
        min_dist = float("inf")
        for d in defenders:
            wrist = d.wrist_position()
            if wrist:
                min_dist = min(min_dist, math.dist(wrist, (bx, by)))
        return min_dist


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """
    Runs YOLOv11 detection + YOLO-Pose on individual frames or full videos.

    Args:
        det_model:  path to YOLOv11 weights, e.g. "models/yolo11n.pt"
                    or a Roboflow-trained basketball model
        pose_model: path to YOLO-Pose weights, e.g. "models/yolo11n-pose.pt"
        device:     "cpu" | "cuda" | "mps"
        hoop_a_x_ratio: fractional x position of hoop A in the frame
        hoop_b_x_ratio: fractional x position of hoop B in the frame
    """

    def __init__(
        self,
        det_model: str = "models/yolo11n.pt",
        pose_model: str = "models/yolo11n-pose.pt",
        device: str = "cpu",
        hoop_a_x_ratio: float = DEFAULT_HOOP_A_X_RATIO,
        hoop_b_x_ratio: float = DEFAULT_HOOP_B_X_RATIO,
        bball_model: str = BBALL_MODEL_PATH,
    ) -> None:
        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        self._det = YOLO(det_model)
        self._pose = YOLO(pose_model)
        self._device = device
        self._hoop_a_x = hoop_a_x_ratio
        self._hoop_b_x = hoop_b_x_ratio
        # Basketball-specific model for ball + hoop detection
        try:
            self._bball = YOLO(bball_model) if os.path.exists(bball_model) else None
        except Exception:
            self._bball = None
        self._bball_ball_classes: set[int] = set()
        self._bball_hoop_classes: set[int] = set()
        self._bball_other_rim_classes: set[int] = set()
        if self._bball is not None:
            self._init_bball_class_map()
        # Rolling hoop position history for smoothing (x_ratio values)
        from collections import deque
        self._hoop_a_history: deque = deque(maxlen=BBALL_HOOP_SMOOTH)
        self._hoop_b_history: deque = deque(maxlen=BBALL_HOOP_SMOOTH)
        self._hoop_a_y_history: deque = deque(maxlen=BBALL_HOOP_SMOOTH)
        self._hoop_b_y_history: deque = deque(maxlen=BBALL_HOOP_SMOOTH)
        # Current-frame hoop positions (None = not detected this frame)
        self._hoop_a_this_frame: Optional[float] = None
        self._hoop_b_this_frame: Optional[float] = None
        self._hoop_a_y: float = 0.42
        self._hoop_b_y: float = 0.42
        self._hoop_a_y_this_frame: Optional[float] = None
        self._hoop_b_y_this_frame: Optional[float] = None
        self._ball_history: deque = deque(maxlen=BALL_TRACK_HISTORY)
        self._ball_velocity: tuple[float, float] = (0.0, 0.0)
        self._ball_misses = 0
        self._last_ball_frame_idx: Optional[int] = None
        self._ball_hsv_mean: Optional[tuple[float, float, float]] = None
        self._ball_hsv_std: Optional[tuple[float, float, float]] = None
        self._ball_learned_radius = 0.0
        self._rim_clusters: list[dict[str, Any]] = []
        self._active_rim_pair: list[dict[str, Any]] = []
        # Fallback ball detector: HSV colour + blob analysis
        try:
            from src.vision.ball_detector_cv import BallDetectorCV
            self._ball_cv: Optional[BallDetectorCV] = BallDetectorCV()
        except Exception:
            self._ball_cv = None
        # Court mask — set via set_court_mask(); used to filter off-court players
        self._court_mask = None


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_court_mask(self, court_mask) -> None:
        """Attach a CourtMask; applied in _parse_detections/_parse_poses to reject off-court players."""
        self._court_mask = court_mask

    def update_court_mask(self, frame) -> None:
        """Forward frame to the attached CourtMask for periodic boundary re-detection."""
        if self._court_mask is not None:
            self._court_mask.update(frame)

    @property
    def hoop_a_x(self) -> float:
        """Best hoop-A x_ratio: current-frame detection if available, else rolling avg."""
        return self._hoop_a_this_frame if self._hoop_a_this_frame is not None else self._hoop_a_x

    @property
    def hoop_b_x(self) -> float:
        """Best hoop-B x_ratio: current-frame detection if available, else rolling avg."""
        return self._hoop_b_this_frame if self._hoop_b_this_frame is not None else self._hoop_b_x

    def _init_bball_class_map(self) -> None:
        """Discover ball/hoop class IDs from the YOLO model."""
        names = getattr(self._bball, "names", {}) or {}
        for cls_id, raw_name in names.items():
            name = str(raw_name).strip().lower().replace("_", " ")
            is_rim = "hoop" in name or "rim" in name or "basket" in name
            if is_rim and any(token in name for token in ("other", "side", "practice", "background")):
                self._bball_other_rim_classes.add(int(cls_id))
            elif is_rim:
                self._bball_hoop_classes.add(int(cls_id))
            elif "ball" in name:
                self._bball_ball_classes.add(int(cls_id))

        # Backwards-compatible fallback for the current basketball_best.pt.
        if not self._bball_ball_classes:
            self._bball_ball_classes.add(0)
        if not self._bball_hoop_classes and len(names) > 1:
            self._bball_hoop_classes.add(1)

    def process_frame(
        self, frame: Any, frame_idx: int, timestamp_ms: int
    ) -> FrameDetections:
        """Detect all objects and keypoints in one frame."""
        h, w = frame.shape[:2]
        # Reset per-frame hoop detections
        self._hoop_a_this_frame = None
        self._hoop_b_this_frame = None
        self._hoop_a_y_this_frame = None
        self._hoop_b_y_this_frame = None
        zones = self._interest_zones(w, h)

        # Split frame: interest zones at full res, centre at half res
        det_frame = self._apply_interest_mask(frame)

        det_result = self._det(
            det_frame, device=self._device, verbose=False,
            conf=min(BALL_CONF, PLAYER_CONF),
            classes=[COCO_PERSON, COCO_SPORTS_BALL, COCO_TV, COCO_CLOCK],
        )[0]

        pose_result = self._pose(
            det_frame, device=self._device, verbose=False, conf=PLAYER_CONF
        )[0]

        detections = FrameDetections(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            frame_width=w,
            frame_height=h,
        )

        # Collect scoreboard exclusion zones (tv + clock detections)
        exclusion_bboxes: list[tuple[float, float, float, float]] = []
        if det_result.boxes is not None:
            for box in det_result.boxes:
                if int(box.cls[0]) in (COCO_TV, COCO_CLOCK):
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    # Expand slightly so nearby blobs are also excluded
                    pad = (x2 - x1) * 0.3
                    exclusion_bboxes.append((x1 - pad, y1 - pad, x2 + pad, y2 + pad))

        self._parse_detections(det_result, detections, zones)
        self._parse_poses(pose_result, detections, zones)

        # Basketball-specific model: ball (primary) + hoop positions
        if self._bball is not None:
            self._apply_bball_model(frame, detections, w, h)

        self._label_hoops(detections, w)

        # If both YOLO and basketball model missed the ball, try HSV fallback
        # but exclude regions the COCO model identified as scoreboard/clock
        if detections.ball is None and self._ball_cv is not None:
            player_bboxes = [p.bbox for p in detections.players]
            # Merge player boxes + exclusion zones so HSV ignores both
            reject_zones = player_bboxes + exclusion_bboxes
            bbox = self._ball_cv.detect(frame, reject_zones)
            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                # Final check: reject if centre falls inside any exclusion zone
                bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                if not any(
                    ex1 <= bcx <= ex2 and ey1 <= bcy <= ey2
                    for ex1, ey1, ex2, ey2 in exclusion_bboxes
                ) and self._accept_ball_candidate(
                    bbox, 0.18, detections.players, w, h, source="hsv"
                ):
                    detections.ball = BallDetection(bbox=bbox, confidence=0.45)
                    detections.ball_source = "hsv"

        if detections.ball is not None:
            self._remember_ball(
                detections.ball,
                frame,
                detections.frame_idx,
                detections.ball_source,
                detections.ball_score,
            )
            self._ball_misses = 0
        else:
            self._ball_misses += 1

        return detections

    def _predicted_ball_pos(self) -> Optional[tuple[float, float, float]]:
        """
        Predict next ball position using last known position + velocity.
        Returns (pred_x, pred_y, search_radius) or None if no history.
        """
        if not self._ball_history:
            return None
        last_x, last_y, _ = self._ball_history[-1]
        vx, vy = self._ball_velocity
        speed  = (vx ** 2 + vy ** 2) ** 0.5
        radius = max(100.0, min(300.0, speed * 2.5 + 80))
        return (last_x + vx, last_y + vy, radius)

    def _apply_bball_model(
        self, frame: Any, detections: FrameDetections, fw: int, fh: int
    ) -> None:
        """
        Run basketball-specific YOLO model on the full frame (players/rims need full frame).
        Ball candidates are then filtered by predicted position — no extra inference pass.
        """
        result = self._bball(frame, verbose=False, conf=BBALL_BALL_LOW_CONF)[0]  # type: ignore[misc]
        if result.boxes is None:
            return

        hoop_candidates: list[tuple[float, float, tuple[float, float, float, float]]] = []
        ball_candidates: list[tuple[float, tuple[float, float, float, float], str]] = []

        pred = self._predicted_ball_pos()   # (pred_x, pred_y, radius) or None

        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            cx = (x1 + x2) / 2
            x_ratio = cx / fw

            if cls in self._bball_ball_classes and conf >= BBALL_BALL_LOW_CONF:
                ball_candidates.append((conf, (x1, y1, x2, y2), "yolo"))

            elif cls in self._bball_hoop_classes and conf >= BBALL_HOOP_CONF:
                cy = (y1 + y2) / 2
                y_ratio = cy / fh
                if (BBALL_HOOP_X_MIN <= x_ratio <= BBALL_HOOP_X_MAX
                        and y_ratio <= BBALL_HOOP_Y_MAX):
                    hoop_candidates.append((x_ratio, conf, (x1, y1, x2, y2)))
            elif cls in self._bball_other_rim_classes:
                continue

        self._process_detection_candidates(frame, detections, fw, fh, ball_candidates, hoop_candidates)

    def _process_detection_candidates(
        self,
        frame: Any,
        detections: FrameDetections,
        fw: int,
        fh: int,
        ball_candidates: list[tuple[float, tuple[float, float, float, float], str]],
        hoop_candidates: list[tuple[float, float, tuple[float, float, float, float]]],
    ) -> None:
        """Shared logic: update hoop tracking + select ball from candidates."""
        detections.raw_ball_candidates = len(ball_candidates)
        detections.raw_hoop_candidates = len(hoop_candidates)

        if hoop_candidates:
            self._update_rim_clusters(hoop_candidates, fw, frame.shape[0])
            detections.active_rims_calibrated = self._active_rims_calibrated()
            top = self._active_rim_candidates(hoop_candidates, fw, frame.shape[0])
            if not top:
                top = self._select_game_hoop_candidates(hoop_candidates, frame.shape[0])
            top.sort(key=lambda t: t[0])

            if len(top) >= 2:
                left_x, _, left_bbox = top[0]
                right_x, _, right_bbox = top[1]
                if right_x - left_x >= 0.25:
                    self._update_hoop("left", left_x, _bbox_y_ratio(left_bbox, frame.shape[0]))
                    self._update_hoop("right", right_x, _bbox_y_ratio(right_bbox, frame.shape[0]))
                else:
                    best = top[0] if top[0][1] >= top[1][1] else top[1]
                    self._update_single_hoop(best[0], _bbox_y_ratio(best[2], frame.shape[0]))
            else:
                self._update_single_hoop(top[0][0], _bbox_y_ratio(top[0][2], frame.shape[0]))

            for x_ratio, conf, bbox in top:
                side = "left" if x_ratio < 0.5 else "right"
                x1, y1, x2, y2 = bbox
                detections.hoops.append(HoopDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    side=side,
                ))

        appearance_candidate = self._appearance_ball_candidate(
            frame, detections.players, fw, fh, detections.frame_idx
        )
        if appearance_candidate is not None:
            ball_candidates.append((BALL_APPEARANCE_SEARCH_CONF, appearance_candidate, "appearance"))

        ball = self._select_ball_candidate(
            ball_candidates, detections.players, detections.hoops, fw, fh
        )
        if ball is not None:
            bbox, conf, score, source = ball
            if detections.ball is None or conf >= detections.ball.confidence:
                detections.ball = BallDetection(bbox=bbox, confidence=conf)
                detections.ball_score = score
                detections.ball_source = source

    def _select_game_hoop_candidates(
        self,
        candidates: list[tuple[float, float, tuple[float, float, float, float]]],
        frame_h: int,
    ) -> list[tuple[float, float, tuple[float, float, float, float]]]:
        """
        Pick the active game hoops from Roboflow rim candidates.

        Once both game hoops have been seen a few times, this becomes a lock:
        future candidates must stay near the learned game-hoop positions. That
        makes the model effectively blind to side baskets after calibration.
        """
        locked = (
            len(self._hoop_a_history) >= BBALL_HOOP_LOCK_MIN_SAMPLES
            and len(self._hoop_b_history) >= BBALL_HOOP_LOCK_MIN_SAMPLES
        )
        if locked:
            locked_candidates = []
            for candidate in candidates:
                x_ratio, _, bbox = candidate
                y_ratio = _bbox_y_ratio(bbox, frame_h)
                left_match = (
                    abs(x_ratio - self._hoop_a_x) <= BBALL_HOOP_LOCK_X_TOL
                    and abs(y_ratio - self._hoop_a_y) <= BBALL_HOOP_LOCK_Y_TOL
                )
                right_match = (
                    abs(x_ratio - self._hoop_b_x) <= BBALL_HOOP_LOCK_X_TOL
                    and abs(y_ratio - self._hoop_b_y) <= BBALL_HOOP_LOCK_Y_TOL
                )
                if left_match or right_match:
                    locked_candidates.append(candidate)
            if locked_candidates:
                candidates = locked_candidates

        return _select_game_hoop_candidates(candidates, frame_h)

    def _update_rim_clusters(
        self,
        candidates: list[tuple[float, float, tuple[float, float, float, float]]],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Accumulate stable rim locations so side/practice baskets can be silenced."""
        for x_ratio, conf, bbox in candidates:
            y_ratio = _bbox_y_ratio(bbox, frame_h)
            best = None
            best_dist = float("inf")
            for cluster in self._rim_clusters:
                dist = math.hypot(x_ratio - cluster["x"], y_ratio - cluster["y"])
                if dist < best_dist:
                    best = cluster
                    best_dist = dist

            if best is None or best_dist > RIM_CLUSTER_RADIUS_RATIO:
                self._rim_clusters.append({
                    "x": x_ratio,
                    "y": y_ratio,
                    "count": 1,
                    "conf_sum": conf,
                })
            else:
                count = int(best["count"]) + 1
                best["x"] = float(best["x"]) + (x_ratio - float(best["x"])) / count
                best["y"] = float(best["y"]) + (y_ratio - float(best["y"])) / count
                best["count"] = count
                best["conf_sum"] = float(best["conf_sum"]) + conf

        self._active_rim_pair = self._select_active_rim_clusters(frame_w, frame_h)

    def _active_rims_calibrated(self) -> bool:
        return len(self._active_rim_pair) == 2

    def _select_active_rim_clusters(self, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
        mature = [
            c for c in self._rim_clusters
            if int(c["count"]) >= RIM_ACTIVE_MIN_OBSERVATIONS
            and BBALL_HOOP_X_MIN <= float(c["x"]) <= BBALL_HOOP_X_MAX
        ]
        if len(mature) < 2:
            return []

        best_pair: list[dict[str, Any]] = []
        best_score = -1.0
        for i, a in enumerate(mature):
            for b in mature[i + 1:]:
                left, right = (a, b) if float(a["x"]) <= float(b["x"]) else (b, a)
                span = float(right["x"]) - float(left["x"])
                if span < 0.25:
                    continue
                y_alignment = max(0.0, 1.0 - abs(float(left["y"]) - float(right["y"])) * 5.0)
                count_score = min(int(left["count"]), int(right["count"])) * 0.45
                conf_score = (
                    float(left["conf_sum"]) / max(1, int(left["count"]))
                    + float(right["conf_sum"]) / max(1, int(right["count"]))
                )
                upper_score = max(0.0, 1.0 - abs((float(left["y"]) + float(right["y"])) / 2 - 0.34) * 2.4)
                score = span * 2.0 + y_alignment * 0.7 + count_score + conf_score + upper_score * 0.35
                if score > best_score:
                    best_score = score
                    best_pair = [left, right]
        return best_pair

    def _active_rim_candidates(
        self,
        candidates: list[tuple[float, float, tuple[float, float, float, float]]],
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[float, float, tuple[float, float, float, float]]]:
        """Return only current-frame rim boxes matching calibrated active rims."""
        if not self._active_rims_calibrated():
            return []

        selected: list[tuple[float, float, tuple[float, float, float, float]]] = []
        used: set[int] = set()
        for cluster in self._active_rim_pair:
            best_idx: Optional[int] = None
            best_score = -1.0
            for idx, candidate in enumerate(candidates):
                if idx in used:
                    continue
                x_ratio, conf, bbox = candidate
                y_ratio = _bbox_y_ratio(bbox, frame_h)
                dist = math.hypot(x_ratio - float(cluster["x"]), y_ratio - float(cluster["y"]))
                if dist > RIM_ACTIVE_MATCH_RADIUS_RATIO:
                    continue
                score = conf - dist * 2.0
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                used.add(best_idx)
                selected.append(candidates[best_idx])
        return selected

    def _update_single_hoop(self, x_ratio: float, y_ratio: float) -> None:
        """Update whichever hoop history matches x_ratio; record as current-frame detection."""
        if x_ratio < 0.5:
            self._update_hoop("left", x_ratio, y_ratio)
        else:
            self._update_hoop("right", x_ratio, y_ratio)

    def _update_hoop(self, side: str, x_ratio: float, y_ratio: float) -> None:
        if side == "left":
            self._hoop_a_this_frame = x_ratio
            self._hoop_a_y_this_frame = y_ratio
            self._hoop_a_history.append(x_ratio)
            self._hoop_a_y_history.append(y_ratio)
            self._hoop_a_x = sum(self._hoop_a_history) / len(self._hoop_a_history)
            self._hoop_a_y = sum(self._hoop_a_y_history) / len(self._hoop_a_y_history)
        else:
            self._hoop_b_this_frame = x_ratio
            self._hoop_b_y_this_frame = y_ratio
            self._hoop_b_history.append(x_ratio)
            self._hoop_b_y_history.append(y_ratio)
            self._hoop_b_x = sum(self._hoop_b_history) / len(self._hoop_b_history)
            self._hoop_b_y = sum(self._hoop_b_y_history) / len(self._hoop_b_y_history)

    def _select_ball_candidate(
        self,
        candidates: list[tuple[float, tuple[float, float, float, float], str]],
        players: list[PlayerDetection],
        hoops: list[HoopDetection],
        frame_w: int,
        frame_h: int,
    ) -> Optional[tuple[tuple[float, float, float, float], float, float, str]]:
        """Choose the most plausible Roboflow ball candidate for this frame."""
        best: Optional[tuple[tuple[float, float, float, float], float, float, str]] = None
        airborne: Optional[tuple[tuple[float, float, float, float], float, float, str]] = None
        top_player_y = min((p.bbox[1] for p in players), default=None)

        for conf, bbox, source in candidates:
            score = self._ball_candidate_score(
                bbox, conf, players, frame_w, frame_h, hoops, source=source
            )
            if top_player_y is not None:
                _, y1, _, y2 = bbox
                cy = (y1 + y2) / 2
                # In shots and jump balls, the true ball can be well above
                # every player and far from hands in a single sampled frame.
                # Check this before temporal gating so one earlier bad frame
                # cannot lock the tracker onto orange clutter.
                if conf >= 0.12 and cy < top_player_y - 30:
                    if airborne is None or score > airborne[2]:
                        airborne = (bbox, conf, score, source)

            if conf >= BALL_STRONG_CONF or score >= BALL_MIN_ACCEPT_SCORE:
                if best is None or score > best[2]:
                    best = (bbox, conf, score, source)

        if airborne is not None:
            return airborne
        if best is None:
            return None
        return best

    def _accept_ball_candidate(
        self,
        bbox: tuple[float, float, float, float],
        conf: float,
        players: list[PlayerDetection],
        frame_w: int,
        frame_h: int,
        source: str,
    ) -> bool:
        """Validate non-model fallback balls with stricter basketball context."""
        score = self._ball_candidate_score(
            bbox, conf, players, frame_w, frame_h, include_frame_prior=False
        )
        if source == "hsv":
            return score >= BALL_MIN_ACCEPT_SCORE + 0.12
        return score >= BALL_MIN_ACCEPT_SCORE

    def _ball_candidate_score(
        self,
        bbox: tuple[float, float, float, float],
        conf: float,
        players: list[PlayerDetection],
        frame_w: int,
        frame_h: int,
        hoops: Optional[list[HoopDetection]] = None,
        include_frame_prior: bool = True,
        source: str = "yolo",
    ) -> float:
        """
        Score a candidate using model confidence plus general basketball priors.

        These checks are camera-agnostic: they avoid fixed gym coordinates and
        instead compare the candidate to visible players, frame scale, floor
        position, and recent ball trajectory.
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        diameter = (bw + bh) / 2
        score = conf

        # Broad court/action prior: raw Roboflow can emit orange-ish edge
        # clutter. Keep normal in-frame candidates cheap, but punish boxes
        # touching walls/score overlays/sidelines unless other context rescues
        # them with strong model confidence.
        x_ratio = cx / frame_w if frame_w else 0.0
        y_ratio = cy / frame_h if frame_h else 0.0
        if include_frame_prior:
            if 0.03 <= x_ratio <= 0.97 and 0.05 <= y_ratio <= 0.94:
                score += 0.08
            else:
                score -= 0.25

        player_heights = [p.bbox[3] - p.bbox[1] for p in players if p.bbox[3] > p.bbox[1]]
        median_player_h = _median(player_heights) if player_heights else 0.0

        near_player = False
        near_feet = False
        inside_player_body = False
        if players:
            min_body_dist = min(_point_box_distance((cx, cy), p.bbox) for p in players)
            min_foot_dist = min(pixel_distance((cx, cy), p.foot_center) for p in players)
            context_radius = max(45.0, median_player_h * 0.65 if median_player_h else frame_w * 0.08)
            near_player = min_body_dist <= context_radius
            near_feet = min_foot_dist <= max(35.0, context_radius * 0.55)
            score += 0.14 if near_player else -0.07
            inside_player_body = any(_point_in_upper_body((cx, cy), p.bbox) for p in players)
            if inside_player_body and not near_feet:
                score -= 0.28

        if median_player_h:
            ratio = diameter / median_player_h
            if 0.025 <= ratio <= 0.22:
                score += 0.06
            elif ratio > 0.32:
                score -= 0.15
            else:
                score -= 0.06

        # Shiny-floor false positives are usually low in frame and detached
        # from players. A real dribble can be low too, so keep it when near feet.
        if cy > frame_h * BALL_FLOOR_Y_RATIO and not near_feet:
            score -= 0.16

        # Airborne balls during shots/tip-offs are often far from hands in a
        # single frame. Avoid letting "near player" dominate when another
        # plausible candidate is above the player cluster.
        if players and not inside_player_body:
            top_y = min(p.bbox[1] for p in players)
            if cy < top_y + max(40.0, median_player_h * 0.30 if median_player_h else 0.0):
                score += 0.11

        if hoops:
            real_hoops = [h for h in hoops if h.confidence >= BBALL_HOOP_CONF]
            if real_hoops:
                rim_dist = min(pixel_distance((cx, cy), h.center) for h in real_hoops)
                if rim_dist <= frame_w * 0.22:
                    score += 0.18

        if source == "appearance":
            score += BALL_APPEARANCE_MATCH_BONUS

        # Spatial continuity: penalise candidates that jumped too far from last known ball.
        # We scale the allowed jump by the ball's recent speed so fast passes are allowed
        # but a "static" ball cannot suddenly teleport to a different orange object.
        if self._ball_history:
            last_x, last_y, last_r = self._ball_history[-1]
            jump = pixel_distance((cx, cy), (last_x, last_y))
            # Estimate recent speed from velocity vector
            vx, vy = self._ball_velocity
            recent_speed = (vx ** 2 + vy ** 2) ** 0.5
            # Allow at most 1.8× recent speed, minimum 60px, maximum 220px
            expected_jump = max(60.0, min(220.0, recent_speed * 1.8))
            if jump <= expected_jump:
                score += 0.12
            elif jump <= expected_jump * 2:
                # Moderate overshoot — small penalty
                score -= 0.15 if conf < BBALL_BALL_CONF else 0.0
            else:
                # Large teleport — hard reject unless model is very confident
                score -= 0.45 if conf < 0.65 else 0.10
            if last_r > 0:
                size_ratio = max(diameter, last_r) / max(1.0, min(diameter, last_r))
                if size_ratio > 4.0 and conf < BBALL_BALL_CONF:
                    score -= 0.08

        return score

    def _remember_ball(
        self,
        ball: BallDetection,
        frame: Any | None = None,
        frame_idx: Optional[int] = None,
        source: str = "",
        score: float = 0.0,
    ) -> None:
        x1, y1, x2, y2 = ball.bbox
        cx, cy = ball.center
        diameter = ((x2 - x1) + (y2 - y1)) / 2
        if self._ball_history:
            last_x, last_y, _ = self._ball_history[-1]
            self._ball_velocity = (cx - last_x, cy - last_y)
        self._ball_history.append((cx, cy, diameter))
        self._last_ball_frame_idx = frame_idx

        trusted_model = source in ("yolo", "coco", "") and ball.confidence >= BALL_APPEARANCE_LEARN_CONF
        if frame is not None and trusted_model:
            self._learn_ball_appearance(frame, ball.bbox)

    def _learn_ball_appearance(
        self,
        frame: Any,
        bbox: tuple[float, float, float, float],
    ) -> None:
        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
        except ImportError:
            return

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(_pad_bbox(bbox, 0.25), w, h)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        center = (mask.shape[1] // 2, mask.shape[0] // 2)
        axes = (max(2, mask.shape[1] // 3), max(2, mask.shape[0] // 3))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        pixels = hsv[mask > 0]
        if len(pixels) < 8:
            return

        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)
        self._ball_hsv_mean = (float(mean[0]), float(mean[1]), float(mean[2]))
        self._ball_hsv_std = (float(std[0]), float(std[1]), float(std[2]))
        self._ball_learned_radius = max(3.0, ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 4)

    def _appearance_ball_candidate(
        self,
        frame: Any,
        players: list[PlayerDetection],
        frame_w: int,
        frame_h: int,
        frame_idx: int,
    ) -> Optional[tuple[float, float, float, float]]:
        if self._ball_hsv_mean is None or self._ball_hsv_std is None:
            return None
        if self._ball_misses > BALL_TRACK_MAX_MISSES:
            return None
        if (
            self._last_ball_frame_idx is not None
            and frame_idx - self._last_ball_frame_idx > BALL_APPEARANCE_MAX_FRAME_GAP
        ):
            return None

        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
        except ImportError:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean = np.array(self._ball_hsv_mean, dtype=np.float32)
        std = np.array(self._ball_hsv_std, dtype=np.float32)
        slack = np.array([
            max(8.0, std[0] * 2.0),
            max(35.0, std[1] * 2.2),
            max(35.0, std[2] * 2.2),
        ], dtype=np.float32)
        lo = np.clip(mean - slack, [0, 20, 20], [179, 255, 255]).astype(np.uint8)
        hi = np.clip(mean + slack, [0, 20, 20], [179, 255, 255]).astype(np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        predicted = self._predicted_ball_center()
        min_area = math.pi * max(3.0, self._ball_learned_radius * 0.45) ** 2
        max_area = math.pi * max(8.0, self._ball_learned_radius * 2.2) ** 2
        best_bbox = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim <= 0:
                continue
            circ = 4.0 * math.pi * area / (perim * perim)
            if circ < 0.45:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            aspect = min(bw, bh) / max(bw, bh)
            if aspect < 0.45:
                continue
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            if cx < frame_w * 0.02 or cx > frame_w * 0.98 or cy < frame_h * 0.03 or cy > frame_h * 0.94:
                continue
            if any(_point_in_upper_body((cx, cy), p.bbox) for p in players):
                continue

            score = circ * 0.35 + aspect * 0.25
            if predicted is not None:
                dist = pixel_distance((cx, cy), predicted)
                score += max(0.0, 1.0 - dist / max(frame_w * 0.22, 1.0)) * 0.45
            if players:
                min_player_dist = min(_point_box_distance((cx, cy), p.bbox) for p in players)
                if min_player_dist <= frame_w * 0.12:
                    score += 0.18

            if score > best_score:
                best_score = score
                best_bbox = (float(bx), float(by), float(bx + bw), float(by + bh))

        return best_bbox

    def _predicted_ball_center(self) -> Optional[tuple[float, float]]:
        if not self._ball_history:
            return None
        last_x, last_y, _ = self._ball_history[-1]
        vx, vy = self._ball_velocity
        decay = max(0.0, 1.0 - self._ball_misses * 0.18)
        return last_x + vx * decay, last_y + vy * decay

    def process_video(
        self,
        video_path: str,
        fps_target: int = 7,
        start_ms: int = 0,
        end_ms: Optional[int] = None,
    ) -> list[FrameDetections]:
        """
        Sample a video at fps_target and return FrameDetections for each
        sampled frame. Uses OpenCV for reading.
        """
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("opencv-python is not installed. Run: pip install opencv-python")

        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, round(native_fps / fps_target))

        results: list[FrameDetections] = []
        frame_idx = 0

        if start_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if end_ms is not None and ts_ms > end_ms:
                break

            if frame_idx % frame_step == 0:
                results.append(self.process_frame(frame, frame_idx, ts_ms))

            frame_idx += 1

        cap.release()
        return results

    # ------------------------------------------------------------------
    # Interest zone helpers
    # ------------------------------------------------------------------

    def _interest_zones(
        self, frame_w: int, frame_h: int
    ) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) interest zone rects in pixels."""
        r = int(frame_w * INTEREST_ZONE_RADIUS_RATIO)
        zones = []
        for x_ratio in (self._hoop_a_x, self._hoop_b_x):
            cx = int(frame_w * x_ratio)
            cy = frame_h // 2
            zones.append((
                max(0, cx - r), max(0, cy - r),
                min(frame_w, cx + r), min(frame_h, cy + r),
            ))
        return zones

    @staticmethod
    def _apply_interest_mask(frame: Any) -> Any:
        """
        Return the frame unmodified — interest zone logic is applied at
        coordinate level after inference. Full-frame inference is simplest
        for correctness; zone-level subsampling can be added for performance.
        """
        return frame

    @staticmethod
    def _in_any_zone(
        cx: float, cy: float, zones: list[tuple[int, int, int, int]]
    ) -> bool:
        return any(z[0] <= cx <= z[2] and z[1] <= cy <= z[3] for z in zones)

    # ------------------------------------------------------------------
    # Parse YOLO results
    # ------------------------------------------------------------------

    def _parse_detections(
        self,
        result,
        detections: FrameDetections,
        zones: list[tuple[int, int, int, int]],
    ) -> None:
        if result.boxes is None:
            return

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            in_zone = self._in_any_zone(cx, cy, zones)

            if cls == COCO_SPORTS_BALL and conf >= BALL_CONF:
                if detections.ball is None or conf > detections.ball.confidence:
                    detections.ball = BallDetection(
                        bbox=(x1, y1, x2, y2), confidence=conf
                    )
                    detections.ball_source = "coco"

            elif cls == COCO_PERSON and conf >= PLAYER_CONF:
                bbox_h = y2 - y1
                bbox_w = x2 - x1
                frame_h = detections.frame_height
                if bbox_h < PLAYER_MIN_HEIGHT_PX:
                    continue
                if bbox_h * bbox_w < PLAYER_MIN_AREA_PX:
                    continue
                if frame_h > 0 and y1 > frame_h * PLAYER_MAX_Y_RATIO:
                    continue
                # Court mask: reject players whose feet land outside the court lines.
                # 15px margin tolerates players momentarily stepping out-of-bounds.
                if (self._court_mask is not None and self._court_mask.valid
                        and not self._court_mask.foot_on_court(cx, y2 - 15)):
                    continue
                detections.players.append(PlayerDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    keypoints=[],
                    in_interest_zone=in_zone,
                ))

    def _parse_poses(
        self,
        result,
        detections: FrameDetections,
        zones: list[tuple[int, int, int, int]],
    ) -> None:
        if result.keypoints is None or result.boxes is None:
            return

        for box, kps in zip(result.boxes, result.keypoints.data):
            conf = float(box.conf[0])
            if conf < PLAYER_CONF:
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            frame_h = detections.frame_height
            if bbox_h < PLAYER_MIN_HEIGHT_PX:
                continue
            if bbox_h * bbox_w < PLAYER_MIN_AREA_PX:
                continue
            if frame_h > 0 and y1 > frame_h * PLAYER_MAX_Y_RATIO:
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if (self._court_mask is not None and self._court_mask.valid
                    and not self._court_mask.foot_on_court(cx, y2 - 15)):
                continue
            in_zone = self._in_any_zone(cx, cy, zones)

            keypoints = [
                Keypoint(x=float(kps[k][0]), y=float(kps[k][1]), confidence=float(kps[k][2]))
                for k in range(min(17, len(kps)))
            ]

            # Match pose result to an existing player detection by bbox overlap
            # If no match, create a new player entry
            matched = self._match_player(detections.players, (x1, y1, x2, y2))
            if matched is not None:
                matched.keypoints = keypoints
                matched.in_interest_zone = in_zone
            else:
                detections.players.append(PlayerDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    keypoints=keypoints,
                    in_interest_zone=in_zone,
                ))

    def _label_hoops(self, detections: FrameDetections, frame_w: int) -> None:
        """
        Ensure downstream code always has one left and one right hoop anchor.
        Real Roboflow hoop detections are preserved; synthetic anchors fill only
        missing sides and are marked with low confidence.
        """
        h = detections.frame_height
        present = {hoop.side for hoop in detections.hoops}
        for side, x_ratio, y_ratio in [
            ("left", self._hoop_a_x, self._hoop_a_y),
            ("right", self._hoop_b_x, self._hoop_b_y),
        ]:
            if side in present:
                continue
            cx = int(frame_w * x_ratio)
            cy = int(h * y_ratio)
            r = int(frame_w * 0.015)
            detections.hoops.append(HoopDetection(
                bbox=(cx - r, cy - r, cx + r, cy + r),
                confidence=0.20,
                side=side,
            ))

    @staticmethod
    def _match_player(
        players: list[PlayerDetection],
        bbox: tuple[float, float, float, float],
    ) -> Optional[PlayerDetection]:
        """Return the player whose bbox overlaps most with the given bbox."""
        best_iou = 0.3   # minimum IoU threshold to count as a match
        best = None
        for p in players:
            iou = _iou(p.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best = p
        return best


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _angle_deg(a: Keypoint, b: Keypoint) -> float:
    """Angle at point b in the triangle origin → a → b (rough elbow angle)."""
    dx, dy = a.x - b.x, a.y - b.y
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def pixel_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _point_box_distance(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> float:
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def _point_in_upper_body(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox
    body_y2 = y1 + (y2 - y1) * 0.72
    return x1 <= px <= x2 and y1 <= py <= body_y2


def _bbox_y_ratio(bbox: tuple[float, float, float, float], frame_h: int) -> float:
    _, y1, _, y2 = bbox
    return ((y1 + y2) / 2) / frame_h if frame_h > 0 else 0.42


def _pad_bbox(
    bbox: tuple[float, float, float, float],
    frac: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * frac
    pad_y = (y2 - y1) * frac
    return x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y


def _clip_bbox(
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, min(float(frame_w), x1)),
        max(0.0, min(float(frame_h), y1)),
        max(0.0, min(float(frame_w), x2)),
        max(0.0, min(float(frame_h), y2)),
    )


def _select_game_hoop_candidates(
    candidates: list[tuple[float, float, tuple[float, float, float, float]]],
    frame_h: int,
) -> list[tuple[float, float, tuple[float, float, float, float]]]:
    """
    Pick the likely game hoops from Roboflow candidates.

    Youth/high-school gyms often have side baskets.  The game hoops are the
    pair with the widest horizontal court span; if only one side is visible,
    use the strongest candidate after basic geometric filtering.
    """
    if len(candidates) <= 2:
        return sorted(candidates, key=lambda t: -t[1])[:2]

    best_pair = None
    best_score = -1.0
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            left, right = (a, b) if a[0] <= b[0] else (b, a)
            span = right[0] - left[0]
            if span < 0.25:
                continue
            left_y = _bbox_y_ratio(left[2], frame_h)
            right_y = _bbox_y_ratio(right[2], frame_h)
            y_alignment = max(0.0, 1.0 - abs(left_y - right_y) * 4.0)
            # Wider span dominates; confidence and similar rim height break ties.
            # Side-wall baskets often have good confidence but do not form a
            # stable left/right pair at the same court depth.
            score = span * 2.0 + left[1] + right[1] + y_alignment * 0.35
            if score > best_score:
                best_score = score
                best_pair = [left, right]

    if best_pair is not None:
        return best_pair
    return sorted(candidates, key=lambda t: -t[1])[:2]


def is_contested(
    shooter: PlayerDetection,
    defenders: list[PlayerDetection],
    ball: Optional[BallDetection],
    threshold_px: float = 80.0,
) -> bool:
    """
    A shot is contested if any defender's wrist is within threshold_px pixels
    of the ball centre at the moment of release.
    """
    if ball is None:
        return any(
            pixel_distance(shooter.center, d.center) < threshold_px * 2
            for d in defenders
        )
    bx, by = ball.center
    for d in defenders:
        wrist = d.wrist_position()
        if wrist and pixel_distance(wrist, (bx, by)) < threshold_px:
            return True
    return False
