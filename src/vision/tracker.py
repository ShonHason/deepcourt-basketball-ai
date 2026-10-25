"""
Tracker — persistent player identity via ByteTrack + jersey OCR via EasyOCR.

Consumes FrameDetections from detector.py and mutates each PlayerDetection's
track_id and jersey_number fields in place.

Processing model:
  • ByteTrack assigns stable track_ids across frames — O(N) overhead per frame.
  • Jersey OCR runs every `ocr_every_n` frames per track_id, then caches the
    result so the same player isn't re-read on every frame.
  • Once a jersey is read with confidence >= OCR_CACHE_CONFIDENCE, OCR stops
    for that track_id for the rest of the clip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from src.vision.detector import FrameDetections, PlayerDetection, _iou

# ---------------------------------------------------------------------------
# Lazy heavy imports — tests can run without any of these installed
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore[import-untyped]
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

try:
    import supervision as sv  # type: ignore[import-untyped]
    _BYTETRACK_AVAILABLE = True
except ImportError:
    sv = None  # type: ignore[assignment]
    _BYTETRACK_AVAILABLE = False

try:
    import easyocr  # type: ignore[import-untyped]
    _EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None  # type: ignore[assignment]
    _EASYOCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Torso crop fractions relative to the player bounding box
JERSEY_CROP_Y_START = 0.20   # skip head; start at 20% of bbox height
JERSEY_CROP_Y_END   = 0.55   # end at 55% (above shorts)
JERSEY_CROP_X_PAD   = 0.10   # trim 10% from each side to reduce arm noise

# Only 0-99 jersey numbers are valid in basketball
_JERSEY_RE = re.compile(r"^\d{1,2}$")

OCR_CONF_THRESHOLD   = 0.65  # minimum EasyOCR confidence to store a reading
OCR_CACHE_CONFIDENCE = 0.80  # at or above this, stop re-reading that track

# Minimum IoU to match a ByteTrack output back to a PlayerDetection
TRACKER_IOU_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class _JerseyRead:
    number: str
    confidence: float
    read_count: int = 1


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Tracker:
    """
    Assigns persistent track_id and jersey_number to PlayerDetection objects.

    Typical usage::

        tracker = Tracker(frame_rate=30)
        for frame_idx, (frame, detections) in enumerate(pipeline):
            tracker.update(detections, frame)
            # detections.players now have .track_id and .jersey_number set

    Args:
        frame_rate:          native video FPS — used to size the ByteTrack buffer
        ocr_every_n:         run jersey OCR every N frames per track_id
        ocr_conf_threshold:  minimum EasyOCR confidence to accept a reading
        use_gpu:             enable GPU for EasyOCR (requires CUDA)
    """

    def __init__(
        self,
        frame_rate: int = 30,
        ocr_every_n: int = 30,
        ocr_conf_threshold: float = OCR_CONF_THRESHOLD,
        use_gpu: bool = False,
        enable_ocr: bool = False,   # disabled by default — models must be pre-downloaded
    ) -> None:
        self._frame_rate = frame_rate
        self._ocr_every_n = ocr_every_n
        self._ocr_conf = ocr_conf_threshold
        self._use_gpu = use_gpu
        self._enable_ocr = enable_ocr
        self._frame_count: int = 0

        self._jersey_cache: dict[int, _JerseyRead] = {}
        self._jersey_votes: dict[int, dict[str, list[float]]] = {}  # track_id → {number: [conf,...]}
        self._ocr_countdown: dict[int, int] = {}   # track_id → frames until next OCR
        self._hue_samples: dict[int, list[float]] = {}  # track_id → [dominant hue, ...]
        # Per-frame snapshots of {track_id: x_norm} for co-occurrence team classification
        self._frame_snapshots: list[dict[int, float]] = []

        self._bt: Any = None          # centroid tracker table (lazy-init)
        self._next_track_id: int = 1  # monotonically increasing
        self._ocr: Any = None         # easyocr.Reader (lazy-init)

        # Appearance Re-ID: color histograms per active and recently-lost track
        self._appearances: dict[int, Any] = {}           # tid → histogram (np.ndarray)
        self._lost_appearances: dict[int, list] = {}     # tid → [histogram, frames_lost]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: FrameDetections, frame: Any = None) -> FrameDetections:
        """
        Mutate PlayerDetection.track_id and PlayerDetection.jersey_number in
        place for every player in detections. Returns detections unchanged.

        Args:
            detections: FrameDetections produced by Detector.process_frame()
            frame:      raw cv2 frame (numpy HxWx3 BGR) for jersey OCR cropping;
                        pass None to skip OCR this frame
        """
        if not detections.players:
            return detections

        self._frame_count += 1
        self._ensure_tracker()

        # Step 1 — assign persistent track IDs (centroid + appearance Re-ID)
        self._assign_track_ids(detections, frame)

        # Step 2 — sample jersey hue for team classification (cheap, every frame)
        if frame is not None and _NUMPY_AVAILABLE:
            self._sample_hues(detections, frame)

        # Step 3 — jersey OCR for players whose countdown has elapsed
        if self._enable_ocr and frame is not None and _EASYOCR_AVAILABLE and _NUMPY_AVAILABLE:
            self._ensure_ocr()
            for player in detections.players:
                if player.track_id is None:
                    continue
                tid = player.track_id
                self._ocr_countdown[tid] = self._ocr_countdown.get(tid, 0) - 1
                if self._ocr_countdown[tid] <= 0:
                    self._read_jersey(player, frame)
                    self._ocr_countdown[tid] = self._ocr_every_n

        # Step 4 — populate jersey_number from cache
        for player in detections.players:
            if player.track_id is not None:
                cached = self._jersey_cache.get(player.track_id)
                if cached is not None:
                    player.jersey_number = cached.number

        return detections

    @property
    def hue_samples(self) -> dict[int, list[float]]:
        """Return a copy of per-track hue samples for team classification."""
        return {tid: list(v) for tid, v in self._hue_samples.items()}

    @property
    def frame_snapshots(self) -> list[dict[int, float]]:
        """Per-frame {track_id: x_norm} snapshots for co-occurrence team classification."""
        return list(self._frame_snapshots)

    def reset(self) -> None:
        """Clear all tracking state — call between clips or quarters."""
        self._bt = None
        self._next_track_id = 1
        self._jersey_cache.clear()
        self._jersey_votes.clear()
        self._ocr_countdown.clear()
        self._hue_samples.clear()
        self._frame_snapshots.clear()
        self._appearances.clear()
        self._lost_appearances.clear()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Jersey hue sampling  (for team classification)
    # ------------------------------------------------------------------

    def _sample_hues(self, detections: FrameDetections, frame: Any) -> None:
        """Sample dominant jersey hue and x-position for each tracked player."""
        fw = detections.frame_width
        snapshot: dict[int, float] = {}

        for player in detections.players:
            if player.track_id is None:
                continue
            hue = self._extract_hue(player, frame)
            if hue is not None:
                buf = self._hue_samples.setdefault(player.track_id, [])
                buf.append(hue)
                if len(buf) > 40:
                    buf.pop(0)
            # Record normalized x-center for court-position team clustering
            if fw > 0:
                cx = (player.bbox[0] + player.bbox[2]) / 2.0
                snapshot[player.track_id] = cx / fw

        # Store snapshot only when enough players visible (skip sparse frames)
        if len(snapshot) >= 6:
            self._frame_snapshots.append(snapshot)
            if len(self._frame_snapshots) > 2000:   # cap memory
                self._frame_snapshots.pop(0)

    @staticmethod
    def _extract_hue(player: Any, frame: Any) -> Optional[float]:
        """
        Return the dominant hue of the player's jersey torso crop.
        Vivid pixels only (S > 40, V > 40) to skip white/black/grey areas.
        Returns None when the crop is too small or too grey (ref/coach uniform).
        """
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            return None

        crop = crop_jersey(frame, player.bbox)
        if crop is None or crop.size == 0:
            return None

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_px = hsv.shape[0] * hsv.shape[1]
        if total_px == 0:
            return None

        # ── Truly colorless dark uniform (referee gray/black) ───────────────
        dark_colorless = (hsv[:, :, 2] < 70) & (hsv[:, :, 1] < 45)
        dark_colorless_ratio = float(np.sum(dark_colorless)) / total_px  # type: ignore[union-attr]
        if dark_colorless_ratio > 0.50:
            return 90.0   # dark/black jersey — synthetic hue far from white (2.0) for k-means

        # ── Stripe pattern detection (black+white referee shirt) ─────────────
        # A striped shirt has BOTH very dark colorless pixels AND very bright
        # colorless pixels in significant amounts. A solid white jersey has only
        # bright pixels; a solid dark jersey has only dark ones.
        very_bright = (hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 55)  # type: ignore[union-attr]
        very_dark   = (hsv[:, :, 2] < 80)  & (hsv[:, :, 1] < 55)  # type: ignore[union-attr]
        if (float(np.mean(very_bright)) > 0.13  # type: ignore[union-attr]
                and float(np.mean(very_dark)) > 0.13):  # type: ignore[union-attr]
            return None  # stripe pattern → referee/official

        # ── Generic color fingerprint ────────────────────────────────────
        # Rather than hard-coding specific colors, return a compact feature
        # vector encoded as a single float:
        #   • High-saturation pixels give a reliable hue signal.
        #   • Low-saturation bright pixels = white/light jersey → synthetic hue near 0.
        #   • Low-saturation dark pixels = dark jersey → synthetic hue near 90.
        # This is color-agnostic: works for any two team color combinations.

        vivid_mask = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
        vivid_count = int(np.sum(vivid_mask))  # type: ignore[union-attr]

        # Bright low-saturation = white/light team
        bright_pale_mask = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 140)
        bright_pale_count = int(np.sum(bright_pale_mask))  # type: ignore[union-attr]
        bright_pale_ratio = bright_pale_count / total_px

        if bright_pale_ratio > 0.20:
            # Predominantly white/light jersey → return near-0 synthetic hue
            return 2.0

        if vivid_count >= 6:
            # Return median hue of the saturated pixels (works for any saturated color)
            return float(np.median(hsv[:, :, 0][vivid_mask]))  # type: ignore[union-attr]

        # Moderate white/light (10-20%) — still classify as light team
        if bright_pale_ratio > 0.10:
            return 2.0

        return None

    # ------------------------------------------------------------------
    # Centroid-distance tracker (replaces ByteTrack for sparse sampling)
    # ------------------------------------------------------------------
    # ByteTrack uses IoU matching which fails at 3 fps because players move
    # too far between sampled frames (IoU → 0 for running players).
    # A simple nearest-centroid greedy match is far more robust here.

    _MATCH_DIST_PX   = 180   # max px distance to re-identify a player across frames
    _MAX_LOST_FRAMES = 20    # sampled frames before dropping a lost track (~6s at 3fps)
    _MAX_VELOCITY_STEPS = 3  # cap velocity extrapolation to avoid runaway predictions
    _CROSSING_ZONE_PX = 80   # radius within which appearance breaks centroid ties
    _CROSS_APP_BONUS  = 45   # max px subtracted from distance for a perfect appearance match

    def _ensure_tracker(self) -> None:
        """Lazy-init the centroid track table."""
        if self._bt is not None:
            return
        # _bt stores: {track_id: {"cx", "cy", "vx", "vy", "lost"}}
        self._bt = {}           # type: ignore[assignment]
        self._next_track_id = 1

    def _assign_track_ids(self, detections: FrameDetections, frame: Any = None) -> None:
        if not _NUMPY_AVAILABLE:
            self._fallback_ids(detections)
            return

        self._ensure_tracker()
        tracks: dict = self._bt  # type: ignore[assignment]

        # Age all existing tracks; move stale ones to lost_appearances
        to_delete = [tid for tid, t in tracks.items() if t["lost"] >= self._MAX_LOST_FRAMES]
        for tid in to_delete:
            if tid in self._appearances:
                self._lost_appearances[tid] = [self._appearances.pop(tid), 0]
            del tracks[tid]
        # Advance frames_lost counter; purge very old appearances
        for tid in list(self._lost_appearances.keys()):
            self._lost_appearances[tid][1] += 1
            if self._lost_appearances[tid][1] > 60:
                del self._lost_appearances[tid]
        for t in tracks.values():
            t["lost"] += 1

        # Compute detection centroids
        det_cx = [(p.bbox[0] + p.bbox[2]) / 2 for p in detections.players]
        det_cy = [(p.bbox[1] + p.bbox[3]) / 2 for p in detections.players]

        track_ids   = list(tracks.keys())
        unmatched_d = set(range(len(detections.players)))

        # --- Stage 1: velocity-predicted distance matching with appearance tiebreak ---
        # Use each track's recent velocity to extrapolate predicted position.
        # When two tracks are within _CROSSING_ZONE_PX of the same detection
        # (i.e. players crossing paths), appearance similarity breaks the tie
        # instead of pure distance — preventing ID swaps at crossing points.
        matched_pairs: list[tuple[float, float]] = []  # (actual dx, dy) of matched pairs

        if track_ids and detections.players:
            dist = np.full((len(track_ids), len(detections.players)), 1e9)
            for ti, tid in enumerate(track_ids):
                lost = int(tracks[tid]["lost"])
                steps = min(lost, self._MAX_VELOCITY_STEPS)
                pred_cx = tracks[tid]["cx"] + tracks[tid]["vx"] * steps
                pred_cy = tracks[tid]["cy"] + tracks[tid]["vy"] * steps
                for di in range(len(detections.players)):
                    dx = det_cx[di] - pred_cx
                    dy = det_cy[di] - pred_cy
                    dist[ti, di] = (dx * dx + dy * dy) ** 0.5

            # Appearance bonus: when tracks are within crossing zone, subtract
            # up to _CROSS_APP_BONUS px from effective distance for visually
            # matching pairs so the correct identity wins even if centroid is
            # momentarily closer to the wrong track.
            if frame is not None:
                det_apps = [
                    self._compute_appearance(frame, detections.players[di].bbox)
                    for di in range(len(detections.players))
                ]
                for ti, tid in enumerate(track_ids):
                    if tid not in self._appearances:
                        continue
                    track_app = self._appearances[tid]
                    for di in range(len(detections.players)):
                        if dist[ti, di] > self._CROSSING_ZONE_PX:
                            continue
                        if det_apps[di] is None:
                            continue
                        sim = float(np.dot(track_app, det_apps[di]))
                        dist[ti, di] = max(0.0, dist[ti, di] - sim * self._CROSS_APP_BONUS)

            assigned_t: set[int] = set()
            while True:
                min_val = float(np.min(dist))
                if min_val > self._MATCH_DIST_PX:
                    break
                ti, di = divmod(int(np.argmin(dist)), len(detections.players))
                tid = track_ids[ti]
                actual_dx = det_cx[di] - tracks[tid]["cx"]
                actual_dy = det_cy[di] - tracks[tid]["cy"]
                matched_pairs.append((actual_dx, actual_dy))
                detections.players[di].track_id = tid
                # EMA velocity update: blend new displacement with prior velocity
                alpha = 0.55
                tracks[tid]["vx"]   = actual_dx * alpha + tracks[tid]["vx"] * (1 - alpha)
                tracks[tid]["vy"]   = actual_dy * alpha + tracks[tid]["vy"] * (1 - alpha)
                tracks[tid]["cx"]   = det_cx[di]
                tracks[tid]["cy"]   = det_cy[di]
                tracks[tid]["lost"] = 0
                assigned_t.add(ti)
                unmatched_d.discard(di)
                dist[ti, :] = 1e9
                dist[:, di] = 1e9

        # --- Stage 2: camera-pan compensation for remaining unmatched ---
        # If ≥3 tracks matched, estimate global frame motion (median dx/dy).
        # Unmatched detections are shifted back by the global motion and
        # re-matched at 2× the normal threshold to recover lost tracks.
        if matched_pairs and unmatched_d and len(matched_pairs) >= 3:
            pan_dx = float(np.median([p[0] for p in matched_pairs]))
            pan_dy = float(np.median([p[1] for p in matched_pairs]))
            if abs(pan_dx) > 5 or abs(pan_dy) > 5:   # only compensate real pans
                still_unmatched: set[int] = set()
                remaining_tids = [tid for tid in track_ids
                                  if tracks.get(tid, {}).get("lost", 0) > 0]
                for di in list(unmatched_d):
                    cx_adj = det_cx[di] - pan_dx
                    cy_adj = det_cy[di] - pan_dy
                    best_tid, best_d = None, self._MATCH_DIST_PX * 2.0
                    for tid in remaining_tids:
                        if tid not in tracks:
                            continue
                        dx = cx_adj - tracks[tid]["cx"]
                        dy = cy_adj - tracks[tid]["cy"]
                        d  = (dx * dx + dy * dy) ** 0.5
                        if d < best_d:
                            best_d, best_tid = d, tid
                    if best_tid is not None:
                        actual_dx = det_cx[di] - tracks[best_tid]["cx"]
                        actual_dy = det_cy[di] - tracks[best_tid]["cy"]
                        detections.players[di].track_id = best_tid
                        alpha = 0.55
                        tracks[best_tid]["vx"]   = actual_dx * alpha + tracks[best_tid]["vx"] * (1 - alpha)
                        tracks[best_tid]["vy"]   = actual_dy * alpha + tracks[best_tid]["vy"] * (1 - alpha)
                        tracks[best_tid]["cx"]   = det_cx[di]
                        tracks[best_tid]["cy"]   = det_cy[di]
                        tracks[best_tid]["lost"] = 0
                        remaining_tids.remove(best_tid)
                    else:
                        still_unmatched.add(di)
                unmatched_d = still_unmatched

        # --- Stage 3: appearance-based re-identification ---
        # For detections still unmatched after centroid+pan stages, compare
        # jersey color histograms against recently-lost tracks. This recovers
        # players who disappeared during wide-shot blackouts and return with
        # a new centroid position outside the normal matching radius.
        if frame is not None and _NUMPY_AVAILABLE and unmatched_d:
            for di in list(unmatched_d):
                app = self._compute_appearance(frame, detections.players[di].bbox)
                if app is None:
                    continue
                best_tid, best_sim = None, 0.84  # cosine similarity threshold
                for lost_tid, (lost_app, frames_lost) in self._lost_appearances.items():
                    sim = float(np.dot(app, lost_app))
                    if sim > best_sim:
                        best_sim, best_tid = sim, lost_tid
                if best_tid is not None:
                    # Recover old track: inherit ID, jersey data, hue samples
                    tracks[best_tid] = {"cx": det_cx[di], "cy": det_cy[di],
                                        "vx": 0.0, "vy": 0.0, "lost": 0}
                    detections.players[di].track_id = best_tid
                    self._appearances[best_tid] = app
                    del self._lost_appearances[best_tid]
                    unmatched_d.discard(di)

        # Start new tracks for truly unmatched detections
        for di in unmatched_d:
            tid = self._next_track_id
            self._next_track_id += 1
            tracks[tid] = {"cx": det_cx[di], "cy": det_cy[di],
                           "vx": 0.0, "vy": 0.0, "lost": 0}
            detections.players[di].track_id = tid
            if frame is not None:
                app = self._compute_appearance(frame, detections.players[di].bbox)
                if app is not None:
                    self._appearances[tid] = app

    def _compute_appearance(self, frame: Any, bbox: tuple) -> Any:
        """
        Compute a 33-dim appearance vector for a player crop:
          [0:16]  — normalised 16-bin hue histogram (vivid/bright pixels)
          [16:24] — normalised 8-bin brightness (V-channel) histogram
          [24]    — ratio of very-dark colorless pixels  (black bands)
          [25]    — ratio of very-bright colorless pixels (white bands)
          [26]    — bimodality score: dark_ratio × bright_ratio × 4  (0→1)
                    High only for striped black/white shirts, low for solid jerseys.

        The brightness histogram + bimodality score let a retrieval system
        distinguish a black/white striped referee (bimodal brightness) from a
        solid white jersey (unimodal bright) even though both have hue ~2.
        Used for appearance Re-ID across detection gaps AND referee profiling.
        """
        if not _NUMPY_AVAILABLE:
            return None
        try:
            import cv2  # type: ignore[import-untyped]
            x1, y1, x2, y2 = (int(v) for v in bbox)
            h_box = y2 - y1
            crop = frame[y1:y1 + int(h_box * 0.70), x1:x2]
            if crop.size == 0:
                return None
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            total = hsv.shape[0] * hsv.shape[1]
            if total == 0:
                return None

            # Hue histogram (vivid or bright pixels)
            mask = ((hsv[:, :, 1] > 40) | (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
            hue_hist = cv2.calcHist([hsv], [0], mask, [16], [0, 180]).flatten()
            hue_norm = float(np.linalg.norm(hue_hist))
            hue_feat = (hue_hist / hue_norm) if hue_norm > 0 else hue_hist

            # Brightness histogram (all pixels)
            val_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            val_norm = float(np.linalg.norm(val_hist))
            val_feat = (val_hist / val_norm) if val_norm > 0 else val_hist

            # Dark/bright colorless ratios for stripe bimodality
            very_dark   = float(np.mean((hsv[:, :, 2] < 80)  & (hsv[:, :, 1] < 55)))
            very_bright = float(np.mean((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 55)))
            bimodality  = min(1.0, very_dark * very_bright * 4.0)

            vec = np.concatenate([
                hue_feat.astype(np.float32),
                val_feat.astype(np.float32),
                np.array([very_dark, very_bright, bimodality], dtype=np.float32),
            ])
            norm = float(np.linalg.norm(vec))
            return (vec / norm).astype(np.float32) if norm > 0 else None
        except Exception:
            return None

    @staticmethod
    def _fallback_ids(detections: FrameDetections) -> None:
        for i, p in enumerate(detections.players):
            if p.track_id is None:
                p.track_id = -(i + 1)

    # ------------------------------------------------------------------
    # EasyOCR jersey reading
    # ------------------------------------------------------------------

    def _ensure_ocr(self) -> None:
        if self._ocr is None and _EASYOCR_AVAILABLE:
            self._ocr = easyocr.Reader(['en'], gpu=self._use_gpu, verbose=False)

    def _read_jersey(self, player: PlayerDetection, frame: Any) -> None:
        if self._ocr is None or player.track_id is None:
            return

        tid = player.track_id

        # Skip once majority is locked: ≥3 reads and same number ≥70% of the time
        if self._jersey_majority_locked(tid):
            return

        crop = crop_jersey(frame, player.bbox)
        if crop is None or crop.size == 0:
            return

        try:
            import cv2  # type: ignore[import-untyped]
            h_c, w_c = crop.shape[:2]
            if h_c <= 0 or w_c <= 0:
                return
            # 3× upscale for legibility at low source resolution
            crop = cv2.resize(crop, (w_c * 3, h_c * 3), interpolation=cv2.INTER_CUBIC)
            # CLAHE contrast enhancement on luminance channel
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            l_ch = clahe.apply(l_ch)
            crop = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
            # Grayscale OCR is more reliable for jersey numbers than colour
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            results = self._ocr.readtext(gray, allowlist="0123456789", detail=1)
        except Exception:
            return

        best_number: Optional[str] = None
        best_conf = 0.0
        for _, text, conf in results:
            text = text.strip()
            if _JERSEY_RE.match(text) and conf > best_conf:
                best_conf = conf
                best_number = text

        if best_number is None or best_conf < self._ocr_conf:
            return

        # Add to vote bank
        track_votes = self._jersey_votes.setdefault(tid, {})
        track_votes.setdefault(best_number, []).append(best_conf)

        # Resolve majority winner and update cache
        winner, winner_conf = self._resolve_jersey(tid)
        if winner is not None:
            existing = self._jersey_cache.get(tid)
            self._jersey_cache[tid] = _JerseyRead(
                number=winner,
                confidence=winner_conf,
                read_count=(existing.read_count + 1) if existing else 1,
            )

    def _jersey_majority_locked(self, tid: int) -> bool:
        """True when the majority vote is stable enough to stop re-reading."""
        votes = self._jersey_votes.get(tid)
        if not votes:
            return False
        total = sum(len(v) for v in votes.values())
        if total < 3:
            return False
        best_count = max(len(v) for v in votes.values())
        return best_count >= 3 and best_count / total >= 0.70

    def _resolve_jersey(self, tid: int) -> tuple[Optional[str], float]:
        """Return (majority_number, avg_confidence) from the vote bank."""
        votes = self._jersey_votes.get(tid)
        if not votes:
            return None, 0.0
        # Most votes wins; tiebreak by average confidence
        best = max(votes, key=lambda n: (len(votes[n]), sum(votes[n]) / len(votes[n])))
        avg_conf = sum(votes[best]) / len(votes[best])
        return best, avg_conf


# ---------------------------------------------------------------------------
# Ball Kalman tracker
# ---------------------------------------------------------------------------

class BallTracker:
    """
    Kalman filter tracker for the basketball.

    Designed for sparse frame sampling (every N frames) and handles detection
    gaps caused by motion blur or occlusion by predicting forward from the
    last known state.

    State vector: [cx, cy, vx, vy]
    - Position advances by velocity * dt each step.
    - High process noise allows for fast direction changes (bounces, passes).
    - If the ball is not detected for more than _MAX_PREDICT_FRAMES consecutive
      sampled frames, the tracker resets on the next valid detection.

    Usage::

        ball_tracker = BallTracker(frame_skip=5)
        for frame_idx, detections in pipeline:
            ball = detections.ball  # may be None
            cx = (ball.bbox[0] + ball.bbox[2]) / 2 if ball else None
            cy = (ball.bbox[1] + ball.bbox[3]) / 2 if ball else None
            estimated = ball_tracker.update(cx, cy, frame_idx)
            # estimated is (cx, cy) or None
    """

    _MAX_PREDICT_FRAMES = 8   # sampled frames without detection before reset
    _MATCH_RADIUS_PX    = 150  # reject detections further than this from prediction

    def __init__(self, frame_skip: int = 5) -> None:
        self._frame_skip = frame_skip
        self._state: Any = None   # np.ndarray [cx, cy, vx, vy]
        self._P: Any = None       # 4×4 covariance matrix
        self._last_frame: int = -999
        self._frames_since_detection: int = 0

    def update(
        self,
        cx: Optional[float],
        cy: Optional[float],
        frame_idx: int,
    ) -> Optional[tuple[float, float]]:
        """
        Feed a detected ball centroid (or None) and return the best position
        estimate. Returns None when there is no reliable state yet.

        Args:
            cx, cy:    ball centroid from the detector, or None if not detected
            frame_idx: current sampled frame index (used to compute dt)
        """
        if not _NUMPY_AVAILABLE:
            return (cx, cy) if cx is not None else None

        # ── First detection ever → initialise ──────────────────────────────
        if self._state is None:
            if cx is None:
                return None
            self._state = np.array([cx, cy, 0.0, 0.0], dtype=float)
            self._P = np.diag([25.0, 25.0, 400.0, 400.0])
            self._last_frame = frame_idx
            self._frames_since_detection = 0
            return (cx, cy)

        frames_elapsed = max(1, frame_idx - self._last_frame)

        # ── Too long without detection → reset on next valid hit ───────────
        if self._frames_since_detection > self._MAX_PREDICT_FRAMES:
            if cx is not None:
                self._state = np.array([cx, cy, 0.0, 0.0], dtype=float)
                self._P = np.diag([25.0, 25.0, 400.0, 400.0])
                self._last_frame = frame_idx
                self._frames_since_detection = 0
                return (cx, cy)
            return None

        dt = float(frames_elapsed)

        # ── Predict ─────────────────────────────────────────────────────────
        F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # Process noise scales with dt: fast ball direction changes unpredictably
        q_pos = 8.0 * dt
        q_vel = 35.0 * dt
        Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        x_pred = F @ self._state
        P_pred = F @ self._P @ F.T + Q

        # ── Update ──────────────────────────────────────────────────────────
        if cx is not None:
            dist = ((cx - x_pred[0]) ** 2 + (cy - x_pred[1]) ** 2) ** 0.5
            if dist <= self._MATCH_RADIUS_PX:
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
                R = np.diag([9.0, 9.0])
                z = np.array([cx, cy])
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                self._state = x_pred + K @ (z - H @ x_pred)
                self._P = (np.eye(4) - K @ H) @ P_pred
                self._frames_since_detection = 0
            else:
                # Ball jumped far (bounce/pass outside prediction cone) → reinit
                self._state = np.array([cx, cy, 0.0, 0.0], dtype=float)
                self._P = np.diag([25.0, 25.0, 400.0, 400.0])
                self._frames_since_detection = 0
            self._last_frame = frame_idx
        else:
            # No detection — use prediction only
            self._state = x_pred
            self._P = P_pred
            self._frames_since_detection += 1

        return (float(self._state[0]), float(self._state[1]))

    @property
    def predicted_position(self) -> Optional[tuple[float, float]]:
        """Current predicted position without advancing the filter state."""
        if self._state is None:
            return None
        return (float(self._state[0]), float(self._state[1]))

    def reset(self) -> None:
        """Clear tracker state — call between quarters or clips."""
        self._state = None
        self._P = None
        self._last_frame = -999
        self._frames_since_detection = 0


# ---------------------------------------------------------------------------
# Geometry helpers (module-level so tests can import them directly)
# ---------------------------------------------------------------------------

def crop_jersey(
    frame: Any, bbox: tuple[float, float, float, float]
) -> Any:
    """
    Return the torso crop of a player frame for jersey-number OCR.
    Returns None if the crop would be empty or numpy is unavailable.
    """
    if not _NUMPY_AVAILABLE:
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_box = y2 - y1
    w_box = x2 - x1

    if h_box <= 0 or w_box <= 0:
        return None

    cy1 = y1 + int(h_box * JERSEY_CROP_Y_START)
    cy2 = y1 + int(h_box * JERSEY_CROP_Y_END)
    cx_pad = int(w_box * JERSEY_CROP_X_PAD)
    cx1 = x1 + cx_pad
    cx2 = x2 - cx_pad

    fh, fw = frame.shape[:2]
    cy1 = max(0, cy1);  cy2 = min(fh, cy2)
    cx1 = max(0, cx1);  cx2 = min(fw, cx2)

    if cy2 <= cy1 or cx2 <= cx1:
        return None

    return frame[cy1:cy2, cx1:cx2]


def jersey_cache_snapshot(tracker: Tracker) -> dict[int, str]:
    """Return a {track_id: jersey_number} copy of the tracker's current cache."""
    return {tid: r.number for tid, r in tracker._jersey_cache.items()}
