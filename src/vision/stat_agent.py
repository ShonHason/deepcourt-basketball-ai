"""
stat_agent.py — Standalone basketball stat-counting agent.

Independently counts stats from raw video without relying on the main pipeline.
Uses YOLO for player/ball detection, a centroid-distance tracker for persistence,
HSV colour for team classification, and EasyOCR for jersey number reading.

Run:
    python3 src/vision/stat_agent.py

Stats counted per player:
    - shots_taken:        ball moves upward ≥30px while this player is nearest
    - possessions:        frames in which this player is nearest to the ball
    - turnovers:          possession-change events where this player loses the ball
"""

from __future__ import annotations

import math
import os
import sys
import sqlite3
import json
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_PATH = str(REPO_ROOT / "data" / "videoMockup" /
                 "Maccabi_Rishon_Neria_vs_Maccabi_Shaham_First_Half_15.12.2021.mp4")
YOLO_MODEL = str(REPO_ROOT / "models" / "yolo11n.pt")
DB_PATH    = str(REPO_ROOT / "data" / "courtlog.db")

# ---------------------------------------------------------------------------
# Sampling / tracking settings
# ---------------------------------------------------------------------------

FRAME_SAMPLE_EVERY  = 10       # process every 10th frame (~3fps for 30fps video)
MAX_SAMPLED_FRAMES  = 500      # stop after this many sampled frames
PROGRESS_EVERY      = 100      # print progress banner every N sampled frames

MATCH_DIST_PX       = 160      # centroid distance threshold to re-identify a player
MAX_LOST_FRAMES     = 20       # sampled frames before dropping a lost track (~6s)

SHOT_UPWARD_PX      = 30       # ball y-delta (upward) to count as a shot attempt
POSSESSION_CHANGE_CONSEC = 2   # consecutive frames of different nearest player = turnover

# HSV team classification (OpenCV hue 0-179)
BLUE_HUE_LOW, BLUE_HUE_HIGH   = 90, 140
BLUE_SAT_MIN, BLUE_VAL_MIN    = 60, 60

# Jersey crop fractions
JERSEY_Y_START = 0.20
JERSEY_Y_END   = 0.55
JERSEY_X_PAD   = 0.10

# ---------------------------------------------------------------------------
# Lazy heavy imports
# ---------------------------------------------------------------------------

try:
    import cv2
    import numpy as np
    _CV_OK = True
except ImportError:
    _CV_OK = False
    print("ERROR: opencv-python / numpy not installed")
    sys.exit(1)

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    print("ERROR: ultralytics not installed")
    sys.exit(1)

try:
    # Some macOS Python installs have SSL cert issues; bypass for model downloads
    import ssl as _ssl
    _ssl._create_default_https_context = _ssl._create_unverified_context  # noqa: SLF001
    import easyocr
    _OCR_OK = True
except ImportError:
    _OCR_OK = False
    print("WARNING: easyocr not installed — jersey OCR will be skipped")

# BallDetectorCV — fallback HSV orange-ball detector
sys.path.insert(0, str(REPO_ROOT))
try:
    from src.vision.ball_detector_cv import BallDetectorCV
    _CVBALL_OK = True
except Exception:
    _CVBALL_OK = False
    print("WARNING: BallDetectorCV import failed — falling back to YOLO only")


# ===========================================================================
# CentroidTracker — lightweight nearest-centroid player ID assignment
# ===========================================================================

class CentroidTracker:
    """Greedy centroid-distance player tracker, tuned for ~3fps sampled video."""

    def __init__(
        self,
        match_dist_px: float = MATCH_DIST_PX,
        max_lost_frames: int = MAX_LOST_FRAMES,
    ) -> None:
        self._match_dist   = match_dist_px
        self._max_lost     = max_lost_frames
        self._next_id      = 1
        # {track_id: {"cx", "cy", "lost"}}
        self._tracks: dict[int, dict] = {}

    def update(self, detections: list[tuple[float, float, float, float]]) -> list[int]:
        """
        Match detections (x1,y1,x2,y2) to existing tracks.
        Returns list of track_ids parallel to detections.
        """
        # Age all tracks; drop stale ones
        to_del = [tid for tid, t in self._tracks.items() if t["lost"] >= self._max_lost]
        for tid in to_del:
            del self._tracks[tid]
        for t in self._tracks.values():
            t["lost"] += 1

        det_cx = [(x1 + x2) / 2 for x1, y1, x2, y2 in detections]
        det_cy = [(y1 + y2) / 2 for x1, y1, x2, y2 in detections]

        n_det = len(detections)
        assigned = [None] * n_det          # type: list[Optional[int]]
        unmatched = set(range(n_det))

        track_ids = list(self._tracks.keys())

        if track_ids and n_det > 0:
            # Build distance matrix
            dist = np.full((len(track_ids), n_det), 1e9, dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                tx, ty = self._tracks[tid]["cx"], self._tracks[tid]["cy"]
                for di in range(n_det):
                    dx = det_cx[di] - tx
                    dy = det_cy[di] - ty
                    dist[ti, di] = math.sqrt(dx * dx + dy * dy)

            # Greedy match
            while True:
                min_val = float(np.min(dist))
                if min_val > self._match_dist:
                    break
                flat_idx = int(np.argmin(dist))
                ti, di = divmod(flat_idx, n_det)
                tid = track_ids[ti]
                assigned[di] = tid
                self._tracks[tid]["cx"]   = det_cx[di]
                self._tracks[tid]["cy"]   = det_cy[di]
                self._tracks[tid]["lost"] = 0
                unmatched.discard(di)
                dist[ti, :] = 1e9
                dist[:, di] = 1e9

        # New tracks for unmatched detections
        for di in unmatched:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"cx": det_cx[di], "cy": det_cy[di], "lost": 0}
            assigned[di] = tid

        return [int(t) for t in assigned]   # type: ignore[arg-type]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ===========================================================================
# Jersey OCR helper
# ===========================================================================

_ocr_reader = None  # lazy-loaded singleton

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None and _OCR_OK:
        print("  [OCR] Initialising EasyOCR reader (first use)…")
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader


def read_jersey_number(frame, bbox: tuple) -> Optional[str]:
    """
    Crop the torso region from bbox, upscale 3x, run digit-only OCR.
    Returns a 1-2 digit string or None.
    """
    if not _OCR_OK:
        return None
    reader = _get_ocr_reader()
    if reader is None:
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_box = y2 - y1
    w_box = x2 - x1
    if h_box <= 0 or w_box <= 0:
        return None

    cy1 = y1 + int(h_box * JERSEY_Y_START)
    cy2 = y1 + int(h_box * JERSEY_Y_END)
    cx_pad = int(w_box * JERSEY_X_PAD)
    cx1, cx2 = x1 + cx_pad, x2 - cx_pad

    fh, fw = frame.shape[:2]
    cy1, cy2 = max(0, cy1), min(fh, cy2)
    cx1, cx2 = max(0, cx1), min(fw, cx2)

    if cy2 <= cy1 or cx2 <= cx1:
        return None

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    # Upscale 3× for better OCR accuracy on low-res sources
    h_c, w_c = crop.shape[:2]
    crop_up = cv2.resize(crop, (w_c * 3, h_c * 3), interpolation=cv2.INTER_CUBIC)

    try:
        results = reader.readtext(crop_up, allowlist="0123456789", detail=1)
    except Exception:
        return None

    import re
    jersey_re = re.compile(r"^\d{1,2}$")
    best_num, best_conf = None, 0.0
    for _, text, conf in results:
        text = text.strip()
        if jersey_re.match(text) and conf > best_conf:
            best_conf = conf
            best_num = text

    return best_num if best_conf >= 0.50 else None


# ===========================================================================
# Team classification by jersey colour
# ===========================================================================

def classify_team(frame, bbox: tuple) -> str:
    """
    Return 'team_a' (blue jerseys) or 'team_b' (white/red/other).
    Uses HSV colour analysis on the jersey torso crop.
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h_box = y2 - y1
    w_box = x2 - x1
    if h_box <= 0 or w_box <= 0:
        return "team_b"

    cy1 = y1 + int(h_box * JERSEY_Y_START)
    cy2 = y1 + int(h_box * JERSEY_Y_END)
    cx_pad = int(w_box * JERSEY_X_PAD)
    cx1, cx2 = x1 + cx_pad, x2 - cx_pad

    fh, fw = frame.shape[:2]
    cy1, cy2 = max(0, cy1), min(fh, cy2)
    cx1, cx2 = max(0, cx1), min(fw, cx2)

    if cy2 <= cy1 or cx2 <= cx1:
        return "team_b"

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return "team_b"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_px = hsv.shape[0] * hsv.shape[1]
    if total_px == 0:
        return "team_b"

    blue_mask = (
        (hsv[:, :, 0] >= BLUE_HUE_LOW) & (hsv[:, :, 0] <= BLUE_HUE_HIGH) &
        (hsv[:, :, 1] > BLUE_SAT_MIN) &
        (hsv[:, :, 2] > BLUE_VAL_MIN)
    )
    blue_ratio = float(np.sum(blue_mask)) / total_px

    return "team_a" if blue_ratio > 0.10 else "team_b"


# ===========================================================================
# StatAgent — main processing loop
# ===========================================================================

class StatAgent:
    """
    Standalone stat-counting agent.  Processes a video and returns a box score.
    """

    def __init__(self) -> None:
        print(f"[StatAgent] Loading YOLO model: {YOLO_MODEL}")
        self._yolo = YOLO(YOLO_MODEL)
        self._tracker = CentroidTracker()
        self._ball_cv = BallDetectorCV() if _CVBALL_OK else None

        # Per-player state: {track_id: {...}}
        self._players: dict[int, dict] = {}

        # Possession tracking
        self._prev_nearest: Optional[int]  = None
        self._poss_change_buf: list[int]   = []   # last 2 nearest track IDs

        # Ball position history for shot detection
        self._prev_ball_cy: Optional[float] = None

        # OCR cache: {track_id: jersey_number}
        self._jersey_cache: dict[int, str] = {}
        # Team cache: {track_id: team}
        self._team_cache: dict[int, list] = {}   # accumulate votes, then majority

        # Team colour vote accumulation per track
        self._team_votes: dict[int, dict[str, int]] = {}

    def _get_player(self, tid: int) -> dict:
        if tid not in self._players:
            self._players[tid] = {
                "player_id":    tid,
                "jersey_number": None,
                "team":         None,
                "shots_taken":  0,
                "possessions":  0,
                "turnovers":    0,
            }
        return self._players[tid]

    def _detect_frame(self, frame):
        """Run YOLO detection, return (player_bboxes, ball_center_or_None)."""
        result = self._yolo(
            frame, verbose=False, conf=0.35,
            classes=[0, 32]   # 0=person, 32=sports ball
        )[0]

        player_bboxes: list[tuple] = []
        ball_bbox: Optional[tuple] = None
        best_ball_conf = 0.0

        if result.boxes is not None:
            for box in result.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])

                if cls == 0 and conf >= 0.50:          # person
                    player_bboxes.append((x1, y1, x2, y2))
                elif cls == 32 and conf >= 0.35:        # sports ball
                    if conf > best_ball_conf:
                        best_ball_conf = conf
                        ball_bbox = (x1, y1, x2, y2)

        # Fallback: HSV orange ball detector
        if ball_bbox is None and self._ball_cv is not None:
            ball_bbox = self._ball_cv.detect(frame, player_bboxes)

        ball_center = None
        if ball_bbox is not None:
            bx1, by1, bx2, by2 = ball_bbox
            ball_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)

        return player_bboxes, ball_center

    def _nearest_player(
        self,
        ball_center: tuple[float, float],
        player_bboxes: list[tuple],
        track_ids: list[int],
    ) -> Optional[int]:
        """Return track_id of the player closest to the ball."""
        if not player_bboxes:
            return None
        bx, by = ball_center
        best_dist = float("inf")
        best_tid  = None
        for bbox, tid in zip(player_bboxes, track_ids):
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            d = math.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
            if d < best_dist:
                best_dist = d
                best_tid  = tid
        return best_tid

    def _update_team_vote(self, tid: int, frame, bbox: tuple) -> None:
        team = classify_team(frame, bbox)
        votes = self._team_votes.setdefault(tid, {"team_a": 0, "team_b": 0})
        votes[team] += 1

    def _resolve_teams(self) -> None:
        """Assign final team to each player based on majority vote."""
        for tid, votes in self._team_votes.items():
            majority = "team_a" if votes["team_a"] >= votes["team_b"] else "team_b"
            self._get_player(tid)["team"] = majority

    def _try_ocr(self, tid: int, frame, bbox: tuple) -> None:
        """Run OCR and cache the jersey number (once per player)."""
        if tid in self._jersey_cache:
            return
        num = read_jersey_number(frame, bbox)
        if num is not None:
            self._jersey_cache[tid] = num
            self._get_player(tid)["jersey_number"] = num

    def process(self) -> list[dict]:
        """
        Main loop: open video, sample every FRAME_SAMPLE_EVERY frames,
        track players, count stats.  Returns list of player box score dicts.
        """
        print(f"[StatAgent] Opening video: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("ERROR: cannot open video")
            return []

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(f"[StatAgent] Native FPS={native_fps:.1f}, total frames≈{total_frames}")
        print(f"[StatAgent] Sampling every {FRAME_SAMPLE_EVERY} frames, "
              f"max {MAX_SAMPLED_FRAMES} sampled frames")

        frame_idx    = 0
        sampled      = 0
        ocr_interval = 30   # try OCR every 30 sampled frames per player

        # Per-player OCR countdown: {tid: frames_until_next_ocr}
        ocr_countdown: dict[int, int] = {}

        while sampled < MAX_SAMPLED_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SAMPLE_EVERY != 0:
                frame_idx += 1
                continue

            # ---- Process this sampled frame --------------------------------

            if sampled % PROGRESS_EVERY == 0:
                pct = 100 * frame_idx / max(total_frames, 1)
                print(f"  [frame {frame_idx:6d}  sampled {sampled:4d}  {pct:5.1f}%]  "
                      f"tracked players: {len(self._players)}")

            player_bboxes, ball_center = self._detect_frame(frame)
            track_ids = self._tracker.update(player_bboxes)

            # ---- Team votes & OCR for each detected player -----------------
            for bbox, tid in zip(player_bboxes, track_ids):
                self._update_team_vote(tid, frame, bbox)

                # OCR: try periodically, stop once cached
                if tid not in self._jersey_cache:
                    countdown = ocr_countdown.get(tid, 0) - 1
                    if countdown <= 0:
                        self._try_ocr(tid, frame, bbox)
                        ocr_countdown[tid] = ocr_interval
                    else:
                        ocr_countdown[tid] = countdown

            # ---- Shot detection (ball moves upward ≥ SHOT_UPWARD_PX) ------
            if ball_center is not None:
                ball_cy = ball_center[1]
                if self._prev_ball_cy is not None:
                    delta_y = self._prev_ball_cy - ball_cy   # positive = upward in image coords
                    if delta_y >= SHOT_UPWARD_PX and track_ids:
                        nearest = self._nearest_player(ball_center, player_bboxes, track_ids)
                        if nearest is not None:
                            self._get_player(nearest)["shots_taken"] += 1
                self._prev_ball_cy = ball_cy

            # ---- Possession counting & turnover detection ------------------
            if ball_center is not None and track_ids:
                nearest = self._nearest_player(ball_center, player_bboxes, track_ids)
                if nearest is not None:
                    self._get_player(nearest)["possessions"] += 1
                    self._poss_change_buf.append(nearest)
                    if len(self._poss_change_buf) > POSSESSION_CHANGE_CONSEC:
                        self._poss_change_buf.pop(0)

                    # Turnover: last POSSESSION_CHANGE_CONSEC frames all have the
                    # SAME new player AND it differs from the frame before that
                    if len(self._poss_change_buf) == POSSESSION_CHANGE_CONSEC:
                        buf = self._poss_change_buf
                        if all(t == buf[-1] for t in buf) and buf[0] != self._prev_nearest:
                            losing_player = self._prev_nearest
                            if losing_player is not None:
                                self._get_player(losing_player)["turnovers"] += 1

                    self._prev_nearest = nearest

            sampled   += 1
            frame_idx += 1

        cap.release()

        # Finalise teams from accumulated votes
        self._resolve_teams()
        for tid, num in self._jersey_cache.items():
            self._get_player(tid)["jersey_number"] = num

        print(f"\n[StatAgent] Done. Processed {sampled} sampled frames.")
        return list(self._players.values())


# ===========================================================================
# DB: pull last pipeline result
# ===========================================================================

def load_pipeline_result() -> Optional[dict]:
    """Return the result_json of the most recent completed pipeline job, or None."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            "SELECT result_json FROM jobs WHERE result_json IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return json.loads(row[0])
    except Exception as exc:
        print(f"  [DB] Warning: {exc}")
    return None


# ===========================================================================
# Comparison report
# ===========================================================================

def print_comparison(agent_results: list[dict], pipeline_result: Optional[dict]) -> None:
    print("\n" + "=" * 70)
    print("  STAT AGENT vs PIPELINE — COMPARISON REPORT")
    print("=" * 70)

    # ---- Agent summary ----
    n_players    = len(agent_results)
    total_shots  = sum(p["shots_taken"] for p in agent_results)
    total_turns  = sum(p["turnovers"]   for p in agent_results)
    teams_found  = sorted({p["team"] for p in agent_results if p["team"]})
    jerseys      = sorted({p["jersey_number"] for p in agent_results if p["jersey_number"]})

    print("\n--- STAT AGENT ---")
    print(f"  Players detected : {n_players}")
    print(f"  Total shots      : {total_shots}")
    print(f"  Total turnovers  : {total_turns}")
    print(f"  Teams identified : {', '.join(teams_found) if teams_found else '(none)'}")
    print(f"  Jersey numbers   : {', '.join(jerseys) if jerseys else '(OCR found none)'}")

    print("\n  Per-player breakdown:")
    print(f"  {'ID':>4}  {'Jersey':>6}  {'Team':>8}  {'Shots':>6}  {'Possess':>8}  {'TOs':>4}")
    print("  " + "-" * 48)
    for p in sorted(agent_results, key=lambda x: -x["shots_taken"]):
        jersey = p["jersey_number"] or "?"
        team   = (p["team"] or "?")
        print(f"  {p['player_id']:>4}  {jersey:>6}  {team:>8}  "
              f"{p['shots_taken']:>6}  {p['possessions']:>8}  {p['turnovers']:>4}")

    # ---- Pipeline summary ----
    print("\n--- PIPELINE (last DB result) ---")
    if pipeline_result is None:
        print("  No completed pipeline jobs found in DB (jobs table is empty).")
        print("  The pipeline has never successfully written a result_json to the DB.")
        pipe_players  = 0
        pipe_shots    = 0
        pipe_turnovers = 0
        pipe_teams    = []
    else:
        # Normalise various possible result_json shapes
        players_list = []
        if isinstance(pipeline_result, list):
            players_list = pipeline_result
        elif isinstance(pipeline_result, dict):
            players_list = (
                pipeline_result.get("players") or
                pipeline_result.get("box_score") or
                pipeline_result.get("stats") or
                []
            )

        pipe_players   = len(players_list)
        pipe_shots     = sum(
            p.get("shots_taken", 0) or p.get("fga", 0) for p in players_list
        )
        pipe_turnovers = sum(
            p.get("turnovers", 0) or p.get("tov", 0) for p in players_list
        )
        pipe_teams     = sorted({
            p.get("team", "") for p in players_list if p.get("team")
        })
        pipe_jerseys   = sorted({
            str(p.get("jersey_number") or p.get("player_name") or "")
            for p in players_list
            if p.get("jersey_number") or p.get("player_name")
        })
        unknown_count  = sum(
            1 for p in players_list
            if str(p.get("player_name", "")).lower().startswith("unknown") or
               str(p.get("jersey_number", "")).lower().startswith("unknown") or
               (not p.get("jersey_number") and not p.get("player_name"))
        )

        print(f"  Players detected : {pipe_players}")
        print(f"  Total shots      : {pipe_shots}")
        print(f"  Total turnovers  : {pipe_turnovers}")
        print(f"  Teams identified : {', '.join(pipe_teams) if pipe_teams else '(none)'}")
        print(f"  Jersey/names     : {', '.join(pipe_jerseys) if pipe_jerseys else '(none)'}")
        print(f"  'Unknown' players: {unknown_count}")

    # ---- Key discrepancies ----
    print("\n--- KEY DISCREPANCIES ---")

    if pipeline_result is None:
        print("  * Pipeline has NO stored results — DB is empty.")
        print("    The pipeline is broken: it has never completed a job successfully.")
    else:
        player_diff = n_players - pipe_players
        shot_diff   = total_shots - pipe_shots
        turn_diff   = total_turns - pipe_turnovers

        def _sign(n): return ("+" if n > 0 else "") + str(n)

        if player_diff != 0:
            print(f"  * Player count: agent={n_players}, pipeline={pipe_players} "
                  f"(delta {_sign(player_diff)})")
        else:
            print(f"  * Player count matches: {n_players}")

        if shot_diff != 0:
            print(f"  * Shot count: agent={total_shots}, pipeline={pipe_shots} "
                  f"(delta {_sign(shot_diff)})")
        else:
            print(f"  * Shot count matches: {total_shots}")

        if turn_diff != 0:
            print(f"  * Turnover count: agent={total_turns}, pipeline={pipe_turnovers} "
                  f"(delta {_sign(turn_diff)})")
        else:
            print(f"  * Turnover count matches: {total_turns}")

        if jerseys and pipeline_result is not None:
            print(f"  * Stat agent found jersey numbers: {', '.join(jerseys)}")
            print(f"    Pipeline jersey/names: {', '.join(pipe_jerseys) if pipe_jerseys else '(none)'}")

    print("\n  ROOT CAUSE HYPOTHESIS:")
    if pipeline_result is None:
        print("  The pipeline never completed: the DB has 0 jobs.  The 'unknown player'")
        print("  issue likely occurs during player naming (OCR / roster lookup) and the")
        print("  pipeline crashes or produces no output before writing to the DB.")
    else:
        if unknown_count > 0:
            print(f"  Pipeline output contains {unknown_count} 'unknown' players, confirming")
            print("  that jersey OCR / team assignment is failing in the main pipeline.")
            print("  Possible causes:")
            print("    1. EasyOCR not running (enable_ocr=False default in Tracker)")
            print("    2. YOLO player confidence too high — players not detected")
            print("    3. ByteTrack not installed (supervision package missing)")
            print("    4. Jersey crop too small at broadcast resolution")
        else:
            print("  Pipeline result looks populated — compare numbers above for drift.")

    print("\n" + "=" * 70)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    agent  = StatAgent()
    results = agent.process()

    pipeline = load_pipeline_result()
    print_comparison(results, pipeline)

    # Persist agent results for inspection
    out_path = REPO_ROOT / "data" / "output" / "stat_agent_box_score.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[StatAgent] Box score saved to: {out_path}")
