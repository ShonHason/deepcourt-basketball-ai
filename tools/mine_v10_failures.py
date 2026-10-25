"""
mine_v10_failures.py — mine failure frames using v10_best.

Failure criteria (any of):
  - Ball detected with low confidence (0.25-0.65)
  - Ball NOT detected but rim IS visible
  - Ball detected but bbox suspiciously large (likely head/player)

Usage:
    python tools/mine_v10_failures.py
"""
from __future__ import annotations
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]
MODEL   = PROJECT / "models/deepcourt_v10_best.pt"
OUT_DIR = PROJECT / "data/output/v10_failures_for_labeling"

VIDEOS = [
    ("nba",       PROJECT / "data/test_v9/Oklahoma City Thunder vs Houston Rockets Q1.mp4",  300),
    ("olympics",  PROJECT / "data/test_v9/USA vs Spain - Men's Gold Final 2012 - Q1.mp4",   300),
    ("euroleague",PROJECT / "data/test_v9/euroleague_partizan_realmadrid.mp4",               200),
    ("fiba",      PROJECT / "data/test_v10",                                                 200),  # new video
]

CONF_LOW       = 0.25
CONF_HIGH      = 0.65   # higher than v9 since v10 is more confident overall
BALL_MAX_AREA  = 0.015
MIN_GAP        = 30
STEP           = 3


def find_video(path: Path) -> Path | None:
    if path.is_file():
        return path
    if path.is_dir():
        for ext in ("*.mp4", "*.mkv", "*.avi", "*.webm"):
            found = list(path.glob(ext))
            if found:
                return found[0]
    return None


def mine_video(model, video_path: Path, tag: str, quota: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{tag}: {total} frames → target {quota}")

    candidates = []
    last_saved  = -MIN_GAP
    frame_idx   = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        if frame_idx % STEP != 0:
            frame_idx += 1; continue

        h, w = frame.shape[:2]
        frame_area = h * w
        result = model(frame, verbose=False, conf=CONF_LOW)[0]

        ball_det = None
        rim_found = False
        detections = []
        yolo_lines = []

        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = (float(v) for v in box.xyxy[0])
            detections.append((cls, conf, (x1,y1,x2,y2)))
            if cls == 1: rim_found = True
            if cls == 0: ball_det = (conf, (x1,y1,x2,y2))
            cx=((x1+x2)/2)/w; cy=((y1+y2)/2)/h
            bw=(x2-x1)/w; bh=(y2-y1)/h
            yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        reason = None
        if ball_det:
            conf, (x1,y1,x2,y2) = ball_det
            area = (x2-x1)*(y2-y1)/frame_area
            if conf < CONF_HIGH:
                reason = f"low conf ({conf:.2f})"
            elif area > BALL_MAX_AREA:
                reason = f"oversized ({area:.3f})"
        elif rim_found:
            reason = "ball missed (rim visible)"

        if reason and (frame_idx - last_saved) >= MIN_GAP:
            candidates.append((frame_idx, reason, detections, yolo_lines, frame.copy()))
            last_saved = frame_idx

        frame_idx += 1
    cap.release()

    if len(candidates) > quota:
        candidates = random.sample(candidates, quota)
    candidates.sort(key=lambda x: x[0])
    return candidates


def save_frames(candidates, tag, out_dir):
    saved = 0
    for frame_idx, reason, dets, ylines, frame in candidates:
        stem = f"{tag}_f{frame_idx:06d}"
        cv2.imwrite(str(out_dir/"images"/f"{stem}.jpg"), frame)
        (out_dir/"labels"/f"{stem}.txt").write_text("\n".join(ylines))
        out = frame.copy()
        cv2.rectangle(out,(0,0),(frame.shape[1],40),(20,20,20),-1)
        cv2.putText(out, f"v10_best | {reason}", (10,26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,180,255), 2)
        for cls,conf,(x1,y1,x2,y2) in dets:
            color = (0,200,255) if cls==0 else (80,255,80)
            cv2.rectangle(out,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
        cv2.imwrite(str(out_dir/"annotated"/f"{stem}.jpg"), out)
        saved += 1
    print(f"  Saved {saved}")
    return saved


def main():
    for sub in ("images", "labels", "annotated"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.txt").write_text("ball\nmain_rim\nother_rim\n")

    model = YOLO(str(MODEL))
    total_saved = 0

    for tag, video_path, quota in VIDEOS:
        resolved = find_video(Path(video_path))
        if not resolved:
            print(f"  SKIP: {video_path} — not found"); continue
        candidates = mine_video(model, resolved, tag, quota)
        total_saved += save_frames(candidates, tag, OUT_DIR)

    print(f"\n✓ Total: {total_saved} frames → {OUT_DIR}")
    print(f"  Open labeler: python tools/local_yolo_labeler.py --images {OUT_DIR}/images --labels {OUT_DIR}/labels")


if __name__ == "__main__":
    main()
