"""
One-frame vision debugger.

Usage:
    python3 -m src.vision.frame_analyzer data/videos/game.mp4 --frame 120

Writes an annotated image and prints JSON with ball, hoop, and player detections.
This is intentionally outside the full stat pipeline so model/court issues can
be diagnosed without waiting for a whole game analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.vision.detector import BBALL_MODEL_PATH, Detector, FrameDetections


def _read_frame(video_path: str, frame_idx: int | None, timestamp_ms: int | None) -> tuple[Any, int, int]:
    import cv2  # type: ignore[import-untyped]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if timestamp_ms is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
    elif frame_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read requested frame")

    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    actual_ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    return frame, max(actual_frame, 0), actual_ts


def _summary(fd: FrameDetections) -> dict:
    return {
        "frame_idx": fd.frame_idx,
        "timestamp_ms": fd.timestamp_ms,
        "frame_size": [fd.frame_width, fd.frame_height],
        "ball": (
            {
                "bbox": [round(v, 2) for v in fd.ball.bbox],
                "center": [round(v, 2) for v in fd.ball.center],
                "confidence": round(fd.ball.confidence, 3),
            }
            if fd.ball is not None
            else None
        ),
        "hoops": [
            {
                "side": h.side,
                "bbox": [round(v, 2) for v in h.bbox],
                "center": [round(v, 2) for v in h.center],
                "confidence": round(h.confidence, 3),
            }
            for h in fd.hoops
        ],
        "players": [
            {
                "bbox": [round(v, 2) for v in p.bbox],
                "center": [round(v, 2) for v in p.center],
                "confidence": round(p.confidence, 3),
                "in_interest_zone": p.in_interest_zone,
            }
            for p in fd.players
        ],
    }


def _annotate(frame: Any, fd: FrameDetections, output_path: Path) -> None:
    import cv2  # type: ignore[import-untyped]

    image = frame.copy()

    for player in fd.players:
        x1, y1, x2, y2 = (int(v) for v in player.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 200, 80), 2)
        cv2.putText(image, "player", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1)

    if fd.ball is not None:
        x1, y1, x2, y2 = (int(v) for v in fd.ball.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 140, 255), 3)
        cv2.putText(image, f"ball {fd.ball.confidence:.2f}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    for hoop in fd.hoops:
        x1, y1, x2, y2 = (int(v) for v in hoop.bbox)
        color = (255, 80, 80) if hoop.confidence >= 0.40 else (160, 160, 160)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{hoop.side} hoop {hoop.confidence:.2f}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write image: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one basketball video frame")
    parser.add_argument("video_path")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--timestamp-ms", type=int, default=None)
    parser.add_argument("--det-model", default="models/yolo11n.pt")
    parser.add_argument("--pose-model", default="models/yolo11n-pose.pt")
    parser.add_argument("--bball-model", default=BBALL_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="data/output/frame_debug.jpg")
    args = parser.parse_args()

    frame, frame_idx, ts_ms = _read_frame(args.video_path, args.frame, args.timestamp_ms)
    detector = Detector(
        det_model=args.det_model,
        pose_model=args.pose_model,
        device=args.device,
        bball_model=args.bball_model,
    )
    fd = detector.process_frame(frame, frame_idx, ts_ms)
    _annotate(frame, fd, Path(args.output))

    summary = _summary(fd)
    summary["annotated_image"] = str(Path(args.output).resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
