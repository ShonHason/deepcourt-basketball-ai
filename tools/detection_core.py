from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.vision.detection_eval import (  # noqa: E402
    BALL_CLASS,
    RIM_CLASS,
    Box,
    evaluate_class,
    merge_metrics,
    read_yolo_labels,
)
from src.vision.detector import Detector  # noqa: E402


DEFAULT_VIDEO = PROJECT / "data/videoMockup/Maccabi_Rishon_Neria_vs_Maccabi_Shaham_First_Half_15.12.2021.mp4"
DEFAULT_OUT = PROJECT / "data/output/detection_core"


def detector_boxes(fd, min_real_hoop_conf: float) -> list[Box]:
    boxes: list[Box] = []
    if fd.ball is not None:
        x1, y1, x2, y2 = fd.ball.bbox
        boxes.append(Box(BALL_CLASS, x1, y1, x2, y2, fd.ball.confidence))

    for hoop in fd.hoops:
        if hoop.confidence < min_real_hoop_conf:
            continue
        x1, y1, x2, y2 = hoop.bbox
        boxes.append(Box(RIM_CLASS, x1, y1, x2, y2, hoop.confidence))
    return boxes


def draw_boxes(img, boxes: list[Box], labels: list[Box] | None = None) -> None:
    colors = {
        BALL_CLASS: (0, 180, 255),
        RIM_CLASS: (255, 80, 80),
    }
    names = {
        BALL_CLASS: "ball",
        RIM_CLASS: "rim",
    }

    for box in labels or []:
        x1, y1, x2, y2 = map(int, box.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(img, f"label {names.get(box.class_id, box.class_id)}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy)
        color = colors.get(box.class_id, (0, 255, 0))
        name = names.get(box.class_id, str(box.class_id))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{name} {box.confidence:.2f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def save_hard_frame(
    hard_out: Path,
    frame,
    review,
    frame_idx: int,
    reason: str,
) -> None:
    raw_dir = hard_out / "images"
    label_dir = hard_out / "labels"
    review_dir = hard_out / "review"
    for path in (raw_dir, label_dir, review_dir):
        path.mkdir(parents=True, exist_ok=True)

    stem = f"frame_{frame_idx:06d}_{reason}"
    cv2.imwrite(str(raw_dir / f"{stem}.jpg"), frame)
    cv2.imwrite(str(review_dir / f"{stem}.jpg"), review)
    (label_dir / f"{stem}.txt").write_text("", encoding="utf-8")
    classes = hard_out / "classes.txt"
    if not classes.exists():
        classes.write_text("ball\nmain_rim\nother_rim\n", encoding="utf-8")


def hard_frame_reason(fd, min_ball_conf: float) -> str:
    reasons = []
    if fd.ball is None:
        reasons.append("missing_ball")
    elif fd.ball.confidence < min_ball_conf:
        reasons.append("weak_ball")
    if fd.raw_ball_candidates > 1:
        reasons.append("multi_ball")
    if fd.raw_hoop_candidates > 2:
        reasons.append("multi_rim")
    if not fd.active_rims_calibrated and fd.raw_hoop_candidates > 2:
        reasons.append("rim_calibration")
    return "_".join(reasons)


def frame_indices(total_frames: int, every: int, limit: int | None) -> list[int]:
    frames = list(range(0, total_frames, every))
    return frames[:limit] if limit is not None else frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detection-only ball/rim workflow: sample video, annotate frames, and compute label metrics."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", type=Path, default=None, help="Optional YOLO label directory for sampled frames.")
    parser.add_argument("--every", type=int, default=30, help="Analyze every Nth frame.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum sampled frames to analyze.")
    parser.add_argument("--start-ms", type=int, default=0, help="Start timestamp in milliseconds.")
    parser.add_argument("--end-ms", type=int, default=None, help="End timestamp in milliseconds.")
    parser.add_argument("--bball-model", type=str, default=None, help="Optional ball/rim YOLO model path.")
    parser.add_argument("--no-images", action="store_true", help="Write JSON/CSV only; skip annotated images.")
    parser.add_argument("--ball-iou", type=float, default=0.35)
    parser.add_argument("--rim-iou", type=float, default=0.50)
    parser.add_argument("--min-real-hoop-conf", type=float, default=0.40)
    parser.add_argument("--hard-out", type=Path, default=None, help="Optional directory for weak/missed frames to relabel.")
    parser.add_argument("--hard-ball-conf", type=float, default=0.25, help="Save frames with ball confidence below this value.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    image_out = args.out / "review"
    if not args.no_images:
        image_out.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = max(0, int(round((args.start_ms / 1000.0) * native_fps)))
    end_frame = total_frames
    if args.end_ms is not None:
        end_frame = min(total_frames, max(start_frame, int(round((args.end_ms / 1000.0) * native_fps))))

    detector_kwargs = {"device": "cpu"}
    if args.bball_model:
        detector_kwargs["bball_model"] = args.bball_model
    detector = Detector(**detector_kwargs)

    rows: list[dict] = []
    ball_metrics = []
    rim_metrics = []
    has_labels = args.labels is not None

    for frame_idx in frame_indices(end_frame - start_frame, args.every, args.limit):
        frame_idx += start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        fd = detector.process_frame(frame, frame_idx, ts_ms)
        preds = detector_boxes(fd, args.min_real_hoop_conf)
        labels: list[Box] = []

        if args.labels is not None:
            label_path = args.labels / f"frame_{frame_idx:06d}.txt"
            if not label_path.exists():
                # Accept the existing hard-frame naming convention too.
                matches = sorted(args.labels.glob(f"frame_{frame_idx:06d}_*.txt"))
                label_path = matches[0] if matches else label_path
            labels = read_yolo_labels(label_path, fd.frame_width, fd.frame_height)
            ball_metrics.append(evaluate_class(preds, labels, BALL_CLASS, args.ball_iou))
            rim_metrics.append(evaluate_class(preds, labels, RIM_CLASS, args.rim_iou))

        review = None
        if not args.no_images or args.hard_out is not None:
            review = frame.copy()
            cv2.rectangle(review, (0, 0), (review.shape[1], 44), (20, 20, 20), -1)
            cv2.putText(review, f"frame {frame_idx}  {ts_ms / 1000:.1f}s", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            draw_boxes(review, preds, labels if has_labels else None)

        if not args.no_images and review is not None:
            cv2.imwrite(str(image_out / f"frame_{frame_idx:06d}.jpg"), review)

        if args.hard_out is not None and review is not None:
            reason = hard_frame_reason(fd, args.hard_ball_conf)
            if reason:
                save_hard_frame(args.hard_out, frame, review, frame_idx, reason)

        ball_pred = next((b for b in preds if b.class_id == BALL_CLASS), None)
        rim_count = sum(1 for b in preds if b.class_id == RIM_CLASS)
        rows.append({
            "frame_idx": frame_idx,
            "timestamp_ms": ts_ms,
            "ball_detected": ball_pred is not None,
            "ball_confidence": None if ball_pred is None else round(ball_pred.confidence, 4),
            "ball_source": fd.ball_source,
            "ball_score": round(fd.ball_score, 4),
            "raw_ball_candidates": fd.raw_ball_candidates,
            "rim_detections": rim_count,
            "raw_rim_candidates": fd.raw_hoop_candidates,
            "active_rims_calibrated": fd.active_rims_calibrated,
            "player_detections": len(fd.players),
            "label_ball_count": sum(1 for b in labels if b.class_id == BALL_CLASS),
            "label_rim_count": sum(1 for b in labels if b.class_id == RIM_CLASS),
        })

    cap.release()

    summary = {
        "video": str(args.video),
        "sampled_frames": len(rows),
        "ball_detected_frames": sum(1 for r in rows if r["ball_detected"]),
        "rim_detected_frames": sum(1 for r in rows if r["rim_detections"] > 0),
        "labels_used": has_labels,
    }
    if rows:
        summary["ball_detected_rate_unlabeled"] = round(summary["ball_detected_frames"] / len(rows), 4)
        summary["rim_detected_rate_unlabeled"] = round(summary["rim_detected_frames"] / len(rows), 4)
    if has_labels:
        summary["ball_metrics"] = merge_metrics(ball_metrics).as_dict()
        summary["rim_metrics"] = merge_metrics(rim_metrics).as_dict()

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame_idx"])
        writer.writeheader()
        writer.writerows(rows)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
