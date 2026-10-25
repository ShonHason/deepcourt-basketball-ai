from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


DEFAULT_VIDEO = PROJECT / "data/videoMockup/Maccabi_Rishon_Neria_vs_Maccabi_Shaham_First_Half_15.12.2021.mp4"
DEFAULT_MODEL = PROJECT / "models/deepcourt_ball_rim_v2_epoch1_best.pt"
DEFAULT_OUT = PROJECT / "data/output/sliced_ball_test"


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(
    boxes: list[tuple[float, float, float, float, float, str]],
    iou_thresh: float,
) -> list[tuple[float, float, float, float, float, str]]:
    selected: list[tuple[float, float, float, float, float, str]] = []
    for box in sorted(boxes, key=lambda item: item[4], reverse=True):
        bbox = box[:4]
        if all(iou(bbox, kept[:4]) < iou_thresh for kept in selected):
            selected.append(box)
    return selected


def tile_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    # The ball is tiny at 640x360, so use half-frame-ish crops. Each crop is
    # resized by YOLO, effectively magnifying small objects without changing
    # the original video.
    windows = [(0, 0, width, height)]
    tile_w = int(width * 0.56)
    tile_h = int(height * 0.62)
    xs = sorted({0, max(0, (width - tile_w) // 2), max(0, width - tile_w)})
    ys = sorted({0, max(0, (height - tile_h) // 2), max(0, height - tile_h)})
    for y in ys:
        for x in xs:
            windows.append((x, y, min(width, x + tile_w), min(height, y + tile_h)))
    return windows


def detect_ball_boxes(
    model: YOLO,
    frame,
    imgsz: int,
    conf: float,
) -> list[tuple[float, float, float, float, float, str]]:
    height, width = frame.shape[:2]
    detections: list[tuple[float, float, float, float, float, str]] = []
    names = getattr(model, "names", {}) or {}
    ball_classes = {int(cls_id) for cls_id, name in names.items() if str(name).lower() == "ball"} or {0}

    for x1_off, y1_off, x2_off, y2_off in tile_windows(width, height):
        crop = frame[y1_off:y2_off, x1_off:x2_off]
        source = "full" if (x1_off, y1_off, x2_off, y2_off) == (0, 0, width, height) else "slice"
        result = model(crop, verbose=False, conf=conf, imgsz=imgsz)[0]
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in ball_classes:
                continue
            score = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append((
                clip(x1 + x1_off, 0, width),
                clip(y1 + y1_off, 0, height),
                clip(x2 + x1_off, 0, width),
                clip(y2 + y1_off, 0, height),
                score,
                source,
            ))
    return nms(detections, 0.35)


def draw(frame, boxes: list[tuple[float, float, float, float, float, str]], frame_idx: int, ts_ms: int) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"frame {frame_idx}  {ts_ms / 1000:.1f}s  sliced ball",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for x1, y1, x2, y2, conf, source in boxes:
        color = (0, 180, 255) if source == "full" else (0, 255, 180)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            f"ball {conf:.2f} {source}",
            (int(x1), max(18, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test sliced/tiled inference for tiny basketball detection.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-ms", type=int, default=600_000)
    parser.add_argument("--end-ms", type=int, default=660_000)
    parser.add_argument("--every", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    review_dir = args.out / "review"
    review_dir.mkdir(exist_ok=True)

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = max(0, int(round((args.start_ms / 1000) * fps)))
    end_frame = int(round((args.end_ms / 1000) * fps))

    rows: list[dict[str, object]] = []
    for frame_idx in range(start_frame, end_frame, args.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        boxes = detect_ball_boxes(model, frame, args.imgsz, args.conf)
        review = frame.copy()
        draw(review, boxes, frame_idx, ts_ms)
        cv2.imwrite(str(review_dir / f"frame_{frame_idx:06d}.jpg"), review)
        best = boxes[0] if boxes else None
        rows.append({
            "frame_idx": frame_idx,
            "timestamp_ms": ts_ms,
            "ball_detected": bool(boxes),
            "ball_count": len(boxes),
            "best_conf": round(best[4], 4) if best else "",
            "best_source": best[5] if best else "",
        })

    cap.release()
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["frame_idx"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "video": str(args.video),
        "model": str(args.model),
        "sampled_frames": len(rows),
        "ball_detected_frames": sum(1 for row in rows if row["ball_detected"]),
        "frames_with_multiple_balls": sum(1 for row in rows if int(row["ball_count"]) > 1),
        "conf": args.conf,
        "imgsz": args.imgsz,
    }
    if rows:
        summary["ball_detected_rate"] = round(summary["ball_detected_frames"] / len(rows), 4)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
