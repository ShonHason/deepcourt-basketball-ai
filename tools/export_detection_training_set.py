from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT / "data/videoMockup/Maccabi_Rishon_Neria_vs_Maccabi_Shaham_First_Half_15.12.2021.mp4"
DEFAULT_MODEL = PROJECT / "models/basketball_best.pt"
DEFAULT_OUT = PROJECT / "data/output/detection_training_set"

BALL_CLASS = 0
RIM_CLASS = 1


def draw_header(img, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 58), (20, 20, 20), -1)
    cv2.putText(
        img,
        text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Correct labels manually: 0 ball, 1 main_rim, 2 other_rim",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def draw_candidate(img, class_id: int, conf: float, xyxy: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    if class_id == BALL_CLASS:
        color = (0, 180, 255)
        label = f"raw ball {conf:.2f}"
    else:
        color = (255, 80, 80)
        label = f"raw rim {conf:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def frame_indices(total_frames: int, every: int, limit: int | None) -> list[int]:
    frames = list(range(0, total_frames, every))
    return frames[:limit] if limit is not None else frames


def priority_reason(ball_count: int, best_ball_conf: float | None, rim_count: int) -> str:
    reasons = []
    if ball_count == 0:
        reasons.append("missing_ball_candidate")
    elif best_ball_conf is not None and best_ball_conf < 0.15:
        reasons.append("weak_ball_candidate")
    if ball_count > 1:
        reasons.append("multiple_ball_candidates")
    if rim_count > 2:
        reasons.append("multiple_rims_mark_main_vs_other")
    if not reasons:
        reasons.append("verify")
    return "+".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export frames for manual Roboflow labeling with classes: "
            "0 ball, 1 main_rim, 2 other_rim. Raw model boxes are drawn only "
            "as review hints; YOLO label files are intentionally empty."
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--every", type=int, default=60)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--conf", type=float, default=0.01)
    args = parser.parse_args()

    images_dir = args.out / "images"
    labels_dir = args.out / "labels"
    review_dir = args.out / "review"
    for path in (images_dir, labels_dir, review_dir):
        path.mkdir(parents=True, exist_ok=True)

    (args.out / "classes.txt").write_text(
        "ball\nmain_rim\nother_rim\n",
        encoding="utf-8",
    )

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    rows: list[dict[str, object]] = []

    for frame_idx in frame_indices(total_frames, args.every, args.limit):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = model(
            frame,
            verbose=False,
            conf=args.conf,
            classes=[BALL_CLASS, RIM_CLASS],
        )[0]

        stem = f"frame_{frame_idx:06d}"
        image_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"
        review_path = review_dir / f"{stem}.jpg"
        cv2.imwrite(str(image_path), frame)
        label_path.write_text("", encoding="utf-8")

        review = frame.copy()
        draw_header(review, f"frame {frame_idx}  {ts_ms / 1000:.1f}s")

        ball_count = 0
        rim_count = 0
        best_ball_conf: float | None = None
        best_rim_conf: float | None = None

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = tuple(float(v) for v in box.xyxy[0])
                draw_candidate(review, class_id, conf, xyxy)
                if class_id == BALL_CLASS:
                    ball_count += 1
                    best_ball_conf = conf if best_ball_conf is None else max(best_ball_conf, conf)
                elif class_id == RIM_CLASS:
                    rim_count += 1
                    best_rim_conf = conf if best_rim_conf is None else max(best_rim_conf, conf)

        cv2.imwrite(str(review_path), review)

        rows.append({
            "image": image_path.name,
            "label": label_path.name,
            "frame_idx": frame_idx,
            "timestamp_ms": ts_ms,
            "raw_ball_candidates": ball_count,
            "best_ball_confidence": "" if best_ball_conf is None else f"{best_ball_conf:.6f}",
            "raw_rim_candidates": rim_count,
            "best_rim_confidence": "" if best_rim_conf is None else f"{best_rim_conf:.6f}",
            "priority": priority_reason(ball_count, best_ball_conf, rim_count),
        })

    cap.release()

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "label",
            "frame_idx",
            "timestamp_ms",
            "raw_ball_candidates",
            "best_ball_confidence",
            "raw_rim_candidates",
            "best_rim_confidence",
            "priority",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} frames")
    print(f"Images: {images_dir}")
    print(f"Empty labels to correct: {labels_dir}")
    print(f"Review overlays: {review_dir}")
    print(f"Manifest: {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
