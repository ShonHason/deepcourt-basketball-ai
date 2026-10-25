"""
mine_test_set.py — mine random frames from test videos for a fixed test set.

Samples frames uniformly (NOT failure-based) from each video.
Skips frame indices already used in failure mining (v8fail_, v9fail_).
Output frames are clean JPEGs ready for labeling.

Usage:
    python tools/mine_test_set.py
"""
from __future__ import annotations
import random
from pathlib import Path

import cv2

PROJECT  = Path(__file__).resolve().parents[1]
MODEL    = PROJECT / "models/deepcourt_v8_best.pt"
OUT_DIR  = PROJECT / "data/output/test_set_for_labeling"
DB_TRAIN = PROJECT / "data/datasets/deepcourt_combined_t12432_v2194/train"

VIDEOS = [
    ("ncaa",   PROJECT / "data/test_v8/ncaa_duke_unc.mp4",             100),
    ("cba",    PROJECT / "data/test_v8/cba_beijing_guangsha.mp4",       100),
    ("fiba_1", PROJECT / "data/test_v8/fiba_women_spain_hungary.mp4",   100),
    ("fiba_2", PROJECT / "data/test_v8/fiba_women_spain_hungary_2.mp4", 100),
]

STEP    = 5   # sample every 5th frame
MIN_GAP = 60  # min frames between saved frames


def get_used_indices(tag: str) -> set[int]:
    """Collect frame indices already in the training set for this video tag."""
    used = set()
    for img in DB_TRAIN.glob(f"images/*{tag}_f*.jpg"):
        try:
            idx = int(img.stem.split("_f")[-1])
            used.add(idx)
        except ValueError:
            pass
    return used


def main():
    for sub in ("images", "labels"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.txt").write_text("ball\nmain_rim\nother_rim\n")

    total_saved = 0

    for tag, video_path, quota in VIDEOS:
        if not video_path.exists():
            print(f"  SKIP (not found): {video_path.name}")
            continue

        used_indices = get_used_indices(tag)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n{tag}: {total_frames} frames, {len(used_indices)} already used → target {quota}")

        # Collect candidate frame indices
        candidates = [
            i for i in range(0, total_frames, STEP)
            if i not in used_indices
        ]
        random.seed(42)
        random.shuffle(candidates)

        # Select with min gap enforced
        selected = []
        last = -MIN_GAP
        for idx in sorted(random.sample(candidates, min(quota * 5, len(candidates)))):
            if idx - last >= MIN_GAP:
                selected.append(idx)
                last = idx
            if len(selected) >= quota:
                break

        selected_set = set(selected)

        # Read and save selected frames
        frame_idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in selected_set:
                stem = f"{tag}_f{frame_idx:06d}"
                cv2.imwrite(str(OUT_DIR / "images" / f"{stem}.jpg"), frame)
                # Empty label — to be filled by labeler
                (OUT_DIR / "labels" / f"{stem}.txt").write_text("")
                saved += 1
            frame_idx += 1

        cap.release()
        total_saved += saved
        print(f"  Saved {saved}/{quota}")

    print(f"\n✓ Total: {total_saved} frames → {OUT_DIR}")
    print(f"  Open labeler: python tools/local_yolo_labeler.py --root {OUT_DIR}")


if __name__ == "__main__":
    main()
