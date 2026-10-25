"""
resplit_dataset.py — shuffle all frames and re-split train/valid.

Collects every image+label from an existing dataset (train + valid),
shuffles them randomly, and writes a new dataset with an 85/15 split.
New labeled frames (e.g. v8fail_, v9fail_) end up spread across both splits.

Usage:
    python tools/resplit_dataset.py
    python tools/resplit_dataset.py --src data/datasets/deepcourt_combined_t12630_v1996 --split 0.85
"""
from __future__ import annotations
import argparse, random, shutil
from pathlib import Path

PROJECT  = Path(__file__).resolve().parents[1]
DEFAULT_SRC = PROJECT / "data/datasets/deepcourt_combined_t12630_v1996"
DATASETS    = PROJECT / "data/datasets"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",   type=Path, default=DEFAULT_SRC)
    ap.add_argument("--split", type=float, default=0.85, help="train fraction")
    ap.add_argument("--seed",  type=int,   default=42)
    args = ap.parse_args()

    src = args.src.resolve()
    print(f"Source: {src.name}")

    # Collect all (image_path, label_path) pairs from train + valid
    pairs: list[tuple[Path, Path]] = []
    for split in ("train", "valid"):
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / img.with_suffix(".txt").name
            if lbl.exists():
                pairs.append((img, lbl))
            else:
                pairs.append((img, None))

    total = len(pairs)
    print(f"Total frames: {total}")

    random.seed(args.seed)
    random.shuffle(pairs)

    n_train = round(total * args.split)
    n_valid = total - n_train
    train_pairs = pairs[:n_train]
    valid_pairs = pairs[n_train:]

    # New dataset name
    new_name = f"deepcourt_combined_t{n_train}_v{n_valid}"
    dst = DATASETS / new_name
    if dst.exists():
        shutil.rmtree(dst)

    for split, split_pairs in (("train", train_pairs), ("valid", valid_pairs)):
        (dst / split / "images").mkdir(parents=True)
        (dst / split / "labels").mkdir(parents=True)
        for img, lbl in split_pairs:
            shutil.copy2(img, dst / split / "images" / img.name)
            if lbl:
                shutil.copy2(lbl, dst / split / "labels" / lbl.name)

    # Write data.yaml
    yaml = f"""train: {dst / 'train' / 'images'}
val:   {dst / 'valid' / 'images'}

nc: 3
names: ['ball', 'main_rim', 'other_rim']
"""
    (dst / "data.yaml").write_text(yaml)

    print(f"\nNew dataset: {new_name}")
    print(f"  train: {n_train}")
    print(f"  valid: {n_valid}")
    print(f"  path:  {dst}")

    # Verify failure frames spread
    fail_train = sum(1 for img, _ in train_pairs if "fail_" in img.name)
    fail_valid = sum(1 for img, _ in valid_pairs if "fail_" in img.name)
    print(f"\nFailure frames distribution:")
    print(f"  train: {fail_train}")
    print(f"  valid: {fail_valid}")

if __name__ == "__main__":
    main()
