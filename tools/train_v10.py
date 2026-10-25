"""Fine-tune v9_last → v10 on deepcourt_tr13282_v2344_ts761 (15,626 frames).

Run on RunPod A40 after uploading:
  - deepcourt_tr13282_v2344_ts761.zip  → extract to /workspace/
  - deepcourt_v9_last.pt               → /workspace/deepcourt_v9_last.pt
  - this script                        → /workspace/train_v10.py

Then: python train_v10.py
"""
from pathlib import Path
from ultralytics import YOLO

WORKSPACE  = Path("/workspace")
DATASET    = WORKSPACE / "deepcourt_tr13282_v2344_ts761"
WEIGHTS    = WORKSPACE / "deepcourt_v9_last.pt"
DATA_YAML  = DATASET / "data_runpod.yaml"

# Write an absolute-path data.yaml for RunPod (avoids relative-path issues)
DATA_YAML.write_text(
    f"train: {DATASET}/train/images\n"
    f"val:   {DATASET}/valid/images\n"
    f"test:  {DATASET}/test/images\n"
    "\n"
    "nc: 3\n"
    "names: ['ball', 'main_rim', 'other_rim']\n"
)

model = YOLO(str(WEIGHTS))
model.train(
    data=str(DATA_YAML),
    imgsz=1280,
    epochs=40,
    batch=16,
    patience=15,
    lr0=0.0003,        # low LR — fine-tune from v9, don't destroy learned weights
    lrf=0.01,          # cosine decay to lr0 * lrf = 3e-6
    warmup_epochs=2,
    device=0,
    workers=4,
    project=str(WORKSPACE / "runs"),
    name="deepcourt_v10",
    exist_ok=True,
    optimizer="AdamW",
    cos_lr=True,
    close_mosaic=10,
    cache=True,
    pretrained=True,
    hsv_h=0.01,
    hsv_s=0.25,
    hsv_v=0.20,
    degrees=0.0,
    translate=0.03,
    scale=0.15,
    flipud=0.0,
    fliplr=0.0,        # no horizontal flip — basket positions are directional
    mosaic=0.10,
    mixup=0.0,
    erasing=0.0,
    plots=True,
    verbose=True,
)
