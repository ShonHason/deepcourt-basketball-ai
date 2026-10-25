# DeepCourt Detection Core

This project is paused at the stat layer until ball and rim detection are reliable.
Points, rebounds, assists, and play-by-play all depend on the same foundation:

1. Where is the ball?
2. Where is the active rim?
3. Are those detections stable over time?

## Current Rule

Do not improve the points/assists/rebounds analyzer until detection metrics pass.
The analyzer code stays in the repo, but new work should go through the
detection-only workflow first.

## Detection-Only Workflow

Run the detector on sampled video frames:

```bash
.venv/bin/python tools/detection_core.py \
  --video data/videoMockup/Maccabi_Rishon_Neria_vs_Maccabi_Shaham_First_Half_15.12.2021.mp4 \
  --every 30 \
  --limit 100
```

Outputs:

- `data/output/detection_core/review/` - annotated frames
- `data/output/detection_core/manifest.csv` - per-frame telemetry
- `data/output/detection_core/summary.json` - detection rates and metrics

If labels exist, pass a YOLO label directory:

```bash
.venv/bin/python tools/detection_core.py \
  --video path/to/video.mp4 \
  --labels path/to/yolo/labels \
  --every 30
```

Expected label classes:

- `0` - basketball
- `1` - active rim

## Success Metrics

Before reconnecting stats, target:

- Ball recall: `>= 0.85` on visible-ball frames
- Ball precision: `>= 0.90`
- Active-rim recall: `>= 0.95` where the basket is visible
- Active-rim precision: `>= 0.95`
- Low false positives on floor reflections, jerseys, signs, lights, and side baskets

Unlabeled detection rate is not true accuracy. It only tells us how often the
detector produced a candidate. Real progress requires labeled frames.

## Dataset Strategy

Train on hard and diverse cases, not just easy frames:

- different gyms and cameras
- close, far, and cropped views
- motion blur
- ball near players
- ball near rim/backboard/net
- ball on floor during dribbles
- no-ball frames
- orange shoes, jerseys, floor reflections, signs, and lights
- side/practice baskets marked as negative for active-rim training

Use `tools/extract_ball_hard_frames.py` to collect low-confidence/missed ball
frames, then correct labels manually before retraining.

## Model Direction

The generic COCO `sports ball` model is not the ball solution. Use the Roboflow
basketball model as the primary ball/rim model, exported through Ultralytics.
Train at a larger image size such as `960` or `1280` when possible because the
ball is tiny in full-court footage.
