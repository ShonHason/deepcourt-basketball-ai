# Detector Model Comparison — ts761 Benchmark

All checkpoints evaluated on the same fixed test set: `deepcourt_tr13282_v2344_ts761/test`
(761 frames, 6 leagues: NCAA, CBA, FIBA, EuroLeague, NBA, Olympics; 402 ball + 393 main_rim labels).
Settings: `imgsz=1280`, default conf/IoU, Ultralytics `val()`. Script: `tools/benchmark_ts761.py`.
Raw per-class numbers: `ts761_results.json`.

| Version | mAP50 | mAP50-95 | Ball R | Ball P | Ball mAP50 | Rim R | Rim mAP50 |
|---|---|---|---|---|---|---|---|
| legacy roboflow (`basketball_best.pt`)* | 0.017 | 0.004 | 0.13 | 0.10 | 0.03 | 0.03 | 0.00 |
| ball_rim v1 (epoch3) | 0.064 | 0.028 | 0.19 | 0.22 | 0.13 | 0.00 | 0.00 |
| ball_rim v2 (epoch1) | 0.077 | 0.038 | 0.22 | 0.25 | 0.15 | 0.00 | 0.00 |
| ball_rim v2 yolo11s | 0.141 | 0.063 | 0.27 | 0.51 | 0.28 | 0.00 | 0.00 |
| ball_rim_best | 0.150 | 0.087 | 0.23 | 0.50 | 0.29 | 0.00 | 0.01 |
| v3 | 0.074 | 0.025 | 0.27 | 0.23 | 0.13 | 0.00 | 0.02 |
| v4 | 0.126 | 0.052 | 0.48 | 0.14 | 0.19 | 0.07 | 0.06 |
| v4 ep56 | 0.240 | 0.075 | 0.35 | 0.23 | 0.16 | 0.33 | 0.32 |
| v5 | 0.516 | 0.290 | 0.55 | 0.58 | 0.55 | 0.53 | 0.49 |
| v6 | 0.586 | 0.371 | 0.55 | 0.70 | 0.59 | 0.58 | 0.58 |
| v7 | 0.638 | 0.430 | 0.55 | 0.76 | 0.61 | 0.67 | 0.67 |
| v8 | 0.709 | 0.514 | 0.60 | 0.78 | 0.66 | 0.74 | 0.76 |
| v9_best | 0.832 | 0.523 | 0.65 | 0.86 | 0.75 | 0.91 | 0.92 |
| v9_last | 0.844 | 0.537 | 0.67 | 0.88 | 0.78 | 0.92 | 0.91 |
| **v10_best** | **0.876** | 0.580 | **0.74** | 0.90 | **0.82** | 0.93 | 0.93 |
| **v10_last** | 0.875 | **0.583** | 0.73 | **0.91** | 0.82 | **0.93** | **0.94** |

\* Different class scheme (Basketball / Basketball Hoop) — comparison is approximate.

## Findings

- **v10_last is the recommended production model.** Ball recall +6 pts over v9_last (≈26 more balls of 402); v10_best vs v10_last difference is noise.
- **Ball detection is still the bottleneck**: 0.73 recall vs 0.85 target. Rim is at 0.93.
- **Validation metrics stored in checkpoints are optimistic.** v9 reports ball recall ≈0.88 on its val set but 0.67 on ts761 — val likely shares distribution (adjacent frames / same games) with train. Compare versions on ts761 only.
- **`other_rim` is not measurable**: ts761 contains zero `other_rim` labels. Add some to the test set if the class matters.
