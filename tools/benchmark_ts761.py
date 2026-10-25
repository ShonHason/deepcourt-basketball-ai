"""Benchmark detector checkpoints on the fixed ts761 test set.

Usage: python tools/benchmark_ts761.py <out_dir> name=path/to/model.pt [name=path ...]
Writes <out_dir>/bench_results.json with overall + per-class P/R/mAP.
"""
import sys, json, os
from ultralytics import YOLO
root = os.getcwd()
os.makedirs(sys.argv[1], exist_ok=True)
yaml = f"{sys.argv[1]}/bench.yaml"
open(yaml,"w").write(f"path: {root}/data/datasets/deepcourt_tr13282_v2344_ts761\ntrain: train/images\nval: test/images\nnc: 3\nnames: ['ball','main_rim','other_rim']\n")
out = {}
for name, path in [a.split("=",1) for a in sys.argv[2:]]:
    m = YOLO(path)
    r = m.val(data=yaml, imgsz=1280, batch=8, device="mps", plots=False, verbose=False, project=sys.argv[1], name=name, exist_ok=True)
    per = {}
    for i, c in enumerate(r.box.ap_class_index):
        p, rc, ap50, ap = r.box.class_result(i)
        per[r.names[int(c)]] = dict(P=round(p,4), R=round(rc,4), mAP50=round(ap50,4), mAP5095=round(ap,4))
    out[name] = dict(P=round(r.box.mp,4), R=round(r.box.mr,4), mAP50=round(r.box.map50,4), mAP5095=round(r.box.map,4), classes=per)
    print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(f"{sys.argv[1]}/bench_results.json","w"), indent=1)
