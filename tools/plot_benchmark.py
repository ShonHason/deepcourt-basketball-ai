import json, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))
vs = ["v3","v4","v5","v6","v7","v8","v9_last","v10_last"]
lab = ["v3","v4","v5","v6","v7","v8","v9","v10"]
series = [("mAP50", lambda r: r["mAP50"], "#2a78d6"),
          ("Ball recall", lambda r: r["classes"]["ball"]["R"], "#eb6834"),
          ("Rim recall", lambda r: r["classes"]["main_rim"]["R"], "#1baf7a")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = range(len(vs))
for name, f, c in series:
    y = [f(d[v]) for v in vs]
    ax.plot(x, y, color=c, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=2, label=name)
    ax.annotate(f"{name}  {y[-1]:.3f}", (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points", va="center", color="#0b0b0b", fontsize=10)
ax.set_xticks(list(x), lab, color="#52514e"); ax.set_ylim(0, 1); ax.set_xlim(-0.3, len(vs) + 0.9)
ax.tick_params(colors="#52514e", length=0)
ax.grid(axis="y", color="#e8e7e3", lw=1); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_visible(False)
ax.axhline(0.85, color="#52514e", lw=1, ls=(0, (4, 3)))
ax.text(2.6, 0.865, "ball recall target 0.85", color="#52514e", fontsize=9)
ax.set_title("Detector progress on the fixed ts761 test set (761 frames, 6 leagues)", loc="left", color="#0b0b0b", fontsize=12, pad=12)
ax.legend(loc="lower right", bbox_to_anchor=(0.86, 0.05), frameon=False, labelcolor="#0b0b0b")
fig.tight_layout(); fig.savefig(sys.argv[2], facecolor="#fcfcfb")
