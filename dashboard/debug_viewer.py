"""
debug_viewer.py — Streamlit viewer for debug_team_vision output.

Run:
    streamlit run dashboard/debug_viewer.py --server.port 8503
"""
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import streamlit as st

st.set_page_config(page_title="DeepCourt — Vision Debug", layout="wide")
st.title("Vision Debug — Team Classification")

OUT_DIR = PROJECT / "data" / "output"
VIDEO   = OUT_DIR / "debug_team_vision.mp4"
TABLE   = OUT_DIR / "debug_team_vision.txt"

# ── run / status ──────────────────────────────────────────────────────────────
col_run, col_status = st.columns([3, 1])

with col_run:
    with st.expander("Run debug analysis on a video", expanded=not VIDEO.exists()):
        video_files = sorted(
            list((PROJECT / "data" / "videos").rglob("*.mp4")) +
            list((PROJECT / "data" / "videoMockup").rglob("*.mp4")),
            key=lambda p: p.stat().st_size, reverse=True,
        )
        video_labels = [str(p.relative_to(PROJECT)) for p in video_files]
        chosen = st.selectbox("Choose video", video_labels)
        seconds = st.slider("Seconds to analyse", 30, 300, 90, step=30)
        fps     = st.select_slider("Sampling FPS", [3, 5, 7], value=5)

        if st.button("▶ Run debug analysis", type="primary"):
            import subprocess
            cmd = [
                sys.executable, "tools/debug_team_vision.py",
                "--video", chosen,
                "--seconds", str(seconds),
                "--fps", str(fps),
            ]
            with st.spinner(f"Running on first {seconds}s of video…"):
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=str(PROJECT)
                )
            if result.returncode == 0:
                st.success("Done!")
                st.text(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
                st.rerun()
            else:
                st.error("Analysis failed")
                st.code(result.stderr[-2000:])

with col_status:
    if VIDEO.exists():
        import time
        age_s = time.time() - VIDEO.stat().st_mtime
        age_str = f"{int(age_s//60)}m ago" if age_s > 60 else "just now"
        st.metric("Last run", age_str)
        size_mb = VIDEO.stat().st_size / 1e6
        st.metric("Video size", f"{size_mb:.1f} MB")

# ── output ────────────────────────────────────────────────────────────────────
if not VIDEO.exists():
    st.info("No debug output yet. Run an analysis above.")
    st.stop()

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Annotated video")
    st.caption("Blue = Team A  |  Red = Team B  |  Yellow = Ref/Staff  |  Gray = Unknown")
    video_bytes = VIDEO.read_bytes()
    st.video(video_bytes)

with right:
    st.subheader("Track classification table")

    if TABLE.exists():
        import pandas as pd
        lines = [l for l in TABLE.read_text().splitlines() if l.strip() and not l.startswith("-")]
        if len(lines) > 1:
            header = [h.strip() for h in lines[0].split("  ") if h.strip()]
            data   = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 5:
                    data.append({
                        "Track":      parts[0],
                        "Team":       parts[1],
                        "Frames":     int(parts[2]),
                        "HueSamples": int(parts[3]),
                        "HueMedian":  parts[4],
                    })
            if data:
                df = pd.DataFrame(data)

                # colour rows by team
                def _row_color(row):
                    colors = {
                        "team_a":       "background-color: #1a3a6e; color: white",
                        "team_b":       "background-color: #6e1a1a; color: white",
                        "REF/staff":    "background-color: #5c5c10; color: white",
                        "unclassified": "background-color: #2a2a2a; color: #aaa",
                    }
                    c = colors.get(row["Team"], "")
                    return [c] * len(row)

                st.dataframe(
                    df.style.apply(_row_color, axis=1),
                    use_container_width=True, height=500,
                )

                # summary pills
                st.divider()
                from collections import Counter
                counts = Counter(df["Team"])
                cols = st.columns(len(counts))
                for i, (team, cnt) in enumerate(sorted(counts.items())):
                    cols[i].metric(team, f"{cnt} tracks")

# ── what to look for ─────────────────────────────────────────────────────────
with st.expander("What to look for"):
    st.markdown("""
**Good signs**
- Team A has ~5 tracks, Team B has ~5 tracks
- Refs and coaches appear as **Yellow (REF/staff)** or **Gray (unknown)** — NOT blue or red
- The same player keeps the same color throughout the video (track IDs are stable)
- Players with jerseys clearly on one team stay that color even when running to the other side

**Bad signs**
- A referee or coach showing up as Blue or Red → dark jersey filter may need tuning
- One team has 10+ tracks → track fragmentation (player losing and re-gaining track ID)
- Lots of Gray tracks → hue sampling not getting enough frames per player
- Team A has 8 players, Team B has 2 → k-means failed (both teams may have similar hue)

**Hue reference (OpenCV HSV, 0–180)**
| Hue | Color |
|-----|-------|
| 0–5 or 175–180 | Red |
| 2 (synthetic) | White/light |
| 10–25 | Orange |
| 25–35 | Yellow |
| 35–85 | Green |
| 85–130 | Blue/cyan |
| 130–160 | Purple/violet |
| n/a | Gray/black (filtered) |
""")
