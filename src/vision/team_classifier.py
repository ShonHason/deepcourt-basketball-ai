"""
TeamClassifier — assigns team IDs to tracked players by jersey color.

Algorithm
─────────
1. Each detected player's jersey torso region is converted to HSV.
2. The dominant hue (median of vivid pixels) is sampled every frame and
   accumulated per track_id.
3. After enough samples are collected, iterative 2-means clustering on
   OpenCV hue (circular, 0–179) splits all tracks into two groups.
4. Tracks whose median hue is farther than `outlier_dist` from *both*
   cluster centroids are classified as None — referee, coach, or staff.

No external ML packages required — uses numpy only.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

OUTLIER_HUE_DIST  = 40   # absolute fallback (used when centroid separation is small)
OUTLIER_FRACTION  = 0.28 # adaptive: outlier if min-dist > 28% of centroid separation
                         # e.g. white(8) vs blue(108) → sep=100 → threshold=28 units
                         # catches green/yellow coaches between the two team colors
MIN_SAMPLES       = 2    # minimum per-track samples before classifying


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

class TeamClassifier:
    """
    Accumulates (track_id, hue) samples during the frame loop, then classifies
    every track as 'team_a', 'team_b', or None (non-player).

    Usage::
        tc = TeamClassifier()
        # inside frame loop:
        tc.add_sample(track_id, hue)
        # after loop:
        assignments = tc.classify()   # {track_id: 'team_a' | 'team_b' | None}
    """

    def __init__(self, outlier_dist: float = OUTLIER_HUE_DIST) -> None:
        self._outlier_dist = outlier_dist
        self._samples: dict[int, list[float]] = {}
        # Referee appearance profiles (33-dim vectors from Tracker._compute_appearance)
        # Accumulated within a clip and optionally persisted across clips.
        self._referee_profiles: list = []   # list of np.ndarray

    def add_sample(self, track_id: int, hue: float) -> None:
        buf = self._samples.setdefault(track_id, [])
        buf.append(hue)
        if len(buf) > 40:       # cap history to keep memory bounded
            buf.pop(0)

    def reclassify_with_appearances(
        self,
        assignments: "dict[int, Optional[str]]",
        appearances: "dict[int, Any]",
        sim_threshold: float = 0.92,
    ) -> "dict[int, Optional[str]]":
        """
        Retroactively reclassify team-labelled tracks as referee/staff using
        appearance similarity to known referee profiles.

        How it works
        ────────────
        1. Collect appearance vectors of tracks already classified as None
           (referee/staff by the hue-outlier test).  These are the "ground truth"
           referee appearances from frames where stripe detection fired cleanly.
        2. Add any stored referee profiles from previous clips (self._referee_profiles).
        3. For every team-labelled track, compute cosine similarity against all
           referee appearances.  If max similarity > sim_threshold → reclassify
           as None.  The bimodality component in the 33-dim vector is what makes
           this work: a striped shirt scores high, a solid jersey scores low, even
           when both share a similar hue distribution.
        4. Store referee appearance vectors in self._referee_profiles so they
           persist across calls (useful for persistent cross-clip storage).

        Args:
            assignments:   output of classify() — mutated in place and returned.
            appearances:   {track_id: 33-dim np.ndarray} from Tracker._appearances.
            sim_threshold: cosine similarity cutoff (0.92 ≈ very similar pattern).
        """
        try:
            import numpy as np
        except ImportError:
            return assignments

        # Step 1: gather appearance vectors of confirmed referees
        ref_vecs = list(self._referee_profiles)
        for tid, label in assignments.items():
            if label is None and tid in appearances and appearances[tid] is not None:
                ref_vecs.append(appearances[tid])

        if not ref_vecs:
            return assignments   # no referee profile available — nothing to do

        # Step 2: reclassify team-labelled tracks with high referee similarity
        changed: list[int] = []
        for tid, label in assignments.items():
            if label is None:
                continue   # already a referee
            vec = appearances.get(tid)
            if vec is None:
                continue
            for ref_vec in ref_vecs:
                sim = float(np.dot(vec, ref_vec))
                if sim >= sim_threshold:
                    changed.append(tid)
                    break

        for tid in changed:
            assignments[tid] = None

        # Step 3: persist new referee profiles for future calls / cross-clip use
        for tid in changed:
            vec = appearances.get(tid)
            if vec is not None:
                self._referee_profiles.append(vec)
        for tid, label in assignments.items():
            if label is None and tid in appearances and appearances[tid] is not None:
                self._referee_profiles.append(appearances[tid])

        return assignments

    def load_referee_profiles(self, path: str) -> None:
        """Load persisted referee appearance profiles from a .npy file."""
        try:
            import numpy as np
            from pathlib import Path
            p = Path(path)
            if p.exists():
                data = np.load(str(p))
                if data.ndim == 2:
                    self._referee_profiles = [data[i] for i in range(len(data))]
        except Exception:
            pass

    def save_referee_profiles(self, path: str) -> None:
        """Persist referee appearance profiles to a .npy file for cross-clip reuse."""
        try:
            import numpy as np
            from pathlib import Path
            if not self._referee_profiles:
                return
            data = np.stack(self._referee_profiles)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            np.save(path, data)
        except Exception:
            pass

    def classify_by_cooccurrence(
        self,
        frame_snapshots: list[dict[int, float]],
        min_coappear: int = 10,
    ) -> dict[int, Optional[str]]:
        """
        Classify players into two teams using court-side co-occurrence.

        For every sampled frame, players on the LEFT half of the court (x_norm < 0.5)
        form one side group and those on the RIGHT form the other.  Over many frames,
        teammates will consistently appear on the SAME side together while opponents
        appear on OPPOSITE sides during transition moments.  The resulting per-player
        similarity profiles are clustered with k-means (k=2).

        frame_snapshots: list of {track_id: x_norm} dicts (one per sampled frame).
        min_coappear:    minimum co-appearance count to trust a similarity score.
        """
        try:
            import numpy as np
        except ImportError:
            # numpy unavailable → fall back to hue-based classify()
            return self.classify()

        all_tids = sorted({t for snap in frame_snapshots for t in snap})
        if len(all_tids) < 4:
            return {tid: "team_a" for tid in all_tids}

        n = len(all_tids)
        tid_idx = {tid: i for i, tid in enumerate(all_tids)}
        same_side   = np.zeros((n, n), dtype=np.int32)
        coappear    = np.zeros((n, n), dtype=np.int32)

        for snap in frame_snapshots:
            visible = [(tid_idx[t], x) for t, x in snap.items() if t in tid_idx]
            if len(visible) < 6:
                continue
            left  = [i for i, x in visible if x < 0.5]
            right = [i for i, x in visible if x >= 0.5]
            vis_i = [i for i, _ in visible]

            # Co-appearance (any pair visible together)
            for a in range(len(vis_i)):
                for b in range(a + 1, len(vis_i)):
                    ia, ib = vis_i[a], vis_i[b]
                    coappear[ia, ib] += 1
                    coappear[ib, ia] += 1

            # Same-side co-occurrence
            for group in (left, right):
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        ia, ib = group[a], group[b]
                        same_side[ia, ib] += 1
                        same_side[ib, ia] += 1

        # Similarity = fraction of co-appearances where both on same side
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.where(
                coappear >= min_coappear,
                same_side.astype(np.float32) / coappear,
                0.5,    # unknown → neutral
            )
        np.fill_diagonal(similarity, 1.0)

        if np.allclose(similarity, 0.5):
            return {tid: "team_a" for tid in all_tids}

        # K-means (k=2) on similarity rows.
        # Initialise from the two most DISSIMILAR players (minimum off-diagonal similarity).
        # Using first/last rows risks both centroids being from the same team.
        min_sim, ci, cj = np.inf, 0, n - 1
        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] < min_sim:
                    min_sim = similarity[i, j]
                    ci, cj = i, j
        c0 = similarity[ci].copy()
        c1 = similarity[cj].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(30):
            new_labels = np.argmin(
                [np.sum((similarity - c0) ** 2, axis=1),
                 np.sum((similarity - c1) ** 2, axis=1)],
                axis=0,
            )
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            if np.any(labels == 0):
                c0 = np.mean(similarity[labels == 0], axis=0)
            if np.any(labels == 1):
                c1 = np.mean(similarity[labels == 1], axis=0)

        # Tracks whose similarity row is almost entirely neutral (0.5) have too little
        # positional data to classify reliably.  Exclude them from the output so the
        # box-score stays balanced (they keep team="unknown" and get filtered later).
        result: dict[int, Optional[str]] = {}
        for tid in all_tids:
            row = similarity[tid_idx[tid]]
            neutral_frac = float(np.mean(np.abs(row - 0.5) < 0.01))
            # Use "unknown"/None when >90% of the row is still at the neutral 0.5 default
            if neutral_frac > 0.90:
                result[tid] = None   # treated as non-player (referee / unknown) by apply_team_assignments
            else:
                result[tid] = "team_a" if labels[tid_idx[tid]] == 0 else "team_b"
        return result

    def classify(self, min_samples: int = MIN_SAMPLES) -> dict[int, Optional[str]]:
        """
        Returns {track_id: 'team_a' | 'team_b' | None}.

        'team_a' and 'team_b' are arbitrary labels — consistent across the clip.
        None means the track is likely a referee, coach, or non-player staff.
        """
        try:
            import numpy as np
        except ImportError:
            return {tid: "team_a" for tid in self._samples}

        # Tracks with enough samples to trust their median hue.
        qualified = {
            tid: float(np.median(hues))
            for tid, hues in self._samples.items()
            if len(hues) >= min_samples
        }
        # Single-sample tracks: use raw value directly (no median needed).
        single = {
            tid: float(hues[0])
            for tid, hues in self._samples.items()
            if len(hues) == 1 and tid not in qualified
        }

        if not qualified:
            return {}

        tids  = list(qualified.keys())
        hues  = np.array([qualified[t] for t in tids], dtype=np.float32)

        if len(hues) == 1:
            # Extend with single-sample tracks using same label
            result = {tids[0]: "team_a"}
            for tid in single:
                result[tid] = "team_a"
            return result

        c0, c1 = _kmeans2(hues)

        # Adaptive outlier threshold: the larger of the fixed threshold or
        # OUTLIER_FRACTION × centroid separation. When teams wear clearly
        # different colors (separation ~100 hue units), this tightens the
        # filter to ~28 units, correctly rejecting coaches/refs in a third color
        # that would otherwise fall within 40 units of the nearer team centroid.
        separation   = _cdist(c0, c1)
        outlier_dist = max(self._outlier_dist, OUTLIER_FRACTION * separation)

        result: dict[int, Optional[str]] = {}
        for i, tid in enumerate(tids):
            h  = float(hues[i])
            da = _cdist(h, c0)
            db = _cdist(h, c1)
            if min(da, db) > outlier_dist:
                result[tid] = None
            elif da <= db:
                result[tid] = "team_a"
            else:
                result[tid] = "team_b"

        # Classify single-sample tracks by nearest centroid — no minimum required.
        for tid, h in single.items():
            da = _cdist(h, c0)
            db = _cdist(h, c1)
            if min(da, db) > outlier_dist:
                result[tid] = None
            elif da <= db:
                result[tid] = "team_a"
            else:
                result[tid] = "team_b"

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cdist(a: float, b: float, max_val: float = 180.0) -> float:
    """Circular distance between two hue values."""
    d = abs(a - b) % max_val
    return float(min(d, max_val - d))


def _cmean(hues) -> float:
    """Circular mean of hue values (numpy array)."""
    import numpy as np
    if len(hues) == 0:
        return 0.0
    angles = hues / 180.0 * 2.0 * np.pi
    mean_a = float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    return float(mean_a % (2.0 * np.pi) / (2.0 * np.pi) * 180.0)


def _kmeans2(hues, max_iter: int = 40):
    """
    Iterative 2-means on circular hue. Returns (centroid_0, centroid_1).
    Initialises from the two most-separated values on the hue circle.
    """
    import numpy as np
    s   = np.sort(hues)
    # Largest gap on the hue circle → split there for initial centroids
    gaps = np.diff(np.concatenate([s, [s[0] + 180.0]]))
    gi  = int(np.argmax(gaps))
    c0  = float(s[(gi + 1) % len(s)])
    c1  = float(s[gi])

    for _ in range(max_iter):
        d0 = np.array([_cdist(h, c0) for h in hues])
        d1 = np.array([_cdist(h, c1) for h in hues])
        labels = d1 < d0                            # True → cluster 1

        nc0 = _cmean(hues[~labels]) if (~labels).any() else c0
        nc1 = _cmean(hues[labels])  if  labels.any()  else c1

        if _cdist(nc0, c0) < 0.5 and _cdist(nc1, c1) < 0.5:
            break
        c0, c1 = nc0, nc1

    return c0, c1
