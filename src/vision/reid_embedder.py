"""
ReidEmbedder — visual player re-identification using MobileNetV2 embeddings.

Problem it solves:
  ByteTrack assigns a new track ID every time it loses a player (camera cut,
  occlusion, angle change). The same physical player ends up with 2-5 different
  track IDs across the game. Their stats get split across "phantom" players.

Approach:
  1. Every EMBED_EVERY_N sampled frames, extract a 1280-dim appearance
     embedding from each player's bounding-box crop using MobileNetV2
     feature maps (no classifier head — pure appearance features).
  2. Only embed players whose FEET are on the court floor (y between
     COURT_Y_MIN_FRAC and COURT_Y_MAX_FRAC of the frame). Sideline
     coaches, spectators, and refs walking the perimeter are excluded.
  3. After the pipeline, compare mean embeddings across all track pairs
     that DON'T overlap in time (can't be the same person on court at once).
  4. Pairs above MERGE_SIM_THRESHOLD are returned as a merge map:
     {secondary_track_id: canonical_track_id}.

The pipeline applies the merge map to box_score stats and events, so
fragmented tracks are combined into one complete player record.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Court-floor region as fraction of frame height.
# Players above COURT_Y_MIN are in the bleachers; below COURT_Y_MAX are at
# the very edge of the camera frame (often out-of-bounds staff).
COURT_Y_MIN_FRAC = 0.18
COURT_Y_MAX_FRAC = 0.93

# Extract an embedding every N sampled frames — enough variation without
# being redundant (player pose changes every few frames).
EMBED_EVERY_N = 8

# Minimum embeddings collected before a track is considered for merging.
MIN_EMBEDDINGS = 5

# Cosine similarity threshold for declaring two tracks the same player.
# 0.72 balances recall (catching split tracks) against precision (false merges).
MERGE_SIM_THRESHOLD = 0.72

# Two tracks overlapping by more than this many ms cannot be the same player.
MAX_OVERLAP_MS = 1_500

# Crop resize for the embedding model (standard Re-ID dimensions)
CROP_H, CROP_W = 128, 64


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class ReidEmbedder:
    """
    Collect player appearance embeddings and detect fragmented tracks.

    Usage in pipeline:
        reid = ReidEmbedder(device=device)
        # inside frame loop:
        reid.sample(frame, fd.players, ts_ms, sampled_count)
        # after loop:
        merge_map = reid.get_merge_map()
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model  = None   # lazy-loaded on first sample
        self._transform = None

        # track_id → list of normalised embedding vectors
        self._embeddings: dict[int, list[np.ndarray]] = {}
        # track_id → (first_ts_ms, last_ts_ms) on court
        self._track_times: dict[int, tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def sample(
        self,
        frame: "np.ndarray",
        players: list,
        timestamp_ms: int,
        sampled_count: int,
    ) -> None:
        """
        Extract embeddings for court-area players in this frame.
        Call once per sampled frame during the main pipeline loop.
        """
        if sampled_count % EMBED_EVERY_N != 0:
            return

        try:
            self._ensure_model()
        except Exception:
            return

        h, w = frame.shape[:2]
        y_min = int(h * COURT_Y_MIN_FRAC)
        y_max = int(h * COURT_Y_MAX_FRAC)

        for p in players:
            if p.track_id is None:
                continue
            if getattr(p, "non_player", False):
                continue

            # Court-floor filter: only sample if feet are on the court
            _, fy = p.foot_center
            if not (y_min <= fy <= y_max):
                continue

            x1, y1, x2, y2 = [int(v) for v in p.bbox]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.shape[0] < 20 or crop.shape[1] < 10:
                continue

            emb = self._embed(crop)
            if emb is None:
                continue

            tid = p.track_id
            self._embeddings.setdefault(tid, []).append(emb)

            if tid not in self._track_times:
                self._track_times[tid] = (timestamp_ms, timestamp_ms)
            else:
                first_ts, _ = self._track_times[tid]
                self._track_times[tid] = (first_ts, timestamp_ms)

    def get_merge_map(self) -> dict[int, int]:
        """
        Compare all track pairs and return {secondary_id: canonical_id}
        for tracks that look like the same physical player.
        """
        mean_embs: dict[int, np.ndarray] = {}
        for tid, embs in self._embeddings.items():
            if len(embs) >= MIN_EMBEDDINGS:
                mean_embs[tid] = _normalize(np.mean(embs, axis=0))

        tids = list(mean_embs.keys())
        raw_merges: list[tuple[float, int, int]] = []

        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                t1, t2 = tids[i], tids[j]

                # Skip tracks that co-existed on court at the same time
                if self._overlap_ms(t1, t2) > MAX_OVERLAP_MS:
                    continue

                sim = float(mean_embs[t1] @ mean_embs[t2])
                if sim >= MERGE_SIM_THRESHOLD:
                    raw_merges.append((sim, t1, t2))

        # Sort by similarity descending — apply greedy merges
        raw_merges.sort(reverse=True)
        merge_map: dict[int, int] = {}

        for _sim, t1, t2 in raw_merges:
            c1 = self._canonical(t1, merge_map)
            c2 = self._canonical(t2, merge_map)
            if c1 == c2:
                continue
            # Merge the one with fewer embeddings into the one with more
            keep  = c1 if len(self._embeddings.get(c1, [])) >= len(self._embeddings.get(c2, [])) else c2
            other = c2 if keep == c1 else c1
            merge_map[other] = keep

        return merge_map

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import ssl
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        # Bypass SSL verification on systems with missing root certificates (e.g. macOS without
        # the Python Certificates.command having been run). Only affects the one-time download.
        _orig_ctx = ssl._create_default_https_context  # type: ignore[attr-defined]
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            base = models.mobilenet_v2(weights=weights)
        finally:
            ssl._create_default_https_context = _orig_ctx
        # Strip the classifier — keep only the feature extractor
        self._model = base.features
        self._model.eval()
        try:
            self._model = self._model.to(self._device)
        except Exception:
            self._model = self._model.to("cpu")
            self._device = "cpu"

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((CROP_H, CROP_W)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _embed(self, crop: "np.ndarray") -> Optional[np.ndarray]:
        try:
            import torch
            import cv2
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model(tensor)
                # Global average pool → flat vector
                vec = feat.mean(dim=[2, 3]).squeeze().cpu().numpy()
            return _normalize(vec)
        except Exception:
            return None

    def _overlap_ms(self, t1: int, t2: int) -> int:
        t1_range = self._track_times.get(t1)
        t2_range = self._track_times.get(t2)
        if t1_range is None or t2_range is None:
            return 0
        t1_first, t1_last = t1_range
        t2_first, t2_last = t2_range
        overlap = min(t1_last, t2_last) - max(t1_first, t2_first)
        return max(0, overlap)

    @staticmethod
    def _canonical(tid: int, merge_map: dict[int, int]) -> int:
        while tid in merge_map:
            tid = merge_map[tid]
        return tid


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)
