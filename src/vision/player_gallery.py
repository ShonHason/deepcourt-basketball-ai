"""
PlayerGallery — jersey-first player identity accumulation.

Identity priority (highest → lowest):
  1. Jersey number (OCR) — deterministic. #7 is always #7.
     Two players with different jersey numbers are ALWAYS different players
     even if they wear identical team colours and look the same to a visual model.
  2. Embedding similarity — fallback for frames where OCR hasn't fired yet.
     Used to link an unidentified crop to a known jersey entry.
  3. Pending entry — when neither OCR nor embedding matches. Accumulates
     until a jersey is read or the entry is retired as noise.

Why jersey-first matters:
  Low-resolution video + same-colour team kit = visual embeddings are nearly
  useless at separating teammates. Jersey numbers are the only reliable signal.

Algorithm per crop:
  a. OCR read jersey number?
       → Look up _jersey_to_gid[number].
       → Found: add embedding to that entry (confirmed identity).
       → Not found: create new entry keyed to this jersey number immediately.
  b. No OCR read:
       → Embedding search against all known entries (jersey-confirmed first).
       → Match found: add crop to that entry.
       → No match: create pending entry (UUID), keep accumulating.
  c. When pending entry accumulates JERSEY_CONFIRM_VOTES OCR reads of same number:
       → Promote: register in _jersey_to_gid, entry is now confirmed.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCH_THRESHOLD      = 0.72   # cosine sim threshold for embedding fallback
JERSEY_CONFIRM_VOTES = 2      # OCR reads of same number to confirm (lower = faster)
MAX_EMBEDDINGS       = 60     # cap embedding history per entry
EMBED_EVERY_N        = 4      # embed every N calls (skip frames to save compute)
CROP_MIN_H           = 28
CROP_MIN_W           = 12

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GalleryEntry:
    gid: str
    embeddings: list[np.ndarray]     = field(default_factory=list)
    mean_emb: Optional[np.ndarray]   = None
    jersey_votes: Counter            = field(default_factory=Counter)
    confirmed_jersey: Optional[str]  = None   # None = pending
    team: Optional[str]              = None
    track_ids: set[int]              = field(default_factory=set)
    crop_count: int                  = 0
    frame_first: int                 = 0
    frame_last: int                  = 0

    @property
    def confirmed(self) -> bool:
        return self.confirmed_jersey is not None

    @property
    def top_jersey_candidate(self) -> Optional[str]:
        """Best jersey candidate even if not yet confirmed."""
        top = self.jersey_votes.most_common(1)
        return top[0][0] if top else None

    @property
    def top_jersey_votes(self) -> int:
        top = self.jersey_votes.most_common(1)
        return top[0][1] if top else 0

    def add_embedding(self, emb: np.ndarray) -> None:
        self.embeddings.append(emb)
        if len(self.embeddings) > MAX_EMBEDDINGS:
            self.embeddings.pop(0)
        n = len(self.embeddings)
        self.mean_emb = emb.copy() if n == 1 else (self.mean_emb * (n - 1) + emb) / n
        self.crop_count += 1

    def vote_jersey(self, number: str) -> bool:
        """Vote for a jersey number. Returns True if this vote confirms the jersey."""
        if not number or number in ("?", ""):
            return False
        self.jersey_votes[number] += 1
        top, count = self.jersey_votes.most_common(1)[0]
        if count >= JERSEY_CONFIRM_VOTES and self.confirmed_jersey is None:
            self.confirmed_jersey = top
            return True
        return False


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

class PlayerGallery:
    """
    Real-time player identity gallery with jersey-number as primary key.

    Usage inside frame loop:
        gallery = PlayerGallery()
        for p in fd.players:
            entry = gallery.update(frame, p, frame_idx=idx)
            label = gallery.label(p.track_id)   # "#7", "#14?", "T42"

    After loop:
        confirmed = gallery.confirmed_players()
        roster    = {e.confirmed_jersey: e.team for e in confirmed}
    """

    def __init__(self) -> None:
        self._entries: dict[str, GalleryEntry]  = {}   # gid → entry
        self._jersey_to_gid: dict[str, str]     = {}   # jersey_number → gid
        self._track_to_gid: dict[int, str]      = {}   # track_id → gid (cache)
        self._model    = None
        self._transform = None
        self._call_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        player,           # PlayerDetection — needs .bbox, .track_id, .jersey_number
        frame_idx: int,
    ) -> Optional[GalleryEntry]:
        """
        Process one player in one frame. Returns the matched or created GalleryEntry.
        """
        self._call_count += 1
        skip_embed = (self._call_count % EMBED_EVERY_N != 0)

        ocr_jersey = _clean_jersey(getattr(player, "jersey_number", None))

        # ── Path 1: OCR read a jersey number ──────────────────────────
        if ocr_jersey:
            entry = self._get_or_create_by_jersey(ocr_jersey, frame_idx)
            entry.track_ids.add(player.track_id)
            entry.frame_last = frame_idx
            self._track_to_gid[player.track_id] = entry.gid
            if not skip_embed:
                emb = self._embed_crop(frame, player.bbox)
                if emb is not None:
                    entry.add_embedding(emb)
            return entry

        # ── Path 2: No OCR — try embedding similarity ──────────────────
        # First check if this track_id is already mapped
        cached_gid = self._track_to_gid.get(player.track_id)
        if cached_gid and cached_gid in self._entries:
            entry = self._entries[cached_gid]
            entry.frame_last = frame_idx
            if not skip_embed:
                emb = self._embed_crop(frame, player.bbox)
                if emb is not None:
                    entry.add_embedding(emb)
            return entry

        if skip_embed:
            return None

        emb = self._embed_crop(frame, player.bbox)
        if emb is None:
            return None

        # Search gallery — prefer jersey-confirmed entries
        match_gid = self._find_match(emb, prefer_confirmed=True)

        if match_gid:
            entry = self._entries[match_gid]
        else:
            # Case A — genuinely new, create pending entry
            gid   = f"p_{str(uuid.uuid4())[:8]}"
            entry = GalleryEntry(gid=gid, frame_first=frame_idx)
            self._entries[gid] = entry

        entry.add_embedding(emb)
        entry.frame_last = frame_idx
        entry.track_ids.add(player.track_id)
        self._track_to_gid[player.track_id] = entry.gid
        return entry

    def label(self, track_id: int) -> str:
        """
        Human-readable label for display:
          '#7'       — jersey confirmed by OCR
          '#14(2)'   — 2 OCR votes, not yet confirmed
          'T42'      — no OCR yet, track ID fallback
        """
        entry = self.resolve_track(track_id)
        if entry is None:
            return f"T{track_id}"
        if entry.confirmed:
            return f"#{entry.confirmed_jersey}"
        cand = entry.top_jersey_candidate
        if cand:
            return f"#{cand}({entry.top_jersey_votes})"
        return f"T{track_id}"

    def resolve_track(self, track_id: int) -> Optional[GalleryEntry]:
        gid = self._track_to_gid.get(track_id)
        return self._entries.get(gid) if gid else None

    def set_team(self, track_id: int, team: Optional[str]) -> None:
        entry = self.resolve_track(track_id)
        if entry and entry.team is None:
            entry.team = team

    def confirmed_players(self) -> list[GalleryEntry]:
        return [e for e in self._entries.values() if e.confirmed]

    def all_entries(self) -> list[GalleryEntry]:
        """All entries that have been seen at least 3 times (filter noise)."""
        return [e for e in self._entries.values() if e.crop_count >= 3]

    def roster_summary(self) -> dict[str, Optional[str]]:
        """Returns {jersey_number: team} for all confirmed players."""
        return {e.confirmed_jersey: e.team for e in self.confirmed_players()}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_by_jersey(self, jersey: str, frame_idx: int) -> GalleryEntry:
        """Look up or create an entry directly by jersey number."""
        if jersey in self._jersey_to_gid:
            gid = self._jersey_to_gid[jersey]
            if gid in self._entries:
                return self._entries[gid]

        # New confirmed-jersey entry
        gid   = f"j{jersey}_{str(uuid.uuid4())[:6]}"
        entry = GalleryEntry(
            gid=gid,
            confirmed_jersey=jersey,   # jersey-keyed entries are immediately confirmed
            frame_first=frame_idx,
        )
        entry.jersey_votes[jersey] = JERSEY_CONFIRM_VOTES  # pre-fill votes
        self._entries[gid]         = entry
        self._jersey_to_gid[jersey] = gid
        return entry

    def _find_match(self, emb: np.ndarray, prefer_confirmed: bool = True) -> Optional[str]:
        """Cosine similarity search. Confirmed (jersey-keyed) entries searched first."""
        best_gid = None
        best_sim = MATCH_THRESHOLD

        # Two-pass: confirmed entries first (better anchors), then pending
        order = sorted(
            self._entries.items(),
            key=lambda kv: (0 if kv[1].confirmed else 1),
        )
        for gid, entry in order:
            if entry.mean_emb is None:
                continue
            n_e = np.linalg.norm(entry.mean_emb)
            n_q = np.linalg.norm(emb)
            if n_e < 1e-8 or n_q < 1e-8:
                continue
            sim = float(np.dot(emb, entry.mean_emb) / (n_q * n_e))
            if sim > best_sim:
                best_sim = sim
                best_gid = gid
                if prefer_confirmed and entry.confirmed and sim > 0.85:
                    break   # strong confirmed match — stop early

        return best_gid

    def _embed_crop(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < CROP_MIN_H or crop.shape[1] < CROP_MIN_W:
            return None
        return self._embed(crop)

    def _embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        try:
            self._ensure_model()
            import torch, cv2
            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0)
            with torch.no_grad():
                feat = self._model(tensor)
                vec  = feat.mean(dim=[2, 3]).squeeze().cpu().numpy()
            n = np.linalg.norm(vec)
            return vec / (n + 1e-8)
        except Exception:
            return None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import ssl
        import torchvision.models as models
        import torchvision.transforms as T

        _orig = ssl._create_default_https_context  # type: ignore
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore
        try:
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        finally:
            ssl._create_default_https_context = _orig

        self._model = base.features
        self._model.eval()
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_jersey(raw: Optional[str]) -> Optional[str]:
    """Normalize OCR jersey string. Returns None if not a valid number."""
    if not raw:
        return None
    s = str(raw).strip().lstrip("#").strip()
    if not s or s == "?" or not s.isdigit():
        return None
    n = int(s)
    if n < 0 or n > 99:   # valid basketball jersey range
        return None
    return s
