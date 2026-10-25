"""
PlayerIdentityBank — centralised player identity store.

Wraps the existing ReidEmbedder + Tracker._jersey_cache into a single
source of truth for player identity across a game. Uses numpy cosine
similarity (no FAISS dependency).

Usage in pipeline.py, after Re-ID merge (step 3b):
    from src.vision.identity_bank import PlayerIdentityBank
    identity_bank = PlayerIdentityBank()
    identity_bank.ingest_reid_embeddings(reid)
    identity_bank.ingest_jersey_ocr(tracker._jersey_cache)
    identity_bank.apply_reid_merges(merge_map)

    for ev in stat_tracker.events:
        identity = identity_bank.resolve_track(int(ev.player_id))
        if identity and identity.jersey_number:
            ev.player_name = f"#{identity.jersey_number}"
        if identity and identity.team:
            ev.team = identity.team
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


MATCH_THRESHOLD = 0.72   # cosine similarity — same as ReidEmbedder.MERGE_SIM_THRESHOLD


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0 on zero vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class PlayerIdentity:
    uid: str
    track_ids: set[int] = field(default_factory=set)
    jersey_number: Optional[str] = None
    team: Optional[str] = None
    embedding: Optional[np.ndarray] = None   # rolling mean (N-dim)
    embedding_count: int = 0
    confidence: float = 0.0                   # 0–1; identity certainty
    last_seen_ms: int = 0

    def update_embedding(self, new_emb: np.ndarray) -> None:
        """Incrementally update the rolling mean embedding."""
        if self.embedding is None:
            self.embedding = new_emb.copy()
            self.embedding_count = 1
        else:
            n = self.embedding_count
            self.embedding = (self.embedding * n + new_emb) / (n + 1)
            self.embedding_count += 1
        self.confidence = min(1.0, self.embedding_count / 10.0)


class PlayerIdentityBank:
    """
    Centralised identity store that merges ReID embeddings and OCR jersey numbers
    into canonical PlayerIdentity objects.
    """

    def __init__(self) -> None:
        self._identities: dict[str, PlayerIdentity] = {}   # uid → PlayerIdentity
        self._track_to_uid: dict[int, str] = {}            # track_id → uid

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_reid_embeddings(self, reid) -> None:
        """Pull all embeddings from ReidEmbedder._embeddings into the bank."""
        embeddings: dict = getattr(reid, "_embeddings", {})
        for track_id, emb_list in embeddings.items():
            tid = int(track_id)
            if not emb_list:
                continue
            mean_emb = np.mean(np.array(emb_list), axis=0)
            uid = self._track_to_uid.get(tid)
            if uid is None:
                # New identity — check if it matches an existing one
                uid = self._find_match(mean_emb)
                if uid is None:
                    uid = f"player_{str(uuid.uuid4())[:8]}"
                    self._identities[uid] = PlayerIdentity(uid=uid)
                self._track_to_uid[tid] = uid
                self._identities[uid].track_ids.add(tid)
            self._identities[uid].update_embedding(mean_emb)

    def ingest_jersey_ocr(self, jersey_cache: dict) -> None:
        """Pull confirmed jersey numbers from Tracker._jersey_cache."""
        for track_id, jersey_read in jersey_cache.items():
            tid = int(track_id)
            number = str(getattr(jersey_read, "number", jersey_read))
            uid = self._track_to_uid.get(tid)
            if uid is None:
                uid = f"player_{str(uuid.uuid4())[:8]}"
                self._identities[uid] = PlayerIdentity(uid=uid)
                self._track_to_uid[tid] = uid
                self._identities[uid].track_ids.add(tid)
            identity = self._identities[uid]
            if identity.jersey_number is None:
                identity.jersey_number = number
                identity.confidence = min(1.0, identity.confidence + 0.3)

    def apply_reid_merges(self, merge_map: dict[int, int]) -> None:
        """
        Apply the merge_map produced by ReidEmbedder.get_merge_map().
        merge_map: {secondary_track_id → canonical_track_id}
        """
        for secondary, canonical in merge_map.items():
            sec_uid = self._track_to_uid.get(int(secondary))
            can_uid = self._track_to_uid.get(int(canonical))
            if sec_uid is None or can_uid is None or sec_uid == can_uid:
                continue
            sec_id = self._identities.get(sec_uid)
            can_id = self._identities.get(can_uid)
            if sec_id is None or can_id is None:
                continue
            # Merge secondary into canonical
            for tid in sec_id.track_ids:
                self._track_to_uid[tid] = can_uid
                can_id.track_ids.add(tid)
            if sec_id.jersey_number and not can_id.jersey_number:
                can_id.jersey_number = sec_id.jersey_number
            if sec_id.team and not can_id.team:
                can_id.team = sec_id.team
            if sec_id.embedding is not None:
                can_id.update_embedding(sec_id.embedding)
            del self._identities[sec_uid]

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_track(self, track_id: int) -> Optional[PlayerIdentity]:
        """Return canonical PlayerIdentity for a track_id, or None."""
        uid = self._track_to_uid.get(track_id)
        return self._identities.get(uid) if uid else None

    def get_all(self) -> list[PlayerIdentity]:
        """All identities with a jersey number or sufficient embedding count."""
        return [
            p for p in self._identities.values()
            if p.jersey_number is not None or p.embedding_count >= 5
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_match(self, emb: np.ndarray) -> Optional[str]:
        """Return uid of an existing identity with cosine sim ≥ MATCH_THRESHOLD."""
        best_uid: Optional[str] = None
        best_sim = MATCH_THRESHOLD
        for uid, identity in self._identities.items():
            if identity.embedding is None:
                continue
            sim = _cosine_sim(emb, identity.embedding)
            if sim > best_sim:
                best_sim = sim
                best_uid = uid
        return best_uid
