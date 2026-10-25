"""
Event Validator — applies basketball rules to raw CV primitives and emits
validated stat events to StatTracker.

Sits between the vision pipeline (detector / tracker) and StatTracker.
The vision layer streams primitives; this module decides what they *mean*.

───────────────────────────────────────────
Raw primitive shapes (all require 'primitive' key):

  pass             — ball left passer's hands toward receiver
  dribble          — single dribble detected for a player
  shot_taken       — player released a shot (outcome not yet known)
  shot_outcome     — ball fate determined (made / missed, sub_type)
  free_throw       — free throw attempt with known outcome
  hand_contact     — defender hand within proximity threshold of ball carrier
  possession_change — ball changed hands between teams
  rebound          — player gained possession after a missed shot

Validated events emitted to StatTracker:
  two_pointer_made / missed
  three_pointer_made / missed
  free_throw_made / missed
  assist
  steal
  turnover_steal
  turnover_unforced
  rebound_offensive
  rebound_defensive
───────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.logic.stat_tracker import StatTracker

# ---------------------------------------------------------------------------
# Rule constants
# ---------------------------------------------------------------------------

ASSIST_WINDOW_MS     = 2_000  # possessor → made shot must occur within this window
                               # 2 s is tight enough to filter rebounds-then-score
                               # but still catches quick pass-and-shoot sequences
ASSIST_MAX_DRIBBLES  = 1      # scorer takes ≤ this many dribbles for an assist
CONTACT_WINDOW_MS    = 1_000  # hand contact must precede possession_change within this
MIN_POSSESSION_MS    = 3_500  # player must hold ball ≥ this long before loss = turnover
                               # shorter transfers are passes, not turnovers
                               # 3.5 s filters out quick passes (0-2 s) while keeping
                               # real turnovers where a player holds under pressure
THREE_POINT_RADIUS   = 6.75   # metres from hoop centre (FIBA arc; NBA is 7.24 m)
# Free throw line is 4.57 m from each baseline; accept ±1.2 m tolerance
FT_DIST_FROM_HOOP_MIN = 3.4
FT_DIST_FROM_HOOP_MAX = 5.8
FT_COURT_WIDTH_TOL    = 2.0   # metres from court centre-line (y = 7.5 m)

# ---------------------------------------------------------------------------
# Internal state records
# ---------------------------------------------------------------------------

@dataclass
class _PendingPass:
    passer_id: str
    passer_name: str
    passer_team: str
    receiver_id: str
    timestamp_ms: int


@dataclass
class _PendingShot:
    player_id: str
    player_name: str
    team: str
    coordinates: dict
    contested: bool
    mismatch: bool
    timestamp_ms: int
    is_three: bool
    reason: str
    prev_possessor_id: Optional[str] = None
    prev_possessor_name: Optional[str] = None
    prev_possessor_ts: int = 0
    potential_blocker_id: Optional[str] = None
    potential_blocker_name: Optional[str] = None


@dataclass
class _ContactRecord:
    defender_id: str
    defender_name: str
    defender_team: str
    ball_holder_id: str
    timestamp_ms: int


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class EventValidator:
    """
    Stateful rules engine. Feed it raw CV primitives via push(); receive
    validated stat event dicts that have already been forwarded to StatTracker.

    Usage:
        tracker   = StatTracker()
        validator = EventValidator(tracker)
        for prim in vision_stream:
            emitted = validator.push(prim)   # list[dict], already tracked
    """

    def __init__(self, tracker: StatTracker, auto_track: bool = True) -> None:
        self._tracker = tracker
        self._auto_track = auto_track  # False in pipeline (game_sm filters before tracking)
        self._pending_passes: list[_PendingPass] = []
        self._dribble_counts: dict[str, int] = {}     # player_id → dribbles since last pass
        self._contact_buffer: list[_ContactRecord] = []
        self._pending_shot: Optional[_PendingShot] = None
        self._last_shot_team: Optional[str] = None    # for rebound side classification
        self._players: dict[str, dict] = {}           # player_id → {name, team}
        # Possession history for assist detection — updated on every possession change
        self._possessor_log: list[tuple[str, str, int]] = []  # (player_id, name, ts_ms)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def push(self, primitive: dict) -> list[dict]:
        """
        Ingest one raw primitive. Returns a list of zero or more validated
        stat events emitted this tick (already forwarded to StatTracker).
        """
        self._register_player(primitive)

        handlers = {
            "pass":              self._on_pass,
            "dribble":           self._on_dribble,
            "shot_taken":        self._on_shot_taken,
            "shot_outcome":      self._on_shot_outcome,
            "free_throw":        self._on_free_throw,
            "hand_contact":      self._on_hand_contact,
            "possession_change": self._on_possession_change,
            "rebound":           self._on_rebound,
        }
        handler = handlers.get(primitive.get("primitive", ""))
        return handler(primitive) if handler else []

    # ------------------------------------------------------------------
    # Primitive handlers
    # ------------------------------------------------------------------

    def _on_pass(self, p: dict) -> list[dict]:
        ts = p["timestamp_ms"]
        self._expire_passes(ts)
        self._pending_passes.append(_PendingPass(
            passer_id=p["passer_id"],
            passer_name=p.get("passer_name", p["passer_id"]),
            passer_team=p.get("passer_team", "unknown"),
            receiver_id=p["receiver_id"],
            timestamp_ms=ts,
        ))
        self._dribble_counts[p["receiver_id"]] = 0
        return []

    def _on_dribble(self, p: dict) -> list[dict]:
        pid = p["player_id"]
        self._dribble_counts[pid] = self._dribble_counts.get(pid, 0) + 1
        return []

    def _on_shot_taken(self, p: dict) -> list[dict]:
        coords = p.get("coordinates", {"x": 0.0, "y": 0.0})
        prev_id   = str(p["prev_possessor_id"])   if p.get("prev_possessor_id")   is not None else None
        prev_name = str(p["prev_possessor_name"]) if p.get("prev_possessor_name") is not None else None
        prev_ts   = int(p.get("prev_possessor_ts", 0))
        self._pending_shot = _PendingShot(
            player_id=p["player_id"],
            player_name=p.get("player_name", p["player_id"]),
            team=p.get("team", "unknown"),
            coordinates=coords,
            contested=bool(p.get("contested", False)),
            mismatch=bool(p.get("mismatch", False)),
            timestamp_ms=p["timestamp_ms"],
            is_three=self._is_three(coords),
            reason=p.get("reason", ""),
            prev_possessor_id=prev_id,
            prev_possessor_name=prev_name,
            prev_possessor_ts=prev_ts,
        )
        return []

    def _on_shot_outcome(self, p: dict) -> list[dict]:
        if self._pending_shot is None:
            return []

        shot = self._pending_shot
        self._pending_shot = None
        made: bool = p.get("made", False)
        sub_type: Optional[str] = p.get("sub_type")
        ts = p.get("timestamp_ms", shot.timestamp_ms)

        reason = p.get("reason") or shot.reason or self._shot_reason(shot, made, sub_type)

        # Free throw: shot from the free throw line area → 1 pt
        if self._is_free_throw(shot.coordinates):
            suffix = "made" if made else "missed"
            event_type = f"free_throw_{suffix}"
        else:
            suffix = "made" if made else "missed"
            event_type = ("three_pointer" if shot.is_three else "two_pointer") + "_" + suffix

        shot_event = self._emit({
            "event_type": event_type,
            "player_id": shot.player_id,
            "player_name": shot.player_name,
            "team": shot.team,
            "timestamp_ms": ts,
            "frame_start": p.get("frame_start", 0),
            "coordinates": shot.coordinates,
            "sub_type": sub_type,
            "contested": shot.contested,
            "mismatch": shot.mismatch,
            "reason": reason,
            "confidence": p.get("confidence", 1.0),
        })

        self._last_shot_team = shot.team
        emitted = [shot_event]

        if made:
            # Assist: previous possessor directly attached to this shot_taken primitive
            # takes priority over the slower possessor-log lookup.
            assist = None
            if (shot.prev_possessor_id
                    and shot.prev_possessor_id != str(shot.player_id)
                    and ts - shot.prev_possessor_ts <= ASSIST_WINDOW_MS):
                assist = {
                    "event_type":  "assist",
                    "player_id":   shot.prev_possessor_id,
                    "player_name": shot.prev_possessor_name or f"#{shot.prev_possessor_id}",
                    "team":        self._players.get(shot.prev_possessor_id, {}).get("team", "unknown"),
                    "timestamp_ms": shot.timestamp_ms,
                    "frame_start":  0,
                    "coordinates":  shot.coordinates,
                    "reason": (
                        f"Last possessor before scorer — direct pass-to-bucket"
                    ),
                }
            if not assist:
                # Only use the pass-log path — the possession-log fallback fires
                # for any player who merely touched the ball (rebound, loose ball)
                # before a score, which generates too many false assists.
                assist = self._resolve_assist(shot.player_id, shot.team, ts)
            if assist:
                emitted.append(self._emit(assist))
        elif not made and shot.potential_blocker_id:
            # Block: a defender touched the ball during the shot and it missed
            emitted.append(self._emit({
                "event_type":  "block",
                "player_id":   shot.potential_blocker_id,
                "player_name": shot.potential_blocker_name or f"#{shot.potential_blocker_id}",
                "team":        self._players.get(shot.potential_blocker_id, {}).get("team", "unknown"),
                "timestamp_ms": ts,
                "frame_start":  0,
                "coordinates":  shot.coordinates,
                "reason": "Hand contact with ball during shot — shot missed",
            }))

        self._dribble_counts.pop(shot.player_id, None)
        self._expire_passes(ts)
        return emitted

    def _on_free_throw(self, p: dict) -> list[dict]:
        made = p.get("made", False)
        event_type = "free_throw_made" if made else "free_throw_missed"
        return [self._emit({
            "event_type": event_type,
            "player_id": p["player_id"],
            "player_name": p.get("player_name", p["player_id"]),
            "team": p.get("team", "unknown"),
            "timestamp_ms": p["timestamp_ms"],
            "frame_start": p.get("frame_start", 0),
            "coordinates": p.get("coordinates", {"x": 0.0, "y": 7.5}),
            "reason": p.get("reason", "Free throw attempt"),
        })]

    def _on_hand_contact(self, p: dict) -> list[dict]:
        self._contact_buffer.append(_ContactRecord(
            defender_id=p["defender_id"],
            defender_name=p.get("defender_name", p["defender_id"]),
            defender_team=p.get("defender_team", "unknown"),
            ball_holder_id=p["ball_holder_id"],
            timestamp_ms=p["timestamp_ms"],
        ))
        # If a shot is in flight and the toucher is NOT the shooter, record as potential block
        if (self._pending_shot is not None
                and str(p["defender_id"]) != str(self._pending_shot.player_id)):
            self._pending_shot.potential_blocker_id   = str(p["defender_id"])
            self._pending_shot.potential_blocker_name = p.get("defender_name")
        return []

    def _on_possession_change(self, p: dict) -> list[dict]:
        ts = p["timestamp_ms"]
        from_id   = p.get("from_player_id", "unknown")
        from_team = p.get("from_team", "unknown")
        new_holder_id = p.get("new_holder_id", p.get("player_id", "unknown"))
        new_holder_name = p.get("player_name", f"#{new_holder_id}")
        coords = p.get("coordinates", {"x": 0.0, "y": 0.0})

        # Always record the new holder for assist detection, regardless of turnover logic
        if new_holder_id and new_holder_id != "unknown":
            self._possessor_log.append((str(new_holder_id), new_holder_name, ts))
            self._possessor_log = [
                (pid, name, t) for pid, name, t in self._possessor_log
                if ts - t <= ASSIST_WINDOW_MS
            ]

        # No prior possessor → not a real turnover
        if from_id == "unknown" or from_id is None:
            return []

        self._expire_contacts(ts)
        contact = self._find_contact(from_id, new_holder_id)

        # Use name directly from primitive; fall back to player registry
        from_name = (
            p.get("from_player_name")
            or self._players.get(from_id, {}).get("name")
            or f"#{from_id}"
        )
        emitted = []

        if contact:
            # Confirmed steal: defender hand was on the ball before possession changed
            emitted.append(self._emit({
                "event_type": "turnover_steal",
                "player_id": from_id,
                "player_name": from_name,
                "team": from_team,
                "timestamp_ms": ts,
                "frame_start": p.get("frame_start", 0),
                "coordinates": coords,
                "reason": f"Ball stripped by {contact.defender_name} — hand contact detected",
            }))
            emitted.append(self._emit({
                "event_type": "steal",
                "player_id": contact.defender_id,
                "player_name": contact.defender_name,
                "team": contact.defender_team,
                "timestamp_ms": ts,
                "frame_start": p.get("frame_start", 0),
                "coordinates": coords,
                "reason": (
                    f"Active hands — stripped {from_name} "
                    f"within {CONTACT_WINDOW_MS}ms contact window"
                ),
            }))
            self._contact_buffer.clear()
            self._dribble_counts.pop(from_id, None)
        # No contact → check if the player held the ball long enough to count as a turnover.
        # Quick transfers (< MIN_POSSESSION_MS) are passes, not turnovers — we can't
        # distinguish teammate passes from opponent steals by position alone, so we
        # use hold duration as the proxy: real turnovers happen after a sustained possession.
        elif p.get("held_ms", MIN_POSSESSION_MS) >= MIN_POSSESSION_MS:
            emitted.append(self._emit({
                "event_type": "turnover_unforced",
                "player_id": from_id,
                "player_name": from_name,
                "team": from_team,
                "timestamp_ms": ts,
                "frame_start": p.get("frame_start", 0),
                "coordinates": coords,
                "confidence": p.get("confidence", 1.0),
                "reason": p.get("reason", "Possession changed without defender contact"),
            }))
            self._dribble_counts.pop(from_id, None)

        return emitted

    def _on_rebound(self, p: dict) -> list[dict]:
        player_team = p.get("player_team", "unknown")
        # Only mark as offensive if both teams are known and match.
        # When either side is "unknown", default to defensive (more common, less misleading).
        known_teams = player_team not in ("unknown", None) and self._last_shot_team not in ("unknown", None)
        event_type = (
            "rebound_offensive"
            if known_teams and player_team == self._last_shot_team
            else "rebound_defensive"
        )
        return [self._emit({
            "event_type": event_type,
            "player_id": p["player_id"],
            "player_name": p.get("player_name", p["player_id"]),
            "team": player_team,
            "timestamp_ms": p["timestamp_ms"],
            "frame_start": p.get("frame_start", 0),
            "coordinates": p.get("coordinates", {"x": 0.0, "y": 0.0}),
            "confidence": p.get("confidence", 1.0),
            "reason": (
                "Same team regained possession after missed shot — offensive board"
                if event_type == "rebound_offensive"
                else "Opposing team secured possession after missed shot — defensive board"
            ),
        })]

    # ------------------------------------------------------------------
    # Assist resolution
    # ------------------------------------------------------------------

    def _resolve_assist(
        self, scorer_id: str, scorer_team: str, shot_ts: int
    ) -> Optional[dict]:
        """Return an assist event dict if a valid pass-to-score sequence exists."""
        for pending in reversed(self._pending_passes):
            if pending.receiver_id != scorer_id:
                continue
            if str(pending.passer_id) == str(scorer_id):
                continue  # self-pass — not an assist
            if pending.passer_team != scorer_team:
                continue
            gap_ms = shot_ts - pending.timestamp_ms
            if gap_ms > ASSIST_WINDOW_MS:
                continue
            dribbles = self._dribble_counts.get(scorer_id, 0)
            if dribbles > ASSIST_MAX_DRIBBLES:
                continue
            return {
                "event_type": "assist",
                "player_id": pending.passer_id,
                "player_name": pending.passer_name,
                "team": pending.passer_team,
                "timestamp_ms": pending.timestamp_ms,
                "frame_start": 0,
                "coordinates": {"x": 0.0, "y": 0.0},
                "reason": (
                    f"Pass-to-score gap = {gap_ms}ms, "
                    f"dribble count = {dribbles} — direct cause of bucket"
                ),
            }
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, raw: dict) -> dict:
        if self._auto_track:
            self._tracker.track(raw)
        return raw

    def _expire_passes(self, current_ts: int) -> None:
        self._pending_passes = [
            p for p in self._pending_passes
            if current_ts - p.timestamp_ms <= ASSIST_WINDOW_MS
        ]

    def _expire_contacts(self, current_ts: int) -> None:
        self._contact_buffer = [
            c for c in self._contact_buffer
            if current_ts - c.timestamp_ms <= CONTACT_WINDOW_MS
        ]

    def _find_contact(
        self, ball_holder_id: str, new_holder_id: str
    ) -> Optional[_ContactRecord]:
        for c in reversed(self._contact_buffer):
            if c.ball_holder_id == ball_holder_id and c.defender_id == new_holder_id:
                return c
        return None

    def _register_player(self, p: dict) -> None:
        pairs = [
            ("player_id",      "player_name",      "team"),
            ("passer_id",      "passer_name",       "passer_team"),
            ("defender_id",    "defender_name",     "defender_team"),
            ("ball_holder_id", None,                "team"),
            ("from_player_id", "from_player_name",  "from_team"),
            ("new_holder_id",  "player_name",       "to_team"),
        ]
        for id_key, name_key, team_key in pairs:
            pid = p.get(id_key)
            if not pid or pid == "unknown":
                continue
            if pid not in self._players:
                name = p.get(name_key, str(pid)) if name_key else str(pid)
                if not name or name == "None":
                    name = f"#{pid}"
                self._players[pid] = {
                    "name": name,
                    "team": p.get(team_key, "unknown"),
                }

    def _resolve_assist_from_possession(
        self, scorer_id: str, shot_ts: int
    ) -> Optional[dict]:
        """
        Scan recent possession log for a player other than the scorer who had
        the ball within a tight window before the made shot.

        Uses a shorter window than ASSIST_WINDOW_MS (2 s vs 4 s) because this
        fallback has no dribble-count evidence — a wider window would attribute
        assists to players who simply had the ball earlier in the possession.
        """
        FALLBACK_ASSIST_MS = 2_000
        scorer_str = str(scorer_id)
        for pid, pname, t in reversed(self._possessor_log):
            if pid == scorer_str:
                continue
            if self._players.get(pid, {}).get("team", "unknown") == "unknown":
                continue  # skip unresolved players to avoid ghost assists
            gap = shot_ts - t
            if gap > FALLBACK_ASSIST_MS:
                break
            # Only fire if the scorer had ZERO possession changes between
            # this passer and the basket (i.e., the log entry immediately
            # before the scorer's own entry).
            return {
                "event_type": "assist",
                "player_id":  pid,
                "player_name": pname,
                "team":       self._players.get(pid, {}).get("team", "unknown"),
                "timestamp_ms": t,
                "frame_start":  0,
                "coordinates":  {"x": 0.0, "y": 7.5},
                "reason": (
                    f"Had possession {gap}ms before basket — direct cause of bucket"
                ),
            }
        return None

    def _is_free_throw(self, coords: dict) -> bool:
        """True when shot coordinates match the free throw line area of either hoop."""
        x = coords.get("x", 0.0)
        y = coords.get("y", 7.5)
        if abs(y - 7.5) > FT_COURT_WIDTH_TOL:
            return False
        for hoop_x in (0.0, 28.0):
            d = abs(x - hoop_x)
            if FT_DIST_FROM_HOOP_MIN <= d <= FT_DIST_FROM_HOOP_MAX:
                return True
        return False

    def _is_three(self, coords: dict) -> bool:
        x = coords.get("x", 0.0)
        y = coords.get("y", 7.5)
        dist_a = ((x - 0.0) ** 2 + (y - 7.5) ** 2) ** 0.5
        dist_b = ((x - 28.0) ** 2 + (y - 7.5) ** 2) ** 0.5
        return min(dist_a, dist_b) > THREE_POINT_RADIUS

    @staticmethod
    def _shot_reason(shot: _PendingShot, made: bool, sub_type: Optional[str]) -> str:
        outcome = "made" if made else "missed"
        contested = "contested" if shot.contested else "uncontested"
        sub = f" ({sub_type})" if sub_type else ""
        return f"Shot {outcome}{sub} — {contested}, zone from coordinates"
