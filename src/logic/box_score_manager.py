"""
Box Score Manager — accumulates all per-player and per-team statistics
from structured event dicts and computes shooting percentages + advanced metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlayerStats:
    player_id: str
    player_name: str
    team: str

    # Scoring
    pts: int = 0
    fg_made: int = 0
    fg_att: int = 0
    three_made: int = 0
    three_att: int = 0
    ft_made: int = 0
    ft_att: int = 0

    # Shot sub-type counts
    swish: int = 0
    bank_shot: int = 0
    rim_in: int = 0

    # Possession
    rebounds_off: int = 0
    rebounds_def: int = 0
    assists: int = 0
    turnovers_steal: int = 0     # turnover caused by a steal (defender contact)
    turnovers_unforced: int = 0  # out-of-bounds, errant pass with no defender

    # Advanced
    contested_shots: int = 0
    uncontested_shots: int = 0
    mismatch_shots: int = 0      # shot taken vs. defender ≥ 4 inches shorter

    # Defensive
    steals: int = 0
    blocks: int = 0

    # Court status
    on_court:    bool = False   # True = on court at end of clip; False = sub
    non_player:  bool = False   # True = referee / coach / staff → excluded from box score
    frames_seen: int  = 0       # sampled frames player was visible — used to filter phantoms
    jersey_ocr:  bool = False   # True = player_name came from EasyOCR jersey read

    # -----------------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------------

    @property
    def fg_pct(self) -> Optional[float]:
        return round(self.fg_made / self.fg_att, 3) if self.fg_att else None

    @property
    def three_pct(self) -> Optional[float]:
        return round(self.three_made / self.three_att, 3) if self.three_att else None

    @property
    def ft_pct(self) -> Optional[float]:
        return round(self.ft_made / self.ft_att, 3) if self.ft_att else None

    @property
    def rebounds_total(self) -> int:
        return self.rebounds_off + self.rebounds_def

    @property
    def turnovers_total(self) -> int:
        return self.turnovers_steal + self.turnovers_unforced

    @property
    def contested_pct(self) -> Optional[float]:
        total = self.contested_shots + self.uncontested_shots
        return round(self.contested_shots / total, 3) if total else None

    @property
    def mismatch_rate(self) -> Optional[float]:
        total = self.fg_att + self.three_att
        return round(self.mismatch_shots / total, 3) if total else None

    @property
    def effective_fg_pct(self) -> Optional[float]:
        """eFG% = (FGM + 0.5 * 3PM) / FGA"""
        total_att = self.fg_att + self.three_att
        if not total_att:
            return None
        return round((self.fg_made + self.three_made + 0.5 * self.three_made) / total_att, 3)

    def to_dict(self) -> dict:
        return {
            "player_id":   self.player_id,
            "player_name": self.player_name,
            "team":        self.team,
            "team_id":     self.team,       # alias so dashboard grouping works
            "pts":         self.pts,
            # Scoring — both short and long names so dashboard can read either
            "fg_made":     self.fg_made,
            "fg_att":      self.fg_att,
            "fg":          f"{self.fg_made}/{self.fg_att}",
            "fg_pct":      self.fg_pct,
            "three_made":  self.three_made,
            "three_att":   self.three_att,
            "3p":          f"{self.three_made}/{self.three_att}",
            "three_pct":   self.three_pct,
            "3p_pct":      self.three_pct,
            "ft_made":     self.ft_made,
            "ft_att":      self.ft_att,
            "ft":          f"{self.ft_made}/{self.ft_att}",
            "ft_pct":      self.ft_pct,
            "effective_fg_pct": self.effective_fg_pct,
            "efg_pct":     self.effective_fg_pct,
            # Possession — both short and long names
            "rebounds_off":   self.rebounds_off,
            "reb_off":        self.rebounds_off,
            "rebounds_def":   self.rebounds_def,
            "reb_def":        self.rebounds_def,
            "rebounds_total": self.rebounds_total,
            "reb_total":      self.rebounds_total,
            "assists":        self.assists,
            "ast":            self.assists,
            "turnovers":      self.turnovers_total,
            "to":             self.turnovers_total,
            "to_steal":       self.turnovers_steal,
            "to_unforced":    self.turnovers_unforced,
            "steals":         self.steals,
            "stl":            self.steals,
            "blocks":         self.blocks,
            "blk":            self.blocks,
            # Advanced
            "shot_types":        {"swish": self.swish, "bank_shot": self.bank_shot, "rim_in": self.rim_in},
            "contested_shots":   self.contested_shots,
            "uncontested_shots": self.uncontested_shots,
            "contested_pct":     self.contested_pct,
            "mismatch_shots":    self.mismatch_shots,
            "mismatch_rate":     self.mismatch_rate,
            # Court status & time
            "on_court":          self.on_court,
            "frames_seen":       self.frames_seen,
            # minutes_played = frames / (3 fps * 60 s) — approximation from sampling rate
            "minutes_played":    round(self.frames_seen / 180, 1),
        }


@dataclass
class TeamStats:
    team_id: str

    pts: int = 0
    fg_made: int = 0
    fg_att: int = 0
    three_made: int = 0
    three_att: int = 0
    ft_made: int = 0
    ft_att: int = 0
    rebounds_off: int = 0
    rebounds_def: int = 0
    assists: int = 0
    turnovers: int = 0
    steals: int = 0
    blocks: int = 0

    @property
    def fg_pct(self) -> Optional[float]:
        return round(self.fg_made / self.fg_att, 3) if self.fg_att else None

    @property
    def three_pct(self) -> Optional[float]:
        return round(self.three_made / self.three_att, 3) if self.three_att else None

    @property
    def ft_pct(self) -> Optional[float]:
        return round(self.ft_made / self.ft_att, 3) if self.ft_att else None

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "pts": self.pts,
            "fg": f"{self.fg_made}/{self.fg_att}",
            "fg_pct": self.fg_pct,
            "3p": f"{self.three_made}/{self.three_att}",
            "3p_pct": self.three_pct,
            "ft": f"{self.ft_made}/{self.ft_att}",
            "ft_pct": self.ft_pct,
            "reb_off": self.rebounds_off,
            "reb_def": self.rebounds_def,
            "reb_total": self.rebounds_off + self.rebounds_def,
            "ast": self.assists,
            "to": self.turnovers,
            "stl": self.steals,
            "blk": self.blocks,
        }


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class BoxScoreManager:
    """
    Consumes structured event dicts and maintains the live box score state.

    Expected event dict shape (all fields optional except 'event'):
    {
        "event":       str,          # see VALID_EVENTS
        "player_id":   str,
        "player_name": str,
        "team":        str,
        "sub_type":    str,          # "swish" | "bank_shot" | "rim_in"
        "contested":   bool,
        "mismatch":    bool,
        "reason":      str,
    }
    """

    SCORING_EVENTS = {
        "free_throw_made", "free_throw_missed",
        "two_pointer_made", "two_pointer_missed",
        "three_pointer_made", "three_pointer_missed",
    }

    POSSESSION_EVENTS = {
        "rebound_offensive", "rebound_defensive",
        "assist",
        "turnover_steal", "turnover_unforced",
        "steal", "block",
    }

    VALID_EVENTS = SCORING_EVENTS | POSSESSION_EVENTS

    def __init__(self) -> None:
        self._players: dict[str, PlayerStats] = {}
        self._teams: dict[str, TeamStats] = {}
        self._event_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_event(self, event: dict) -> None:
        """Process one event dict and update all affected stats."""
        event_type = event.get("event")
        if event_type not in self.VALID_EVENTS:
            raise ValueError(f"Unknown event type: {event_type!r}")

        player = self._get_or_create_player(event)
        team = self._get_or_create_team(event.get("team", "unknown"))

        handler = getattr(self, f"_handle_{event_type}", None)
        if handler:
            handler(event, player, team)

        self._event_log.append(event)

    def ingest_events(self, events: list[dict]) -> None:
        for e in events:
            self.ingest_event(e)

    def get_player_stats(self, player_id: str) -> Optional[PlayerStats]:
        return self._players.get(player_id)

    def set_on_court(self, player_ids: set) -> None:
        """Mark which players are on court at end of clip. All others = bench."""
        for ps in self._players.values():
            ps.on_court = ps.player_id in player_ids

    def apply_team_assignments(self, assignments: dict) -> None:
        """
        Apply color-based team classification results.

        assignments: {track_id → 'team_a' | 'team_b' | None}
            None  = referee / coach / staff → mark as non_player (excluded from output)
            str   = update the player's team label

        Only updates players that appear in the assignments dict; players with
        too few color samples (not in the dict) are left unchanged.
        """
        for pid, label in assignments.items():
            ps = self._players.get(pid) or self._players.get(str(pid))
            if ps is None:
                continue
            if label is None:
                # Hue classifier explicitly rejected this track (gray/black uniform).
                # Mark non_player unless they have real stats — real players can
                # temporarily have no hue reads if they're far from camera.
                meaningful = ps.pts + ps.assists + ps.steals + ps.blocks + ps.rebounds_total
                if meaningful == 0:
                    ps.non_player = True
            else:
                # Even with a team label, referees who got misclassified (e.g. dark
                # uniform with slight hue) produce zero scoring activity.
                # Flag them — get_box_score() will still include them only if they
                # pass the frames+stats gate, keeping real bench players.
                has_ball_activity = (
                    ps.fg_att + ps.three_att + ps.ft_att
                    + ps.rebounds_total + ps.steals + ps.assists
                ) > 0
                if not has_ball_activity and ps.frames_seen < 30:
                    ps.non_player = True
                else:
                    ps.team = label

    # Minimum sampled frames a player must appear in to be included in output.
    # At 5 fps, 150 frames ≈ 30 seconds — long enough to rule out crowd false-positives
    # and referee detections while still catching short-stint bench players.
    MIN_FRAMES_SEEN = 80

    def get_box_score(self) -> dict:
        """
        Return box score filtered to real players.

        Excluded: non_player (refs/coaches), phantom tracks, jersey numbers > 99.
        Team stats are re-derived from player stats so team assignment applied
        after event ingestion is reflected correctly.
        """
        real_players = [
            p for p in self._players.values()
            if not p.non_player and self._is_real_player(p)
        ]

        def _pct(m: int, a: int) -> Optional[float]:
            return round(m / a, 3) if a else None

        # Merge players with the same OCR'd jersey number (track-fragmentation fix)
        # Do this BEFORE aggregating team stats so cross-team merges are resolved first.
        real_players = self._merge_by_jersey(real_players)

        # Enforce exactly 5 on-court players per team.
        # If tracker noise gave >5, trim to top-5 by frames_seen.
        # If too few were seen at end of clip (<5), fill up with top players by frames_seen.
        from collections import defaultdict
        _team_all: dict[str, list] = defaultdict(list)
        for p in real_players:
            if p.team in ("team_a", "team_b"):
                _team_all[p.team].append(p)
        for _, plist in _team_all.items():
            plist.sort(key=lambda x: x.frames_seen, reverse=True)
            on_count = sum(1 for p in plist if p.on_court)
            if on_count > 5:
                # Too many — keep only the 5 most-seen
                for p in plist:
                    p.on_court = False
                for p in plist[:5]:
                    p.on_court = True
            elif on_count < 5:
                # Too few — promote most-seen players until we have 5
                for p in plist:
                    if on_count >= 5:
                        break
                    if not p.on_court:
                        p.on_court = True
                        on_count += 1

        # Re-aggregate team stats from post-merge player data so that cross-team
        # merges don't create phantom contributions to the wrong team bucket.
        team_buckets: dict[str, dict] = {}
        for p in real_players:
            tid = p.team
            if tid not in team_buckets:
                team_buckets[tid] = dict(
                    team_id=tid, pts=0,
                    fg_made=0, fg_att=0,
                    three_made=0, three_att=0,
                    ft_made=0, ft_att=0,
                    rebounds_off=0, rebounds_def=0,
                    assists=0, turnovers=0,
                    steals=0, blocks=0,
                )
            b = team_buckets[tid]
            b["pts"]          += p.pts
            b["fg_made"]      += p.fg_made
            b["fg_att"]       += p.fg_att
            b["three_made"]   += p.three_made
            b["three_att"]    += p.three_att
            b["ft_made"]      += p.ft_made
            b["ft_att"]       += p.ft_att
            b["rebounds_off"] += p.rebounds_off
            b["rebounds_def"] += p.rebounds_def
            b["assists"]      += p.assists
            b["turnovers"]    += p.turnovers_total
            b["steals"]       += p.steals
            b["blocks"]       += p.blocks

        teams = []
        for b in team_buckets.values():
            fa, fm = b["fg_att"], b["fg_made"]
            ta, tm = b["three_att"], b["three_made"]
            xfa, xft = b["ft_att"], b["ft_made"]
            teams.append({
                "team_id":   b["team_id"],
                "pts":       b["pts"],
                "fg":        f"{fm}/{fa}",
                "fg_pct":    _pct(fm, fa),
                "3p":        f"{tm}/{ta}",
                "3p_pct":    _pct(tm, ta),
                "ft":        f"{xft}/{xfa}",
                "ft_pct":    _pct(xft, xfa),
                "reb_off":   b["rebounds_off"],
                "reb_def":   b["rebounds_def"],
                "reb_total": b["rebounds_off"] + b["rebounds_def"],
                "ast":       b["assists"],
                "to":        b["turnovers"],
                "stl":       b["steals"],
                "blk":       b["blocks"],
            })

        return {"players": [p.to_dict() for p in real_players], "teams": teams}

    def _merge_by_jersey(self, players: list) -> list:
        """Merge PlayerStats entries that share the same OCR-confirmed jersey number."""
        from collections import defaultdict
        jersey_groups: dict[str, list] = defaultdict(list)
        no_jersey: list = []
        for p in players:
            # Only merge entries where the name was set by OCR (jersey_ocr=True)
            if p.jersey_ocr:
                name = (p.player_name or "").lstrip("#").strip()
                jersey_groups[name].append(p)
            else:
                no_jersey.append(p)

        merged: list = []
        for _, group in jersey_groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue
            # Primary = most frames seen; merge others into it
            primary = max(group, key=lambda p: p.frames_seen)
            for other in group:
                if other.player_id == primary.player_id:
                    continue
                primary.pts              += other.pts
                primary.fg_made          += other.fg_made
                primary.fg_att           += other.fg_att
                primary.three_made       += other.three_made
                primary.three_att        += other.three_att
                primary.ft_made          += other.ft_made
                primary.ft_att           += other.ft_att
                primary.rebounds_off     += other.rebounds_off
                primary.rebounds_def     += other.rebounds_def
                primary.assists          += other.assists
                primary.turnovers_steal  += other.turnovers_steal
                primary.turnovers_unforced += other.turnovers_unforced
                primary.steals           += other.steals
                primary.blocks           += other.blocks
                primary.contested_shots  += other.contested_shots
                primary.uncontested_shots += other.uncontested_shots
                primary.frames_seen      += other.frames_seen
                if other.on_court:
                    primary.on_court = True
            merged.append(primary)

        # Final pass: merge any remaining duplicates by player_name (catches
        # cases where a no_jersey fallback name collides with an OCR'd jersey)
        final: dict[str, "PlayerStats"] = {}
        for p in merged + no_jersey:
            key = p.player_name or str(p.player_id)
            if key in final:
                ex = final[key]
                ex.pts               += p.pts
                ex.fg_made           += p.fg_made
                ex.fg_att            += p.fg_att
                ex.three_made        += p.three_made
                ex.three_att         += p.three_att
                ex.ft_made           += p.ft_made
                ex.ft_att            += p.ft_att
                ex.rebounds_off      += p.rebounds_off
                ex.rebounds_def      += p.rebounds_def
                ex.assists           += p.assists
                ex.turnovers_steal   += p.turnovers_steal
                ex.turnovers_unforced += p.turnovers_unforced
                ex.steals            += p.steals
                ex.blocks            += p.blocks
                ex.contested_shots   += p.contested_shots
                ex.uncontested_shots += p.uncontested_shots
                ex.frames_seen       += p.frames_seen
                if p.on_court:
                    ex.on_court = True
                if p.jersey_ocr:
                    ex.jersey_ocr = True
            else:
                final[key] = p
        return list(final.values())

    def _is_real_player(self, p: PlayerStats) -> bool:
        """True if player has enough evidence to include in box score."""
        if str(p.player_id).lower() == "unknown" or str(p.player_name).lower() == "unknown":
            return False
        # Basketball jersey numbers are 0–99. Anything higher is a raw track ID
        # that OCR never resolved — treat it as a phantom and exclude it.
        name = (p.player_name or "").lstrip("#").strip()
        if name.isdigit() and int(name) > 99:
            return False
        # Must be visible for at least MIN_FRAMES_SEEN frames to be a real player.
        # This filters out audience members, referees, and camera flicker detections.
        return p.frames_seen >= self.MIN_FRAMES_SEEN

    def export_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.get_box_score(), f, indent=2)

    # ------------------------------------------------------------------
    # Event handlers — one per event type
    # ------------------------------------------------------------------

    def _handle_free_throw_made(self, _: dict, player: PlayerStats, team: TeamStats) -> None:
        player.pts += 1
        player.ft_made += 1
        player.ft_att += 1
        team.pts += 1
        team.ft_made += 1
        team.ft_att += 1

    def _handle_free_throw_missed(self, _: dict, player: PlayerStats, team: TeamStats) -> None:
        player.ft_att += 1
        team.ft_att += 1

    def _handle_two_pointer_made(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.pts += 2
        player.fg_made += 1
        player.fg_att += 1
        team.pts += 2
        team.fg_made += 1
        team.fg_att += 1
        self._record_shot_meta(event, player)

    def _handle_two_pointer_missed(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.fg_att += 1
        team.fg_att += 1
        self._record_shot_meta(event, player)

    def _handle_three_pointer_made(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.pts += 3
        player.fg_made += 1         # 3-pointers count toward overall FGM
        player.three_made += 1
        player.three_att += 1
        player.fg_att += 1
        team.pts += 3
        team.fg_made += 1
        team.three_made += 1
        team.three_att += 1
        team.fg_att += 1
        self._record_shot_meta(event, player)

    def _handle_three_pointer_missed(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.three_att += 1
        player.fg_att += 1
        team.three_att += 1
        team.fg_att += 1
        self._record_shot_meta(event, player)

    def _handle_rebound_offensive(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.rebounds_off += 1
        team.rebounds_off += 1

    def _handle_rebound_defensive(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.rebounds_def += 1
        team.rebounds_def += 1

    def _handle_assist(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.assists += 1
        team.assists += 1

    def _handle_turnover_steal(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.turnovers_steal += 1
        team.turnovers += 1

    def _handle_turnover_unforced(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.turnovers_unforced += 1
        team.turnovers += 1

    def _handle_steal(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.steals += 1
        team.steals += 1

    def _handle_block(self, event: dict, player: PlayerStats, team: TeamStats) -> None:
        player.blocks += 1
        team.blocks += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_shot_meta(self, event: dict, player: PlayerStats) -> None:
        sub = event.get("sub_type")
        if sub == "swish":
            player.swish += 1
        elif sub == "bank_shot":
            player.bank_shot += 1
        elif sub == "rim_in":
            player.rim_in += 1

        if event.get("contested"):
            player.contested_shots += 1
        else:
            player.uncontested_shots += 1

        if event.get("mismatch"):
            player.mismatch_shots += 1

    def _get_or_create_player(self, event: dict) -> PlayerStats:
        pid = str(event.get("player_id", "unknown"))
        if pid not in self._players:
            self._players[pid] = PlayerStats(
                player_id=pid,
                player_name=event.get("player_name", pid),
                team=event.get("team", "unknown"),
            )
        return self._players[pid]

    def _get_or_create_team(self, team_id: str) -> TeamStats:
        if team_id not in self._teams:
            self._teams[team_id] = TeamStats(team_id=team_id)
        return self._teams[team_id]
