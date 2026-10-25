"""
Pipeline — end-to-end CourtLog.ai video analysis.

Wires:
    Video → Detector → Tracker → PrimitiveGenerator
        → EventValidator (owns StatTracker internally)
        → Narrator + FeedBuilder
        → result dict
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional

from src.logic.event_validator import EventValidator
from src.logic.game_state import GameStateMachine
from src.logic.stat_tracker import StatTracker
from src.narrative.narrator import Narrator
from src.social.feed_builder import FeedBuilder
from src.vision.primitive_generator import PrimitiveGenerator


def _apply_reid_merges(stat_tracker, merge_map: dict) -> None:
    """
    Merge fragmented ByteTrack IDs that the Re-ID model identified as the same
    physical player.  merge_map: {secondary_track_id (int): canonical_track_id (int)}.
    """
    bsm = stat_tracker.box_score
    for secondary, canonical in merge_map.items():
        s_key = str(secondary)
        c_key = str(canonical)
        ps_s = bsm._players.get(s_key)
        ps_c = bsm._players.get(c_key)
        if ps_s is None or ps_c is None:
            continue

        # Sum every counting stat from the secondary track into the canonical one
        ps_c.pts                += ps_s.pts
        ps_c.fg_made            += ps_s.fg_made
        ps_c.fg_att             += ps_s.fg_att
        ps_c.three_made         += ps_s.three_made
        ps_c.three_att          += ps_s.three_att
        ps_c.ft_made            += ps_s.ft_made
        ps_c.ft_att             += ps_s.ft_att
        ps_c.rebounds_off       += ps_s.rebounds_off
        ps_c.rebounds_def       += ps_s.rebounds_def
        ps_c.assists            += ps_s.assists
        ps_c.turnovers_steal    += ps_s.turnovers_steal
        ps_c.turnovers_unforced += ps_s.turnovers_unforced
        ps_c.steals             += ps_s.steals
        ps_c.blocks             += ps_s.blocks
        ps_c.contested_shots    += ps_s.contested_shots
        ps_c.uncontested_shots  += ps_s.uncontested_shots
        ps_c.swish              += ps_s.swish
        ps_c.bank_shot          += ps_s.bank_shot
        ps_c.rim_in             += ps_s.rim_in
        ps_c.frames_seen        += ps_s.frames_seen
        ps_c.mismatch_shots     += ps_s.mismatch_shots
        if ps_s.on_court:
            ps_c.on_court = True
        # If secondary had OCR-confirmed jersey and canonical doesn't, inherit it
        if ps_s.jersey_ocr and not ps_c.jersey_ocr:
            ps_c.jersey_ocr  = True
            ps_c.player_name = ps_s.player_name

        del bsm._players[s_key]

        # Re-attribute all events from secondary → canonical
        for ev in stat_tracker.events:
            if str(ev.player_id) == s_key:
                ev.player_id   = c_key
                ev.player_name = ps_c.player_name


def _best_device() -> str:
    """Return the fastest available inference device: cuda > mps > cpu."""
    try:
        import torch  # type: ignore[import-untyped]
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_pipeline(
    video_path: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    fps_target: int = 5,
    det_model: str = "models/yolo11n.pt",
    pose_model: str = "models/yolo11n-pose.pt",
    device: Optional[str] = None,
    game_id: Optional[str] = None,
    quarter_duration_ms: Optional[int] = None,
) -> dict:
    """
    Process a basketball video end-to-end and return a result dict.

    quarter_duration_ms: period length in milliseconds.
        FIBA / EuroLeague: 10 * 60 * 1000  (default)
        NBA:               12 * 60 * 1000
        NCAA halves:       20 * 60 * 1000
        HS (8 min):         8 * 60 * 1000
        If None, the narrator falls back to FIBA (10 min).

    Returns dict with keys:
        video_path, game_id, box_score, events, feeds, narrative
    """
    game_id = game_id or f"game_{uuid.uuid4().hex[:8]}"
    device  = device or _best_device()

    def _cb(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    # ── 0. Open video ────────────────────────────────────────────────────────
    _cb(0.0, "Opening video…")
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("opencv-python not installed — run: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step   = max(1, round(native_fps / fps_target))
    sampled      = max(1, total_frames // frame_step)
    cap.release()

    # ── 1. Load vision models ────────────────────────────────────────────────
    _cb(0.02, "Loading vision models…")
    vision_ok = False
    detector  = None
    tracker   = None
    try:
        from src.vision.detector import Detector
        from src.vision.tracker import Tracker
        detector  = Detector(det_model=det_model, pose_model=pose_model, device=device)
        tracker   = Tracker(frame_rate=fps_target, enable_ocr=True, ocr_every_n=15)
        # Court mask: filters bench/crowd geometrically using painted court lines.
        # Attached to the detector so _parse_detections/_parse_poses apply it
        # automatically without changing the rest of the frame loop.
        try:
            from src.vision.court_mask import CourtMask
            detector.set_court_mask(CourtMask())
        except Exception:
            pass
        vision_ok = True
        _cb(0.05, f"Vision models loaded  ({device.upper()})")
    except Exception as exc:
        _cb(0.05, f"Vision unavailable ({exc}) — stat-only mode")

    # ── 2. Logic components ──────────────────────────────────────────────────
    stat_tracker = StatTracker()
    validator    = EventValidator(stat_tracker, auto_track=False)

    game_sm = GameStateMachine()

    # CourtCalibrator: auto-detects hoop pixel positions from the first 200 frames
    # so coordinate mapping works correctly for any camera angle (side, end, etc.)
    court_cal = None
    try:
        from src.vision.court_calibrator import CourtCalibrator
        court_cal = CourtCalibrator()
    except Exception:
        pass

    prim_gen = PrimitiveGenerator()

    # Scoreboard reader: runs OCR on the physical LED board visible in frames
    scoreboard = None
    try:
        from src.vision.scoreboard_reader import ScoreboardReader
        scoreboard = ScoreboardReader(use_gpu=(device != "cpu"))
    except Exception:
        pass

    # Re-ID embedder: collects per-player appearance embeddings to merge
    # fragmented ByteTrack IDs that belong to the same physical player.
    reid = None
    try:
        from src.vision.reid_embedder import ReidEmbedder
        reid = ReidEmbedder(device=device)
    except Exception:
        pass

    # Player gallery: jersey-first identity accumulation with embedding fallback.
    # Runs in-parallel with ReID; team assignment is injected after Pass 1.
    gallery = None
    try:
        from src.vision.player_gallery import PlayerGallery
        gallery = PlayerGallery()
    except Exception:
        pass

    # ── 3. Frame loop ────────────────────────────────────────────────────────
    game_start_ms: int = 0   # offset subtracted from all clock times
    processed = 0
    _calib_applied = False   # True once CourtCalibrator result has been applied

    if vision_ok and detector is not None and tracker is not None:
        cap = cv2.VideoCapture(video_path)
        _game_started = False

        # Seek directly to each target frame instead of reading every frame —
        # avoids decoding 14-out-of-15 frames that are immediately discarded.
        target_frame = 0
        while target_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            try:
                detector.update_court_mask(frame)   # periodic boundary re-detection
                fd = detector.process_frame(frame, target_frame, ts_ms)
                tracker.update(fd, frame)

                # Detect game start: first sampled frame where ball + ≥6 players visible
                if not _game_started and fd.ball is not None and len(fd.players) >= 6:
                    game_start_ms = ts_ms
                    _game_started = True

                # Court calibration: feed this frame's ball position into the
                # calibrator.  Once it has enough samples it locks in the hoop
                # pixel positions and we update prim_gen's coordinate mapping.
                if court_cal is not None:
                    court_cal.update(fd)
                    if not _calib_applied and court_cal.ready:
                        hoop_a, hoop_b = court_cal.result()
                        prim_gen._hoop_a_x = hoop_a
                        prim_gen._hoop_b_x = hoop_b
                        _calib_applied = True
                        _cb(processed / sampled * 0.85 + 0.05,
                            f"Court calibrated: left hoop={hoop_a:.2f}  right hoop={hoop_b:.2f}")

                # Record which players are visible this sampled frame
                for p in fd.players:
                    if p.track_id is not None:
                        stat_tracker.mark_seen(p.track_id, processed)

                # Compute ball court coordinates for game state machine
                ball_court = None
                if fd.ball is not None:
                    ball_court = prim_gen._to_court(fd.ball.center, fd)

                prim_gen.push(fd)
                for prim in prim_gen.drain():
                    raw_events = validator.push(prim)
                    for ev in raw_events:
                        ev = game_sm.on_event(ev, ball_court)
                        if ev is not None:
                            stat_tracker.track(ev)

                # Update game state machine with this frame's vision signals
                game_sm.on_frame(fd, ball_court)

                # Collect Re-ID appearance embeddings for court-area players
                if reid is not None:
                    try:
                        reid.sample(frame, fd.players, ts_ms, processed)
                    except Exception:
                        pass

                # Feed player crops into gallery (jersey-first identity accumulation)
                if gallery is not None:
                    try:
                        for _gp in fd.players:
                            if _gp.track_id is not None:
                                gallery.update(frame, _gp, processed)
                    except Exception:
                        pass

                # Read scoreboard every 60 sampled frames (~12 s at 5 fps).
                # The physical scoreboard is only visible when the camera faces
                # that end of the court — sparse sampling catches it without
                # wasting OCR time on frames where it's out of view.
                if scoreboard is not None and processed % 60 == 0:
                    try:
                        scoreboard.read_frame(frame, ts_ms)
                        # Notify game state machine of period transitions
                        period_ts = scoreboard.get_period_timestamps()
                        if period_ts:
                            latest_period = max(period_ts.keys())
                            latest_ts = period_ts[latest_period]
                            if abs(latest_ts - ts_ms) < 2000:
                                game_sm.notify_period_boundary(latest_period, ts_ms)
                    except Exception:
                        pass
            except Exception:
                pass  # never abort on a single bad frame

            # Explicitly release frame + inference buffers to keep memory flat
            del frame

            # Flush GPU/MPS memory every 100 frames
            if processed % 100 == 0:
                try:
                    import torch  # type: ignore[import-untyped]
                    if hasattr(torch, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            processed += 1
            frac = 0.05 + 0.85 * (processed / sampled)
            _cb(frac, f"Frame {processed:,} / {sampled:,}")

            target_frame += frame_step

        cap.release()

    # ── 3b. Re-ID merge ──────────────────────────────────────────────────────
    # Compare mean appearance embeddings across all track pairs that never
    # co-existed on court.  High-similarity pairs → same physical player →
    # merge their stats and re-attribute their events before the box score passes.
    if reid is not None:
        try:
            _cb(0.90, "Re-ID: merging fragmented tracks…")
            merge_map = reid.get_merge_map()
            if merge_map:
                _apply_reid_merges(stat_tracker, merge_map)
        except Exception:
            pass

    # ── 3c. Player Identity Bank ─────────────────────────────────────────────
    # Build a centralised identity store from ReID embeddings + OCR jersey
    # numbers, then re-attribute events to canonical identities.
    if reid is not None and vision_ok and tracker is not None:
        try:
            from src.vision.identity_bank import PlayerIdentityBank
            identity_bank = PlayerIdentityBank()
            identity_bank.ingest_reid_embeddings(reid)
            identity_bank.ingest_jersey_ocr(tracker._jersey_cache)
            if reid is not None:
                identity_bank.apply_reid_merges(reid.get_merge_map() if reid else {})
            for ev in stat_tracker.events:
                try:
                    identity = identity_bank.resolve_track(int(ev.player_id))
                except (ValueError, TypeError):
                    identity = None
                if identity is None:
                    continue
                if identity.jersey_number:
                    ev.player_name = f"#{identity.jersey_number}"
                if identity.team and ev.team in ("unknown", ""):
                    ev.team = identity.team
        except Exception:
            pass

    # ── 4. Box score ─────────────────────────────────────────────────────────
    _cb(0.92, "Building box score…")

    # Push per-player seen counts into PlayerStats for phantom-track filtering
    stat_tracker.apply_seen_counts()

    # Determine who was on court at end of clip (last 15 % of sampled frames)
    on_court_ids = stat_tracker.compute_on_court(processed)
    stat_tracker.box_score.set_on_court(on_court_ids)

    # Mark OCR-confirmed jersey players so phantom-filter treats them leniently
    if vision_ok and tracker is not None:
        try:
            for tid, jersey_read in tracker._jersey_cache.items():
                pid      = str(tid)
                new_name = f"#{jersey_read.number}"
                ps = stat_tracker.box_score._players.get(pid)
                if ps is not None:
                    ps.jersey_ocr  = True
                    ps.player_name = new_name
                # Retroactively patch any events logged before OCR confirmed the jersey.
                # The frame loop runs OCR throughout the whole video, so a 3-pointer at
                # frame 200 may be attributed to "Player 104" even though frame 500 later
                # confirmed the same track is jersey #7.
                for ev in stat_tracker.events:
                    if ev.player_id == pid:
                        ev.player_name = new_name
        except Exception:
            pass

    # ── Pass 1: team classification ──────────────────────────────────────────
    # Jersey hue k-means → two clusters (team_a / team_b / None).
    # Then retroactive referee reclassification: tracks whose 33-dim appearance
    # vector (brightness bimodality + hue) matches a known referee profile get
    # flipped to None regardless of their hue reading.
    # Profiles are persisted to disk so referees recognised in previous clips
    # are immediately flagged in future clips without needing per-frame detection.
    if vision_ok and tracker is not None:
        try:
            import os
            from pathlib import Path as _Path
            from src.vision.team_classifier import TeamClassifier

            _referee_profile_path = str(
                _Path(video_path).parent.parent / "output" / "referee_profiles.npy"
            )

            tc = TeamClassifier()
            tc.load_referee_profiles(_referee_profile_path)

            for tid, hues in tracker.hue_samples.items():
                for h in hues:
                    tc.add_sample(tid, h)

            assignments = tc.classify()

            # Retroactive reclassification using appearance embeddings
            assignments = tc.reclassify_with_appearances(
                assignments, tracker._appearances
            )

            tc.save_referee_profiles(_referee_profile_path)

            # Second pass: court-position co-occurrence clustering.
            # Fills in tracks that had ambiguous hue (e.g. dark uniforms on dark floor)
            # using which side of the court each player consistently occupied.
            try:
                co_assignments = tc.classify_by_cooccurrence(tracker.frame_snapshots)
                for _tid, _team in co_assignments.items():
                    if assignments.get(_tid) is None and _team is not None:
                        assignments[_tid] = _team
            except Exception:
                pass

            stat_tracker.box_score.apply_team_assignments(assignments)

            # Flow team labels into PlayerGallery so identity lookups are team-aware.
            # Knowing a player's team before searching for embedding matches prevents
            # false merges between players on the same-coloured kit.
            if gallery is not None:
                try:
                    for _tid, _team in assignments.items():
                        gallery.set_team(_tid, _team)
                except Exception:
                    pass
        except Exception:
            pass

    # ── Pass 2: attacking-basket assignment ──────────────────────────────────
    # Basketball rule: each team attacks ONE basket the whole half.
    # Identify which basket each team attacks by looking at where classified
    # players shoot from.  Then classify unknown players the same way.
    _basket_team: dict[str, str] = {}
    try:
        # Step A — build basket→team map from already-classified events.
        # Events are stored with team="unknown" so look up the player's team
        # from box_score (updated by Pass 1) rather than from the event itself.
        _basket_votes: dict[str, dict[str, int]] = {"left": {}, "right": {}}
        for ev in stat_tracker.events:
            if ev.event_type not in (
                "two_pointer_made", "two_pointer_missed",
                "three_pointer_made", "three_pointer_missed",
            ):
                continue
            ps = stat_tracker.box_score._players.get(str(ev.player_id))
            if ps is None or ps.team not in ("team_a", "team_b"):
                continue
            x = ev.coordinates.get("x", 14.0)
            side = "left" if x < 14.0 else "right"
            _basket_votes[side][ps.team] = _basket_votes[side].get(ps.team, 0) + 1

        # Determine dominant team per basket side (majority vote)
        for side, votes in _basket_votes.items():
            if votes:
                _basket_team[side] = max(votes, key=lambda t: votes[t])

        # Step B — classify unknown players using their own shot events.
        if len(_basket_team) >= 1:
            for ps in stat_tracker.box_score._players.values():
                if ps.team != "unknown":
                    continue
                pid = ps.player_id
                p_events = stat_tracker.get_player_events(str(pid))
                shot_xs = [
                    e.coordinates.get("x", 14.0)
                    for e in p_events
                    if e.event_type in (
                        "two_pointer_made", "two_pointer_missed",
                        "three_pointer_made", "three_pointer_missed",
                    )
                ]
                if shot_xs:
                    avg_x = sum(shot_xs) / len(shot_xs)
                    side = "left" if avg_x < 14.0 else "right"
                    if side in _basket_team:
                        ps.team = _basket_team[side]
    except Exception:
        pass

    # ── Pass 3: jersey-number → team registry ───────────────────────────────
    # Same jersey number on different track IDs = same physical player (or rival
    # with the same number, but that's handled by the hue classifier already).
    # After passes 1+2, build jersey→team map and propagate to remaining unknowns.
    try:
        jersey_team: dict[str, str] = {}
        for ps in stat_tracker.box_score._players.values():
            if ps.team in ("team_a", "team_b") and ps.player_name.startswith("#"):
                jnum = ps.player_name.lstrip("#").strip()
                if jnum:
                    jersey_team[jnum] = ps.team

        for ps in stat_tracker.box_score._players.values():
            if ps.team == "unknown" and ps.player_name.startswith("#"):
                jnum = ps.player_name.lstrip("#").strip()
                if jnum in jersey_team:
                    ps.team = jersey_team[jnum]
    except Exception:
        pass

    # ── Pass 4: event-context assignment using basketball rules ─────────────
    # Assist  → assister is on the SAME team as the scorer.
    # Block   → blocker is on the OPPOSITE team from the shooter.
    # These are direct basketball rules — no colour or position data needed.
    try:
        for ps in stat_tracker.box_score._players.values():
            if ps.team != "unknown":
                continue
            pid = str(ps.player_id)
            p_events = stat_tracker.get_player_events(pid)
            for ev in p_events:
                if ev.event_type == "assist":
                    # Find the scoring event closest in time
                    ts = ev.timestamp_ms
                    best_score = None
                    for sev in stat_tracker.events:
                        if sev.event_type not in ("two_pointer_made", "three_pointer_made", "free_throw_made"):
                            continue
                        if sev.team in ("team_a", "team_b") and abs(sev.timestamp_ms - ts) < 2000:
                            best_score = sev
                            break
                    if best_score and best_score.team in ("team_a", "team_b"):
                        ps.team = best_score.team
                        break
                elif ev.event_type == "block":
                    # Find shot attempt near this timestamp — blocker is on opposite team
                    ts = ev.timestamp_ms
                    for sev in stat_tracker.events:
                        if sev.event_type not in (
                            "two_pointer_made", "two_pointer_missed",
                            "three_pointer_made", "three_pointer_missed"
                        ):
                            continue
                        if sev.team in ("team_a", "team_b") and abs(sev.timestamp_ms - ts) < 3000:
                            ps.team = "team_b" if sev.team == "team_a" else "team_a"
                            break
                    if ps.team != "unknown":
                        break
    except Exception:
        pass

    # ── Pass 5: court-position fallback for any remaining unknowns ───────────
    # Use average x-position in frame snapshots to determine which basket side
    # a player spent time near.  Works for any remaining uncategorised tracks.
    if vision_ok and tracker is not None and _basket_team:
        try:
            snapshots = tracker.frame_snapshots
            for ps in stat_tracker.box_score._players.values():
                if ps.team != "unknown":
                    continue
                try:
                    uid = int(ps.player_id)
                except (ValueError, TypeError):
                    continue
                xs = [snap[uid] * 28.0 for snap in snapshots if uid in snap]
                if len(xs) < 3:
                    continue
                avg_x = sum(xs) / len(xs)
                # A player spending most time on the left third → near left basket
                side = "left" if avg_x < 12.0 else ("right" if avg_x > 16.0 else None)
                if side and side in _basket_team:
                    ps.team = _basket_team[side]
        except Exception:
            pass

    # ── Pass 5b: backfill ev.team on all EnrichedEvents from resolved PlayerStats ─
    # Passes 1–5 resolve team on box_score._players but never copy it back onto
    # the already-emitted EnrichedEvent objects.  Do that now so every event has
    # a real team string before any downstream pass or JSON export reads it.
    for ev in stat_tracker.events:
        if ev.team in ("unknown", ""):
            ps = stat_tracker.box_score._players.get(str(ev.player_id))
            if ps is not None and ps.team not in ("unknown", ""):
                ev.team = ps.team

    # ── Pass 6: retroactive rebound offensive/defensive reclassification ────
    # Use shot coordinates + attacking-basket map (_basket_team from Pass 2) to classify
    # each rebound without relying on shooter's track_id (fragile after track fragmentation).
    # Rebounder's team vs the team that attacks the basket where the shot came from
    # → same team = offensive rebound, different team = defensive rebound.
    try:
        # Only missed shots produce rebounds
        shot_events = [
            e for e in stat_tracker.events
            if e.event_type in ("two_pointer_missed", "three_pointer_missed")
        ]
        shot_events.sort(key=lambda e: e.timestamp_ms)

        for ev in stat_tracker.events:
            if ev.event_type not in ("rebound_offensive", "rebound_defensive"):
                continue
            rebounder_ps = stat_tracker.box_score._players.get(str(ev.player_id))
            if rebounder_ps is None or rebounder_ps.team not in ("team_a", "team_b"):
                continue
            # Find the closest preceding miss within 15 seconds
            ts = ev.timestamp_ms
            preceding = [s for s in shot_events if s.timestamp_ms <= ts and ts - s.timestamp_ms <= 15_000]
            if not preceding:
                continue
            shooter_ev = max(preceding, key=lambda s: s.timestamp_ms)
            # Determine which basket the shot was aimed at from shot coordinates
            shot_x = (shooter_ev.coordinates or {}).get("x", 14.0)
            basket_side = "left" if shot_x < 14.0 else "right"
            attacking_team = _basket_team.get(basket_side)
            if attacking_team not in ("team_a", "team_b"):
                # _basket_team not populated — fall back to shooter's team
                shooter_ps = stat_tracker.box_score._players.get(str(shooter_ev.player_id))
                if shooter_ps is None or shooter_ps.team not in ("team_a", "team_b"):
                    continue
                attacking_team = shooter_ps.team
            correct_type = (
                "rebound_offensive"
                if rebounder_ps.team == attacking_team
                else "rebound_defensive"
            )
            if correct_type == ev.event_type:
                continue
            # Reclassify — update event, reason, and box_score player stats
            ev.event_type = correct_type  # type: ignore[misc]
            ev.reason = (
                "Same team regained possession after missed shot — offensive board"
                if correct_type == "rebound_offensive"
                else "Opposing team secured possession after missed shot — defensive board"
            )
            if correct_type == "rebound_offensive":
                rebounder_ps.rebounds_off += 1
                rebounder_ps.rebounds_def -= 1
            else:
                rebounder_ps.rebounds_off -= 1
                rebounder_ps.rebounds_def += 1
    except Exception:
        pass

    # ── Pass 6.5: post-calibration shot distance filter ──────────────────────
    # Shots detected before court calibration used default hoop positions and
    # may have wrong coordinates that slipped through the per-frame distance
    # filter.  Re-check every shot event against the calibrated court geometry.
    try:
        import math as _math
        _COURT_L = 28.0
        _COURT_W = 15.0
        _MAX_SHOT_DIST_M = 8.5
        _half_w = _COURT_W / 2
        _shot_types = {
            "two_pointer_made", "two_pointer_missed",
            "three_pointer_made", "three_pointer_missed",
            "free_throw_made", "free_throw_missed",
        }
        _events_to_remove: set[int] = set()
        for _i, _ev in enumerate(stat_tracker._events):
            if _ev.event_type not in _shot_types:
                continue
            if _ev.event_type in ("free_throw_made", "free_throw_missed"):
                continue  # FT is near the line, always valid
            _coords = getattr(_ev, "coordinates", None) or {}
            _cx = _coords.get("x", 0.0)
            _cy = _coords.get("y", _half_w)
            _da = _math.hypot(_cx, _cy - _half_w)
            _db = _math.hypot(_cx - _COURT_L, _cy - _half_w)
            if min(_da, _db) > _MAX_SHOT_DIST_M:
                _events_to_remove.add(_i)
        if _events_to_remove:
            # stat_tracker.events is a read-only property — mutate _events directly
            stat_tracker._events = [
                e for i, e in enumerate(stat_tracker._events)
                if i not in _events_to_remove
            ]
    except Exception:
        pass

    # ── Pass 7: reconcile blocks & steals from events → box_score ───────────
    # Re-ID merges and _merge_by_jersey can silently drop block/steal increments
    # when a secondary track is deleted before its stat is added to the canonical.
    # Recount both stats directly from the authoritative events[] list.
    try:
        from collections import Counter as _Counter
        _blk_counts: dict = _Counter(
            ev.player_id for ev in stat_tracker.events if ev.event_type == "block"
        )
        _stl_counts: dict = _Counter(
            ev.player_id for ev in stat_tracker.events if ev.event_type == "steal"
        )
        for pid, cnt in _blk_counts.items():
            ps = stat_tracker.box_score._players.get(str(pid))
            if ps is not None:
                ps.blocks = cnt
        for pid, cnt in _stl_counts.items():
            ps = stat_tracker.box_score._players.get(str(pid))
            if ps is not None:
                ps.steals = cnt
    except Exception:
        pass

    # ── Gallery persistence ──────────────────────────────────────────────────
    # Write per-player metadata + embedding matrix to data/output/ so identity
    # data survives beyond the pipeline run and can be queried offline.
    if gallery is not None:
        try:
            import json as _json
            import numpy as _np
            from pathlib import Path as _Path

            _out_dir = _Path(video_path).parent.parent / "output"
            _out_dir.mkdir(parents=True, exist_ok=True)

            _gallery_data = []
            for _entry in gallery.all_entries():
                _gallery_data.append({
                    "gid":              _entry.gid,
                    "jersey":           _entry.confirmed_jersey,
                    "jersey_candidate": _entry.top_jersey_candidate,
                    "jersey_votes":     _entry.top_jersey_votes,
                    "team":             _entry.team,
                    "confirmed":        _entry.confirmed,
                    "crop_count":       _entry.crop_count,
                    "track_ids":        list(_entry.track_ids),
                    "frame_first":      _entry.frame_first,
                    "frame_last":       _entry.frame_last,
                })

            _gallery_path = str(_out_dir / f"{game_id}_gallery.json")
            with open(_gallery_path, "w") as _f:
                _json.dump({"game_id": game_id, "players": _gallery_data}, _f, indent=2)

            # Embedding matrix: rows = players, label file maps row index → jersey/gid
            _emb_rows = [
                (_entry.confirmed_jersey or _entry.gid, _entry.mean_emb)
                for _entry in gallery.all_entries()
                if _entry.mean_emb is not None
            ]
            if _emb_rows:
                _emb_labels = [label for label, _ in _emb_rows]
                _emb_matrix = _np.stack([vec for _, vec in _emb_rows])
                _np.save(str(_out_dir / f"{game_id}_gallery_emb.npy"), _emb_matrix)
                with open(str(_out_dir / f"{game_id}_gallery_labels.json"), "w") as _f:
                    _json.dump(_emb_labels, _f)
        except Exception:
            pass

    box_score = stat_tracker.box_score.get_box_score()

    # ── Post-merge event re-attribution ──────────────────────────────────────
    # _merge_by_jersey() inside get_box_score() consolidates duplicate tracks
    # (e.g. track 23 + track 104 → single #7 row).  Re-attribute all events so
    # the play-by-play and feeds use the canonical merged player_id / player_name.
    try:
        # Build: player_id → (canonical_player_id, player_name) from final box score
        _pid_to_canonical: dict[str, tuple[str, str]] = {}
        for bp in box_score.get("players", []):
            canonical_pid  = str(bp.get("player_id", ""))
            canonical_name = bp.get("player_name", "")
            _pid_to_canonical[canonical_pid] = (canonical_pid, canonical_name)

        # jersey_name → canonical_pid (for updating events whose pid was merged away)
        _jersey_to_canonical: dict[str, str] = {}
        for pid, (cpid, cname) in _pid_to_canonical.items():
            if cname.startswith("#"):
                _jersey_to_canonical[cname] = cpid

        # Also build canonical_pids set for fast "is this pid still in box score?" lookup
        _canonical_pids: set[str] = set(_pid_to_canonical.keys())

        # Patch events: sync player_id / player_name with the final box score.
        # Three cases:
        #   1. player_name is a jersey ("#N") known to the box score → use that canonical pid
        #   2. player_id is in the box score → update player_name to match
        #   3. player_id was merged away (not in box score) → look up by player_name
        for ev in stat_tracker.events:
            if ev.player_name in _jersey_to_canonical:
                # Case 1: jersey confirmed — use canonical pid
                ev.player_id = _jersey_to_canonical[ev.player_name]
            elif ev.player_id in _canonical_pids:
                # Case 2: pid still exists in box score — sync name only
                _, cname = _pid_to_canonical[ev.player_id]
                if cname:
                    ev.player_name = cname
            else:
                # Case 3: pid was absorbed by _merge_by_jersey — re-attribute by name
                # If player_name was later patched to a jersey, use that canonical
                if ev.player_name in _jersey_to_canonical:
                    ev.player_id = _jersey_to_canonical[ev.player_name]
                # If we still can't resolve, leave as-is (event is marked needs_review)
    except Exception:
        pass

    # ── 5. Narrative ─────────────────────────────────────────────────────────
    _cb(0.94, "Generating play-by-play…")
    period_timestamps = {}
    scoreboard_score  = None
    if scoreboard is not None:
        period_timestamps = scoreboard.get_period_timestamps()
        scoreboard_score  = scoreboard.infer_final_score()
    narrator  = Narrator(
        game_start_ms=game_start_ms,
        period_timestamps=period_timestamps,
        quarter_duration_ms=quarter_duration_ms,
    )
    narrative = narrator.narrate(stat_tracker.events)
    narrative_dict = {
        "full_text":              narrative.full_text,
        "pbp_lines":              [l.to_dict() for l in narrative.pbp_lines],
        "key_moments":            [
            {"timestamp_ms": km.timestamp_ms, "clock": km.clock,
             "quarter": km.quarter, "kind": km.kind, "description": km.description}
            for km in narrative.key_moments
        ],
        "final_score":            narrative.final_score,
        "total_lead_changes":     narrative.total_lead_changes,
        "total_learning_moments": narrative.total_learning_moments,
    }

    # ── 5b. AI enrichment (optional — requires ANTHROPIC_API_KEY) ───────────
    ai_insights: dict = {}
    try:
        from src.narrative.ai_enhancer import enhance_game as _enhance
        _cb(0.95, "AI enrichment…")
        ai_insights = _enhance(
            [e.to_dict() for e in stat_tracker.events],
            box_score,
        )
        if ai_insights.get("_tokens"):
            t = ai_insights["_tokens"]
            print(f"[ai_enhancer] tokens in={t['input']} cache_read={t['cache_read']} "
                  f"out={t['output']} cost=${t['estimated_cost_usd']:.5f}")
    except Exception:
        pass

    # ── 6. Social feeds ──────────────────────────────────────────────────────
    _cb(0.96, "Building feeds…")
    feed_builder = FeedBuilder(stat_tracker, game_id=game_id)
    feeds = {
        "game":     feed_builder.build_game_feed().to_dict(),
        "plays":    feed_builder.build_plays_feed().to_dict(),
        "learning": feed_builder.build_learning_feed().to_dict(),
    }

    _cb(1.0, "Done ✓")
    return {
        "video_path":       video_path,
        "game_id":          game_id,
        "box_score":        box_score,
        "events":           [e.to_dict() for e in stat_tracker.events],
        "feeds":            feeds,
        "narrative":        narrative_dict,
        "scoreboard_score": scoreboard_score,   # (home, away) from OCR, None if unreadable
        "period_timestamps": period_timestamps, # {period: video_ts_ms}
        "ai_insights":       ai_insights,        # game_summary, player_insights, standout_moments
    }
