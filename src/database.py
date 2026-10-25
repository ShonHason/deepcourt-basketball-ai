"""
database.py — SQLite persistence for CourtLog.ai.

Tables
──────
players  — roster entries (jersey #, name, position, email, photo)
jobs     — analysis jobs (video path, status, progress, result JSON)

All writes use `INSERT OR REPLACE` so upserts are safe.
`get_conn()` is thread-safe via check_same_thread=False.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Optional


class _SafeEncoder(json.JSONEncoder):
    """Serialize numpy scalars and other non-standard types that the pipeline may emit."""
    def default(self, obj):  # type: ignore[override]
        try:
            import numpy as np  # type: ignore[import-untyped]
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
        return repr(obj)

DB_PATH = Path("data/courtlog.db")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads/writes
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            jersey_number  TEXT PRIMARY KEY,
            name           TEXT NOT NULL DEFAULT '',
            position       TEXT NOT NULL DEFAULT '—',
            email          TEXT NOT NULL DEFAULT '',
            photo_b64      TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id             TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            video_path     TEXT NOT NULL DEFAULT '',
            status         TEXT NOT NULL DEFAULT 'queued',
            progress       REAL NOT NULL DEFAULT 0.0,
            status_msg     TEXT NOT NULL DEFAULT '',
            result_json    TEXT,
            error          TEXT,
            user_id        TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS other_videos (
            id             TEXT PRIMARY KEY,
            user_id        TEXT NOT NULL,
            title          TEXT NOT NULL DEFAULT '',
            file_path      TEXT NOT NULL DEFAULT '',
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS users (
            id             TEXT PRIMARY KEY,
            auth_provider  TEXT NOT NULL DEFAULT 'email',
            email          TEXT NOT NULL UNIQUE,
            password_hash  TEXT,
            display_name   TEXT NOT NULL DEFAULT '',
            height_cm      INTEGER,
            position       TEXT NOT NULL DEFAULT '—',
            dominant_hand  TEXT NOT NULL DEFAULT 'right',
            birth_date     TEXT,
            competition_track TEXT NOT NULL DEFAULT 'boys',
            age_group      TEXT,
            club_name      TEXT,
            photo_b64      TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS teams (
            id             TEXT PRIMARY KEY,
            name           TEXT NOT NULL UNIQUE,
            created_by     TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS team_memberships (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id        TEXT NOT NULL,
            user_id        TEXT NOT NULL,
            role           TEXT NOT NULL DEFAULT 'player',
            created_at     TEXT DEFAULT (datetime('now')),
            UNIQUE(team_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS games (
            id             TEXT PRIMARY KEY,
            source_job_id  TEXT,
            host_user_id   TEXT,
            team_id        TEXT,
            title          TEXT NOT NULL DEFAULT '',
            video_path     TEXT NOT NULL DEFAULT '',
            game_scope     TEXT NOT NULL DEFAULT 'full_game',
            quarter_ms     INTEGER,
            overtime_count INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS game_invites (
            id             TEXT PRIMARY KEY,
            game_id        TEXT NOT NULL,
            user_id        TEXT,
            invited_email  TEXT,
            status         TEXT NOT NULL DEFAULT 'pending',
            created_at     TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS player_game_links (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id              TEXT NOT NULL,
            user_id              TEXT NOT NULL,
            detected_player_id   TEXT,
            detected_player_name TEXT,
            team_side            TEXT,
            created_at           TEXT DEFAULT (datetime('now')),
            UNIQUE(game_id, user_id)
        );
        """)

        # Lightweight migrations for existing local DBs
        existing_user_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "birth_date" not in existing_user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN birth_date TEXT")
        if "competition_track" not in existing_user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN competition_track TEXT NOT NULL DEFAULT 'boys'")
        if "age_group" not in existing_user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN age_group TEXT")
        if "club_name" not in existing_user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN club_name TEXT")
        if "dominant_hand" not in existing_user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN dominant_hand TEXT NOT NULL DEFAULT 'right'")

        existing_job_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        if "user_id" not in existing_job_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN user_id TEXT")


# ---------------------------------------------------------------------------
# Player operations
# ---------------------------------------------------------------------------

def save_player(
    jersey: str,
    name: str,
    position: str = "—",
    email: str = "",
    photo_b64: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO players
               (jersey_number, name, position, email, photo_b64)
               VALUES (?, ?, ?, ?, ?)""",
            (jersey.strip(), name.strip(), position, email.strip(), photo_b64),
        )


def load_all_players() -> dict[str, dict]:
    """Return {jersey_number: player_dict}."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM players ORDER BY CAST(jersey_number AS INTEGER)"
        ).fetchall()
    return {r["jersey_number"]: dict(r) for r in rows}


def delete_player(jersey: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM players WHERE jersey_number = ?", (jersey,))


def clear_players() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM players")


def delete_job(job_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


# ---------------------------------------------------------------------------
# Job operations
# ---------------------------------------------------------------------------

def save_job(job: dict) -> None:
    result_json = json.dumps(job["result"], cls=_SafeEncoder) if job.get("result") else None
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO jobs
               (id, name, video_path, status, progress, status_msg, result_json, error, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job["id"],
                job["name"],
                job.get("path", ""),
                job.get("status", "queued"),
                job.get("progress", 0.0),
                job.get("status_msg", ""),
                result_json,
                job.get("error"),
                job.get("user_id"),
            ),
        )


def _decode_job(row) -> dict:
    j = dict(row)
    j["result"] = json.loads(j["result_json"]) if j.get("result_json") else None
    del j["result_json"]
    j["path"] = j.pop("video_path", "")
    return j


def load_all_jobs() -> list[dict]:
    """Return all jobs ordered newest first, with result_json decoded."""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [_decode_job(r) for r in rows]


def load_jobs_by_user(user_id: str) -> list[dict]:
    """Return jobs belonging to a specific user, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    return [_decode_job(r) for r in rows]


def load_job(job_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _decode_job(row) if row else None


# ---------------------------------------------------------------------------
# Other-videos operations
# ---------------------------------------------------------------------------

def save_other_video(user_id: str, title: str, file_path: str, video_id: Optional[str] = None) -> str:
    video_id = video_id or uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO other_videos (id, user_id, title, file_path)
               VALUES (?, ?, ?, ?)""",
            (video_id, user_id, title.strip(), file_path),
        )
    return video_id


def load_other_videos_by_user(user_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM other_videos WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def delete_other_video(video_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM other_videos WHERE id = ?", (video_id,))


# ---------------------------------------------------------------------------
# Product-layer operations
# ---------------------------------------------------------------------------

def save_user(
    email: str,
    display_name: str,
    auth_provider: str = "email",
    position: str = "—",
    height_cm: Optional[int] = None,
    dominant_hand: str = "right",
    birth_date: Optional[str] = None,
    competition_track: str = "boys",
    age_group: Optional[str] = None,
    club_name: Optional[str] = None,
    photo_b64: Optional[str] = None,
    user_id: Optional[str] = None,
    password_hash: Optional[str] = None,
) -> str:
    user_id = user_id or uuid.uuid4().hex
    existing = load_user_by_email(email)
    if existing:
        user_id = existing["id"]
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO users
               (id, auth_provider, email, password_hash, display_name, height_cm, position,
                dominant_hand, birth_date, competition_track, age_group, club_name, photo_b64)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                auth_provider,
                email.strip().lower(),
                password_hash,
                display_name.strip(),
                height_cm,
                position,
                dominant_hand,
                birth_date,
                competition_track,
                age_group,
                club_name.strip() if club_name else None,
                photo_b64,
            ),
        )
    return user_id


def patch_user(user_id: str, **fields) -> None:
    allowed = {"height_cm", "club_name", "age_group", "position", "photo_b64"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE users SET {set_clause} WHERE id = ?",
            (*updates.values(), user_id),
        )


def load_all_users() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC, display_name ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def load_user(user_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def load_user_by_email(email: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return dict(row) if row else None


def save_team(name: str, created_by: Optional[str] = None, team_id: Optional[str] = None) -> str:
    team_id = team_id or uuid.uuid4().hex
    existing = load_team_by_name(name)
    if existing:
        team_id = existing["id"]
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO teams
               (id, name, created_by)
               VALUES (?, ?, ?)""",
            (team_id, name.strip(), created_by),
        )
    return team_id


def load_team_by_name(name: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM teams WHERE name = ?", (name.strip(),)).fetchone()
    return dict(row) if row else None


def load_all_teams() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM teams ORDER BY name COLLATE NOCASE").fetchall()
    return [dict(r) for r in rows]


def add_team_member(team_id: str, user_id: str, role: str = "player") -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO team_memberships
               (id, team_id, user_id, role)
               VALUES (
                   COALESCE((SELECT id FROM team_memberships WHERE team_id = ? AND user_id = ?), NULL),
                   ?, ?, ?
               )""",
            (team_id, user_id, team_id, user_id, role),
        )


def load_teammates(user_id: str) -> list[dict]:
    """Return all other users who share at least one team with user_id."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT u.id, u.display_name, u.position, u.club_name, u.photo_b64
            FROM team_memberships tm
            JOIN team_memberships tm2 ON tm2.team_id = tm.team_id AND tm2.user_id != ?
            JOIN users u ON u.id = tm2.user_id
            WHERE tm.user_id = ?
            ORDER BY u.display_name COLLATE NOCASE
            """,
            (user_id, user_id),
        ).fetchall()
    return [dict(r) for r in rows]


def load_team_memberships() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT tm.*, t.name AS team_name, u.display_name, u.email
            FROM team_memberships tm
            JOIN teams t ON t.id = tm.team_id
            JOIN users u ON u.id = tm.user_id
            ORDER BY t.name COLLATE NOCASE, u.display_name COLLATE NOCASE
            """
        ).fetchall()
    return [dict(r) for r in rows]


def save_game(
    title: str,
    video_path: str = "",
    game_scope: str = "full_game",
    quarter_ms: Optional[int] = None,
    overtime_count: int = 0,
    host_user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    source_job_id: Optional[str] = None,
    game_id: Optional[str] = None,
) -> str:
    game_id = game_id or uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO games
               (id, source_job_id, host_user_id, team_id, title, video_path, game_scope, quarter_ms, overtime_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id,
                source_job_id,
                host_user_id,
                team_id,
                title.strip(),
                video_path,
                game_scope,
                quarter_ms,
                overtime_count,
            ),
        )
    return game_id


def load_all_games() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT g.*,
                   u.display_name AS host_name,
                   t.name AS team_name
            FROM games g
            LEFT JOIN users u ON u.id = g.host_user_id
            LEFT JOIN teams t ON t.id = g.team_id
            ORDER BY g.created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def save_game_invite(
    game_id: str,
    user_id: Optional[str] = None,
    invited_email: Optional[str] = None,
    status: str = "pending",
    invite_id: Optional[str] = None,
) -> str:
    invite_id = invite_id or uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO game_invites
               (id, game_id, user_id, invited_email, status)
               VALUES (?, ?, ?, ?, ?)""",
            (invite_id, game_id, user_id, (invited_email or "").strip().lower() or None, status),
        )
    return invite_id


def load_game_invites() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT gi.*, g.title AS game_title, u.display_name, u.email
            FROM game_invites gi
            JOIN games g ON g.id = gi.game_id
            LEFT JOIN users u ON u.id = gi.user_id
            ORDER BY gi.created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def save_player_game_link(
    game_id: str,
    user_id: str,
    detected_player_id: Optional[str] = None,
    detected_player_name: Optional[str] = None,
    team_side: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO player_game_links
               (id, game_id, user_id, detected_player_id, detected_player_name, team_side)
               VALUES (
                   COALESCE((SELECT id FROM player_game_links WHERE game_id = ? AND user_id = ?), NULL),
                   ?, ?, ?, ?, ?
               )""",
            (
                game_id,
                user_id,
                game_id,
                user_id,
                detected_player_id,
                detected_player_name,
                team_side,
            ),
        )


def load_player_game_links() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT pgl.*, g.title AS game_title, u.display_name
            FROM player_game_links pgl
            JOIN games g ON g.id = pgl.game_id
            JOIN users u ON u.id = pgl.user_id
            ORDER BY pgl.created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]
