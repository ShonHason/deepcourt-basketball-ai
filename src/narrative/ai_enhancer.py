"""
ai_enhancer.py — Claude Haiku enrichment layer on top of rule-based narrator.

Cost optimisations:
  1. Haiku 4.5 (~20× cheaper than Sonnet)
  2. Events compressed to ~30 chars each instead of full JSON
  3. Prompt caching on system prompt (90% discount on repeat calls)
  4. Single API call for the whole game — no per-event calls

Estimated cost: ~$0.005-0.01 per game.

Usage:
    from src.narrative.ai_enhancer import enhance_game

    enriched = enhance_game(events, box_score)
    print(enriched["game_summary"])
    print(enriched["player_insights"]["#7"])
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Compact event formatter  (30-40 chars per event instead of 500+ JSON chars)
# ---------------------------------------------------------------------------

_SHORT = {
    "two_pointer_made":    "2PT MADE",
    "two_pointer_missed":  "2PT MISS",
    "three_pointer_made":  "3PT MADE",
    "three_pointer_missed":"3PT MISS",
    "free_throw_made":     "FT MADE",
    "free_throw_missed":   "FT MISS",
    "assist":              "AST",
    "rebound_offensive":   "OREB",
    "rebound_defensive":   "DREB",
    "steal":               "STL",
    "block":               "BLK",
    "turnover_steal":      "TOV",
    "turnover_unforced":   "TOV",
}

_ZONE_SHORT = {
    "restricted_area": "rim",
    "paint":           "paint",
    "mid_range":       "mid",
    "three_point":     "3pt",
    "backcourt":       "backcourt",
}


def _compress_events(events: list[dict]) -> str:
    """
    Convert events list → compact multi-line string.
    Example line: "02m15s #7 A 2PT MADE paint"
    """
    lines = []
    for ev in events:
        ts    = ev.get("timestamp_ms", 0)
        mm    = ts // 60000
        ss    = (ts // 1000) % 60
        clock = f"{mm:02d}m{ss:02d}s"
        player = (ev.get("player_name") or ev.get("player_id") or "?")[:4]
        team   = (ev.get("team") or "?")[0].upper()
        etype  = _SHORT.get(ev.get("event_type", ""), ev.get("event_type", ""))
        zone   = _ZONE_SHORT.get(ev.get("court_zone", ""), "")
        cont   = "C" if ev.get("contested") else ""
        lm     = "LM" if ev.get("is_learning_moment") else ""
        parts  = [clock, player, team, etype]
        if zone: parts.append(zone)
        if cont: parts.append(cont)
        if lm:   parts.append(lm)
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _compress_box_score(box_score: dict) -> str:
    """Compact box score: one player per line."""
    lines = []
    for player, stats in (box_score.get("players") or {}).items():
        pts  = stats.get("pts", 0)
        reb  = stats.get("reb", 0)
        ast  = stats.get("ast", 0)
        stl  = stats.get("stl", 0)
        blk  = stats.get("blk", 0)
        fgm  = stats.get("fgm", 0)
        fga  = stats.get("fga", 0)
        tov  = stats.get("tov", 0)
        lines.append(
            f"{player}: {pts}pts {reb}reb {ast}ast {stl}stl "
            f"{blk}blk {fgm}/{fga}FG {tov}to"
        )
    return "\n".join(lines) if lines else "No box score available."


# ---------------------------------------------------------------------------
# System prompt (cached — paid once per session)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a basketball analyst assistant for DeepCourt.ai, a youth basketball platform.

Your job: given a compact event log + box score from an auto-detected basketball game,
produce THREE outputs in JSON:

1. "game_summary" (2-3 sentences): flowing narrative summary of the game.
2. "player_insights": dict of {player_name: one coaching sentence focusing on one strength
   and one improvement opportunity}. Keep it encouraging — every miss is an "Optimization
   Opportunity", not a failure.
3. "standout_moments": list of up to 3 short strings describing the most memorable plays.

Rules:
- Output ONLY valid JSON, no markdown fences, no extra text.
- Keep language simple — these are youth players and coaches.
- If a player has 0 points but good rebounds/assists, highlight that.
- Use "LM" markers in the event log as learning moment indicators.
- Contested shots ("C") that go in are highlights; contested misses are learning moments.

JSON schema:
{
  "game_summary": "string",
  "player_insights": {"#7": "string", ...},
  "standout_moments": ["string", ...]
}"""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def enhance_game(
    events: list[dict],
    box_score: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Enrich game data with Claude Haiku.

    Returns dict with keys: game_summary, player_insights, standout_moments.
    Falls back to empty strings if API key is missing or call fails.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {"game_summary": "", "player_insights": {}, "standout_moments": []}

    try:
        import anthropic
    except ImportError:
        return {"game_summary": "", "player_insights": {}, "standout_moments": []}

    compressed_events = _compress_events(events)
    compressed_box    = _compress_box_score(box_score or {})

    user_content = (
        f"EVENT LOG:\n{compressed_events}\n\n"
        f"BOX SCORE:\n{compressed_box}"
    )

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # cached after first call
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        import json
        text = response.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)

        # Log token usage so caller can track cost
        usage = response.usage
        input_tokens  = getattr(usage, "input_tokens", 0)
        cache_read    = getattr(usage, "cache_read_input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        # Haiku pricing: $0.80/MTok input, $0.08/MTok cache-read, $4/MTok output
        cost = (
            (input_tokens - cache_read) * 0.80 / 1_000_000
            + cache_read               * 0.08 / 1_000_000
            + output_tokens            * 4.00 / 1_000_000
        )
        result["_tokens"] = {
            "input": input_tokens,
            "cache_read": cache_read,
            "output": output_tokens,
            "estimated_cost_usd": round(cost, 5),
        }
        return result

    except Exception as e:
        return {
            "game_summary": "",
            "player_insights": {},
            "standout_moments": [],
            "_error": str(e),
        }
