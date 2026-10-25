"""
Video Fetcher Agent — searches YouTube for amateur/low-level basketball footage
and downloads full games or full quarters into data/videoMockup/.

Default target is amateur basketball (street, high school, youth, local leagues).
Pro/college are optional. Always enforces full-game or full-quarter duration.

Usage (CLI):
    python -m src.agents.video_fetcher                              # interactive
    python -m src.agents.video_fetcher "under 16 league" --level amateur
    python -m src.agents.video_fetcher "high school championship" --quality medium
    python -m src.agents.video_fetcher "NCAA college" --level college --quality high
    python -m src.agents.video_fetcher --url "https://youtube.com/..." --quality low
    python -m src.agents.video_fetcher --list

Quality presets:
    low    → 360p  (~80-300 MB for a full amateur game)
    medium → 720p  (~300-700 MB)
    high   → 1080p (~700 MB-2 GB)

Level presets:
    amateur  → street ball, high school, youth, local leagues  (DEFAULT)
    college  → NCAA, university leagues
    pro      → NBA, EuroLeague, FIBA international

Valid downloads:
    Full game   ≥ 35 min   (amateur games run 40-80 min with breaks)
    Full quarter ≥ 8 min   (a single 8-10 min quarter)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yt_dlp  # type: ignore[import-untyped]
    _YTDLP_AVAILABLE = True
except ImportError:
    _YTDLP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "videoMockup"

# ---------------------------------------------------------------------------
# Quality format strings
# ---------------------------------------------------------------------------

QUALITY_FORMATS = {
    "low":    "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best[height<=360]",
    "medium": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
    "high":   "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]",
}

QUALITY_LABELS = {
    "low":    "≤360p  (~80-300 MB for a full amateur game)",
    "medium": "≤720p  (~300-700 MB)",
    "high":   "≤1080p (~700 MB-2 GB)",
}

# ---------------------------------------------------------------------------
# Level-specific search context terms
# ---------------------------------------------------------------------------

LEVEL_CONTEXT = {
    "amateur": [
        "street basketball",
        "high school basketball",
        "youth basketball",
        "local league basketball",
        "recreational basketball",
        "amateur basketball",
        "under 16 basketball",
        "under 18 basketball",
        "pickup basketball game",
    ],
    "college": [
        "college basketball",
        "NCAA basketball",
        "university basketball",
        "JUCO basketball",
    ],
    "pro": [
        "NBA basketball",
        "EuroLeague basketball",
        "FIBA basketball",
        "professional basketball",
    ],
}

LEVEL_LABELS = {
    "amateur": "Amateur / Street / High School / Youth",
    "college": "College / NCAA / University",
    "pro":     "Professional / NBA / EuroLeague",
}

# ---------------------------------------------------------------------------
# Duration thresholds
# ---------------------------------------------------------------------------

FULL_GAME_MIN_SEC    = 35 * 60   # ≥35 min — covers short amateur games
FULL_HALF_MIN_SEC    = 18 * 60   # ≥18 min — a full half (18-35 min)
FULL_QUARTER_MIN_SEC =  8 * 60   # ≥8 min  — a single quarter (8-12 min)
FULL_QUARTER_MAX_SEC = 18 * 60   # <18 min — anything longer is at least a half

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VideoResult:
    index: int
    title: str
    url: str
    duration_sec: int
    uploader: str
    view_count: int
    thumbnail: str
    description: str

    @property
    def duration_str(self) -> str:
        h = self.duration_sec // 3600
        m = (self.duration_sec % 3600) // 60
        s = self.duration_sec % 60
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

    @property
    def is_full_game(self) -> bool:
        return self.duration_sec >= FULL_GAME_MIN_SEC

    @property
    def is_full_half(self) -> bool:
        return FULL_HALF_MIN_SEC <= self.duration_sec < FULL_GAME_MIN_SEC

    @property
    def is_full_quarter(self) -> bool:
        return FULL_QUARTER_MIN_SEC <= self.duration_sec < FULL_QUARTER_MAX_SEC

    @property
    def is_usable(self) -> bool:
        """True if this is a full game, full half, or at least a full quarter."""
        return self.is_full_game or self.is_full_half or self.is_full_quarter

    @property
    def content_label(self) -> str:
        if self.is_full_game:
            return "[FULL GAME]"
        if self.is_full_half:
            return "[FULL HALF]"
        if self.is_full_quarter:
            return "[FULL QUARTER]"
        return "[SHORT CLIP]"

    def display(self) -> str:
        views = f"{self.view_count:,}" if self.view_count else "N/A"
        usable_marker = "✓" if self.is_usable else "✗"
        return (
            f"[{self.index}] {usable_marker} {self.content_label}  {self.title}\n"
            f"     Duration: {self.duration_str}  |  Views: {views}\n"
            f"     Uploader: {self.uploader}\n"
            f"     URL: {self.url}"
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class VideoFetcher:
    """
    Searches YouTube for amateur basketball footage (full games or full quarters)
    and downloads to data/videoMockup/.

    No API key required — uses yt-dlp's built-in YouTube search.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        if not _YTDLP_AVAILABLE:
            raise ImportError("yt-dlp is not installed. Run:  pip install yt-dlp")
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        quality: str = "low",
        level: str = "amateur",
        limit: int = 10,
        usable_only: bool = True,
    ) -> list[VideoResult]:
        """
        Search YouTube for basketball footage.

        Args:
            query:       e.g. "under 16 championship game", "street ball NYC"
            quality:     "low" | "medium" | "high"
            level:       "amateur" | "college" | "pro"
            limit:       results to fetch before filtering (max 20)
            usable_only: if True, hide results shorter than one full quarter

        Returns:
            VideoResult list — usable entries (✓) listed first.
        """
        if quality not in QUALITY_FORMATS:
            raise ValueError(f"quality must be one of {list(QUALITY_FORMATS)}")
        if level not in LEVEL_CONTEXT:
            raise ValueError(f"level must be one of {list(LEVEL_CONTEXT)}")

        limit = min(limit, 20)
        search_query = self._build_query(query, level)

        print(f"\nSearching YouTube: \"{search_query}\"")
        print(f"Level:   {LEVEL_LABELS[level]}")
        print(f"Quality: {QUALITY_LABELS[quality]}\n")

        ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            raw = ydl.extract_info(f"ytsearch{limit}:{search_query}", download=False)

        entries = raw.get("entries", []) if raw else []
        results = []
        idx = 1

        for entry in entries:
            if not entry:
                continue
            r = VideoResult(
                index=idx,
                title=entry.get("title", "Unknown"),
                url=(entry.get("url")
                     or f"https://www.youtube.com/watch?v={entry.get('id', '')}"),
                duration_sec=int(entry.get("duration") or 0),
                uploader=entry.get("uploader") or entry.get("channel", "Unknown"),
                view_count=entry.get("view_count") or 0,
                thumbnail=entry.get("thumbnail", ""),
                description=(entry.get("description") or "")[:200],
            )
            if usable_only and not r.is_usable:
                continue
            results.append(r)
            idx += 1

        # Sort: full games → full halves → full quarters
        def _sort_key(r: VideoResult) -> tuple[int, int]:
            if r.is_full_game:    return (0, -r.duration_sec)
            if r.is_full_half:    return (1, -r.duration_sec)
            if r.is_full_quarter: return (2, -r.duration_sec)
            return (3, -r.duration_sec)

        results.sort(key=_sort_key)
        for i, r in enumerate(results, 1):
            r.index = i

        return results

    def download(
        self,
        video: VideoResult,
        quality: str = "low",
        filename: Optional[str] = None,
    ) -> Path:
        """Download a VideoResult to data/videoMockup/."""
        if quality not in QUALITY_FORMATS:
            raise ValueError(f"quality must be one of {list(QUALITY_FORMATS)}")

        safe_title = _safe_filename(filename or video.title)
        output_template = str(self.output_dir / f"{safe_title}.%(ext)s")

        print(f"\nDownloading: {video.title}")
        print(f"Content:     {video.content_label}  ({video.duration_str})")
        print(f"Quality:     {QUALITY_LABELS[quality]}")
        print(f"Destination: {self.output_dir}\n")

        ydl_opts = {
            "format": QUALITY_FORMATS[quality],
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "progress_hooks": [_progress_hook],
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video.url])

        matches = [
            p for p in self.output_dir.glob(f"{safe_title}.*")
            if p.suffix != ".json"
        ]
        downloaded = matches[0] if matches else self.output_dir / f"{safe_title}.mp4"
        print(f"\nSaved to: {downloaded}")
        self._write_metadata(video, downloaded, quality)
        return downloaded

    def download_by_url(
        self, url: str, quality: str = "low", filename: Optional[str] = None
    ) -> Path:
        """Download a specific URL directly — no search needed."""
        ydl_opts = {"quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            raise ValueError(f"Could not fetch metadata for {url}")
        result = VideoResult(
            index=1,
            title=info.get("title", "video"),
            url=url,
            duration_sec=int(info.get("duration") or 0),
            uploader=info.get("uploader") or info.get("channel", "Unknown"),
            view_count=info.get("view_count") or 0,
            thumbnail=info.get("thumbnail", ""),
            description=(info.get("description") or "")[:200],
        )
        return self.download(result, quality=quality, filename=filename)

    def list_downloads(self) -> list[dict]:
        """Return metadata for all previously downloaded videos."""
        results = []
        for f in sorted(self.output_dir.glob("*.meta.json")):
            with open(f) as fp:
                results.append(json.load(fp))
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(raw: str, level: str) -> str:
        """
        Enrich the user's query with level-appropriate context and
        ensure it targets full games or full quarters.
        """
        lower = raw.lower()

        # Inject level context if the query doesn't already mention it
        context_terms = LEVEL_CONTEXT[level]
        has_context = any(t.split()[0] in lower for t in context_terms)
        if not has_context:
            context = random.choice(context_terms[:3])   # pick from top-3 terms
            raw = f"{context} {raw}"

        # Ensure the query targets full content, not highlights
        has_duration_hint = any(
            w in lower for w in ["full game", "full quarter", "full match", "complete game"]
        )
        if not has_duration_hint:
            raw = f"{raw} full game OR full quarter"

        return raw

    def _write_metadata(self, video: VideoResult, path: Path, quality: str) -> None:
        meta = {
            "title": video.title,
            "url": video.url,
            "duration_sec": video.duration_sec,
            "content_label": video.content_label,
            "uploader": video.uploader,
            "view_count": video.view_count,
            "quality": quality,
            "local_path": str(path),
        }
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_filename(name: str, max_len: int = 80) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._-")
    cleaned = "".join(c if c in keep else "_" for c in name)
    cleaned = "_".join(cleaned.split())
    return cleaned[:max_len]


def _progress_hook(d: dict) -> None:
    if d["status"] == "downloading":
        pct   = d.get("_percent_str", "?%").strip()
        speed = d.get("_speed_str", "?").strip()
        eta   = d.get("_eta_str", "?").strip()
        print(f"\r  {pct}  speed: {speed}  eta: {eta}    ", end="", flush=True)
    elif d["status"] == "finished":
        print("\r  Download complete. Merging streams...        ")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_interactive_search(
    agent: VideoFetcher, query: str, quality: str, level: str, limit: int
) -> None:
    results = agent.search(query, quality=quality, level=level, limit=limit)

    if not results:
        print("No usable results found (all clips were too short).")
        print("Try broadening your query or removing specific team names.")
        return

    full_games    = [r for r in results if r.is_full_game]
    full_quarters = [r for r in results if r.is_full_quarter]

    print(f"Found {len(results)} usable result(s):  "
          f"{len(full_games)} full game(s), {len(full_quarters)} full quarter(s)\n")

    for r in results:
        print(r.display())
        print()

    print("Enter result number to download (or 'q' to quit): ", end="")
    choice = input().strip()
    if choice.lower() == "q":
        return

    try:
        idx = int(choice)
        selected = next(r for r in results if r.index == idx)
    except (ValueError, StopIteration):
        print("Invalid choice.")
        return

    agent.download(selected, quality=quality)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepCourt.ai — Search and download basketball footage"
    )
    parser.add_argument(
        "query", nargs="?",
        help="What to search, e.g. 'high school championship game' or 'street ball NYC'"
    )
    parser.add_argument(
        "--level", choices=["amateur", "college", "pro"], default="amateur",
        help="Game level (default: amateur)"
    )
    parser.add_argument(
        "--quality", choices=["low", "medium", "high"], default="low",
        help="Video quality (default: low)"
    )
    parser.add_argument("--limit", type=int, default=10, help="Results to fetch (default: 10)")
    parser.add_argument("--url", help="Download a specific URL directly")
    parser.add_argument("--list", action="store_true", help="List already downloaded videos")

    args = parser.parse_args()
    fetcher = VideoFetcher()

    if args.list:
        downloads = fetcher.list_downloads()
        if not downloads:
            print("No videos downloaded yet.")
        for d in downloads:
            print(f"  {d['content_label']}  {d['title']}  [{d['quality']}]")
            print(f"           → {d['local_path']}")

    elif args.url:
        fetcher.download_by_url(args.url, quality=args.quality)

    elif args.query:
        _run_interactive_search(
            fetcher, args.query,
            quality=args.quality, level=args.level, limit=args.limit
        )

    else:
        # Fully interactive
        print("DeepCourt.ai — Video Fetcher")
        print("=" * 45)
        print("\nGame level:")
        for k, v in LEVEL_LABELS.items():
            print(f"  {k:8s}  {v}")
        print("\nVideo quality:")
        for k, v in QUALITY_LABELS.items():
            print(f"  {k:8s}  {v}")

        print("\nSearch query (e.g. 'under 16 tournament', 'street ball NYC'): ", end="")
        query = input().strip()
        if not query:
            print("No query entered.")
        else:
            print(f"Level [amateur/college/pro] (default: amateur): ", end="")
            level = input().strip() or "amateur"
            if level not in LEVEL_CONTEXT:
                level = "amateur"

            print(f"Quality [low/medium/high] (default: low): ", end="")
            quality = input().strip() or "low"
            if quality not in QUALITY_FORMATS:
                quality = "low"

            _run_interactive_search(fetcher, query, quality=quality, level=level, limit=10)
