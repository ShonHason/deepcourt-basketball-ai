"""
ScoreboardReader — reads game clock and score from physical LED scoreboards.

Finds the scoreboard by detecting bright LED-coloured pixels (yellow clock,
red scores) in the upper portion of each frame. When found, runs EasyOCR on
the cropped region and parses clock (MM:SS) and score (two integers).

Quarter boundaries are inferred from clock resets: if the clock jumps from
< 30 s to > 8:30, a new period has started.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScoreboardReading:
    video_ts_ms:   int    # video timestamp when reading was captured
    game_clock_ms: int    # time remaining in current period (ms)
    period:        int    # estimated period number (1-based)
    home_score:    int
    away_score:    int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ROI: search for scoreboard in top N% of frame
SCOREBOARD_ROI_FRAC = 0.28

# Minimum LED cluster size to be considered a scoreboard (pixels in ROI)
MIN_LED_WIDTH  = 20
MIN_LED_HEIGHT = 8

# Upscale factor applied before OCR (LED digits at source resolution are tiny)
OCR_UPSCALE = 4

# Period boundary: clock jumps from < this → new period started
CLOCK_RESET_THRESHOLD_MS = 30_000   # < 30 s left = "end of period" territory
CLOCK_NEW_PERIOD_MS      = 510_000  # > 8:30 = new period has begun (handles warmup delay)

PERIOD_DURATION_MS = 10 * 60 * 1000  # 10-min FIBA quarters


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class ScoreboardReader:
    """
    Feed frames via read_frame().  After the pipeline completes, call
    get_period_timestamps() to get period → video_ts_ms mapping, and
    infer_score() to get the final scoreboard-confirmed score.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        import easyocr
        self._ocr = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        self._readings: list[ScoreboardReading] = []
        self._current_period: int = 1
        self._prev_clock_ms: Optional[int] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def read_frame(self, frame: np.ndarray, video_ts_ms: int) -> Optional[ScoreboardReading]:
        """
        Attempt to read the scoreboard from *frame*.
        Returns a ScoreboardReading if successful, None if scoreboard not visible.
        """
        crop = self._find_scoreboard(frame)
        if crop is None:
            return None

        text = self._ocr_crop(crop)
        reading = self._parse(text, video_ts_ms)
        if reading is None:
            return None

        # Detect period transitions
        if self._prev_clock_ms is not None:
            if (self._prev_clock_ms < CLOCK_RESET_THRESHOLD_MS
                    and reading.game_clock_ms > CLOCK_NEW_PERIOD_MS):
                self._current_period += 1

        reading.period = self._current_period
        self._prev_clock_ms = reading.game_clock_ms
        self._readings.append(reading)
        return reading

    def get_period_timestamps(self) -> dict[int, int]:
        """
        Return {period: video_ts_ms_when_period_started}.
        Period 1 always starts at 0.
        """
        result: dict[int, int] = {1: 0}
        seen = set()
        for r in sorted(self._readings, key=lambda x: x.video_ts_ms):
            if r.period not in seen:
                if r.period > 1:
                    result[r.period] = r.video_ts_ms
                seen.add(r.period)
        return result

    def infer_final_score(self) -> Optional[tuple[int, int]]:
        """Return (home, away) from the last reliable scoreboard reading."""
        if not self._readings:
            return None
        last = max(self._readings, key=lambda r: r.video_ts_ms)
        return last.home_score, last.away_score

    def get_readings(self) -> list[ScoreboardReading]:
        return list(self._readings)

    # ------------------------------------------------------------------
    # Internal: detection
    # ------------------------------------------------------------------

    def _find_scoreboard(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the LED scoreboard by locating a dark rectangle that contains
        bright digits — a universal property of all physical LED scoreboards
        regardless of colour scheme or arena.
        """
        h, w = frame.shape[:2]
        roi_h = int(h * SCOREBOARD_ROI_FRAC)

        # Center 70% horizontally — avoids court-side banners / backboard arms
        x_lo = int(w * 0.15)
        x_hi = int(w * 0.85)
        roi  = frame[:roi_h, x_lo:x_hi]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Step 1: find dark regions — LED housing is nearly black
        dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
        dark = cv2.dilate(dark, np.ones((5, 8), np.uint8), iterations=2)

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_crop: Optional[np.ndarray] = None
        best_score = 0

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # Plausible scoreboard size: not too small, not too large
            if bw < MIN_LED_WIDTH or bh < MIN_LED_HEIGHT:
                continue
            if bw > w // 2 or bh > roi_h // 2:
                continue

            patch_gray = gray[y:y+bh, x:x+bw]

            # Step 2: require bright pixels inside the dark housing
            bright_count = int((patch_gray > 140).sum())
            dark_count   = int((patch_gray < 50).sum())
            total        = bw * bh

            # LED board: roughly 20-70 % dark housing + some bright digits
            dark_frac   = dark_count / total
            bright_frac = bright_count / total
            if dark_frac < 0.15 or bright_frac < 0.03:
                continue

            # Score: favour compact regions with more bright pixels
            score = bright_count
            if score > best_score:
                best_score = score
                pad = 5
                y1 = max(0, y - pad)
                y2 = min(roi_h, y + bh + pad)
                x1 = max(0, x - pad)
                x2 = min(x_hi - x_lo, x + bw + pad)
                best_crop = roi[y1:y2, x1:x2]

        return best_crop

    def _ocr_crop(self, crop: np.ndarray) -> str:
        target_h = max(crop.shape[0] * OCR_UPSCALE, 60)
        scale = target_h / crop.shape[0]
        resized = cv2.resize(
            crop,
            (int(crop.shape[1] * scale), target_h),
            interpolation=cv2.INTER_CUBIC,
        )
        # Increase contrast for LED digits
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = self._ocr.readtext(thresh, detail=0, allowlist="0123456789: ")
        return " ".join(results)

    # ------------------------------------------------------------------
    # Internal: parsing
    # ------------------------------------------------------------------

    def _parse(self, text: str, video_ts_ms: int) -> Optional[ScoreboardReading]:
        if not text.strip():
            return None

        # Game clock: MM:SS or M:SS
        clock_match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
        if not clock_match:
            return None
        mins = int(clock_match.group(1))
        secs = int(clock_match.group(2))
        if mins > 10 or secs > 59:
            return None
        game_clock_ms = (mins * 60 + secs) * 1000

        # Scores: all other integers in the text
        remainder = text[:clock_match.start()] + text[clock_match.end():]
        numbers = [int(n) for n in re.findall(r"\b(\d{1,3})\b", remainder)
                   if int(n) <= 199]
        if len(numbers) < 2:
            return None

        return ScoreboardReading(
            video_ts_ms=video_ts_ms,
            game_clock_ms=game_clock_ms,
            period=self._current_period,
            home_score=numbers[0],
            away_score=numbers[1],
        )
