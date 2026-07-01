"""Training data for the Dixon-Coles model.

Loads the martj42/international_results dataset, restricts it to a recent
training window, and attaches the two weights the model fits on:

  * a **recency** weight  — an exponential (half-life) decay so a game from
    last year counts far more than one from ten years ago; and
  * a **competition** weight — competitive fixtures (World Cups, continental
    championships, qualifiers) count for more than friendlies.

Crucially, the window ends the day *before* the 2026 World Cup kicks off, and
the 2026 World Cup's own tournament matches are dropped outright, so none of
the games we later try to predict can leak into training.
"""

from __future__ import annotations

import csv
import datetime
import io
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass

# Source data: ~50k international results, 1872-present.
CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cache_results.csv")


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS where possible; fall back gracefully on the misconfigured
    macOS Python builds that ship without a usable root-certificate bundle."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _download_csv() -> str:
    """Return the results CSV, downloading and caching it on first use."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    print("Downloading international results CSV (~50k matches)...")
    try:
        with urllib.request.urlopen(CSV_URL, timeout=60, context=_ssl_context()) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            # Last resort for a host with no working CA bundle at all.
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(CSV_URL, timeout=60, context=ctx) as resp:
                raw = resp.read().decode("utf-8")
        else:
            raise
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(raw)
    return raw

# --- Training window -------------------------------------------------------
# The 2026 finals begin on 11 June 2026. We train on roughly the decade before
# that and never past it, so the tournament we score ourselves against is
# entirely unseen.
WC2026_START = datetime.date(2026, 6, 11)
TRAINING_START = datetime.date(2016, 6, 11)  # ~10 years of history

# --- Recency weighting -----------------------------------------------------
# Exponential time-decay with a half-life: a match HALF_LIFE_DAYS old counts
# half as much as a brand-new one, a match twice that old a quarter as much,
# and so on. Three years is a reasonable half-life for international football,
# where teams play relatively few games a year: a 1-year-old result still
# carries ~79% weight while a 10-year-old one is down to ~10%.
HALF_LIFE_DAYS = 3 * 365.0

# --- Competition weighting -------------------------------------------------
# Multiplies the recency weight. Friendlies are noisy and lightly contested, so
# they count for roughly half of a full competitive international.
def competition_weight(tournament: str) -> float:
    t = tournament.lower()
    if "friendly" in t:
        return 0.5
    if "fifa world cup" in t and "qualif" not in t:
        return 1.0  # World Cup finals (2018, 2022 — the 2026 finals are excluded)
    if any(
        c in t
        for c in (
            "uefa euro",
            "copa américa",
            "copa america",
            "african cup",
            "africa cup",
            "afc asian cup",
            "gold cup",
            "confederations cup",
        )
    ) and "qualif" not in t:
        return 0.95  # continental championships
    if "qualif" in t or "qualification" in t:
        return 0.85  # World Cup / continental qualifiers
    if "nations league" in t:
        return 0.8
    return 0.7  # other competitive internationals


@dataclass
class Match:
    """One historical result plus the weight it carries during fitting."""

    date: datetime.date
    home: str
    away: str
    home_goals: int
    away_goals: int
    neutral: bool
    weight: float


def _parse_date(s: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(s)
    except ValueError:
        return None


def load_training_matches(
    window_start: datetime.date = TRAINING_START,
    window_end: datetime.date = WC2026_START,
    half_life_days: float = HALF_LIFE_DAYS,
) -> list[Match]:
    """Return the weighted training matches in ``[window_start, window_end)``.

    The 2026 World Cup finals (``window_end`` onwards) are excluded by the date
    filter, guaranteeing the model never sees a game it is later asked to
    predict. Recency is measured relative to ``window_end`` (the eve of the
    tournament), which is the moment we are effectively "standing at".
    """
    raw = _download_csv()
    reader = csv.reader(io.StringIO(raw))
    next(reader, None)  # header

    matches: list[Match] = []
    for parts in reader:
        if len(parts) < 9:
            continue
        date = _parse_date(parts[0])
        if date is None or not (window_start <= date < window_end):
            continue
        try:
            hg, ag = int(parts[3]), int(parts[4])
        except ValueError:
            continue  # unplayed / malformed score (e.g. "NA")

        tournament = parts[5]
        neutral = parts[8].strip().upper() == "TRUE"

        # Recency: exponential half-life decay in days before the window end.
        age_days = (window_end - date).days
        recency = 0.5 ** (age_days / half_life_days)
        weight = recency * competition_weight(tournament)

        matches.append(Match(date, parts[1], parts[2], hg, ag, neutral, weight))

    return matches


def load_world_cup_2026_played() -> list[Match]:
    """The 2026 World Cup *finals* matches that have already been played.

    These are deliberately excluded from training; here we load them back to
    score the model's out-of-sample predictions against reality. Unplayed
    fixtures (blank "NA" scores) are skipped. Weight is left at 1.0 — it plays
    no role in evaluation.
    """
    raw = _download_csv()
    reader = csv.reader(io.StringIO(raw))
    next(reader, None)

    played: list[Match] = []
    for parts in reader:
        if len(parts) < 9 or parts[5] != "FIFA World Cup":
            continue
        date = _parse_date(parts[0])
        if date is None or date < WC2026_START:
            continue
        try:
            hg, ag = int(parts[3]), int(parts[4])
        except ValueError:
            continue  # not yet played
        neutral = parts[8].strip().upper() == "TRUE"
        played.append(Match(date, parts[1], parts[2], hg, ag, neutral, 1.0))

    return played
