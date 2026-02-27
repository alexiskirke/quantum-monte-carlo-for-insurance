"""Automated data pipeline: download, cache, fallback, and distribution fitting.

Provides two data sources:
  1. **Real NOAA Storm Events** – Property damage records from 2020–2024,
     dynamically discovered from the NOAA/NCEI FTP directory.
  2. **Synthetic Pareto** – Fallback heavy-tailed data for offline use.

Both are fitted to a lognormal distribution for quantum state preparation.
"""

from __future__ import annotations

import json
import re
import gzip
import io
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import stats

from fairy_queen.logging_config import get_logger

CACHE_DIR = Path("data/cache")

NOAA_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
NOAA_YEARS = [2020, 2021, 2022, 2023, 2024]


def _parse_noaa_damage(raw: str) -> float:
    """Convert NOAA damage string like '25.00K' or '1.50M' to a float in USD."""
    if not raw or raw.strip() == "":
        return 0.0
    raw = raw.strip().upper()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
    for suffix, mult in multipliers.items():
        if raw.endswith(suffix):
            try:
                return float(raw[:-1]) * mult
            except ValueError:
                return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _discover_noaa_urls(years: list[int] | None = None) -> list[str]:
    """Scrape the NOAA directory listing to find the latest detail files."""
    import requests

    if years is None:
        years = NOAA_YEARS

    log = get_logger()
    log.info("Discovering NOAA Storm Events files from %s ...", NOAA_BASE_URL)
    resp = requests.get(NOAA_BASE_URL, timeout=30)
    resp.raise_for_status()

    pattern = r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz"
    matches = re.findall(pattern, resp.text)

    latest: dict[int, str] = {}
    for year_str, compiled in matches:
        yr = int(year_str)
        if yr in years:
            if yr not in latest or compiled > latest[yr]:
                latest[yr] = compiled

    urls = []
    for yr in sorted(latest):
        fname = f"StormEvents_details-ftp_v1.0_d{yr}_c{latest[yr]}.csv.gz"
        urls.append(NOAA_BASE_URL + fname)
        log.info("  Found: %s", fname)

    return urls


def download_and_cache_noaa(
    min_loss: float = 1_000.0,
    years: list[int] | None = None,
) -> np.ndarray:
    """Download NOAA Storm Events data across multiple years, cache locally.

    Returns an array of positive loss values (USD) above *min_loss*.
    """
    import requests

    log = get_logger()
    cache_file = CACHE_DIR / "noaa_real_losses.npy"
    meta_file = CACHE_DIR / "noaa_real_meta.json"

    if cache_file.exists():
        log.info("Loading cached real NOAA loss data from %s", cache_file)
        losses = np.load(cache_file)
        if len(losses) > 100:
            return losses
        log.warning("Cached data too small (%d rows); re-downloading.", len(losses))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    urls = _discover_noaa_urls(years)

    all_damages: list[float] = []
    for url in urls:
        log.info("Downloading %s ...", url.split("/")[-1])
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        import csv
        with gzip.open(io.BytesIO(resp.content), "rt", encoding="latin-1") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                val = _parse_noaa_damage(row.get("DAMAGE_PROPERTY", ""))
                if val >= min_loss:
                    all_damages.append(val)

    losses = np.array(all_damages, dtype=np.float64)
    log.info("Downloaded %d real loss records (>= $%.0f) across %d years.",
             len(losses), min_loss, len(urls))

    np.save(cache_file, losses)
    downloaded_years = []
    for u in urls:
        m = re.search(r"_d(\d{4})_c", u)
        if m:
            downloaded_years.append(int(m.group(1)))
    with open(meta_file, "w") as f:
        json.dump({"source": "noaa_storm_events",
                    "years": sorted(set(downloaded_years)),
                    "n_records": len(losses), "min_loss": min_loss}, f)
    return losses


def download_and_cache_losses(
    url: str | None = None,
    min_loss: float = 1_000.0,
) -> np.ndarray:
    """Download NOAA data or fall back to synthetic. For backward compatibility."""
    log = get_logger()
    cache_file = CACHE_DIR / "noaa_losses.npy"
    meta_file = CACHE_DIR / "noaa_meta.json"

    if cache_file.exists():
        log.info("Loading cached loss data from %s", cache_file)
        losses = np.load(cache_file)
        if len(losses) > 100:
            return losses

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        losses = download_and_cache_noaa(min_loss=min_loss)
        np.save(cache_file, losses)
        with open(meta_file, "w") as f:
            json.dump({"source": "noaa_real", "n_records": len(losses)}, f)
        return losses
    except Exception as exc:
        log.warning("NOAA download failed (%s). Generating synthetic data.", exc)
        return _generate_synthetic_losses(cache_file, meta_file)


def _generate_synthetic_losses(
    cache_file: Path, meta_file: Path
) -> np.ndarray:
    """Fallback: synthetic Pareto-tailed losses typical for cat-risk.

    Parameters chosen to mimic US hurricane property damage:
      shape (alpha) = 1.5  (heavy tail)
      scale          = 50_000 USD  (minimum modelled loss)
    """
    log = get_logger()
    rng = np.random.default_rng(42)

    pareto_alpha = 1.5
    pareto_scale = 50_000.0
    n_samples = 20_000

    raw = (rng.pareto(pareto_alpha, size=n_samples) + 1) * pareto_scale
    losses = raw.astype(np.float64)
    log.info(
        "Generated %d synthetic Pareto losses (alpha=%.1f, scale=$%.0f).",
        n_samples, pareto_alpha, pareto_scale,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, losses)
    with open(meta_file, "w") as f:
        json.dump({"source": "synthetic_pareto", "n_records": n_samples,
                    "alpha": pareto_alpha, "scale": pareto_scale}, f)
    return losses


def get_synthetic_loss_data() -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Generate synthetic Pareto losses and fit lognormal."""
    cache_file = CACHE_DIR / "synthetic_losses.npy"
    meta_file = CACHE_DIR / "synthetic_meta.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        losses = np.load(cache_file)
    else:
        losses = _generate_synthetic_losses(cache_file, meta_file)

    params = fit_lognormal(losses)
    return losses, params


def get_real_loss_data() -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Download real NOAA Storm Events data and fit lognormal."""
    losses = download_and_cache_noaa()
    params = fit_lognormal(losses)
    return losses, params


def fit_lognormal(losses: np.ndarray) -> Tuple[float, float, float]:
    """Fit a lognormal distribution to the loss data via MLE.

    Returns (shape_sigma, loc, scale) as used by scipy.stats.lognorm.
    The lognormal PDF is parameterised so that log(X - loc) ~ N(mu, sigma^2)
    with mu = log(scale) and sigma = shape.
    """
    log = get_logger()
    shape, loc, scale = stats.lognorm.fit(losses, floc=0)
    mu = np.log(scale)
    sigma = shape
    log.info(
        "Lognormal fit: mu=%.4f, sigma=%.4f  (median=$%.0f)",
        mu, sigma, np.exp(mu),
    )
    return shape, loc, scale


def get_loss_data() -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """High-level entry: download/cache data, fit distribution, return both."""
    losses = download_and_cache_losses()
    params = fit_lognormal(losses)
    return losses, params
