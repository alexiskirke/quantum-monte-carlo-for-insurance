"""Automated data pipeline: download, cache, fallback, and distribution fitting.

Attempts to download NOAA Storm Events bulk data (property-damage column).
Falls back to a synthetic heavy-tailed (Pareto) dataset when the download
fails.  Fits the empirical data to a lognormal distribution and returns
the parameters needed for quantum state preparation.
"""

from __future__ import annotations

import json
import os
import gzip
import io
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import stats

from fairy_queen.logging_config import get_logger

CACHE_DIR = Path("data/cache")

NOAA_URL = (
    "https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
    "StormEvents_details-ftp_v1.0_d2022_c20231017.csv.gz"
)


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


def download_and_cache_losses(
    url: str = NOAA_URL,
    min_loss: float = 1_000.0,
) -> np.ndarray:
    """Download NOAA Storm Events data, extract property damage, cache locally.

    Returns an array of positive loss values (USD) above *min_loss*.
    """
    log = get_logger()
    cache_file = CACHE_DIR / "noaa_losses.npy"
    meta_file = CACHE_DIR / "noaa_meta.json"

    if cache_file.exists():
        log.info("Loading cached NOAA loss data from %s", cache_file)
        losses = np.load(cache_file)
        if len(losses) > 100:
            return losses
        log.warning("Cached data too small (%d rows); re-downloading.", len(losses))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import requests

        log.info("Downloading NOAA Storm Events from %s ...", url)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        with gzip.open(io.BytesIO(resp.content), "rt", encoding="latin-1") as fh:
            import csv
            reader = csv.DictReader(fh)
            raw_damages = []
            for row in reader:
                val = _parse_noaa_damage(row.get("DAMAGE_PROPERTY", ""))
                if val >= min_loss:
                    raw_damages.append(val)

        losses = np.array(raw_damages, dtype=np.float64)
        log.info("Downloaded %d loss records (>= $%.0f).", len(losses), min_loss)

        if len(losses) < 100:
            raise ValueError("Too few records after filtering.")

        np.save(cache_file, losses)
        with open(meta_file, "w") as f:
            json.dump({"source": url, "n_records": len(losses),
                        "min_loss": min_loss}, f)
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
