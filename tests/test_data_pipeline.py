"""Tests for the data caching and distribution fitting pipeline."""

import os
import shutil
import numpy as np
import pytest
from pathlib import Path

from fairy_queen.data_pipeline import (
    download_and_cache_losses,
    fit_lognormal,
    _parse_noaa_damage,
    CACHE_DIR,
)


class TestDamageParser:
    def test_thousands(self):
        assert _parse_noaa_damage("25.00K") == 25_000.0

    def test_millions(self):
        assert _parse_noaa_damage("1.5M") == 1_500_000.0

    def test_empty(self):
        assert _parse_noaa_damage("") == 0.0

    def test_plain_number(self):
        assert _parse_noaa_damage("5000") == 5000.0


class TestCaching:
    @pytest.fixture(autouse=True)
    def _setup_teardown(self, tmp_path, monkeypatch):
        """Point CACHE_DIR to a temp directory for isolated testing."""
        import fairy_queen.data_pipeline as dp
        self._orig = dp.CACHE_DIR
        dp.CACHE_DIR = tmp_path / "cache"
        dp.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        yield
        dp.CACHE_DIR = self._orig

    def test_fallback_generates_data(self):
        """When download fails, synthetic data should be generated and cached."""
        import fairy_queen.data_pipeline as dp
        losses = dp.download_and_cache_losses(url="http://invalid.example.com/nope")
        assert len(losses) > 1000
        assert (dp.CACHE_DIR / "noaa_losses.npy").exists()

    def test_cached_data_reused(self):
        """Second call should load from cache without re-downloading."""
        import fairy_queen.data_pipeline as dp
        losses1 = dp.download_and_cache_losses(url="http://invalid.example.com/nope")
        losses2 = dp.download_and_cache_losses(url="http://invalid.example.com/nope")
        np.testing.assert_array_equal(losses1, losses2)


class TestDistributionFit:
    def test_lognormal_params_valid(self):
        rng = np.random.default_rng(0)
        losses = rng.lognormal(mean=11, sigma=1.5, size=5000)
        shape, loc, scale = fit_lognormal(losses)
        assert shape > 0, "shape (sigma) must be positive"
        assert scale > 0, "scale must be positive"
        assert loc >= 0, "loc should be non-negative (floc=0)"

    def test_fit_recovers_approximate_params(self):
        rng = np.random.default_rng(1)
        mu_true, sigma_true = 11.0, 1.5
        losses = rng.lognormal(mean=mu_true, sigma=sigma_true, size=10_000)
        shape, loc, scale = fit_lognormal(losses)
        mu_fit = np.log(scale)
        assert abs(mu_fit - mu_true) < 0.5, f"mu_fit={mu_fit} too far from {mu_true}"
        assert abs(shape - sigma_true) < 0.5, f"sigma_fit={shape} too far from {sigma_true}"
