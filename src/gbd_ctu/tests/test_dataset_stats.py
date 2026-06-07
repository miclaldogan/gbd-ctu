"""Tests for CTU-13 dataset statistics functions.

All tests use synthetic flow DataFrames with known properties and run
without real CTU-13 data files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gbd_ctu.data.ctu13_loader import (
    SCENARIO_FAMILY,
    dataset_statistics_table,
    scenario_statistics,
)


# ---------------------------------------------------------------------------
# Synthetic flow frame helpers
# ---------------------------------------------------------------------------

def _make_flow_frame(
    n_botnet: int = 30,
    n_normal: int = 50,
    n_background: int = 20,
    n_src_ips: int = 10,
    n_dst_ips: int = 15,
    n_dst_ports: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic normalised flow DataFrame with known properties."""
    rng = np.random.default_rng(seed)
    total = n_botnet + n_normal + n_background

    src_pool = [f"10.0.0.{i}" for i in range(n_src_ips)]
    dst_pool = [f"192.168.1.{i}" for i in range(n_dst_ips)]
    port_pool = list(range(1000, 1000 + n_dst_ports))

    rows = []
    base_time = pd.Timestamp("2011-08-10 09:00:00")

    for i in range(total):
        if i < n_botnet:
            label = "Botnet"
            lb = 1
        elif i < n_botnet + n_normal:
            label = "Normal"
            lb = 0
        else:
            label = "Background"
            lb = 0

        rows.append({
            "start_time": base_time + pd.Timedelta(seconds=int(rng.integers(0, 3600))),
            "duration": float(rng.uniform(0.0, 60.0)),
            "proto": "TCP",
            "src_addr": rng.choice(src_pool),
            "src_port": int(rng.integers(1024, 65535)),
            "direction": "->",
            "dst_addr": rng.choice(dst_pool),
            "dst_port": int(rng.choice(port_pool)),
            "state": "FIN",
            "src_tos": 0,
            "dst_tos": 0,
            "packets": int(rng.integers(1, 100)),
            "bytes": int(rng.integers(64, 65536)),
            "src_bytes": int(rng.integers(32, 32768)),
            "label": label,
            "label_binary": lb,
            "scenario_name": "synthetic",
            "scenario_id": 1,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# scenario_statistics
# ---------------------------------------------------------------------------

class TestScenarioStatistics:
    def test_returns_dict(self):
        frame = _make_flow_frame()
        stats = scenario_statistics(frame)
        assert isinstance(stats, dict)

    def test_all_required_keys_present(self):
        frame = _make_flow_frame()
        stats = scenario_statistics(frame)
        required = {
            "total_flows", "botnet_flows", "botnet_frac",
            "normal_flows", "normal_frac",
            "background_flows", "background_frac",
            "unique_src_ips", "unique_dst_ips",
            "mean_duration", "max_duration",
            "botnet_host_ips", "unique_dst_ports",
            "start_time", "end_time",
        }
        assert required.issubset(set(stats.keys()))

    def test_total_flows_correct(self):
        frame = _make_flow_frame(n_botnet=30, n_normal=50, n_background=20)
        stats = scenario_statistics(frame)
        assert stats["total_flows"] == 100

    def test_botnet_flows_correct(self):
        frame = _make_flow_frame(n_botnet=30, n_normal=50, n_background=20)
        stats = scenario_statistics(frame)
        assert stats["botnet_flows"] == 30

    def test_botnet_frac_correct(self):
        frame = _make_flow_frame(n_botnet=25, n_normal=75, n_background=0)
        stats = scenario_statistics(frame)
        assert stats["botnet_frac"] == pytest.approx(0.25, abs=1e-5)

    def test_normal_flows_correct(self):
        frame = _make_flow_frame(n_botnet=30, n_normal=50, n_background=20)
        stats = scenario_statistics(frame)
        assert stats["normal_flows"] == 50

    def test_background_flows_correct(self):
        frame = _make_flow_frame(n_botnet=30, n_normal=50, n_background=20)
        stats = scenario_statistics(frame)
        assert stats["background_flows"] == 20

    def test_fracs_sum_to_one(self):
        frame = _make_flow_frame(n_botnet=20, n_normal=50, n_background=30)
        stats = scenario_statistics(frame)
        total_frac = stats["botnet_frac"] + stats["normal_frac"] + stats["background_frac"]
        assert total_frac == pytest.approx(1.0, abs=1e-4)

    def test_unique_src_ips_bounded(self):
        frame = _make_flow_frame(n_src_ips=10)
        stats = scenario_statistics(frame)
        assert 1 <= stats["unique_src_ips"] <= 10

    def test_unique_dst_ips_bounded(self):
        frame = _make_flow_frame(n_dst_ips=15)
        stats = scenario_statistics(frame)
        assert 1 <= stats["unique_dst_ips"] <= 15

    def test_unique_dst_ports_bounded(self):
        frame = _make_flow_frame(n_dst_ports=8)
        stats = scenario_statistics(frame)
        assert 1 <= stats["unique_dst_ports"] <= 8

    def test_mean_duration_nonnegative(self):
        frame = _make_flow_frame()
        stats = scenario_statistics(frame)
        assert stats["mean_duration"] >= 0.0

    def test_max_duration_geq_mean(self):
        frame = _make_flow_frame()
        stats = scenario_statistics(frame)
        assert stats["max_duration"] >= stats["mean_duration"]

    def test_botnet_host_ips_leq_unique_src_ips(self):
        frame = _make_flow_frame()
        stats = scenario_statistics(frame)
        assert stats["botnet_host_ips"] <= stats["unique_src_ips"]

    def test_botnet_frac_equals_label_binary_mean(self):
        """botnet_frac must equal (label_binary == 1).mean() per the acceptance criteria."""
        frame = _make_flow_frame(n_botnet=40, n_normal=60, n_background=0)
        stats = scenario_statistics(frame)
        expected = float((frame["label_binary"] == 1).mean())
        assert stats["botnet_frac"] == pytest.approx(expected, abs=1e-5)

    def test_start_time_before_end_time(self):
        frame = _make_flow_frame()
        stats = scenario_statistics(frame)
        if stats["start_time"] and stats["end_time"]:
            assert stats["start_time"] <= stats["end_time"]

    def test_empty_frame_returns_zeros(self):
        """Empty DataFrame must not raise and must return zero values."""
        frame = _make_flow_frame(n_botnet=0, n_normal=0, n_background=0)
        # pandas will have 0 rows
        empty = frame.iloc[:0].copy()
        stats = scenario_statistics(empty)
        assert stats["total_flows"] == 0

    def test_all_botnet_frame(self):
        frame = _make_flow_frame(n_botnet=100, n_normal=0, n_background=0)
        stats = scenario_statistics(frame)
        assert stats["botnet_frac"] == pytest.approx(1.0, abs=1e-5)
        assert stats["normal_flows"] == 0

    def test_no_botnet_frame(self):
        frame = _make_flow_frame(n_botnet=0, n_normal=80, n_background=20)
        stats = scenario_statistics(frame)
        assert stats["botnet_flows"] == 0
        assert stats["botnet_frac"] == pytest.approx(0.0, abs=1e-5)
        assert stats["botnet_host_ips"] == 0


# ---------------------------------------------------------------------------
# dataset_statistics_table
# ---------------------------------------------------------------------------

class TestDatasetStatisticsTable:
    def _make_mock_loader(self, n_scenarios: int = 3) -> MagicMock:
        loader = MagicMock()
        loader.available_scenarios.return_value = list(range(1, n_scenarios + 1))
        loader.load_scenario.side_effect = lambda sid: _make_flow_frame(seed=sid)
        return loader

    def test_returns_dataframe(self):
        loader = self._make_mock_loader()
        df = dataset_statistics_table(loader)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_scenario(self):
        loader = self._make_mock_loader(n_scenarios=5)
        df = dataset_statistics_table(loader)
        assert len(df) == 5

    def test_scenario_id_column_present(self):
        loader = self._make_mock_loader()
        df = dataset_statistics_table(loader)
        assert "scenario_id" in df.columns

    def test_family_column_present(self):
        loader = self._make_mock_loader()
        df = dataset_statistics_table(loader)
        assert "family" in df.columns

    def test_family_from_scenario_family_map(self):
        loader = self._make_mock_loader(n_scenarios=1)
        df = dataset_statistics_table(loader)
        expected = SCENARIO_FAMILY.get(1, "Unknown")
        assert df.iloc[0]["family"] == expected

    def test_sorted_by_scenario_id(self):
        loader = self._make_mock_loader(n_scenarios=4)
        df = dataset_statistics_table(loader)
        assert df["scenario_id"].tolist() == sorted(df["scenario_id"].tolist())

    def test_contains_stat_columns(self):
        loader = self._make_mock_loader()
        df = dataset_statistics_table(loader)
        for col in ("total_flows", "botnet_flows", "botnet_frac",
                    "unique_src_ips", "mean_duration"):
            assert col in df.columns, f"Missing column: {col}"

    def test_exportable_to_latex(self):
        loader = self._make_mock_loader(n_scenarios=2)
        df = dataset_statistics_table(loader)
        tex = df.to_latex(index=False, float_format="%.4f")
        assert "\\begin{tabular}" in tex

    def test_exportable_to_csv(self, tmp_path):
        loader = self._make_mock_loader(n_scenarios=2)
        df = dataset_statistics_table(loader)
        csv_path = tmp_path / "stats.csv"
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(df)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _env_with_src():
    import os
    env = os.environ.copy()
    src_root = str(Path(__file__).parents[3] / "src")
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_dataset_stats_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "gbd_ctu.scripts.dataset_stats", "--help"],
        capture_output=True, text=True, env=_env_with_src(),
    )
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
    assert "--output" in result.stdout
    assert "--latex" in result.stdout


def test_dataset_stats_cli_missing_dir_exits_nonzero(tmp_path):
    missing = tmp_path / "no_such_dir"
    result = subprocess.run(
        [sys.executable, "-m", "gbd_ctu.scripts.dataset_stats",
         "--data-dir", str(missing)],
        capture_output=True, text=True, env=_env_with_src(),
    )
    assert result.returncode != 0
