"""Tests for cross-dataset evaluation utilities.

Groups
------
- ``TestUNSWNB15Loader``: loading and remapping UNSW-NB15 CSVs using synthetic
  data; no real dataset required.
- ``TestTransformExternal``: ``FlowFeatureExtractor.transform_external()`` and
  ``feature_means_``; no torch or PyG required.
- ``TestCLI``: smoke tests for the ``run_cross_dataset`` CLI.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic UNSW-NB15-like CSV helper
# ---------------------------------------------------------------------------

_UNSW_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "spkts", "dpkts", "stime", "label", "attack_cat",
]


def _make_unsw_csv(tmp_path: Path, n: int = 50, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "srcip": [f"10.0.0.{i % 20}" for i in range(n)],
        "sport": rng.integers(1024, 65535, n),
        "dstip": [f"192.168.1.{i % 15}" for i in range(n)],
        "dsport": rng.integers(1, 1024, n),
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "state": rng.choice(["FIN", "CON", "INT"], n),
        "dur": rng.uniform(0, 60, n),
        "sbytes": rng.integers(64, 32768, n),
        "dbytes": rng.integers(64, 32768, n),
        "spkts": rng.integers(1, 50, n),
        "dpkts": rng.integers(1, 50, n),
        "stime": np.arange(1_420_000_000, 1_420_000_000 + n, dtype=float),
        "label": rng.integers(0, 2, n),
        "attack_cat": rng.choice(["Normal", "Backdoor", "DoS"], n),
    })
    path = tmp_path / "unsw_test.csv"
    df.to_csv(path, index=False)
    return path


def _make_ctu_like_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "duration": rng.uniform(0, 60, n),
        "proto": rng.choice(["TCP", "UDP", "ICMP"], n),
        "src_port": rng.integers(0, 65535, n),
        "dst_port": rng.integers(0, 65535, n),
        "direction": rng.choice(["->", "<-", "<->"], n),
        "state": rng.choice(["FIN", "CON", "INT"], n),
        "src_tos": rng.integers(0, 4, n),
        "dst_tos": rng.integers(0, 4, n),
        "packets": rng.integers(1, 100, n),
        "bytes": rng.integers(64, 65536, n),
        "src_bytes": rng.integers(32, 32768, n),
        "label": rng.choice(["Botnet", "Normal"], n),
        "start_time": pd.date_range("2011-01-01", periods=n, freq="s"),
    })


# ---------------------------------------------------------------------------
# TestUNSWNB15Loader
# ---------------------------------------------------------------------------


class TestUNSWNB15Loader:
    def test_load_returns_dataframe(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert isinstance(df, pd.DataFrame)

    def test_load_has_label_binary(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert "label_binary" in df.columns

    def test_label_binary_is_01(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert set(df["label_binary"].unique()).issubset({0, 1})

    def test_load_has_src_addr(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert "src_addr" in df.columns

    def test_load_has_dst_addr(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert "dst_addr" in df.columns

    def test_load_has_duration(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert "duration" in df.columns

    def test_load_has_proto(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert "proto" in df.columns

    def test_bytes_is_sum_of_sbytes_dbytes(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path, n=10, seed=7)
        raw = pd.read_csv(csv)
        df = UNSWN15Loader().load(csv)
        expected = raw["sbytes"].iloc[0] + raw["dbytes"].iloc[0]
        assert df["bytes"].iloc[0] == pytest.approx(expected, abs=1)

    def test_packets_is_sum_of_spkts_dpkts(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path, n=10, seed=8)
        raw = pd.read_csv(csv)
        df = UNSWN15Loader().load(csv)
        expected = raw["spkts"].iloc[0] + raw["dpkts"].iloc[0]
        assert df["packets"].iloc[0] == pytest.approx(expected, abs=1)

    def test_direction_column_present(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert "direction" in df.columns

    def test_start_time_is_datetime(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert pd.api.types.is_datetime64_any_dtype(df["start_time"])

    def test_scenario_name_column(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load(csv)
        assert "scenario_name" in df.columns
        assert (df["scenario_name"] == "UNSW-NB15").all()

    def test_nonexistent_file_raises(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        with pytest.raises(FileNotFoundError):
            UNSWN15Loader().load(tmp_path / "missing.csv")

    def test_empty_after_remap_raises(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="No recognised"):
            UNSWN15Loader().load(bad)

    def test_load_botnet_vs_normal_filters(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv = _make_unsw_csv(tmp_path)
        df = UNSWN15Loader().load_botnet_vs_normal(csv)
        assert set(df["label_binary"].unique()).issubset({0, 1})

    def test_case_insensitive_columns(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        csv_path = tmp_path / "upper.csv"
        pd.DataFrame({
            "SRCIP": ["1.2.3.4"], "SPORT": [80], "DSTIP": ["5.6.7.8"],
            "DSPORT": [443], "PROTO": ["tcp"], "DUR": [1.0],
            "SBYTES": [100], "DBYTES": [200], "SPKTS": [5], "DPKTS": [5],
            "STIME": [1_420_000_000.0], "LABEL": [0], "ATTACK_CAT": ["Normal"],
            "STATE": ["FIN"],
        }).to_csv(csv_path, index=False)
        df = UNSWN15Loader().load(csv_path)
        assert "src_addr" in df.columns

    def test_unsw_feature_map_is_dict(self):
        from gbd_ctu.data.unsw_nb15_loader import UNSW_FEATURE_MAP
        assert isinstance(UNSW_FEATURE_MAP, dict)
        assert len(UNSW_FEATURE_MAP) > 5

    def test_missing_flow_features_list_exists(self):
        from gbd_ctu.data.unsw_nb15_loader import UNSW_MISSING_FLOW_FEATURES
        assert "tos_src" in UNSW_MISSING_FLOW_FEATURES
        assert "tos_dst" in UNSW_MISSING_FLOW_FEATURES


# ---------------------------------------------------------------------------
# TestTransformExternal
# ---------------------------------------------------------------------------


class TestFeatureMeans:
    def test_feature_means_set_after_fit(self):
        from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
        ffe = FlowFeatureExtractor()
        ffe.fit(_make_ctu_like_frame())
        assert ffe.feature_means_ is not None

    def test_feature_means_shape(self):
        from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
        ffe = FlowFeatureExtractor()
        ffe.fit(_make_ctu_like_frame())
        assert ffe.feature_means_.shape == (22,)

    def test_feature_means_set_after_fit_transform(self):
        from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
        ffe = FlowFeatureExtractor()
        ffe.fit_transform(_make_ctu_like_frame())
        assert ffe.feature_means_ is not None

    def test_feature_means_is_float32(self):
        from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
        ffe = FlowFeatureExtractor()
        ffe.fit(_make_ctu_like_frame())
        assert ffe.feature_means_.dtype == np.float32

    def test_feature_means_none_before_fit(self):
        from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
        ffe = FlowFeatureExtractor()
        assert ffe.feature_means_ is None


class TestTransformExternal:
    def _fitted_extractor(self):
        from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
        ffe = FlowFeatureExtractor()
        ffe.fit(_make_ctu_like_frame(n=200, seed=0))
        return ffe

    def test_returns_correct_shape(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        ffe = self._fitted_extractor()
        csv = _make_unsw_csv(tmp_path, n=30)
        df = UNSWN15Loader().load(csv)
        out = ffe.transform_external(df)
        assert out.shape == (len(df), 22)

    def test_returns_float32(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        ffe = self._fitted_extractor()
        csv = _make_unsw_csv(tmp_path, n=20)
        df = UNSWN15Loader().load(csv)
        out = ffe.transform_external(df)
        assert out.dtype == np.float32

    def test_missing_features_filled_with_means(self, tmp_path):
        from gbd_ctu.data.feature_extractor import FLOW_FEATURE_COLUMNS
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        ffe = self._fitted_extractor()
        csv = _make_unsw_csv(tmp_path, n=30)
        df = UNSWN15Loader().load(csv)

        out_with_means = ffe.transform_external(df, missing_features=["tos_src", "tos_dst"])
        out_with_zeros = ffe.transform_external(df, missing_features=[])

        tos_src_idx = FLOW_FEATURE_COLUMNS.index("tos_src")
        # Columns filled with means vs zeros should differ
        assert not np.allclose(
            out_with_means[:, tos_src_idx],
            out_with_zeros[:, tos_src_idx],
        ) or float(ffe.feature_means_[tos_src_idx]) == pytest.approx(0.0, abs=1e-3)

    def test_raises_without_fit(self):
        from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
        ffe = FlowFeatureExtractor()
        frame = _make_ctu_like_frame(n=5)
        with pytest.raises(RuntimeError, match="fitted"):
            ffe.transform_external(frame)

    def test_empty_missing_features_works(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        ffe = self._fitted_extractor()
        csv = _make_unsw_csv(tmp_path, n=20)
        df = UNSWN15Loader().load(csv)
        out = ffe.transform_external(df, missing_features=[])
        assert out.shape == (len(df), 22)

    def test_unknown_feature_name_skipped(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        ffe = self._fitted_extractor()
        csv = _make_unsw_csv(tmp_path, n=20)
        df = UNSWN15Loader().load(csv)
        # Should not crash for a name not in FLOW_FEATURE_COLUMNS
        out = ffe.transform_external(df, missing_features=["nonexistent_col"])
        assert out.shape == (len(df), 22)

    def test_no_nans_in_output(self, tmp_path):
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        ffe = self._fitted_extractor()
        csv = _make_unsw_csv(tmp_path, n=40)
        df = UNSWN15Loader().load(csv)
        out = ffe.transform_external(df)
        assert not np.isnan(out).any()

    def test_default_missing_features_is_tos(self, tmp_path):
        """Calling without missing_features should default to tos_src / tos_dst."""
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
        ffe = self._fitted_extractor()
        csv = _make_unsw_csv(tmp_path, n=20)
        df = UNSWN15Loader().load(csv)
        out1 = ffe.transform_external(df)
        out2 = ffe.transform_external(df, missing_features=["tos_src", "tos_dst"])
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


def _env_with_src():
    import os
    env = os.environ.copy()
    src_root = str(Path(__file__).parents[3] / "src")
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


class TestCLI:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_cross_dataset", "--help"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode == 0

    def test_help_shows_key_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_cross_dataset", "--help"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        for flag in ("--ctu13-model", "--unsw-data", "--output", "--dry-run"):
            assert flag in result.stdout, f"Missing {flag!r} in --help"

    def test_dry_run_exits_zero(self, tmp_path):
        out = tmp_path / "result.json"
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_cross_dataset",
             "--dry-run", "--output", str(out)],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode == 0, result.stderr

    def test_dry_run_writes_json(self, tmp_path):
        out = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_cross_dataset",
             "--dry-run", "--output", str(out)],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert out.exists()

    def test_dry_run_json_has_auc(self, tmp_path):
        out = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_cross_dataset",
             "--dry-run", "--output", str(out)],
            capture_output=True, text=True, env=_env_with_src(),
        )
        data = json.loads(out.read_text())
        assert "auc" in data

    def test_dry_run_json_has_dataset_key(self, tmp_path):
        out = tmp_path / "result.json"
        subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_cross_dataset",
             "--dry-run", "--output", str(out)],
            capture_output=True, text=True, env=_env_with_src(),
        )
        data = json.loads(out.read_text())
        assert "dataset" in data

    def test_missing_unsw_data_exits_nonzero(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_cross_dataset",
             "--unsw-data", str(tmp_path / "missing.csv"),
             "--output", str(tmp_path / "out.json")],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode != 0

    def test_no_args_without_dry_run_exits_nonzero(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_cross_dataset",
             "--output", str(tmp_path / "out.json")],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode != 0
