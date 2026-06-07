"""Tests for reproducibility utilities (seed_everything, ExperimentManifest)."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from gbd_ctu.training.seed import seed_everything
from gbd_ctu.training.manifest import ExperimentManifest, save_manifest


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_sets_pythonhashseed(self):
        seed_everything(7)
        assert os.environ["PYTHONHASHSEED"] == "7"

    def test_random_module_deterministic(self):
        seed_everything(42)
        a = [random.random() for _ in range(5)]
        seed_everything(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_deterministic(self):
        seed_everything(42)
        a = np.random.rand(10).tolist()
        seed_everything(42)
        b = np.random.rand(10).tolist()
        assert a == b

    def test_torch_deterministic(self):
        torch = pytest.importorskip("torch")
        seed_everything(42)
        a = torch.rand(5).tolist()
        seed_everything(42)
        b = torch.rand(5).tolist()
        assert a == b

    def test_different_seeds_different_values(self):
        seed_everything(1)
        a = random.random()
        seed_everything(2)
        b = random.random()
        assert a != b

    def test_default_seed_is_42(self):
        seed_everything()
        assert os.environ["PYTHONHASHSEED"] == "42"

    def test_returns_none(self):
        result = seed_everything(0)
        assert result is None


# ---------------------------------------------------------------------------
# ExperimentManifest
# ---------------------------------------------------------------------------

class TestExperimentManifest:
    def test_required_fields(self):
        m = ExperimentManifest(experiment_name="test_run")
        assert m.experiment_name == "test_run"
        assert m.seed == 42
        assert isinstance(m.timestamp, str)

    def test_custom_seed(self):
        m = ExperimentManifest(experiment_name="x", seed=7)
        assert m.seed == 7

    def test_environment_auto_populated(self):
        m = ExperimentManifest(experiment_name="x")
        assert "python" in m.environment
        assert "platform" in m.environment
        assert "numpy" in m.environment

    def test_hyperparameters_dict(self):
        m = ExperimentManifest(
            experiment_name="x",
            hyperparameters={"epochs": 50, "lr": 0.001},
        )
        assert m.hyperparameters["epochs"] == 50

    def test_data_provenance(self):
        m = ExperimentManifest(
            experiment_name="x",
            data={"graph_dir": "data/processed", "scenario_ids": [1, 2, 3]},
        )
        assert m.data["graph_dir"] == "data/processed"

    def test_model_type(self):
        m = ExperimentManifest(experiment_name="x", model_type="graphsage")
        assert m.model_type == "graphsage"

    def test_timestamp_is_iso8601(self):
        from datetime import datetime, timezone
        m = ExperimentManifest(experiment_name="x")
        dt = datetime.fromisoformat(m.timestamp)
        assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# save_manifest
# ---------------------------------------------------------------------------

class TestSaveManifest:
    def test_writes_json_file(self, tmp_path):
        m = ExperimentManifest(experiment_name="test", seed=1)
        path = save_manifest(m, tmp_path)
        assert path.exists()
        assert path.name == "manifest.json"

    def test_json_is_valid(self, tmp_path):
        m = ExperimentManifest(experiment_name="test", seed=2)
        path = save_manifest(m, tmp_path)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_json_contains_expected_keys(self, tmp_path):
        m = ExperimentManifest(experiment_name="run42", seed=42)
        path = save_manifest(m, tmp_path)
        data = json.loads(path.read_text())
        for key in ("experiment_name", "seed", "timestamp", "environment",
                    "hyperparameters", "data", "model_type"):
            assert key in data, f"Missing key: {key}"

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        m = ExperimentManifest(experiment_name="x")
        save_manifest(m, nested)
        assert nested.exists()

    def test_returns_path(self, tmp_path):
        m = ExperimentManifest(experiment_name="x")
        result = save_manifest(m, tmp_path)
        assert isinstance(result, Path)

    def test_roundtrip(self, tmp_path):
        m = ExperimentManifest(
            experiment_name="roundtrip",
            seed=99,
            model_type="gat",
            hyperparameters={"epochs": 10},
        )
        path = save_manifest(m, tmp_path)
        data = json.loads(path.read_text())
        assert data["seed"] == 99
        assert data["model_type"] == "gat"
        assert data["hyperparameters"]["epochs"] == 10
