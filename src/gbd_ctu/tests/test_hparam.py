"""Tests for hyperparameter search utilities.

Structure
---------
- Group 1: No special dependencies — tests SEARCH_SPACE and mock-trial
  based ``sample_params``.
- Group 2: Requires ``optuna`` — tests ``run_hparam_search`` integration.
- Group 3: Requires ``optuna`` + ``torch_geometric`` — tests ``objective``
  on a tiny synthetic graph.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

_optuna_available = importlib.util.find_spec("optuna") is not None
_pyg_available = importlib.util.find_spec("torch_geometric") is not None

_skip_optuna = pytest.mark.skipif(
    not _optuna_available, reason="optuna not installed"
)
_skip_optuna_pyg = pytest.mark.skipif(
    not (_optuna_available and _pyg_available),
    reason="optuna or torch_geometric not installed",
)

# ---------------------------------------------------------------------------
# Minimal mock trial (no optuna required)
# ---------------------------------------------------------------------------


class _MockTrial:
    """Minimal optuna.Trial lookalike for testing ``sample_params``."""

    def __init__(self, number: int = 0) -> None:
        self.number = number
        self.params: dict = {}

    def suggest_categorical(self, name: str, choices: list):
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, log: bool = False):
        value = low
        self.params[name] = value
        return value

    def report(self, value: float, step: int) -> None:
        pass

    def should_prune(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Group 1: SEARCH_SPACE and sample_params (no optuna needed)
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_search_space_importable(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        assert isinstance(SEARCH_SPACE, dict)

    def test_required_keys_present(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        required = {
            "hidden_channels", "dropout", "learning_rate", "weight_decay",
            "heads", "focal_gamma", "focal_alpha", "num_layers",
        }
        assert required.issubset(set(SEARCH_SPACE.keys()))

    def test_hidden_channels_values(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        assert set(SEARCH_SPACE["hidden_channels"]) == {32, 64, 128, 256}

    def test_dropout_values(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        assert set(SEARCH_SPACE["dropout"]) == {0.1, 0.2, 0.3, 0.5}

    def test_heads_values(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        assert set(SEARCH_SPACE["heads"]) == {1, 2, 4, 8}

    def test_focal_gamma_values(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        for v in [0.5, 1.0, 2.0, 4.0]:
            assert v in SEARCH_SPACE["focal_gamma"]

    def test_focal_alpha_values(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        for v in [0.5, 0.65, 0.75, 0.9]:
            assert v in SEARCH_SPACE["focal_alpha"]

    def test_num_layers_values(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        assert set(SEARCH_SPACE["num_layers"]) == {2, 3, 4}

    def test_lr_range_is_tuple(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        lr = SEARCH_SPACE["learning_rate"]
        assert isinstance(lr, (tuple, list))

    def test_weight_decay_range_is_tuple(self):
        from gbd_ctu.training.hparam_search import SEARCH_SPACE
        wd = SEARCH_SPACE["weight_decay"]
        assert isinstance(wd, (tuple, list))


class TestSampleParams:
    def test_hybrid_contains_heads_and_num_layers(self):
        from gbd_ctu.training.hparam_search import sample_params
        trial = _MockTrial()
        params = sample_params(trial, model_type="hybrid")
        assert "heads" in params
        assert "num_layers" in params

    def test_gat_contains_heads_no_num_layers(self):
        from gbd_ctu.training.hparam_search import sample_params
        trial = _MockTrial()
        params = sample_params(trial, model_type="gat")
        assert "heads" in params
        assert "num_layers" not in params

    def test_graphsage_contains_num_layers_no_heads(self):
        from gbd_ctu.training.hparam_search import sample_params
        trial = _MockTrial()
        params = sample_params(trial, model_type="graphsage")
        assert "num_layers" in params
        assert "heads" not in params

    def test_returns_dict(self):
        from gbd_ctu.training.hparam_search import sample_params
        trial = _MockTrial()
        params = sample_params(trial, model_type="hybrid")
        assert isinstance(params, dict)

    def test_common_keys_always_present(self):
        from gbd_ctu.training.hparam_search import sample_params
        for mt in ("hybrid", "gat", "graphsage"):
            trial = _MockTrial()
            params = sample_params(trial, model_type=mt)
            for key in ("hidden_channels", "dropout", "learning_rate",
                        "weight_decay", "focal_gamma", "focal_alpha"):
                assert key in params, f"Missing {key!r} for model_type={mt!r}"

    def test_hidden_channels_in_valid_range(self):
        from gbd_ctu.training.hparam_search import sample_params
        trial = _MockTrial()
        params = sample_params(trial, model_type="hybrid")
        assert params["hidden_channels"] in (32, 64, 128, 256)

    def test_dropout_in_valid_range(self):
        from gbd_ctu.training.hparam_search import sample_params
        trial = _MockTrial()
        params = sample_params(trial, model_type="hybrid")
        assert params["dropout"] in (0.1, 0.2, 0.3, 0.5)

    def test_focal_gamma_in_valid_range(self):
        from gbd_ctu.training.hparam_search import sample_params
        trial = _MockTrial()
        params = sample_params(trial, model_type="hybrid")
        assert params["focal_gamma"] in (0.5, 1.0, 2.0, 4.0)

    def test_focal_alpha_in_valid_range(self):
        from gbd_ctu.training.hparam_search import sample_params
        trial = _MockTrial()
        params = sample_params(trial, model_type="hybrid")
        assert params["focal_alpha"] in (0.5, 0.65, 0.75, 0.9)


class TestModuleImportable:
    def test_module_imports_without_optuna(self):
        import gbd_ctu.training.hparam_search  # noqa: F401

    def test_has_optuna_flag_is_bool(self):
        from gbd_ctu.training.hparam_search import _HAS_OPTUNA
        assert isinstance(_HAS_OPTUNA, bool)

    def test_run_hparam_search_raises_import_error_without_optuna(self):
        """If optuna is absent, calling run_hparam_search must raise ImportError."""
        if _optuna_available:
            pytest.skip("optuna is installed; skip absence test")
        from gbd_ctu.training.hparam_search import run_hparam_search
        with pytest.raises(ImportError, match="optuna"):
            run_hparam_search(n_trials=1, dry_run=True)


# ---------------------------------------------------------------------------
# Group 2: Requires optuna
# ---------------------------------------------------------------------------


@_skip_optuna
class TestRunHparamSearchIntegration:
    def test_returns_study_object(self, tmp_path):
        import optuna
        from gbd_ctu.training.hparam_search import run_hparam_search

        if not _pyg_available:
            pytest.skip("torch_geometric not installed")

        study = run_hparam_search(
            graph_dir=None,
            model_type="hybrid",
            n_trials=2,
            seed=0,
            output_dir=str(tmp_path),
            dry_run=True,
        )
        assert isinstance(study, optuna.Study)

    def test_best_params_json_written(self, tmp_path):
        from gbd_ctu.training.hparam_search import run_hparam_search

        if not _pyg_available:
            pytest.skip("torch_geometric not installed")

        run_hparam_search(
            graph_dir=None,
            model_type="hybrid",
            n_trials=2,
            seed=0,
            output_dir=str(tmp_path),
            dry_run=True,
        )
        params_file = tmp_path / "best_params.json"
        assert params_file.exists(), "best_params.json not written"

    def test_best_params_json_structure(self, tmp_path):
        from gbd_ctu.training.hparam_search import run_hparam_search

        if not _pyg_available:
            pytest.skip("torch_geometric not installed")

        run_hparam_search(
            graph_dir=None,
            model_type="hybrid",
            n_trials=2,
            seed=0,
            output_dir=str(tmp_path),
            dry_run=True,
        )
        data = json.loads((tmp_path / "best_params.json").read_text())
        for key in ("params", "value", "trial_number", "model_type", "n_trials"):
            assert key in data, f"Missing key {key!r} in best_params.json"

    def test_best_params_value_is_float(self, tmp_path):
        from gbd_ctu.training.hparam_search import run_hparam_search

        if not _pyg_available:
            pytest.skip("torch_geometric not installed")

        run_hparam_search(
            graph_dir=None,
            model_type="hybrid",
            n_trials=2,
            seed=0,
            output_dir=str(tmp_path),
            dry_run=True,
        )
        data = json.loads((tmp_path / "best_params.json").read_text())
        assert isinstance(data["value"], float)

    def test_model_type_recorded_in_json(self, tmp_path):
        from gbd_ctu.training.hparam_search import run_hparam_search

        if not _pyg_available:
            pytest.skip("torch_geometric not installed")

        run_hparam_search(
            graph_dir=None,
            model_type="gat",
            n_trials=2,
            seed=0,
            output_dir=str(tmp_path),
            dry_run=True,
        )
        data = json.loads((tmp_path / "best_params.json").read_text())
        assert data["model_type"] == "gat"

    def test_resume_loads_existing_study(self, tmp_path):
        from gbd_ctu.training.hparam_search import run_hparam_search
        import optuna

        if not _pyg_available:
            pytest.skip("torch_geometric not installed")

        db_url = f"sqlite:///{tmp_path}/study.db"
        run_hparam_search(
            graph_dir=None,
            model_type="hybrid",
            n_trials=2,
            seed=0,
            storage=db_url,
            study_name="test_resume",
            output_dir=str(tmp_path),
            dry_run=True,
        )
        study2 = run_hparam_search(
            graph_dir=None,
            model_type="hybrid",
            n_trials=1,
            seed=0,
            storage=db_url,
            study_name="test_resume",
            output_dir=str(tmp_path),
            dry_run=True,
        )
        assert len(study2.trials) == 3


# ---------------------------------------------------------------------------
# Group 3: Requires optuna + torch_geometric
# ---------------------------------------------------------------------------


@_skip_optuna_pyg
class TestObjectiveDryRun:
    def test_objective_returns_float(self):
        import optuna
        from gbd_ctu.training.hparam_search import objective

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = objective(trial, model_type="hybrid", dry_run=True, seed=0)
        study.tell(trial, result)
        assert isinstance(result, float)

    def test_objective_value_in_zero_one(self):
        import optuna
        from gbd_ctu.training.hparam_search import objective

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = objective(trial, model_type="hybrid", dry_run=True, seed=0)
        study.tell(trial, result)
        assert 0.0 <= result <= 1.0

    def test_two_trials_deterministic(self):
        import optuna
        from gbd_ctu.training.hparam_search import objective

        def _run_study():
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            for _ in range(2):
                trial = study.ask()
                val = objective(trial, model_type="hybrid", dry_run=True, seed=42)
                study.tell(trial, val)
            return [t.value for t in study.trials]

        values_a = _run_study()
        values_b = _run_study()
        assert values_a == values_b, "Objective must be deterministic given same seeds"

    def test_graphsage_objective_returns_float(self):
        import optuna
        from gbd_ctu.training.hparam_search import objective

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = objective(trial, model_type="graphsage", dry_run=True, seed=1)
        study.tell(trial, result)
        assert isinstance(result, float)

    def test_gat_objective_returns_float(self):
        import optuna
        from gbd_ctu.training.hparam_search import objective

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = objective(trial, model_type="gat", dry_run=True, seed=2)
        study.tell(trial, result)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# CLI smoke tests (no special deps needed)
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
            [sys.executable, "-m", "gbd_ctu.scripts.run_hparam_search", "--help"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode == 0

    def test_help_shows_key_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_hparam_search", "--help"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        for flag in ("--model-type", "--n-trials", "--storage", "--resume",
                     "--seed", "--graph-dir"):
            assert flag in result.stdout, f"Expected {flag!r} in --help output"

    def test_resume_without_storage_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_hparam_search",
             "--resume", "--dry-run", "--n-trials", "1"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode != 0

    def test_missing_graph_dir_exits_nonzero(self, tmp_path):
        missing = tmp_path / "no_such_dir"
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.run_hparam_search",
             "--graph-dir", str(missing)],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode != 0
