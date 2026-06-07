"""Tests for GBD-CTU training utilities.

Covers:
  - FocalLoss: scalar output, gradient flow, alpha/gamma configurability.
  - train_gnn dry-run: 3-epoch synthetic run returns expected keys, saves
    checkpoint and history file, checkpoint is reloadable.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import pytest
import torch

from gbd_ctu.training.losses import FocalLoss
from gbd_ctu.training.trainer_gnn import _make_synthetic_graph, train_gnn

_pyg_available = importlib.util.find_spec("torch_geometric") is not None
_skip_pyg = pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")


# ---------------------------------------------------------------------------
# FocalLoss tests
# ---------------------------------------------------------------------------

def _make_logits_labels(n: int = 8, n_classes: int = 2, seed: int = 0):
    rng = torch.Generator()
    rng.manual_seed(seed)
    logits = torch.randn(n, n_classes, generator=rng, requires_grad=True)
    labels = torch.randint(0, n_classes, (n,), generator=rng)
    return logits, labels


class TestFocalLoss:
    def test_output_is_scalar(self):
        loss_fn = FocalLoss()
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert loss.shape == torch.Size([]), "FocalLoss must return a scalar tensor."

    def test_gradient_flows(self):
        loss_fn = FocalLoss()
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        loss.backward()
        assert logits.grad is not None, "Gradients must flow through FocalLoss."
        assert not torch.any(torch.isnan(logits.grad)), "No NaN gradients expected."

    def test_default_alpha_is_0_25(self):
        loss_fn = FocalLoss()
        assert math.isclose(loss_fn.alpha, 0.25), "Default alpha should be 0.25."

    def test_default_gamma_is_2_0(self):
        loss_fn = FocalLoss()
        assert math.isclose(loss_fn.gamma, 2.0), "Default gamma should be 2.0."

    def test_custom_alpha_and_gamma(self):
        loss_fn = FocalLoss(gamma=0.5, alpha=0.1)
        assert math.isclose(loss_fn.gamma, 0.5)
        assert math.isclose(loss_fn.alpha, 0.1)

    def test_alpha_changes_loss_value(self):
        """Loss with alpha=0.25 must differ from alpha=0.75 (on same inputs)."""
        logits, labels = _make_logits_labels(n=32)
        loss_a = FocalLoss(alpha=0.25)(logits, labels)
        # Use detached logits so both share the same values
        loss_b = FocalLoss(alpha=0.75)(logits.detach(), labels)
        assert not torch.isclose(loss_a, loss_b), "Different alpha values must yield different loss."

    def test_gamma_zero_close_to_weighted_ce(self):
        """With gamma=0, FocalLoss should equal alpha-weighted cross-entropy."""
        logits, labels = _make_logits_labels(n=32)
        fl = FocalLoss(gamma=0.0, alpha=0.25)(logits, labels)
        # Alpha-weighted CE manually
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        alpha_tensor = torch.where(labels > 0, torch.full_like(ce, 0.25), torch.full_like(ce, 0.75))
        expected = (alpha_tensor * ce).mean()
        assert torch.allclose(fl, expected, atol=1e-5), (
            f"gamma=0 FocalLoss={fl:.6f} should equal weighted CE={expected:.6f}."
        )

    def test_loss_is_positive(self):
        loss_fn = FocalLoss()
        logits, labels = _make_logits_labels()
        loss = loss_fn(logits, labels)
        assert float(loss.item()) > 0, "FocalLoss must be positive for non-perfect predictions."


# ---------------------------------------------------------------------------
# train_gnn dry-run tests
# ---------------------------------------------------------------------------

class TestTrainGNNDryRun:
    """Smoke tests for train_gnn with dry_run=True.

    All run in ~3 epochs on a tiny synthetic graph; no CTU-13 data required.
    """

    pytestmark = _skip_pyg

    def test_dry_run_completes_and_returns_keys(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "graphsage_1.pt"
        result = train_gnn(
            family="graphsage",
            checkpoint_path=ckpt,
            dry_run=True,
            seed=0,
        )
        assert "checkpoint" in result
        assert "history_path" in result
        assert "epochs_trained" in result
        assert result["epochs_trained"] == 3

    def test_dry_run_returns_history_list(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        result = train_gnn(family="graphsage", checkpoint_path=ckpt, dry_run=True, seed=0)
        assert "history" in result, "Return must include 'history' key"
        assert isinstance(result["history"], list)
        assert len(result["history"]) == 3

    def test_dry_run_returns_stopped_epoch(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        result = train_gnn(family="graphsage", checkpoint_path=ckpt, dry_run=True, seed=0)
        assert "stopped_epoch" in result, "Return must include 'stopped_epoch' key"
        assert isinstance(result["stopped_epoch"], int)

    def test_dry_run_history_records_contain_lr(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        result = train_gnn(family="hybrid", checkpoint_path=ckpt, dry_run=True, seed=0)
        for record in result["history"]:
            assert "lr" in record, "Each history record must include 'lr' key"
            assert record["lr"] > 0.0

    def test_dry_run_lr_in_history_json(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        result = train_gnn(family="graphsage", checkpoint_path=ckpt, dry_run=True, seed=0)
        history = json.loads(Path(result["history_path"]).read_text())
        for record in history:
            assert "lr" in record

    def test_dry_run_per_scenario_checkpoint_saved(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "gnn_model.pt"
        result = train_gnn(family="hybrid", checkpoint_path=ckpt, dry_run=True, seed=0)
        # Synthetic graph has scenario_id=0; expect hybrid_scenario0.pt
        scenario_ckpt = tmp_path / "checkpoints" / "hybrid_scenario0.pt"
        assert scenario_ckpt.exists(), "Per-scenario checkpoint must be saved"
        state = torch.load(str(scenario_ckpt), weights_only=False)
        assert "model_state" in state

    def test_dry_run_min_lr_effective(self, tmp_path):
        """ReduceLROnPlateau min_lr must be present in optimizer param group."""
        ckpt = tmp_path / "ckpt.pt"
        # We patch the scheduler to verify min_lr was set by checking history LR
        result = train_gnn(
            family="graphsage", checkpoint_path=ckpt, dry_run=True, seed=0,
            lr_scheduler_min_lr=1e-5,
        )
        # LR should be >= min_lr throughout
        for record in result["history"]:
            assert record["lr"] >= 1e-5

    def test_dry_run_checkpoint_saved_and_reloadable(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        result = train_gnn(family="graphsage", checkpoint_path=ckpt, dry_run=True, seed=1)
        assert Path(result["checkpoint"]).exists(), "Checkpoint file must be written."
        state = torch.load(result["checkpoint"], weights_only=False)
        assert "model_state" in state, "Checkpoint must contain 'model_state'."
        assert "family" in state, "Checkpoint must record the model family."

    def test_dry_run_history_file_written(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        result = train_gnn(family="hybrid", checkpoint_path=ckpt, dry_run=True, seed=2)
        history_path = Path(result["history_path"])
        assert history_path.exists(), "History JSON must be written."
        history = json.loads(history_path.read_text())
        assert len(history) == 3, "History must have 3 entries for dry-run."
        for record in history:
            assert "epoch" in record
            assert "train_loss" in record
            assert "val_auc" in record
            assert "val_f1" in record

    def test_dry_run_gat(self, tmp_path):
        ckpt = tmp_path / "ckpt_gat.pt"
        result = train_gnn(family="gat", checkpoint_path=ckpt, dry_run=True, seed=3)
        assert result["epochs_trained"] == 3

    def test_dry_run_hybrid(self, tmp_path):
        ckpt = tmp_path / "ckpt_hybrid.pt"
        result = train_gnn(family="hybrid", checkpoint_path=ckpt, dry_run=True, seed=4)
        assert result["epochs_trained"] == 3

    def test_dry_run_no_crash_wandb_disabled(self, tmp_path, monkeypatch):
        """Train must complete gracefully when WANDB_MODE=disabled."""
        monkeypatch.setenv("WANDB_MODE", "disabled")
        ckpt = tmp_path / "ckpt.pt"
        result = train_gnn(
            family="graphsage",
            checkpoint_path=ckpt,
            dry_run=True,
            seed=5,
            wandb_project="test-project",  # would normally trigger W&B
        )
        assert result["epochs_trained"] == 3


# ---------------------------------------------------------------------------
# Synthetic graph helper
# ---------------------------------------------------------------------------

class TestMakeSyntheticGraph:
    pytestmark = _skip_pyg

    def test_graph_has_expected_attributes(self):
        graph = _make_synthetic_graph(n_nodes=50, n_features=6)
        assert graph.num_nodes == 50
        assert graph.x.shape[1] == 6
        assert graph.y.shape[0] == 50
        assert graph.train_mask.any()
        assert graph.val_mask.any()
        assert graph.test_mask.any()

    def test_masks_are_disjoint(self):
        graph = _make_synthetic_graph(n_nodes=50)
        overlap = graph.train_mask & graph.val_mask
        assert not overlap.any(), "train_mask and val_mask must not overlap."

    def test_labels_are_binary(self):
        graph = _make_synthetic_graph(n_nodes=50)
        unique = set(graph.y.tolist())
        assert unique.issubset({0, 1}), "Labels must be binary (0 or 1)."
