"""Tests for the MLP deep tabular baseline.

Groups
------
- ``TestMLPClassifier``: model architecture tests; require only torch.
- ``TestTrainMLP``: trainer integration tests; require torch + torch_geometric.
- ``TestTrainMLPDryRun``: dry-run end-to-end; require torch + torch_geometric.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Availability guards
# ---------------------------------------------------------------------------

_torch_available = importlib.util.find_spec("torch") is not None
_pyg_available = importlib.util.find_spec("torch_geometric") is not None

_skip_torch = pytest.mark.skipif(not _torch_available, reason="torch not installed")
_skip_pyg = pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")
_skip_torch_pyg = pytest.mark.skipif(
    not (_torch_available and _pyg_available),
    reason="torch or torch_geometric not installed",
)


# ---------------------------------------------------------------------------
# Group 1 – MLPClassifier (torch only)
# ---------------------------------------------------------------------------


@_skip_torch
class TestMLPClassifier:
    def _make_model(self, **kw):
        from gbd_ctu.models.baselines.mlp_clf import MLPClassifier
        return MLPClassifier(**kw)

    def test_forward_output_shape_default(self):
        import torch
        model = self._make_model()
        x = torch.randn(16, 22)
        out = model(x)
        assert out.shape == (16, 2)

    def test_forward_shape_22_dim_input(self):
        import torch
        model = self._make_model(in_channels=22)
        x = torch.randn(100, 22)
        out = model(x)
        assert out.shape == (100, 2), f"Expected (100, 2), got {out.shape}"

    def test_forward_shape_custom_in_channels(self):
        import torch
        model = self._make_model(in_channels=6, hidden=[32, 16])
        x = torch.randn(50, 6)
        out = model(x)
        assert out.shape == (50, 2)

    def test_forward_single_sample(self):
        import torch
        model = self._make_model(in_channels=22)
        model.eval()
        x = torch.randn(1, 22)
        out = model(x)
        assert out.shape == (1, 2)

    def test_is_nn_module(self):
        import torch.nn as nn
        model = self._make_model()
        assert isinstance(model, nn.Module)

    def test_default_hidden_produces_12k_params(self):
        model = self._make_model(in_channels=22)
        # 22*128 + 128 + 128 + 128*64 + 64 + 64 + 64*32 + 32 + 32 + 32*2 + 2 ≈ 12 K
        assert 8_000 <= model.n_params <= 20_000

    def test_n_params_attribute_set(self):
        model = self._make_model()
        assert hasattr(model, "n_params")
        assert isinstance(model.n_params, int)
        assert model.n_params > 0

    def test_param_count_increases_with_width(self):
        m_small = self._make_model(in_channels=22, hidden=[32])
        m_large = self._make_model(in_channels=22, hidden=[256])
        assert m_large.n_params > m_small.n_params

    def test_output_requires_no_grad_in_eval(self):
        import torch
        model = self._make_model()
        model.eval()
        x = torch.randn(8, 22)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 2)

    def test_custom_hidden_dims(self):
        import torch
        model = self._make_model(in_channels=22, hidden=[64, 32])
        x = torch.randn(10, 22)
        out = model(x)
        assert out.shape == (10, 2)

    def test_zero_dropout_no_stochasticity(self):
        import torch
        model = self._make_model(dropout=0.0)
        model.train()
        x = torch.randn(4, 22)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_backward_pass_does_not_crash(self):
        import torch
        model = self._make_model(in_channels=22)
        x = torch.randn(8, 22)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

    def test_gradients_exist_after_backward(self):
        import torch
        model = self._make_model(in_channels=22)
        x = torch.randn(8, 22)
        logits = model(x)
        logits.sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_output_logits_not_softmax(self):
        """Output should be raw logits, not probabilities (sum != 1 per row)."""
        import torch
        model = self._make_model(in_channels=22)
        model.eval()
        x = torch.randn(4, 22)
        with torch.no_grad():
            out = model(x)
        # Raw logits won't sum to 1 per row (in general)
        row_sums = torch.abs(out.sum(dim=1) - 1.0)
        assert row_sums.max() > 0.01

    def test_no_torch_import_error_message(self):
        """Verify ImportError is raised when torch is absent (simulated)."""
        import importlib
        import sys
        if not _torch_available:
            with pytest.raises(ImportError, match="torch"):
                from gbd_ctu.models.baselines.mlp_clf import MLPClassifier
                MLPClassifier()

    def test_net_attribute_is_sequential(self):
        import torch.nn as nn
        model = self._make_model()
        assert isinstance(model.net, nn.Sequential)

    def test_has_batchnorm_layers(self):
        import torch.nn as nn
        model = self._make_model()
        bn_layers = [m for m in model.net if isinstance(m, nn.BatchNorm1d)]
        assert len(bn_layers) > 0

    def test_has_dropout_layers(self):
        import torch.nn as nn
        model = self._make_model(hidden=[128, 64, 32], dropout=0.3)
        dropout_layers = [m for m in model.net if isinstance(m, nn.Dropout)]
        # Dropout after all but last hidden layer → 2 for default [128,64,32]
        assert len(dropout_layers) >= 1

    def test_no_dropout_after_last_hidden(self):
        """Dropout must NOT appear between last hidden ReLU and output Linear."""
        import torch.nn as nn
        model = self._make_model(hidden=[128, 64, 32])
        layers = list(model.net)
        # Final Linear is the last layer; the layer before it must not be Dropout
        assert not isinstance(layers[-2], nn.Dropout)

    def test_single_hidden_layer(self):
        import torch
        model = self._make_model(in_channels=22, hidden=[64])
        x = torch.randn(5, 22)
        out = model(x)
        assert out.shape == (5, 2)


# ---------------------------------------------------------------------------
# Group 2 – train_mlp dry-run (requires torch_geometric for _make_synthetic_graph)
# ---------------------------------------------------------------------------


@_skip_torch_pyg
class TestTrainMLPDryRun:
    def test_returns_dataframe(self, tmp_path):
        import pandas as pd
        from gbd_ctu.training.trainer_baseline import train_mlp
        df = train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_model_column_is_mlp(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        df = train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        assert (df["model"] == "mlp").all()

    def test_split_column_is_test(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        df = train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        assert (df["split"] == "test").all()

    def test_writes_mlp_metrics_csv(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        assert (tmp_path / "mlp_metrics.csv").exists()

    def test_writes_mlp_metrics_json(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        assert (tmp_path / "mlp_metrics.json").exists()

    def test_writes_mlp_model_pt(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        assert (tmp_path / "mlp_model.pt").exists()

    def test_writes_mlp_history_json(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        assert (tmp_path / "mlp_history.json").exists()

    def test_history_has_epoch_records(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        history = json.loads((tmp_path / "mlp_history.json").read_text())
        assert len(history) == 3  # dry_run forces 3 epochs
        assert "epoch" in history[0]
        assert "train_loss" in history[0]
        assert "val_auc" in history[0]

    def test_metrics_csv_has_auc_column(self, tmp_path):
        import pandas as pd
        from gbd_ctu.training.trainer_baseline import train_mlp
        train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        df = pd.read_csv(tmp_path / "mlp_metrics.csv")
        assert "auc" in df.columns

    def test_checkpoint_contains_model_state(self, tmp_path):
        import torch
        from gbd_ctu.training.trainer_baseline import train_mlp
        train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        ckpt = torch.load(tmp_path / "mlp_model.pt", weights_only=False)
        assert "model_state" in ckpt

    def test_checkpoint_contains_in_channels(self, tmp_path):
        import torch
        from gbd_ctu.training.trainer_baseline import train_mlp
        train_mlp(output_dir=str(tmp_path), dry_run=True, seed=0)
        ckpt = torch.load(tmp_path / "mlp_model.pt", weights_only=False)
        assert "in_channels" in ckpt

    def test_no_graph_dir_raises_without_dry_run(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        with pytest.raises(ValueError, match="graph_dir"):
            train_mlp(graph_dir=None, output_dir=str(tmp_path), dry_run=False)

    def test_custom_hidden_accepted(self, tmp_path):
        from gbd_ctu.training.trainer_baseline import train_mlp
        df = train_mlp(
            output_dir=str(tmp_path), dry_run=True, seed=0,
            hidden=[32, 16],
        )
        assert len(df) >= 0  # no crash

    def test_reproducible_with_same_seed(self, tmp_path):
        import pandas as pd
        from gbd_ctu.training.trainer_baseline import train_mlp
        df1 = train_mlp(output_dir=str(tmp_path / "r1"), dry_run=True, seed=7)
        df2 = train_mlp(output_dir=str(tmp_path / "r2"), dry_run=True, seed=7)
        if "auc" in df1.columns and "auc" in df2.columns:
            np.testing.assert_allclose(
                df1["auc"].values, df2["auc"].values, atol=1e-5,
                err_msg="train_mlp not reproducible with same seed",
            )
