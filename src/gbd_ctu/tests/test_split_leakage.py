"""Split leakage and mask integrity tests.

These tests verify that PyG graphs produced by ``IPGraphBuilder`` have
train/val/test masks that are mutually exclusive and collectively exhaustive
(MECE). All tests run against synthetic flow frames — no real CTU-13 data
is required.
"""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
from gbd_ctu.data.graph_builder import GraphBuildArtifact, IPGraphBuilder


# ---------------------------------------------------------------------------
# Synthetic flow frame factory
# ---------------------------------------------------------------------------

def _make_frame(n: int = 6) -> pd.DataFrame:
    """Return a minimal synthetic flow frame with ``n`` rows."""
    base = {
        "StartTime": "2011-08-10 10:00:00",
        "Dur": 1.0,
        "Proto": "tcp",
        "Sport": 12345,
        "Dport": 80,
        "State": "CON",
        "Dir": "->",
        "sTos": 0,
        "dTos": 0,
        "TotPkts": 10,
        "TotBytes": 1024,
        "SrcBytes": 512,
    }
    rows = []
    for i in range(n):
        row = dict(base)
        row["SrcAddr"] = f"10.0.0.{i % 3}"
        row["DstAddr"] = f"192.168.0.{i % 4}"
        row["Label"] = "Botnet" if i % 3 == 0 else "Background"
        rows.append(row)
    return pd.DataFrame(rows)


def _build_graph(split_name: str) -> GraphBuildArtifact:
    builder = IPGraphBuilder(feature_extractor=FlowFeatureExtractor())
    return builder.build_from_frame(_make_frame(), scenario_id=1, split_name=split_name)


# ---------------------------------------------------------------------------
# Per-split: masks are MECE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_masks_are_mutually_exclusive(split: str) -> None:
    """For a graph built for any split, train ∩ val, train ∩ test, val ∩ test must all be empty."""
    artifact = _build_graph(split)
    g = artifact.graph

    assert not (g.train_mask & g.val_mask).any(), (
        f"split={split!r}: train_mask and val_mask overlap"
    )
    assert not (g.train_mask & g.test_mask).any(), (
        f"split={split!r}: train_mask and test_mask overlap"
    )
    assert not (g.val_mask & g.test_mask).any(), (
        f"split={split!r}: val_mask and test_mask overlap"
    )


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_masks_cover_all_nodes(split: str) -> None:
    """train_mask | val_mask | test_mask must be True for every node."""
    artifact = _build_graph(split)
    g = artifact.graph
    union = g.train_mask | g.val_mask | g.test_mask
    assert union.all(), (
        f"split={split!r}: {(~union).sum().item()} node(s) not covered by any mask"
    )


# ---------------------------------------------------------------------------
# Specific split semantics
# ---------------------------------------------------------------------------

def test_train_split_sets_only_train_mask() -> None:
    """When split_name='train', all nodes belong to train_mask exclusively."""
    g = _build_graph("train").graph
    assert g.train_mask.all(), "Not all nodes in train_mask for split='train'"
    assert not g.val_mask.any(), "val_mask is non-empty for split='train'"
    assert not g.test_mask.any(), "test_mask is non-empty for split='train'"


def test_val_split_sets_only_val_mask() -> None:
    """When split_name='val', all nodes belong to val_mask exclusively."""
    g = _build_graph("val").graph
    assert g.val_mask.all(), "Not all nodes in val_mask for split='val'"
    assert not g.train_mask.any(), "train_mask is non-empty for split='val'"
    assert not g.test_mask.any(), "test_mask is non-empty for split='val'"


def test_test_split_sets_only_test_mask() -> None:
    """When split_name='test', all nodes belong to test_mask exclusively."""
    g = _build_graph("test").graph
    assert g.test_mask.all(), "Not all nodes in test_mask for split='test'"
    assert not g.train_mask.any(), "train_mask is non-empty for split='test'"
    assert not g.val_mask.any(), "val_mask is non-empty for split='test'"


# ---------------------------------------------------------------------------
# Mask dtype and shape
# ---------------------------------------------------------------------------

def test_masks_are_bool_tensors() -> None:
    """All three masks must be boolean PyTorch tensors."""
    g = _build_graph("train").graph
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(g, name)
        assert mask.dtype == torch.bool, (
            f"{name} has dtype {mask.dtype}, expected torch.bool"
        )


def test_mask_shape_matches_num_nodes() -> None:
    """Each mask length must equal graph.num_nodes."""
    g = _build_graph("train").graph
    n = g.num_nodes
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(g, name)
        assert mask.shape[0] == n, (
            f"{name} length {mask.shape[0]} != num_nodes {n}"
        )
