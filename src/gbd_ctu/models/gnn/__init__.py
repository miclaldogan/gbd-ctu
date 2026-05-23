"""GBD-CTU graph neural network models.

This package provides GraphSAGE, GAT, and hybrid architectures for node-level
botnet classification on CTU-13 IP communication graphs.

Factory
-------
Use :func:`build_gnn_from_config` to instantiate any model from a YAML config
that contains a ``model_type`` key (``graphsage`` | ``gat`` | ``hybrid``).
"""

from __future__ import annotations

from gbd_ctu.models.gnn.gat import GATNodeClassifier
from gbd_ctu.models.gnn.graphsage import GraphSAGENodeClassifier
from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier

__all__ = [
    "GATNodeClassifier",
    "GraphSAGENodeClassifier",
    "GraphSageGATHybridClassifier",
    "build_gnn_from_config",
]


def build_gnn_from_config(in_channels: int, config: dict):
    """Instantiate a GNN model from a YAML-derived config dict.

    The config must contain a ``model_type`` key whose value is one of
    ``graphsage``, ``gat``, or ``hybrid``.  Model-specific hyper-parameters
    are read from the corresponding sub-dict (e.g. ``config['gat']``).

    Parameters
    ----------
    in_channels:
        Number of input node feature dimensions.
    config:
        Dictionary loaded from ``configs/gnn.yaml``.

    Returns
    -------
    nn.Module
        The instantiated classifier.

    Raises
    ------
    ValueError
        If ``model_type`` is missing or unknown.
    """
    model_type = config.get("model_type", "hybrid")
    if model_type == "graphsage":
        cfg = config.get("graphsage", {})
        return GraphSAGENodeClassifier(
            in_channels=in_channels,
            hidden_channels=cfg.get("hidden_dim", 128),
            out_channels=cfg.get("out_channels", 2),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.3),
        )
    if model_type == "gat":
        cfg = config.get("gat", {})
        return GATNodeClassifier(
            in_channels=in_channels,
            hidden_channels=cfg.get("hidden_channels", 64),
            embed_channels=cfg.get("embed_channels", 32),
            out_channels=cfg.get("out_channels", 2),
            heads=cfg.get("heads", 4),
            dropout=cfg.get("dropout", 0.3),
        )
    if model_type == "hybrid":
        cfg = config.get("hybrid", {})
        return GraphSageGATHybridClassifier(
            in_channels=in_channels,
            embed_channels=cfg.get("embed_channels", 32),
            out_channels=cfg.get("out_channels", 2),
            dropout=cfg.get("dropout", 0.3),
        )
    raise ValueError(
        f"Unknown model_type '{model_type}'. Choose from: graphsage, gat, hybrid."
    )
