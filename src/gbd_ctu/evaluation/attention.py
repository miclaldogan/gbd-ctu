"""GBD-CTU GAT attention weight extraction and analysis.

Public API
----------
``extract_attention_weights``
    Run one forward pass with ``return_attention=True``; return per-layer
    alpha tensors of shape ``[num_edges, num_heads]``.

``attention_edge_heatmap``
    Draw a NetworkX graph with edge colours proportional to mean attention
    weight; save a 300 DPI PNG.

``attention_correlation``
    Pearson correlation between each edge-feature column and the mean
    attention weight across heads and layers.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention_weights(
    model,
    graph,
) -> dict[str, "torch.Tensor"]:  # noqa: F821
    """Run inference and collect per-layer GAT attention coefficients.

    Parameters
    ----------
    model:
        A ``GATNodeClassifier`` or ``GraphSageGATHybridClassifier`` instance
        whose ``forward()`` accepts ``return_attention=True``.
    graph:
        PyG ``Data`` object passed to ``model.forward()``.

    Returns
    -------
    dict mapping ``"layer_0"``, ``"layer_1"`` … to tensors of shape
    ``[num_edges, num_heads]``.  Self-loop entries added internally by
    ``GATConv`` are excluded so the length matches ``graph.edge_index``.

    Raises
    ------
    ValueError
        When the model's ``forward()`` does not return a tuple
        ``(logits, attention_dict)``.
    ImportError
        When ``torch`` / ``torch_geometric`` are not installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for extract_attention_weights.") from exc

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = model.forward(graph, return_attention=True)
    finally:
        if was_training:
            model.train()

    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            "model.forward(graph, return_attention=True) must return "
            "(logits, attention_dict).  Did you update the forward() method?"
        )

    _, attn = result
    if not isinstance(attn, dict):
        raise ValueError("attention_dict must be a dict mapping layer names to tensors.")

    return attn


# ---------------------------------------------------------------------------
# Edge heatmap
# ---------------------------------------------------------------------------

def attention_edge_heatmap(
    graph,
    attention: dict[str, Any],
    scenario_id: int | str,
    output_dir: Path,
    layer: str = "layer_0",
    dpi: int = 300,
    max_nodes_spring: int = 200,
    spring_k: float = 0.15,
) -> Path:
    """Draw a graph with edge colours proportional to mean attention weight.

    Parameters
    ----------
    graph:
        PyG ``Data`` object with ``edge_index``, ``y`` (binary node labels),
        and ``num_nodes``.
    attention:
        Dict as returned by :func:`extract_attention_weights`.
    scenario_id:
        Used in the figure title and output filename.
    output_dir:
        Destination directory (created if missing).
    layer:
        Which attention layer to visualise (default ``"layer_0"``).
    dpi:
        PNG resolution (default 300).
    max_nodes_spring:
        For graphs with more nodes than this, Kamada-Kawai is replaced by
        spring layout with ``k=spring_k`` for speed.
    spring_k:
        Spring layout ``k`` parameter used for dense graphs.

    Returns
    -------
    Path to the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for attention_edge_heatmap.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_tensor = attention.get(layer)
    if alpha_tensor is None:
        available = list(attention.keys())
        if available:
            alpha_tensor = attention[available[0]]
            _logger.warning("Layer '%s' not found; using '%s' instead.", layer, available[0])
        else:
            raise ValueError("attention dict is empty.")

    alpha_np = alpha_tensor.detach().cpu().numpy() if hasattr(alpha_tensor, "detach") else np.asarray(alpha_tensor)
    edge_weights = alpha_np.mean(axis=-1)  # [E]

    # Normalise to [0, 1]
    w_min, w_max = edge_weights.min(), edge_weights.max()
    if w_max > w_min:
        edge_weights_norm = (edge_weights - w_min) / (w_max - w_min)
    else:
        edge_weights_norm = np.zeros_like(edge_weights)

    # Build NetworkX graph
    ei = graph.edge_index
    if hasattr(ei, "cpu"):
        ei = ei.cpu().numpy()
    else:
        ei = np.asarray(ei)

    G = nx.DiGraph()
    n = int(graph.num_nodes)
    G.add_nodes_from(range(n))

    y = getattr(graph, "y", None)
    if y is not None:
        labels = y.cpu().numpy() if hasattr(y, "numpy") else np.asarray(y)
    else:
        labels = np.zeros(n, dtype=int)

    E = ei.shape[1] if ei.ndim == 2 else 0
    for idx in range(E):
        src, dst = int(ei[0, idx]), int(ei[1, idx])
        w = float(edge_weights_norm[idx]) if idx < len(edge_weights_norm) else 0.0
        G.add_edge(src, dst, weight=w)

    # Layout
    if n <= max_nodes_spring:
        try:
            pos = nx.kamada_kawai_layout(G.to_undirected())
        except Exception:
            pos = nx.spring_layout(G, k=spring_k, seed=42)
    else:
        _logger.info("Graph has %d nodes; using spring layout (k=%.2f).", n, spring_k)
        pos = nx.spring_layout(G, k=spring_k, seed=42)

    # Node colours: red = botnet, light grey = benign
    node_colors = ["#d62728" if int(labels[i]) == 1 else "#aec7e8" for i in range(n)]

    # Edge colours from attention weights
    cmap = cm.YlOrRd
    edge_data = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    edge_list = [(u, v) for u, v, _ in edge_data]
    edge_color_vals = [w for _, _, w in edge_data]
    edge_colors = [mcolors.to_hex(cmap(w)) for w in edge_color_vals]

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list,
        edge_color=edge_colors, width=0.8, alpha=0.7,
        arrows=True, arrowsize=8, ax=ax,
    )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean attention weight (normalised)")

    # Legend for node types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Botnet"),
        Patch(facecolor="#aec7e8", label="Benign"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_title(f"GAT Attention Edge Heatmap — Scenario {scenario_id}", fontsize=13)
    ax.axis("off")
    fig.tight_layout()

    out_path = output_dir / f"attention_heatmap_scenario_{scenario_id}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Feature–attention correlation
# ---------------------------------------------------------------------------

def attention_correlation(
    edge_attr: np.ndarray,
    attention_weights: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Pearson correlation between edge features and mean attention weight.

    Parameters
    ----------
    edge_attr:
        2-D array of shape ``[num_edges, num_features]``.
    attention_weights:
        1-D array of shape ``[num_edges]`` — mean attention across heads.
    feature_names:
        Names for each feature column (length must equal
        ``edge_attr.shape[1]``).

    Returns
    -------
    pd.DataFrame with columns ``feature``, ``pearson_r``, ``p_value``,
    sorted descending by ``|pearson_r|``.
    """
    try:
        from scipy.stats import pearsonr
    except ImportError as exc:
        raise ImportError("scipy is required for attention_correlation.") from exc

    if edge_attr.ndim != 2:
        raise ValueError(f"edge_attr must be 2-D, got shape {edge_attr.shape}.")
    if attention_weights.ndim != 1:
        raise ValueError(f"attention_weights must be 1-D, got shape {attention_weights.shape}.")
    n_edges, n_features = edge_attr.shape
    if len(attention_weights) != n_edges:
        raise ValueError(
            f"edge_attr has {n_edges} rows but attention_weights has {len(attention_weights)} entries."
        )
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but edge_attr has {n_features} columns."
        )

    records = []
    for i, name in enumerate(feature_names):
        col = edge_attr[:, i].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r, p = pearsonr(col, attention_weights.astype(float))
            except Exception:
                r, p = float("nan"), float("nan")
        records.append({"feature": name, "pearson_r": float(r), "p_value": float(p)})

    df = pd.DataFrame(records)
    df = df.reindex(df["pearson_r"].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True)
