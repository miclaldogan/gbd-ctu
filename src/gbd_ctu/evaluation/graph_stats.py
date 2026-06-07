"""GBD-CTU graph topology analysis.

Public API
----------
``scenario_topology_stats``
    Compute 12 topology statistics for one CTU-13 scenario graph.

``topology_stats_table``
    Aggregate per-scenario dicts into a tidy DataFrame for LaTeX export.

``degree_distribution_plot``
    Log-log degree distribution with a power-law fit overlay (300 DPI PNG).
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

# Keys returned by scenario_topology_stats — in display order
STAT_KEYS = [
    "scenario_id",
    "n_nodes",
    "n_edges",
    "botnet_node_frac",
    "mean_degree",
    "median_degree",
    "max_degree",
    "global_clustering",
    "botnet_clustering",
    "lcc_diameter",
    "lcc_avg_path",
    "botnet_density",
    "full_density",
]

# For large LCCs, sample this many source nodes for avg-path approximation
_AVG_PATH_MAX_NODES = 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_networkx(graph) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a PyG-style Data object.

    Accepts any object with ``num_nodes``, ``edge_index``, and optionally
    ``y`` attributes.  Works whether or not torch_geometric is installed.
    """
    G: nx.DiGraph = nx.DiGraph()
    n = int(graph.num_nodes)
    G.add_nodes_from(range(n))

    y = getattr(graph, "y", None)
    if y is not None:
        labels = y.cpu().numpy() if hasattr(y, "numpy") else np.asarray(y)
        for i, label in enumerate(labels):
            G.nodes[i]["botnet"] = int(label)

    ei = getattr(graph, "edge_index", None)
    if ei is not None:
        arr = ei.cpu().numpy() if hasattr(ei, "numpy") else np.asarray(ei)
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] > 0:
            for src, dst in zip(arr[0], arr[1]):
                G.add_edge(int(src), int(dst))

    return G


def _lcc_diameter_and_avgpath(undirected: nx.Graph) -> tuple[float, float]:
    """Compute diameter and average path length on the LCC.

    Falls back to NaN with a log warning when the LCC has no edges or when
    exact computation would be prohibitively expensive (>``_AVG_PATH_MAX_NODES``
    nodes — uses sampled BFS average instead).
    """
    if undirected.number_of_nodes() == 0:
        return float("nan"), float("nan")

    components = list(nx.connected_components(undirected))
    if not components:
        return float("nan"), float("nan")

    lcc_nodes = max(components, key=len)
    lcc = undirected.subgraph(lcc_nodes)

    if lcc.number_of_nodes() <= 1:
        return 0.0, 0.0

    if lcc.number_of_edges() == 0:
        _logger.warning("LCC has nodes but no edges; path stats undefined.")
        return float("nan"), float("nan")

    try:
        diameter = float(nx.diameter(lcc))
    except nx.NetworkXError:
        _logger.warning("Graph disconnected inside LCC view; diameter unavailable.")
        diameter = float("nan")

    n_lcc = lcc.number_of_nodes()
    if n_lcc <= _AVG_PATH_MAX_NODES:
        try:
            avg_path = float(nx.average_shortest_path_length(lcc))
        except nx.NetworkXError:
            avg_path = float("nan")
    else:
        # Sample-based approximation: BFS from a random subset of source nodes
        rng = np.random.default_rng(0)
        sample = rng.choice(list(lcc_nodes), size=_AVG_PATH_MAX_NODES, replace=False)
        total, count = 0.0, 0
        for src in sample:
            lengths = nx.single_source_shortest_path_length(lcc, src)
            total += sum(lengths.values())
            count += len(lengths)
        avg_path = total / count if count > 0 else float("nan")
        _logger.info(
            "LCC too large (%d nodes); avg path approximated from %d sources.",
            n_lcc, _AVG_PATH_MAX_NODES,
        )

    return diameter, avg_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scenario_topology_stats(
    graph,
    networkx_graph: nx.DiGraph | None = None,
) -> dict[str, Any]:
    """Compute topology statistics for one CTU-13 scenario graph.

    Parameters
    ----------
    graph:
        PyG ``Data`` object (or any object with ``num_nodes``, ``edge_index``,
        ``y``).  Used for node count, edge count, and label extraction.
    networkx_graph:
        Pre-built ``nx.DiGraph``.  When ``None`` it is derived automatically
        from ``graph.edge_index`` and ``graph.y``.

    Returns
    -------
    dict with keys:
        ``n_nodes``, ``n_edges``, ``botnet_node_frac``,
        ``mean_degree``, ``median_degree``, ``max_degree``,
        ``global_clustering``, ``botnet_clustering``,
        ``lcc_diameter``, ``lcc_avg_path``,
        ``botnet_density``, ``full_density``
    """
    if networkx_graph is None:
        networkx_graph = _to_networkx(graph)

    n_nodes = int(graph.num_nodes)

    ei = getattr(graph, "edge_index", None)
    if ei is not None:
        arr = ei.cpu().numpy() if hasattr(ei, "numpy") else np.asarray(ei)
        n_edges = int(arr.shape[1])
    else:
        n_edges = networkx_graph.number_of_edges()

    # Botnet fraction
    y = getattr(graph, "y", None)
    if y is not None:
        labels = y.cpu().numpy() if hasattr(y, "numpy") else np.asarray(y)
        botnet_node_frac = float(labels.mean()) if len(labels) > 0 else float("nan")
        botnet_nodes = [i for i, l in enumerate(labels) if int(l) == 1]
    else:
        botnet_node_frac = float("nan")
        botnet_nodes = []

    # Undirected view for degree / clustering / density
    undirected = networkx_graph.to_undirected()

    degrees = [d for _, d in undirected.degree()]
    if degrees:
        mean_degree = float(np.mean(degrees))
        median_degree = float(np.median(degrees))
        max_degree = float(max(degrees))
    else:
        mean_degree = median_degree = max_degree = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        global_clustering = float(nx.average_clustering(undirected))

    # Botnet subgraph stats
    if len(botnet_nodes) > 1:
        botnet_sub = undirected.subgraph(botnet_nodes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            botnet_clustering = float(nx.average_clustering(botnet_sub))
        n_b = len(botnet_nodes)
        max_edges_b = n_b * (n_b - 1) / 2
        botnet_density = botnet_sub.number_of_edges() / max_edges_b if max_edges_b > 0 else 0.0
    else:
        botnet_clustering = 0.0
        botnet_density = 0.0

    full_density = float(nx.density(undirected))
    lcc_diameter, lcc_avg_path = _lcc_diameter_and_avgpath(undirected)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "botnet_node_frac": botnet_node_frac,
        "mean_degree": mean_degree,
        "median_degree": median_degree,
        "max_degree": max_degree,
        "global_clustering": global_clustering,
        "botnet_clustering": botnet_clustering,
        "lcc_diameter": lcc_diameter,
        "lcc_avg_path": lcc_avg_path,
        "botnet_density": botnet_density,
        "full_density": full_density,
    }


def topology_stats_table(all_stats: list[dict[str, Any]]) -> pd.DataFrame:
    """Produce a scenario × metric DataFrame ready for LaTeX export.

    Parameters
    ----------
    all_stats:
        List of dicts as returned by ``scenario_topology_stats``.  Each dict
        may optionally contain ``scenario_id`` and ``scenario`` keys for
        labelling rows.

    Returns
    -------
    pd.DataFrame sorted by ``scenario_id`` (if present), columns ordered per
    ``STAT_KEYS`` (minus columns absent in the input).
    """
    df = pd.DataFrame(all_stats)
    # Reorder to canonical column order, dropping absent columns
    ordered_cols = [c for c in STAT_KEYS if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining]

    if "scenario_id" in df.columns:
        df = df.sort_values("scenario_id").reset_index(drop=True)

    return df


def degree_distribution_plot(
    graph,
    scenario_id: int | str,
    output_dir: Path,
    dpi: int = 300,
) -> Path:
    """Plot log-log degree distribution with a power-law fit overlay.

    Parameters
    ----------
    graph:
        PyG Data object or equivalent (needs ``edge_index``, ``num_nodes``).
    scenario_id:
        Used in the figure title and output filename.
    output_dir:
        Destination directory (created if missing).
    dpi:
        PNG resolution (default 300).

    Returns
    -------
    Path to the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for degree_distribution_plot.") from exc

    G = _to_networkx(graph).to_undirected()
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    degrees = degrees[degrees > 0]  # exclude isolates for log-log

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    if len(degrees) > 0:
        # Degree distribution as a sorted frequency plot
        unique, counts = np.unique(degrees, return_counts=True)
        ax.scatter(unique, counts, s=20, alpha=0.7, color="steelblue", label="Observed")
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Power-law fit via linear regression on log-log data
        if len(unique) >= 3:
            log_k = np.log10(unique)
            log_c = np.log10(counts)
            coeffs = np.polyfit(log_k, log_c, 1)
            gamma = -coeffs[0]
            fit_line = 10 ** np.polyval(coeffs, log_k)
            ax.plot(unique, fit_line, "r--", linewidth=1.5,
                    label=rf"Power-law fit $\gamma={gamma:.2f}$")

    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Count $P(k)$")
    ax.set_title(f"Degree Distribution — Scenario {scenario_id}")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = output_dir / f"degree_dist_scenario_{scenario_id}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
