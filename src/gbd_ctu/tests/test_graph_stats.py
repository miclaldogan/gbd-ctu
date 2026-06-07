"""Tests for GBD-CTU graph topology analysis.

Most tests use a synthetic 10-node graph with known properties so they run
without torch or torch_geometric.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from gbd_ctu.evaluation.graph_stats import (
    STAT_KEYS,
    _to_networkx,
    degree_distribution_plot,
    scenario_topology_stats,
    topology_stats_table,
)


# ---------------------------------------------------------------------------
# Synthetic graph fixtures
# ---------------------------------------------------------------------------

def _make_mock_graph(
    n_nodes: int = 10,
    edges: list[tuple[int, int]] | None = None,
    labels: list[int] | None = None,
) -> SimpleNamespace:
    """Minimal mock graph that satisfies the interface expected by graph_stats."""
    if edges is None:
        # Simple cycle: 0→1→2→...→(n-1)→0
        edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    if labels is None:
        # First 3 nodes are botnet
        labels = [1 if i < 3 else 0 for i in range(n_nodes)]

    ei = np.array(edges, dtype=np.int64).reshape(-1, 2).T if edges else np.empty((2, 0), dtype=np.int64)

    g = SimpleNamespace()
    g.num_nodes = n_nodes
    g.edge_index = ei
    g.y = np.array(labels, dtype=np.int64)
    return g


def _make_known_graph() -> SimpleNamespace:
    """10-node graph with fully predictable topology.

    Topology:
    - Nodes 0-4: complete graph K5 (botnet, all connected to each other)
    - Nodes 5-9: simple path 5-6-7-8-9 (benign)
    - Bridge: 4→5

    Known properties:
    - n_nodes = 10, labels = [1,1,1,1,1, 0,0,0,0,0]
    - botnet_node_frac = 0.5
    - max_degree >= 4 (K5 nodes have degree 4 in undirected)
    """
    edges = []
    # K5 directed (both ways)
    for i in range(5):
        for j in range(5):
            if i != j:
                edges.append((i, j))
    # Path 5-6-7-8-9
    for i in range(5, 9):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    # Bridge
    edges.append((4, 5))

    labels = [1] * 5 + [0] * 5
    return _make_mock_graph(n_nodes=10, edges=edges, labels=labels)


# ---------------------------------------------------------------------------
# _to_networkx
# ---------------------------------------------------------------------------

class TestToNetworkx:
    def test_correct_node_count(self):
        g = _make_mock_graph(n_nodes=10)
        G = _to_networkx(g)
        assert G.number_of_nodes() == 10

    def test_correct_edge_count(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        g = _make_mock_graph(n_nodes=3, edges=edges)
        G = _to_networkx(g)
        assert G.number_of_edges() == 3

    def test_botnet_attribute_set(self):
        g = _make_mock_graph(n_nodes=5, labels=[1, 0, 1, 0, 0])
        G = _to_networkx(g)
        assert G.nodes[0]["botnet"] == 1
        assert G.nodes[1]["botnet"] == 0

    def test_returns_digraph(self):
        g = _make_mock_graph()
        G = _to_networkx(g)
        assert isinstance(G, nx.DiGraph)

    def test_no_y_attribute(self):
        g = _make_mock_graph()
        del g.y
        G = _to_networkx(g)
        assert G.number_of_nodes() == g.num_nodes


# ---------------------------------------------------------------------------
# scenario_topology_stats — return shape
# ---------------------------------------------------------------------------

class TestScenarioTopologyStats:
    def test_returns_dict(self):
        g = _make_mock_graph()
        stats = scenario_topology_stats(g)
        assert isinstance(stats, dict)

    def test_all_12_keys_present(self):
        g = _make_mock_graph()
        stats = scenario_topology_stats(g)
        required = {
            "n_nodes", "n_edges", "botnet_node_frac",
            "mean_degree", "median_degree", "max_degree",
            "global_clustering", "botnet_clustering",
            "lcc_diameter", "lcc_avg_path",
            "botnet_density", "full_density",
        }
        assert required.issubset(set(stats.keys()))

    def test_n_nodes_correct(self):
        g = _make_mock_graph(n_nodes=10)
        stats = scenario_topology_stats(g)
        assert stats["n_nodes"] == 10

    def test_n_edges_correct(self):
        edges = [(i, (i + 1) % 5) for i in range(5)]
        g = _make_mock_graph(n_nodes=5, edges=edges)
        stats = scenario_topology_stats(g)
        assert stats["n_edges"] == 5

    def test_botnet_node_frac_correct(self):
        labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        g = _make_mock_graph(n_nodes=10, labels=labels)
        stats = scenario_topology_stats(g)
        assert stats["botnet_node_frac"] == pytest.approx(0.2)

    def test_max_degree_known_graph(self):
        g = _make_known_graph()
        stats = scenario_topology_stats(g)
        # K5 nodes each have degree 4 (undirected)
        assert stats["max_degree"] >= 4

    def test_full_density_in_range(self):
        g = _make_mock_graph()
        stats = scenario_topology_stats(g)
        assert 0.0 <= stats["full_density"] <= 1.0

    def test_botnet_density_in_range(self):
        g = _make_known_graph()
        stats = scenario_topology_stats(g)
        assert 0.0 <= stats["botnet_density"] <= 1.0

    def test_botnet_clustering_in_range(self):
        g = _make_known_graph()
        stats = scenario_topology_stats(g)
        assert 0.0 <= stats["botnet_clustering"] <= 1.0

    def test_global_clustering_in_range(self):
        g = _make_mock_graph()
        stats = scenario_topology_stats(g)
        assert 0.0 <= stats["global_clustering"] <= 1.0

    def test_k5_botnet_density_is_1(self):
        """Fully connected K5 subgraph → botnet density == 1.0."""
        g = _make_known_graph()
        stats = scenario_topology_stats(g)
        assert stats["botnet_density"] == pytest.approx(1.0, abs=1e-6)

    def test_lcc_diameter_nonnegative(self):
        g = _make_known_graph()
        stats = scenario_topology_stats(g)
        d = stats["lcc_diameter"]
        assert np.isnan(d) or d >= 0

    def test_accepts_networkx_graph_directly(self):
        """When networkx_graph is supplied, it is used without rebuilding."""
        g = _make_mock_graph(n_nodes=5)
        G = nx.DiGraph()
        G.add_nodes_from(range(5))
        G.add_edge(0, 1)
        stats = scenario_topology_stats(g, networkx_graph=G)
        # n_edges comes from graph.edge_index; G is used for clustering etc.
        assert isinstance(stats, dict)

    def test_disconnected_graph_does_not_raise(self):
        """Disconnected graphs (LCC fallback) must not crash."""
        edges = [(0, 1), (3, 4)]  # two disconnected components
        g = _make_mock_graph(n_nodes=5, edges=edges)
        stats = scenario_topology_stats(g)
        assert isinstance(stats, dict)

    def test_single_node_graph(self):
        g = _make_mock_graph(n_nodes=1, edges=[], labels=[0])
        stats = scenario_topology_stats(g)
        assert stats["n_nodes"] == 1
        assert stats["n_edges"] == 0


# ---------------------------------------------------------------------------
# topology_stats_table
# ---------------------------------------------------------------------------

class TestTopologyStatsTable:
    def test_returns_dataframe(self):
        g = _make_mock_graph()
        stats = scenario_topology_stats(g)
        df = topology_stats_table([stats])
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_graph(self):
        rows = [scenario_topology_stats(_make_mock_graph(n_nodes=i + 5))
                for i in range(3)]
        df = topology_stats_table(rows)
        assert len(df) == 3

    def test_columns_in_canonical_order(self):
        g = _make_known_graph()
        stats = scenario_topology_stats(g)
        df = topology_stats_table([stats])
        stat_cols = [c for c in STAT_KEYS if c in df.columns]
        df_stat_cols = [c for c in df.columns if c in STAT_KEYS]
        assert df_stat_cols == stat_cols

    def test_sorted_by_scenario_id(self):
        rows = []
        for sid in [3, 1, 2]:
            s = scenario_topology_stats(_make_mock_graph())
            s["scenario_id"] = sid
            rows.append(s)
        df = topology_stats_table(rows)
        assert df["scenario_id"].tolist() == [1, 2, 3]

    def test_exportable_to_latex(self):
        g = _make_mock_graph()
        stats = scenario_topology_stats(g)
        df = topology_stats_table([stats])
        tex = df.to_latex(index=False, float_format="%.4f")
        assert isinstance(tex, str)
        assert "\\begin{tabular}" in tex


# ---------------------------------------------------------------------------
# degree_distribution_plot
# ---------------------------------------------------------------------------

class TestDegreeDistributionPlot:
    def test_png_created(self, tmp_path):
        g = _make_known_graph()
        path = degree_distribution_plot(g, scenario_id=1, output_dir=tmp_path)
        assert Path(path).exists()

    def test_filename_contains_scenario_id(self, tmp_path):
        g = _make_known_graph()
        path = degree_distribution_plot(g, scenario_id=7, output_dir=tmp_path)
        assert "7" in Path(path).name

    def test_output_dir_created(self, tmp_path):
        g = _make_known_graph()
        nested = tmp_path / "a" / "b"
        assert not nested.exists()
        degree_distribution_plot(g, scenario_id=1, output_dir=nested)
        assert nested.exists()

    def test_returns_path(self, tmp_path):
        g = _make_mock_graph()
        result = degree_distribution_plot(g, scenario_id=1, output_dir=tmp_path)
        assert isinstance(result, Path)

    def test_isolated_nodes_do_not_crash(self, tmp_path):
        """Graph with no edges (all isolates) must not raise."""
        g = _make_mock_graph(n_nodes=5, edges=[])
        degree_distribution_plot(g, scenario_id=0, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _env_with_src():
    import os
    env = os.environ.copy()
    src_root = str(Path(__file__).parents[3] / "src")
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_analyze_topology_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "gbd_ctu.scripts.analyze_topology", "--help"],
        capture_output=True, text=True, env=_env_with_src(),
    )
    assert result.returncode == 0
    assert "--graph-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--no-plots" in result.stdout


def test_analyze_topology_empty_dir_exits_nonzero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "gbd_ctu.scripts.analyze_topology",
         "--graph-dir", str(empty), "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, env=_env_with_src(),
    )
    assert result.returncode != 0
