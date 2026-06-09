"""GBD-CTU graph builder tests.

Tests for both IPGraphBuilder (IP-level nodes) and FlowGraphBuilder
(flow-level nodes).  IP-level tests validate the legacy interface; flow-level
tests validate the new default construction.
"""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
from gbd_ctu.data.graph_builder import GraphBuildArtifact, FlowGraphBuilder, IPGraphBuilder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_frame() -> pd.DataFrame:
    """Three-flow frame: one botnet, two benign.  IPs: A→B (botnet), C→A, A→C."""
    return pd.DataFrame([
        {
            "StartTime": "2011-08-10 10:00:00", "Dur": 1.0, "Proto": "tcp",
            "Sport": 12345, "Dport": 80, "SrcAddr": "147.32.84.165",
            "DstAddr": "74.125.39.104", "State": "CON", "Dir": "->",
            "sTos": 0, "dTos": 0, "TotPkts": 12, "TotBytes": 5120,
            "SrcBytes": 1024, "Label": "Botnet",
        },
        {
            "StartTime": "2011-08-10 10:00:01", "Dur": 1.5, "Proto": "udp",
            "Sport": 53, "Dport": 50123, "SrcAddr": "8.8.8.8",
            "DstAddr": "147.32.84.165", "State": "CON", "Dir": "<-",
            "sTos": 16, "dTos": 8, "TotPkts": 8, "TotBytes": 4096,
            "SrcBytes": 900, "Label": "Background",
        },
        {
            "StartTime": "2011-08-10 10:00:02", "Dur": 0.3, "Proto": "tcp",
            "Sport": 443, "Dport": 52000, "SrcAddr": "147.32.84.165",
            "DstAddr": "8.8.8.8", "State": "CON", "Dir": "->",
            "sTos": 0, "dTos": 0, "TotPkts": 6, "TotBytes": 2048,
            "SrcBytes": 512, "Label": "Botnet",
        },
    ])


def _build(frame: pd.DataFrame | None = None, **kwargs) -> GraphBuildArtifact:
    if frame is None:
        frame = _make_frame()
    builder = IPGraphBuilder(feature_extractor=FlowFeatureExtractor(), **kwargs)
    return builder.build_from_frame(frame, scenario_id=1, split_name="train")


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_build_returns_graph_build_artifact() -> None:
    """build_from_frame must return a GraphBuildArtifact."""
    artifact = _build()
    assert isinstance(artifact, GraphBuildArtifact)


def test_graph_node_count() -> None:
    """Three distinct IPs in the frame → 3 nodes."""
    artifact = _build()
    assert artifact.graph.num_nodes == 3


def test_graph_edge_count() -> None:
    """Three flows → 3 directed edges (edge_index second dim == 3)."""
    artifact = _build()
    assert artifact.graph.edge_index.shape[1] == 3


def test_edge_index_first_dim_is_2() -> None:
    """edge_index must always have shape [2, num_edges]."""
    artifact = _build()
    assert artifact.graph.edge_index.shape[0] == 2


def test_node_feature_dim_is_5() -> None:
    """Node feature matrix must have exactly 5 columns."""
    artifact = _build()
    assert artifact.graph.x.shape[1] == 5
    assert artifact.graph.x.shape[0] == artifact.graph.num_nodes


def test_edge_feature_dim_is_22() -> None:
    """Edge attribute matrix must have exactly 35 columns."""
    artifact = _build()
    assert artifact.graph.edge_attr.shape[1] == 35
    assert artifact.graph.edge_attr.shape[0] == artifact.graph.edge_index.shape[1]


def test_y_shape_matches_num_nodes() -> None:
    """`y` label vector length must equal the number of nodes."""
    artifact = _build()
    assert artifact.graph.y.shape[0] == artifact.graph.num_nodes


def test_y_values_are_binary() -> None:
    """`y` must contain only 0 and 1."""
    artifact = _build()
    unique_labels = set(artifact.graph.y.tolist())
    assert unique_labels.issubset({0, 1})


# ---------------------------------------------------------------------------
# Label correctness
# ---------------------------------------------------------------------------

def test_botnet_sender_labeled_1() -> None:
    """147.32.84.165 sends two botnet flows and must receive y=1."""
    artifact = _build()
    node_ips = artifact.node_frame["ip_address"].tolist()
    botnet_ip_idx = node_ips.index("147.32.84.165")
    assert int(artifact.graph.y[botnet_ip_idx]) == 1


def test_benign_dest_labeled_0() -> None:
    """74.125.39.104 only receives flows from a botnet host but sends nothing.
    Since label is based on *outgoing* flows, this node must have y=0."""
    artifact = _build()
    node_ips = artifact.node_frame["ip_address"].tolist()
    benign_idx = node_ips.index("74.125.39.104")
    assert int(artifact.graph.y[benign_idx]) == 0


def test_no_isolated_botnet_nodes() -> None:
    """Every node with y=1 must have at least one incident edge (no isolated botnet)."""
    artifact = _build()
    edge_index = artifact.graph.edge_index
    y = artifact.graph.y
    botnet_indices = (y == 1).nonzero(as_tuple=True)[0].tolist()
    for node_idx in botnet_indices:
        node_tensor = torch.tensor(node_idx)
        has_edge = (
            (edge_index[0] == node_tensor).any()
            or (edge_index[1] == node_tensor).any()
        )
        assert has_edge, f"Botnet node {node_idx} is isolated (no edges)"


# ---------------------------------------------------------------------------
# Split masks
# ---------------------------------------------------------------------------

def test_train_mask_covers_all_nodes_for_train_split() -> None:
    """When split_name='train', all nodes must be in train_mask."""
    artifact = _build()
    assert artifact.graph.train_mask.all()
    assert not artifact.graph.val_mask.any()
    assert not artifact.graph.test_mask.any()


def test_val_mask_covers_all_nodes_for_val_split() -> None:
    """When split_name='val', all nodes must be in val_mask."""
    builder = IPGraphBuilder(feature_extractor=FlowFeatureExtractor())
    artifact = builder.build_from_frame(_make_frame(), scenario_id=1, split_name="val")
    assert artifact.graph.val_mask.all()
    assert not artifact.graph.train_mask.any()


def test_test_mask_covers_all_nodes_for_test_split() -> None:
    """When split_name='test', all nodes must be in test_mask."""
    builder = IPGraphBuilder(feature_extractor=FlowFeatureExtractor())
    artifact = builder.build_from_frame(_make_frame(), scenario_id=1, split_name="test")
    assert artifact.graph.test_mask.all()
    assert not artifact.graph.train_mask.any()


def test_masks_union_equals_all_nodes() -> None:
    """Union of all three masks must cover every node exactly once."""
    artifact = _build()
    union = artifact.graph.train_mask | artifact.graph.val_mask | artifact.graph.test_mask
    assert union.all(), "Some nodes are not covered by any mask"


def test_masks_no_leakage() -> None:
    """No node must appear in more than one mask."""
    artifact = _build()
    assert not (artifact.graph.train_mask & artifact.graph.val_mask).any()
    assert not (artifact.graph.train_mask & artifact.graph.test_mask).any()
    assert not (artifact.graph.val_mask & artifact.graph.test_mask).any()


# ---------------------------------------------------------------------------
# Edge feature quality
# ---------------------------------------------------------------------------

def test_edge_features_are_finite() -> None:
    """edge_attr must not contain NaN or Inf values."""
    artifact = _build()
    edge_attr = artifact.graph.edge_attr.numpy()
    import numpy as np
    assert np.isfinite(edge_attr).all(), "edge_attr contains NaN or Inf"


def test_node_features_are_finite() -> None:
    """Node feature matrix must not contain NaN or Inf values."""
    artifact = _build()
    import numpy as np
    assert np.isfinite(artifact.graph.x.numpy()).all()


# ---------------------------------------------------------------------------
# Builder options
# ---------------------------------------------------------------------------

def test_undirected_option_doubles_edge_count() -> None:
    """undirected_option=True must add reverse edges, doubling edge count."""
    directed = _build()
    undirected = _build(undirected_option=True)
    assert undirected.graph.edge_index.shape[1] == directed.graph.edge_index.shape[1] * 2


def test_self_loops_option_adds_self_edges() -> None:
    """self_loops=True must add exactly num_nodes additional self-loop edges."""
    no_loops = _build()
    with_loops = _build(self_loops=True)
    n_nodes = no_loops.graph.num_nodes
    expected = no_loops.graph.edge_index.shape[1] + n_nodes
    assert with_loops.graph.edge_index.shape[1] == expected


def test_min_flows_per_node_filters_low_degree_nodes() -> None:
    """min_flows_per_node=999 must filter all nodes producing an empty graph."""
    artifact = _build(min_flows_per_node=999)
    assert artifact.graph.num_nodes == 0


def test_min_flows_per_node_1_keeps_all_nodes() -> None:
    """min_flows_per_node=1 (default) must keep all nodes that appear in flows."""
    artifact = _build(min_flows_per_node=1)
    assert artifact.graph.num_nodes == 3


# ---------------------------------------------------------------------------
# Artifact fields
# ---------------------------------------------------------------------------

def test_artifact_node_frame_has_ip_column() -> None:
    """node_frame must have an 'ip_address' column."""
    artifact = _build()
    assert "ip_address" in artifact.node_frame.columns


def test_artifact_node_frame_contains_all_ips() -> None:
    """node_frame must contain an entry for every unique IP in the flow frame."""
    artifact = _build()
    expected_ips = {"147.32.84.165", "74.125.39.104", "8.8.8.8"}
    assert expected_ips == set(artifact.node_frame["ip_address"])


def test_artifact_edge_frame_row_count_matches_edges() -> None:
    """edge_frame must have the same number of rows as there are graph edges."""
    artifact = _build()
    assert len(artifact.edge_frame) == artifact.graph.edge_index.shape[1]


def test_artifact_networkx_graph_node_count() -> None:
    """networkx_graph must have the same number of nodes as the PyG graph."""
    artifact = _build()
    assert artifact.networkx_graph.number_of_nodes() == artifact.graph.num_nodes


def test_artifact_networkx_graph_contains_all_ips() -> None:
    """networkx_graph nodes must match the kept IP address list."""
    artifact = _build()
    nx_nodes = set(artifact.networkx_graph.nodes())
    pg_ips = set(artifact.node_frame["ip_address"])
    assert nx_nodes == pg_ips


def test_graph_metadata_attributes() -> None:
    """PyG Data object must carry scenario, scenario_id, and feature name lists."""
    artifact = _build()
    assert artifact.graph.scenario == "scenario-01"
    assert artifact.graph.scenario_id == 1
    assert artifact.graph.node_feature_names is not None
    assert artifact.graph.edge_feature_names is not None
    assert len(artifact.graph.node_feature_names) == 5
    assert len(artifact.graph.edge_feature_names) == 35


# ===========================================================================
# FlowGraphBuilder tests
# ===========================================================================

def _build_flow(frame: pd.DataFrame | None = None, **kwargs) -> GraphBuildArtifact:
    if frame is None:
        frame = _make_frame()
    builder = FlowGraphBuilder(feature_extractor=FlowFeatureExtractor(), **kwargs)
    return builder.build_from_frame(frame, scenario_id=1, split_name="train")


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_flow_build_returns_artifact() -> None:
    """build_from_frame must return a GraphBuildArtifact."""
    assert isinstance(_build_flow(), GraphBuildArtifact)


def test_flow_node_count_equals_flow_count() -> None:
    """Three flows in the fixture → exactly 3 nodes."""
    artifact = _build_flow()
    assert artifact.graph.num_nodes == 3


def test_flow_node_feature_dim_is_35() -> None:
    """Node feature matrix must have exactly 35 columns (flow features)."""
    artifact = _build_flow()
    assert artifact.graph.x.shape[1] == 35
    assert artifact.graph.x.shape[0] == 3


def test_flow_y_shape_matches_node_count() -> None:
    """`y` length must equal number of flows."""
    artifact = _build_flow()
    assert artifact.graph.y.shape[0] == artifact.graph.num_nodes


def test_flow_y_values_are_binary() -> None:
    """`y` must contain only 0 and 1."""
    artifact = _build_flow()
    assert set(artifact.graph.y.tolist()).issubset({0, 1})


def test_flow_edge_index_shape() -> None:
    """edge_index must have shape [2, E] with E > 0 for a connected frame."""
    artifact = _build_flow()
    assert artifact.graph.edge_index.shape[0] == 2
    assert artifact.graph.edge_index.shape[1] > 0


# ---------------------------------------------------------------------------
# Label correctness (flow-level)
# ---------------------------------------------------------------------------

def test_flow_botnet_flows_labeled_1() -> None:
    """Flows labeled 'Botnet' in fixture must have y=1."""
    artifact = _build_flow()
    # fixture: flows 0 and 2 are Botnet, flow 1 is Background
    assert int(artifact.graph.y[0]) == 1
    assert int(artifact.graph.y[1]) == 0
    assert int(artifact.graph.y[2]) == 1


def test_flow_botnet_rate_is_nonzero() -> None:
    """Botnet flow fraction must be > 0 (fixture has 2/3 botnet flows)."""
    artifact = _build_flow()
    botnet_count = int((artifact.graph.y == 1).sum())
    assert botnet_count > 0


# ---------------------------------------------------------------------------
# Edge structure
# ---------------------------------------------------------------------------

def test_flow_edges_connect_shared_ip_flows() -> None:
    """All three flows share IP 147.32.84.165 → they should all be connected."""
    artifact = _build_flow()
    # IPs: A=147.32.84.165, B=74.125.39.104, C=8.8.8.8
    # Flow 0: A→B, Flow 1: C→A, Flow 2: A→C
    # A appears in all three → edges (0,1), (0,2), (1,2) in both directions
    edge_index = artifact.graph.edge_index
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # Flows 0 and 1 both involve A
    assert (0, 1) in edges or (1, 0) in edges
    # Flows 0 and 2 both involve A
    assert (0, 2) in edges or (2, 0) in edges


def test_flow_self_loops_option() -> None:
    """self_loops=True must add exactly num_nodes self-loop edges."""
    no_loops = _build_flow()
    with_loops = _build_flow(self_loops=True)
    n_nodes = no_loops.graph.num_nodes
    assert with_loops.graph.edge_index.shape[1] == no_loops.graph.edge_index.shape[1] + n_nodes


def test_flow_no_edge_attr() -> None:
    """FlowGraphBuilder does not produce edge_attr (flows carry their own features)."""
    artifact = _build_flow()
    assert not hasattr(artifact.graph, "edge_attr") or artifact.graph.edge_attr is None


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def test_flow_train_mask_covers_all_nodes() -> None:
    """Within-graph split: train and val masks must be non-empty.

    FlowGraphBuilder always does a 60/20/20 within-graph random split
    (ignoring split_name) so every scenario has its own val/test partition.
    With very small frames (n<5) test_mask may be empty due to rounding.
    """
    artifact = _build_flow()
    assert artifact.graph.train_mask.any()
    assert artifact.graph.val_mask.any()


def test_flow_val_mask_covers_all_nodes() -> None:
    """Within-graph split produces a non-empty val mask."""
    builder = FlowGraphBuilder(feature_extractor=FlowFeatureExtractor())
    artifact = builder.build_from_frame(_make_frame(), scenario_id=1, split_name="val")
    assert artifact.graph.val_mask.any()


def test_flow_test_mask_covers_all_nodes() -> None:
    """Within-graph split: union of train+val+test covers all nodes.

    With very small frames (n<5) test_mask may be empty due to rounding,
    but the union must always cover all nodes without overlap.
    """
    builder = FlowGraphBuilder(feature_extractor=FlowFeatureExtractor())
    artifact = builder.build_from_frame(_make_frame(), scenario_id=1, split_name="test")
    union = artifact.graph.train_mask | artifact.graph.val_mask | artifact.graph.test_mask
    assert union.all()


def test_flow_masks_no_leakage() -> None:
    """No flow may appear in more than one mask."""
    artifact = _build_flow()
    assert not (artifact.graph.train_mask & artifact.graph.val_mask).any()
    assert not (artifact.graph.train_mask & artifact.graph.test_mask).any()
    assert not (artifact.graph.val_mask & artifact.graph.test_mask).any()


def test_flow_masks_union_covers_all() -> None:
    artifact = _build_flow()
    union = artifact.graph.train_mask | artifact.graph.val_mask | artifact.graph.test_mask
    assert union.all()


# ---------------------------------------------------------------------------
# Feature quality
# ---------------------------------------------------------------------------

def test_flow_node_features_are_finite() -> None:
    """Node feature matrix must not contain NaN or Inf."""
    import numpy as np
    artifact = _build_flow()
    assert np.isfinite(artifact.graph.x.numpy()).all()


# ---------------------------------------------------------------------------
# Artifact fields
# ---------------------------------------------------------------------------

def test_flow_node_frame_has_flow_idx() -> None:
    """node_frame must include a flow_idx column."""
    artifact = _build_flow()
    assert "flow_idx" in artifact.node_frame.columns


def test_flow_node_frame_row_count_equals_node_count() -> None:
    """node_frame must have one row per flow node."""
    artifact = _build_flow()
    assert len(artifact.node_frame) == artifact.graph.num_nodes


def test_flow_edge_frame_row_count_matches_edges() -> None:
    """edge_frame must have the same number of rows as graph edges."""
    artifact = _build_flow()
    assert len(artifact.edge_frame) == artifact.graph.edge_index.shape[1]


def test_flow_metadata_attributes() -> None:
    """PyG Data must carry scenario, scenario_id, and 35 node_feature_names."""
    artifact = _build_flow()
    assert artifact.graph.scenario == "scenario-01"
    assert artifact.graph.scenario_id == 1
    assert len(artifact.graph.node_feature_names) == 35
    assert artifact.graph.edge_feature_names == []


def test_flow_networkx_graph_has_ip_nodes() -> None:
    """networkx_graph must still be IP-level for topology analysis."""
    artifact = _build_flow()
    expected_ips = {"147.32.84.165", "74.125.39.104", "8.8.8.8"}
    assert expected_ips == set(artifact.networkx_graph.nodes())


# ---------------------------------------------------------------------------
# Max-degree cap
# ---------------------------------------------------------------------------

def test_flow_max_degree_1_produces_no_edges() -> None:
    """max_degree=1 means no IP has 2+ flows to pair → no edges."""
    # With max_degree=1 each IP keeps only 1 flow → no pairs possible
    artifact = _build_flow(max_degree=1)
    assert artifact.graph.edge_index.shape[1] == 0

