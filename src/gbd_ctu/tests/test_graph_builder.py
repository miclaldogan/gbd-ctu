"""GBD-CTU graph builder tests.

These tests validate that CTU-13-like flow samples produce an IP-level graph
with the requested node and edge feature dimensions.
"""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
from gbd_ctu.data.graph_builder import IPGraphBuilder


def test_build_ip_graph_data_creates_ip_nodes_and_edges() -> None:
    """A small CTU-like sample should produce a valid IP-level PyG graph."""

    frame = pd.DataFrame(
        [
            {
                "StartTime": "2011-08-10 10:00:00",
                "Dur": 1.0,
                "Proto": "tcp",
                "Sport": 12345,
                "Dport": 80,
                "SrcAddr": "147.32.84.165",
                "DstAddr": "74.125.39.104",
                "State": "CON",
                "Dir": "->",
                "sTos": 0,
                "dTos": 0,
                "TotPkts": 12,
                "TotBytes": 5120,
                "SrcBytes": 1024,
                "Label": "Botnet",
            },
            {
                "StartTime": "2011-08-10 10:00:01",
                "Dur": 1.5,
                "Proto": "udp",
                "Sport": 53,
                "Dport": 50123,
                "SrcAddr": "8.8.8.8",
                "DstAddr": "147.32.84.165",
                "State": "CON",
                "Dir": "<-",
                "sTos": 16,
                "dTos": 8,
                "TotPkts": 8,
                "TotBytes": 4096,
                "SrcBytes": 900,
                "Label": "Background",
            },
        ]
    )
    builder = IPGraphBuilder(feature_extractor=FlowFeatureExtractor())
    artifact = builder.build_from_frame(frame, scenario_id=1, split_name="train")
    assert artifact.graph.num_nodes == 3
    assert artifact.graph.edge_index.shape[1] == 2
    assert artifact.graph.edge_attr.shape[0] == 2
    assert artifact.graph.edge_attr.shape[1] == 22
    assert artifact.graph.x.shape[1] == 6
    assert int(artifact.graph.train_mask.sum() + artifact.graph.val_mask.sum() + artifact.graph.test_mask.sum()) == 3
    assert "147.32.84.165" in set(artifact.node_frame["ip_address"])
