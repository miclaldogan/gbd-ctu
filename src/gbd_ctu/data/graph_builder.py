"""GBD-CTU IP-level graph construction.

This module builds PyTorch Geometric graphs for CTU-13 scenarios. Inputs are a
scenario flow DataFrame or a CTU13Loader plus scenario ids; outputs are
torch_geometric.data.Data objects whose nodes are IP addresses, edges are
directed flows, node features are aggregated IP statistics, and edge features
are 22-dimensional flow vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

try:
    import torch
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None
    Data = None

from gbd_ctu.data.ctu13_loader import CTU13Loader
from gbd_ctu.data.feature_extractor import EDGE_FEATURE_COLUMNS, NODE_FEATURE_COLUMNS, FlowFeatureExtractor, standardize_flow_frame
from gbd_ctu.data.utils import safe_divide


@dataclass
class GraphBuildArtifact:
    """In-memory representation of a built CTU-13 scenario graph."""

    graph: Data
    node_frame: pd.DataFrame
    edge_frame: pd.DataFrame
    networkx_graph: nx.DiGraph


def _require_torch_geometric() -> None:
    if torch is None or Data is None:
        raise ImportError("torch and torch-geometric are required for graph construction.")


def _aggregate_node_features(feature_frame: pd.DataFrame, ip_addresses: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, object]] = []
    for ip_address in ip_addresses:
        outgoing = feature_frame[feature_frame["src_addr"] == ip_address]
        incoming = feature_frame[feature_frame["dst_addr"] == ip_address]
        record = {
            "ip_address": ip_address,
            "flow_count": int(outgoing.shape[0] + incoming.shape[0]),
            "mean_duration": float(pd.concat([outgoing["duration"], incoming["duration"]]).mean()) if not outgoing.empty or not incoming.empty else 0.0,
            "total_bytes_sent": float(outgoing["bytes"].sum()) if not outgoing.empty else 0.0,
            "total_bytes_recv": float(incoming["bytes"].sum()) if not incoming.empty else 0.0,
            "unique_dst_count": int(outgoing["dst_addr"].nunique()) if not outgoing.empty else 0,
            "botnet_flow_ratio": float(outgoing["label_binary"].mean()) if not outgoing.empty else 0.0,
            "label_binary": int(outgoing["label_binary"].max()) if not outgoing.empty else 0,
        }
        records.append(record)
    node_frame = pd.DataFrame.from_records(records)
    node_array = node_frame[NODE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return node_frame, node_array


def _build_networkx_graph(feature_frame: pd.DataFrame, ip_addresses: list[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(ip_addresses)
    for row in feature_frame.itertuples(index=False):
        src_addr = row.src_addr
        dst_addr = row.dst_addr
        if src_addr not in graph or dst_addr not in graph:
            continue
        if graph.has_edge(src_addr, dst_addr):
            graph[src_addr][dst_addr]["flow_count"] += 1
            graph[src_addr][dst_addr]["byte_count"] += float(row.bytes)
            graph[src_addr][dst_addr]["packet_count"] += float(row.packets)
            graph[src_addr][dst_addr]["label_binary"] = max(graph[src_addr][dst_addr]["label_binary"], int(row.label_binary))
        else:
            graph.add_edge(
                src_addr,
                dst_addr,
                flow_count=1,
                byte_count=float(row.bytes),
                packet_count=float(row.packets),
                label_binary=int(row.label_binary),
            )
    return graph


class IPGraphBuilder:
    """Build IP-level PyG graphs from CTU-13 scenario flow tables."""

    def __init__(
        self,
        loader: CTU13Loader | None = None,
        feature_extractor: FlowFeatureExtractor | None = None,
        min_flows_per_node: int = 1,
        self_loops: bool = False,
        undirected_option: bool = False,
        train_size: float = 0.6,
        val_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.loader = loader
        self.feature_extractor = feature_extractor or FlowFeatureExtractor()
        self.min_flows_per_node = min_flows_per_node
        self.self_loops = self_loops
        self.undirected_option = undirected_option
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed

    def _split_scenarios(self, scenario_ids: list[int]) -> tuple[list[int], list[int], list[int]]:
        if not scenario_ids:
            return [], [], []
        train_cutoff = max(1, int(round(self.train_size * len(scenario_ids))))
        remaining = scenario_ids[train_cutoff:]
        val_count = int(round(self.val_size * len(scenario_ids))) if remaining else 0
        train_ids = scenario_ids[:train_cutoff]
        val_ids = remaining[:val_count]
        test_ids = remaining[val_count:]
        if not test_ids and val_ids:
            test_ids = [val_ids.pop()]
        if not val_ids and len(train_ids) > 1:
            val_ids = [train_ids.pop()]
        return train_ids, val_ids, test_ids

    def _mask_for_split(self, node_count: int, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_mask = np.zeros(node_count, dtype=bool)
        val_mask = np.zeros(node_count, dtype=bool)
        test_mask = np.zeros(node_count, dtype=bool)
        if split_name == "train":
            train_mask[:] = True
        elif split_name == "val":
            val_mask[:] = True
        else:
            test_mask[:] = True
        return train_mask, val_mask, test_mask

    def build_from_frame(
        self,
        flow_frame: pd.DataFrame,
        scenario_id: int,
        split_name: str = "train",
    ) -> GraphBuildArtifact:
        """Build a single IP graph from a scenario flow frame."""

        _require_torch_geometric()
        scenario_label = f"scenario-{scenario_id:02d}"
        feature_frame = standardize_flow_frame(flow_frame, scenario=scenario_label)
        if not self.feature_extractor.is_fitted:
            self.feature_extractor.fit(feature_frame, scenario=scenario_label)
        edge_feature_array = self.feature_extractor.transform(feature_frame, scenario=scenario_label)

        ip_addresses = sorted(set(feature_frame["src_addr"]).union(set(feature_frame["dst_addr"])))
        node_frame, node_array = _aggregate_node_features(feature_frame, ip_addresses)
        if self.min_flows_per_node > 1:
            node_frame = node_frame[node_frame["flow_count"] >= self.min_flows_per_node].reset_index(drop=True)
        kept_ips = node_frame["ip_address"].tolist()
        node_array = node_frame[NODE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        node_index = {ip_address: index for index, ip_address in enumerate(kept_ips)}

        edge_frame = feature_frame[["src_addr", "dst_addr", "label", "label_binary", "start_time"]].copy()
        for feature_idx, column in enumerate(EDGE_FEATURE_COLUMNS):
            edge_frame[column] = edge_feature_array[:, feature_idx]
        edge_frame = edge_frame[
            edge_frame["src_addr"].isin(node_index) & edge_frame["dst_addr"].isin(node_index)
        ].reset_index(drop=True)

        edge_pairs = edge_frame[["src_addr", "dst_addr"]].replace(node_index).to_numpy(dtype=np.int64)
        edge_attr = edge_frame[EDGE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if self.undirected_option and edge_pairs.size:
            reversed_pairs = edge_pairs[:, ::-1]
            edge_pairs = np.vstack([edge_pairs, reversed_pairs])
            edge_attr = np.vstack([edge_attr, edge_attr])
        if self.self_loops and node_array.shape[0] > 0:
            self_loops = np.column_stack([np.arange(node_array.shape[0]), np.arange(node_array.shape[0])])
            zero_edge_attr = np.zeros((node_array.shape[0], len(EDGE_FEATURE_COLUMNS)), dtype=np.float32)
            edge_pairs = np.vstack([edge_pairs, self_loops]) if edge_pairs.size else self_loops
            edge_attr = np.vstack([edge_attr, zero_edge_attr]) if edge_attr.size else zero_edge_attr

        labels = node_frame["label_binary"].to_numpy(dtype=np.int64)
        train_mask, val_mask, test_mask = self._mask_for_split(len(node_frame), split_name=split_name)
        graph = Data(
            x=torch.tensor(node_array, dtype=torch.float32),
            edge_index=torch.tensor(edge_pairs.T, dtype=torch.long) if edge_pairs.size else torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr.size else torch.empty((0, len(EDGE_FEATURE_COLUMNS)), dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.long),
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            val_mask=torch.tensor(val_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool),
            scenario=scenario_label,
            scenario_id=int(scenario_id),
            node_ips=kept_ips,
            node_feature_names=NODE_FEATURE_COLUMNS,
            edge_feature_names=EDGE_FEATURE_COLUMNS,
        )
        networkx_graph = _build_networkx_graph(feature_frame, kept_ips)
        return GraphBuildArtifact(graph=graph, node_frame=node_frame, edge_frame=edge_frame, networkx_graph=networkx_graph)

    def build_scenario(self, scenario_id: int) -> Data:
        """Load and build a single CTU-13 scenario graph."""

        if self.loader is None:
            raise ValueError("IPGraphBuilder.build_scenario requires a CTU13Loader instance.")
        frame = self.loader.load_scenario(scenario_id)
        artifact = self.build_from_frame(frame, scenario_id=scenario_id, split_name="train")
        return artifact.graph

    def build_all_scenarios(self, scenario_ids: Iterable[int] | None = None) -> list[Data]:
        """Build graph Data objects for multiple CTU-13 scenarios."""

        if self.loader is None:
            raise ValueError("IPGraphBuilder.build_all_scenarios requires a CTU13Loader instance.")
        selected_ids = sorted(list(scenario_ids) if scenario_ids is not None else self.loader.available_scenarios())
        loaded_frames = {scenario_id: self.loader.load_scenario(scenario_id) for scenario_id in selected_ids}
        train_ids, val_ids, test_ids = self._split_scenarios(selected_ids)
        fit_ids = train_ids or selected_ids
        concatenated_train = pd.concat([loaded_frames[scenario_id] for scenario_id in fit_ids], ignore_index=True)
        self.feature_extractor.fit(concatenated_train, scenario="train")

        graphs: list[Data] = []
        for scenario_id in selected_ids:
            if scenario_id in train_ids:
                split_name = "train"
            elif scenario_id in val_ids:
                split_name = "val"
            else:
                split_name = "test"
            artifact = self.build_from_frame(loaded_frames[scenario_id], scenario_id=scenario_id, split_name=split_name)
            graphs.append(artifact.graph)
        return graphs


def build_ip_graph_data(
    flow_frame: pd.DataFrame,
    scenario: str | int,
    feature_extractor: FlowFeatureExtractor | None = None,
    min_flows_per_node: int = 1,
    self_loops: bool = False,
    undirected_option: bool = False,
) -> GraphBuildArtifact:
    """Backward-compatible wrapper that builds an artifact from a raw flow frame."""

    if isinstance(scenario, int):
        scenario_id = scenario
    else:
        match = re.search(r"(\d+)$", str(scenario))
        scenario_id = int(match.group(1)) if match else 0
    builder = IPGraphBuilder(
        feature_extractor=feature_extractor,
        min_flows_per_node=min_flows_per_node,
        self_loops=self_loops,
        undirected_option=undirected_option,
    )
    return builder.build_from_frame(flow_frame, scenario_id=scenario_id, split_name="train")


def save_graph_artifact(artifact: GraphBuildArtifact, output_dir: str | Path, scenario: str) -> Path:
    """Persist a built graph and its supporting tables to disk."""

    _require_torch_geometric()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    graph_path = destination / f"{scenario}.pt"
    node_path = destination / f"{scenario}_nodes.csv"
    edge_path = destination / f"{scenario}_edges.csv"
    nx_path = destination / f"{scenario}_ip.graphml"
    torch.save(artifact.graph, graph_path)
    artifact.node_frame.to_csv(node_path, index=False)
    artifact.edge_frame.to_csv(edge_path, index=False)
    nx.write_graphml(artifact.networkx_graph, nx_path)
    return graph_path


def load_graphs(graph_dir: str | Path) -> list[Data]:
    """Load serialized PyG graphs from disk."""

    _require_torch_geometric()
    directory = Path(graph_dir)
    graph_paths = sorted(directory.glob("*.pt"))
    if not graph_paths:
        raise FileNotFoundError(f"No serialized graphs found in {directory}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in graph_paths]
