"""GBD-CTU graph construction.

Two builders are provided:

* ``FlowGraphBuilder`` — **recommended**.  Each node is one NetFlow record
  (22-dim features).  Edges connect flows sharing an IP endpoint.  Labels are
  per-flow (1 = botnet).  CTU-13 has 7–20 % botnet flows, so class balance is
  far better than the IP-level alternative.

* ``IPGraphBuilder`` — legacy.  Each node is a unique IP address (5-dim
  aggregated features).  Edges are directed flows.  Labels are per-IP
  (1 = botnet host).  Class balance is extremely poor (0.001–0.010 % botnet).
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
from gbd_ctu.data.feature_extractor import (
    EDGE_FEATURE_COLUMNS,
    FLOW_FEATURE_COLUMNS,
    NODE_FEATURE_COLUMNS,
    FlowFeatureExtractor,
    standardize_flow_frame,
)
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
    src_grouped = feature_frame.groupby("src_addr")
    src_agg = src_grouped.agg(
        outgoing_count=("bytes", "count"),
        total_bytes_sent=("bytes", "sum"),
        unique_dst_count=("dst_addr", "nunique"),
        botnet_flow_ratio=("label_binary", "mean"),
        label_binary_max=("label_binary", "max"),
        src_dur_sum=("duration", "sum")
    )
    
    dst_grouped = feature_frame.groupby("dst_addr")
    dst_agg = dst_grouped.agg(
        incoming_count=("bytes", "count"),
        total_bytes_recv=("bytes", "sum"),
        dst_dur_sum=("duration", "sum")
    )
    
    df_ips = pd.DataFrame(index=ip_addresses)
    df_ips = df_ips.join(src_agg, how="left")
    df_ips = df_ips.join(dst_agg, how="left")
    
    df_ips.fillna(0, inplace=True)
    
    df_ips["ip_address"] = df_ips.index
    df_ips["flow_count"] = (df_ips["outgoing_count"] + df_ips["incoming_count"]).astype(int)
    
    total_dur_sum = df_ips["src_dur_sum"] + df_ips["dst_dur_sum"]
    df_ips["mean_duration"] = np.where(df_ips["flow_count"] > 0, total_dur_sum / df_ips["flow_count"], 0.0)
    
    df_ips["total_bytes_sent"] = df_ips["total_bytes_sent"].astype(float)
    df_ips["total_bytes_recv"] = df_ips["total_bytes_recv"].astype(float)
    df_ips["unique_dst_count"] = df_ips["unique_dst_count"].astype(int)
    df_ips["botnet_flow_ratio"] = df_ips["botnet_flow_ratio"].astype(float)
    df_ips["label_binary"] = df_ips["label_binary_max"].astype(int)
    
    node_frame = df_ips[[
        "ip_address", "flow_count", "mean_duration", "total_bytes_sent",
        "total_bytes_recv", "unique_dst_count", "botnet_flow_ratio", "label_binary"
    ]].reset_index(drop=True)
    
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

        edge_pairs = np.column_stack([
            edge_frame["src_addr"].map(node_index).to_numpy(dtype=np.int64),
            edge_frame["dst_addr"].map(node_index).to_numpy(dtype=np.int64)
        ])
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


# ---------------------------------------------------------------------------
# Flow-level graph construction
# ---------------------------------------------------------------------------

def _stratified_sample(
    feature_frame: pd.DataFrame,
    max_flows: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified sample of at most ``max_flows`` rows, preserving botnet ratio.

    Always retains all botnet flows if they number fewer than half of
    ``max_flows``, then fills the remainder with background/normal flows.
    This ensures the minority class is never under-sampled.
    """
    if len(feature_frame) <= max_flows:
        return feature_frame
    botnet = feature_frame[feature_frame["label_binary"] == 1]
    normal = feature_frame[feature_frame["label_binary"] == 0]
    rng = np.random.default_rng(seed)
    # Keep all botnet rows (if they fit), fill rest with normal
    n_botnet = min(len(botnet), max_flows // 2)
    n_normal = max_flows - n_botnet
    bot_idx = rng.choice(len(botnet), size=n_botnet, replace=False) if n_botnet < len(botnet) else np.arange(len(botnet))
    nor_idx = rng.choice(len(normal), size=min(n_normal, len(normal)), replace=False)
    sampled = pd.concat([botnet.iloc[bot_idx], normal.iloc[nor_idx]]).sample(frac=1, random_state=seed)
    return sampled.reset_index(drop=True)


def _build_flow_edges(
    feature_frame: pd.DataFrame,
    max_degree: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Build undirected edges between flows sharing an IP endpoint.

    For each unique IP address, all flow indices where that IP appears as src
    or dst are collected (capped at ``max_degree`` to prevent hub-IP explosion).
    Edges are drawn between every pair within each set.

    Returns
    -------
    edge_src, edge_dst : np.ndarray of shape (2 * n_unique_pairs,)
        Symmetric edge set (both directions).
    """
    n_flows = len(feature_frame)
    if n_flows == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    flow_idx = np.arange(n_flows, dtype=np.int64)
    src_entries = pd.DataFrame({"ip": feature_frame["src_addr"].values, "flow_idx": flow_idx})
    dst_entries = pd.DataFrame({"ip": feature_frame["dst_addr"].values, "flow_idx": flow_idx})
    ip_flow = (
        pd.concat([src_entries, dst_entries])
        .drop_duplicates(subset=["ip", "flow_idx"])
    )

    # Cap each IP to at most max_degree flows (preserves temporal order)
    ip_flow_capped = (
        ip_flow.sort_values(["ip", "flow_idx"])
        .groupby("ip", sort=False)
        .head(max_degree)
        .reset_index(drop=True)
    )

    # Self-join: get all (a, b) pairs sharing an IP, keep a < b to avoid dups
    merged = ip_flow_capped.merge(ip_flow_capped, on="ip", suffixes=("_a", "_b"))
    pairs = (
        merged.loc[merged["flow_idx_a"] < merged["flow_idx_b"], ["flow_idx_a", "flow_idx_b"]]
        .drop_duplicates()
    )

    if pairs.empty:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    a = pairs["flow_idx_a"].to_numpy(dtype=np.int64)
    b = pairs["flow_idx_b"].to_numpy(dtype=np.int64)
    return np.concatenate([a, b]), np.concatenate([b, a])


class FlowGraphBuilder:
    """Build flow-level PyG graphs from CTU-13 scenario flow tables.

    Each node is one NetFlow record with 22-dimensional features (same
    feature set previously used as edge features by ``IPGraphBuilder``).
    Edges connect flows sharing at least one IP endpoint.  Node labels are
    per-flow botnet indicators (1 = botnet, 0 = benign/background).

    CTU-13 scenarios have 7–20 % botnet flows, giving far better class balance
    than the IP-level alternative (0.001–0.010 % botnet IPs).

    Parameters
    ----------
    loader:
        Optional ``CTU13Loader`` used by :meth:`build_scenario` and
        :meth:`build_all_scenarios`.
    feature_extractor:
        Pre-configured ``FlowFeatureExtractor``.  A fresh one is created when
        omitted.
    max_degree:
        Maximum number of flows per IP address considered for edge
        construction.  Caps edge density for hub IPs (routers, gateways).
    self_loops:
        When ``True`` add a self-loop for every node.
    train_size / val_size:
        Scenario-level train/val fractions used by
        :meth:`build_all_scenarios`.
    seed:
        Random seed (currently unused — kept for API parity with
        ``IPGraphBuilder``).
    """

    def __init__(
        self,
        loader: CTU13Loader | None = None,
        feature_extractor: FlowFeatureExtractor | None = None,
        max_degree: int = 50,
        max_flows: int | None = None,
        self_loops: bool = False,
        train_size: float = 0.6,
        val_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.loader = loader
        self.feature_extractor = feature_extractor or FlowFeatureExtractor()
        self.max_degree = max_degree
        self.max_flows = max_flows
        self.self_loops = self_loops
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

    def _mask_for_split(self, n_flows: int, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Always do a within-graph random node split so each scenario graph
        # has its own train/val/test partition (60/20/20 by default).
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n_flows)
        n_train = int(round(self.train_size * n_flows))
        n_val = int(round(self.val_size * n_flows))
        train_mask = np.zeros(n_flows, dtype=bool)
        val_mask = np.zeros(n_flows, dtype=bool)
        test_mask = np.zeros(n_flows, dtype=bool)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True
        return train_mask, val_mask, test_mask

    def build_from_frame(
        self,
        flow_frame: pd.DataFrame,
        scenario_id: int,
        split_name: str = "train",
    ) -> GraphBuildArtifact:
        """Build a flow-level graph from a scenario flow frame.

        Parameters
        ----------
        flow_frame:
            Raw CTU-13-format DataFrame (column names may be in original
            capitalised form; ``standardize_flow_frame`` normalises them).
        scenario_id:
            Integer scenario identifier (written into graph metadata).
        split_name:
            ``"train"`` | ``"val"`` | ``"test"`` — all nodes in this graph
            receive the corresponding mask.
        """
        _require_torch_geometric()
        scenario_label = f"scenario-{scenario_id:02d}"
        feature_frame = standardize_flow_frame(flow_frame, scenario=scenario_label)
        if self.max_flows is not None and len(feature_frame) > self.max_flows:
            feature_frame = _stratified_sample(feature_frame, self.max_flows, seed=self.seed)

        if not self.feature_extractor.is_fitted:
            self.feature_extractor.fit(feature_frame, scenario=scenario_label)
        node_feature_array = self.feature_extractor.transform(feature_frame, scenario=scenario_label)

        labels = feature_frame["label_binary"].to_numpy(dtype=np.int64)
        n_flows = len(feature_frame)

        edge_src, edge_dst = _build_flow_edges(feature_frame, max_degree=self.max_degree)
        if self.self_loops and n_flows > 0:
            sl = np.arange(n_flows, dtype=np.int64)
            edge_src = np.concatenate([edge_src, sl])
            edge_dst = np.concatenate([edge_dst, sl])

        if edge_src.size > 0:
            edge_index = torch.tensor(np.stack([edge_src, edge_dst]), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        train_mask, val_mask, test_mask = self._mask_for_split(n_flows, split_name)
        graph = Data(
            x=torch.tensor(node_feature_array, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            val_mask=torch.tensor(val_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool),
            scenario=scenario_label,
            scenario_id=int(scenario_id),
            node_feature_names=FLOW_FEATURE_COLUMNS,
            edge_feature_names=[],
        )

        node_frame = feature_frame[
            ["src_addr", "dst_addr", "label", "label_binary", "start_time"]
        ].copy().reset_index(drop=True)
        node_frame["flow_idx"] = np.arange(n_flows)

        if edge_src.size > 0:
            edge_frame = pd.DataFrame({"flow_idx_a": edge_src, "flow_idx_b": edge_dst})
        else:
            edge_frame = pd.DataFrame(columns=["flow_idx_a", "flow_idx_b"])

        # NetworkX IP-level graph kept for topology analysis backward-compat
        ip_addresses = sorted(set(feature_frame["src_addr"]).union(set(feature_frame["dst_addr"])))
        networkx_graph = _build_networkx_graph(feature_frame, ip_addresses)

        return GraphBuildArtifact(
            graph=graph,
            node_frame=node_frame,
            edge_frame=edge_frame,
            networkx_graph=networkx_graph,
        )

    def build_scenario(self, scenario_id: int) -> Data:
        """Load and build a single flow-level scenario graph."""
        if self.loader is None:
            raise ValueError("FlowGraphBuilder.build_scenario requires a CTU13Loader instance.")
        frame = self.loader.load_scenario(scenario_id)
        return self.build_from_frame(frame, scenario_id=scenario_id, split_name="train").graph

    def build_all_scenarios(self, scenario_ids: Iterable[int] | None = None) -> list[Data]:
        """Build flow-level graphs for multiple CTU-13 scenarios."""
        if self.loader is None:
            raise ValueError("FlowGraphBuilder.build_all_scenarios requires a CTU13Loader instance.")
        selected_ids = sorted(
            list(scenario_ids) if scenario_ids is not None else self.loader.available_scenarios()
        )
        loaded_frames = {sid: self.loader.load_scenario(sid) for sid in selected_ids}
        train_ids, val_ids, test_ids = self._split_scenarios(selected_ids)
        fit_ids = train_ids or selected_ids
        concatenated_train = pd.concat([loaded_frames[sid] for sid in fit_ids], ignore_index=True)
        self.feature_extractor.fit(concatenated_train, scenario="train")

        graphs: list[Data] = []
        for sid in selected_ids:
            if sid in train_ids:
                split_name = "train"
            elif sid in val_ids:
                split_name = "val"
            else:
                split_name = "test"
            artifact = self.build_from_frame(loaded_frames[sid], scenario_id=sid, split_name=split_name)
            graphs.append(artifact.graph)
        return graphs


def build_flow_graph_data(
    flow_frame: pd.DataFrame,
    scenario: str | int,
    feature_extractor: FlowFeatureExtractor | None = None,
    max_degree: int = 50,
    max_flows: int | None = None,
    self_loops: bool = False,
) -> GraphBuildArtifact:
    """Convenience wrapper that builds a flow-level artifact from a raw flow frame."""
    if isinstance(scenario, int):
        scenario_id = scenario
    else:
        match = re.search(r"(\d+)$", str(scenario))
        scenario_id = int(match.group(1)) if match else 0
    builder = FlowGraphBuilder(
        feature_extractor=feature_extractor,
        max_degree=max_degree,
        max_flows=max_flows,
        self_loops=self_loops,
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
