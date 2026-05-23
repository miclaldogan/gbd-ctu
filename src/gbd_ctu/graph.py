"""Backward-compatible graph helpers.

This module preserves the original flat import surface while delegating to the
production graph builder under `gbd_ctu.data.graph_builder`.
"""

from gbd_ctu.data.graph_builder import GraphBuildArtifact as GraphBuildResult
from gbd_ctu.data.graph_builder import build_ip_graph_data as build_flow_graph
from gbd_ctu.data.graph_builder import load_graphs, save_graph_artifact as save_graph_artifacts
