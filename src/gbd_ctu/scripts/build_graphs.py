"""GBD-CTU graph building script.

Builds PyG graph artifacts from CTU-13 scenario flow files.

Two modes:
  flow  (default) — flow-level nodes with 22-dim features and shared-IP edges.
                    Better class balance (~7–20 % botnet flows vs <0.01 % IPs).
  ip              — IP-level nodes with 5-dim aggregated features (legacy).
"""

from __future__ import annotations

import argparse

from gbd_ctu.data.ctu13_loader import discover_flow_files, load_flow_table
from gbd_ctu.data.graph_builder import (
    build_flow_graph_data,
    build_ip_graph_data,
    save_graph_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CTU-13 scenario graphs.")
    parser.add_argument("--data-root", default="data/ctu13")
    parser.add_argument("--output-dir", default="artifacts/graphs")
    parser.add_argument(
        "--mode",
        choices=["flow", "ip"],
        default="flow",
        help="flow = flow-level nodes (recommended); ip = IP-level nodes (legacy).",
    )
    # flow-level options
    parser.add_argument("--max-degree", type=int, default=50,
                        help="Max flows per IP for edge construction (flow mode).")
    parser.add_argument("--max-flows", type=int, default=None,
                        help="Stratified cap on flows per scenario (flow mode). "
                             "Keeps all botnet flows, samples normal flows to fit.")
    # ip-level options
    parser.add_argument("--min-flows-per-node", type=int, default=1)
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--self-loops", action="store_true")
    args = parser.parse_args()

    for path in discover_flow_files(args.data_root):
        frame = load_flow_table(path)
        if args.mode == "flow":
            artifact = build_flow_graph_data(
                frame,
                scenario=path.stem,
                max_degree=args.max_degree,
                max_flows=args.max_flows,
                self_loops=args.self_loops,
            )
        else:
            artifact = build_ip_graph_data(
                frame,
                scenario=path.stem,
                min_flows_per_node=args.min_flows_per_node,
                self_loops=args.self_loops,
                undirected_option=args.undirected,
            )
        save_graph_artifact(artifact, args.output_dir, path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
