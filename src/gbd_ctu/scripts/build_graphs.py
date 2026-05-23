"""GBD-CTU graph building script.

This script loads CTU-13 scenario files, builds IP-level graphs, and writes PyG
artifacts to disk. Inputs are a dataset root and output directory; outputs are
serialized `.pt`, `.csv`, and `.graphml` files per scenario.
"""

from __future__ import annotations

import argparse

from gbd_ctu.data.ctu13_loader import discover_flow_files, load_flow_table
from gbd_ctu.data.graph_builder import build_ip_graph_data, save_graph_artifact


def main() -> int:
    """Build IP-level graphs for all discovered CTU-13 scenarios."""

    parser = argparse.ArgumentParser(description="Build IP-level graphs for CTU-13 scenarios.")
    parser.add_argument("--data-root", default="data/ctu13", help="Directory containing CTU-13 flow files.")
    parser.add_argument("--output-dir", default="artifacts/graphs", help="Directory for graph artifacts.")
    parser.add_argument("--min-flows-per-node", type=int, default=1)
    parser.add_argument("--self-loops", action="store_true")
    parser.add_argument("--undirected", action="store_true")
    args = parser.parse_args()

    for path in discover_flow_files(args.data_root):
        frame = load_flow_table(path)
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
