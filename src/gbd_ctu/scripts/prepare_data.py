"""GBD-CTU dataset preparation script.

This script downloads CTU-13, optionally extracts it, selects scenarios, and can
build PyG graph objects saved under `data/processed/`. Inputs are CLI flags and
filesystem paths; outputs are extracted flow files and serialized `.pt` graphs.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None

from gbd_ctu.data.ctu13_loader import CTU13Loader
from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
from gbd_ctu.data.graph_builder import IPGraphBuilder

DEFAULT_CTU13_INDEX = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/"
DEFAULT_CTU13_ARCHIVE = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2"


def _parse_scenarios(raw_value: str | None, all_selected: bool) -> list[int] | None:
    if all_selected:
        return None
    if not raw_value:
        return None
    return sorted({int(chunk.strip()) for chunk in raw_value.split(",") if chunk.strip()})


def download_archive(url: str, destination: str | Path) -> Path:
    """Download the CTU-13 archive to a local path with a tqdm progress bar."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(unit="B", unit_scale=True, desc="Downloading CTU-13") as progress_bar:
        def _reporthook(block_count: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress_bar.total = total_size
            progress_bar.update(block_count * block_size - progress_bar.n)

        urllib.request.urlretrieve(url, destination_path, reporthook=_reporthook)
    return destination_path


def extract_archive(archive_path: str | Path, output_dir: str | Path) -> Path:
    """Extract a CTU-13 tar.bz2 archive into an output directory with tqdm."""

    archive = Path(archive_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:bz2") as handle:
        members = handle.getmembers()
        for member in tqdm(members, desc="Extracting CTU-13", unit="file"):
            handle.extract(member, output)
    return output


def save_processed_graphs(graphs: list, output_dir: str | Path) -> list[Path]:
    """Persist built PyG scenario graphs under the processed data directory."""

    if torch is None:
        raise ImportError("torch is required to save processed graph objects.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for graph in tqdm(graphs, desc="Saving graphs", unit="graph"):
        graph_path = destination / f"scenario_{int(graph.scenario_id):02d}.pt"
        torch.save(graph, graph_path)
        saved_paths.append(graph_path)
    return saved_paths


def _print_dry_run_summary(loader: CTU13Loader, scenario_ids: list[int] | None) -> None:
    """Load scenarios and print shape and label distribution without side effects."""

    available = loader.available_scenarios()
    targets = scenario_ids if scenario_ids is not None else available
    missing = [sid for sid in targets if sid not in available]
    if missing:
        print(f"[dry-run] WARNING: scenarios not found on disk: {missing}")
        targets = [sid for sid in targets if sid in available]

    if not targets:
        print("[dry-run] No scenarios to display.")
        return

    header = f"{'Scenario':>10}  {'Flows':>8}  {'Botnet':>8}  {'Benign':>8}  {'Botnet%':>8}"
    print(header)
    print("-" * len(header))
    for sid in sorted(targets):
        frame = loader.load_scenario(sid)
        total = len(frame)
        botnet = int(frame["label_binary"].sum())
        benign = total - botnet
        pct = 100.0 * botnet / total if total else 0.0
        print(f"{sid:>10}  {total:>8}  {botnet:>8}  {benign:>8}  {pct:>7.2f}%")


def main() -> int:
    """Parse CLI arguments and run download and graph-building tasks."""

    parser = argparse.ArgumentParser(description="Prepare CTU-13 raw data and processed graphs.")
    parser.add_argument("--download", action="store_true", help=f"Download CTU-13 from {DEFAULT_CTU13_INDEX}")
    parser.add_argument("--url", default=DEFAULT_CTU13_ARCHIVE, help="Archive URL for the CTU-13 tarball.")
    parser.add_argument("--archive", default="data/raw/ctu13/CTU-13-Dataset.tar.bz2", help="Local archive path.")
    parser.add_argument("--data-root", default="data/raw/ctu13", help="Directory containing extracted CTU-13 flow files.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for serialized `.pt` graphs.")
    parser.add_argument("--build-graphs", action="store_true", help="Build PyG graphs and save them to the processed directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print scenario shapes and label distributions without building graphs.")
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--scenarios", default=None, help="Comma-separated scenario ids to process, e.g. 1,2,3.")
    selection_group.add_argument("--all", action="store_true", help="Process all discovered scenarios.")
    parser.add_argument("--min-flows-per-node", type=int, default=1)
    parser.add_argument("--self-loops", action="store_true")
    parser.add_argument("--undirected", action="store_true")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if args.download:
        download_archive(args.url, archive_path)
        extract_archive(archive_path, args.data_root)
    elif archive_path.exists() and not Path(args.data_root).exists():
        extract_archive(archive_path, args.data_root)
    elif args.download and not archive_path.exists():
        raise FileNotFoundError(f"Archive not found after download: {archive_path}")

    if args.dry_run:
        loader = CTU13Loader(data_root=args.data_root)
        scenario_ids = _parse_scenarios(args.scenarios, all_selected=args.all)
        _print_dry_run_summary(loader, scenario_ids)
        return 0

    if args.build_graphs:
        loader = CTU13Loader(data_root=args.data_root)
        scenario_ids = _parse_scenarios(args.scenarios, all_selected=args.all)
        extractor = FlowFeatureExtractor()
        builder = IPGraphBuilder(
            loader=loader,
            feature_extractor=extractor,
            min_flows_per_node=args.min_flows_per_node,
            self_loops=args.self_loops,
            undirected_option=args.undirected,
        )
        graphs = builder.build_all_scenarios(scenario_ids=scenario_ids)
        save_processed_graphs(graphs, args.processed_dir)

    if not args.download and not args.build_graphs and not args.dry_run and not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
