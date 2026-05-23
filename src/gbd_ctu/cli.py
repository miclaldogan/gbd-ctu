from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbd_ctu.config import load_config
from gbd_ctu.data.ctu13_loader import discover_flow_files, load_flow_table
from gbd_ctu.data.graph_builder import build_ip_graph_data, save_graph_artifact
from gbd_ctu.evaluation.compare import compare_reports
from gbd_ctu.evaluation.scenario_eval import evaluate_gnn_checkpoint
from gbd_ctu.training.trainer_baseline import train_baselines
from gbd_ctu.training.trainer_gnn import train_gnn


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GBD-CTU command line interface")
    parser.add_argument("--config", default=None, help="Optional path to a YAML override config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-graphs", help="Build graph artifacts from CTU-13 flow files.")
    build_parser.add_argument("--data-root", default=None, help="Directory containing CTU-13 flow files.")
    build_parser.add_argument("--output-dir", default=None, help="Directory for serialized graph artifacts.")

    train_parser = subparsers.add_parser("train", help="Train the GraphSAGE + GAT classifier.")
    train_parser.add_argument("--graph-dir", default=None, help="Directory containing serialized graph files.")
    train_parser.add_argument("--checkpoint", default=None, help="Path to the output model checkpoint.")
    train_parser.add_argument("--epochs", type=int, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained GNN checkpoint.")
    eval_parser.add_argument("--graph-dir", default=None, help="Directory containing serialized graph files.")
    eval_parser.add_argument("--checkpoint", default=None, help="Path to a trained checkpoint.")
    eval_parser.add_argument("--output", default=None, help="CSV path for scenario-wise metrics.")

    compare_parser = subparsers.add_parser(
        "compare-baselines",
        help="Train classical baselines and merge them with the GNN evaluation report.",
    )
    compare_parser.add_argument("--graph-dir", default=None, help="Directory containing serialized graph files.")
    compare_parser.add_argument("--report-dir", default=None, help="Output directory for reports.")
    compare_parser.add_argument("--gnn-report", default=None, help="Path to a GNN evaluation CSV.")

    alias_parser = subparsers.add_parser("compare", help="Alias for compare-baselines.")
    alias_parser.add_argument("--graph-dir", default=None, help="Directory containing serialized graph files.")
    alias_parser.add_argument("--report-dir", default=None, help="Output directory for reports.")
    alias_parser.add_argument("--gnn-report", default=None, help="Path to a GNN evaluation CSV.")
    return parser


def main() -> int:
    parser = _base_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    data_root = args.data_root if hasattr(args, "data_root") and args.data_root else config["paths"]["data_root"]
    graph_dir = args.graph_dir if hasattr(args, "graph_dir") and args.graph_dir else config["paths"]["graph_dir"]
    checkpoint = args.checkpoint if hasattr(args, "checkpoint") and args.checkpoint else str(Path(config["paths"]["checkpoint_dir"]) / "gnn_model.pt")
    report_dir = args.report_dir if hasattr(args, "report_dir") and args.report_dir else config["paths"]["report_dir"]

    if args.command == "build-graphs":
        files = discover_flow_files(data_root)
        created = []
        for path in files:
            frame = load_flow_table(path)
            scenario = path.stem
            result = build_ip_graph_data(
                frame,
                scenario=scenario,
                min_flows_per_node=config["graph"]["min_flows_per_node"],
                self_loops=config["graph"]["self_loops"],
                undirected_option=config["graph"]["undirected_option"],
            )
            created.append(str(save_graph_artifact(result, graph_dir, scenario)))
        print(json.dumps({"graphs": created}, indent=2))
        return 0

    if args.command == "train":
        result = train_gnn(
            graph_dir=graph_dir,
            checkpoint_path=checkpoint,
            family=config["model"]["family"],
            epochs=args.epochs or config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            hidden_channels=config["model"]["hidden_channels"],
            out_channels=config["model"]["out_channels"],
            heads=config["model"]["heads"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            focal_gamma=config["training"]["focal_gamma"],
            focal_alpha=config["training"]["focal_alpha"],
            seed=config["project"]["seed"],
            wandb_project=config["training"]["wandb_project"],
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "evaluate":
        output = args.output or str(Path(report_dir) / "gnn_metrics.csv")
        report = evaluate_gnn_checkpoint(graph_dir=graph_dir, checkpoint_path=checkpoint, output_path=output)
        print(report.to_string(index=False))
        return 0

    if args.command in {"compare-baselines", "compare"}:
        destination_dir = Path(report_dir)
        baseline_report = train_baselines(
            graph_dir=graph_dir,
            output_dir=destination_dir,
            random_state=config["baselines"]["random_state"],
            rf_n_estimators=config["baselines"]["random_forest"]["n_estimators"],
            rf_max_depth=config["baselines"]["random_forest"]["max_depth"],
            xgb_n_estimators=config["baselines"]["xgboost"]["n_estimators"],
            xgb_max_depth=config["baselines"]["xgboost"]["max_depth"],
            xgb_learning_rate=config["baselines"]["xgboost"]["learning_rate"],
            xgb_subsample=config["baselines"]["xgboost"]["subsample"],
            xgb_colsample_bytree=config["baselines"]["xgboost"]["colsample_bytree"],
            xgb_tree_method=config["baselines"]["xgboost"]["tree_method"],
        )
        gnn_report = args.gnn_report or str(destination_dir / "gnn_metrics.csv")
        comparison = compare_reports(
            gnn_report_path=gnn_report,
            baseline_report_path=destination_dir / "baseline_metrics.csv",
            output_path=destination_dir / "comparison.csv",
        )
        print(comparison.to_string(index=False))
        return 0

    parser.print_help()
    return 1
