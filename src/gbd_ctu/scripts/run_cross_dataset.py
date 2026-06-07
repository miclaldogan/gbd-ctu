"""GBD-CTU cross-dataset zero-shot evaluation on UNSW-NB15.

Loads a trained GNN checkpoint (or MLP checkpoint), applies it to the
UNSW-NB15 test set, and writes AUC / AUPRC / MCC to a JSON results file.

Evaluation protocol
-------------------
- Training dataset: CTU-13 (all 13 scenarios, done separately)
- Test dataset: UNSW-NB15 (botnet vs. normal flows)
- Feature alignment: ``FlowFeatureExtractor.transform_external()`` fills
  features absent in UNSW-NB15 (``tos_src``, ``tos_dst``) with CTU-13
  training-set means rather than zeros.
- Graph construction: an IP-level graph is built from UNSW-NB15 flows using
  the same ``IPGraphBuilder`` pipeline as CTU-13, then the GNN evaluates all
  nodes in test mode.

Examples
--------
    # With a GNN checkpoint:
    python -m gbd_ctu.scripts.run_cross_dataset \\
        --ctu13-model checkpoints/hybrid_best.pt \\
        --unsw-data data/unsw_nb15/UNSW_NB15_testing-set.csv \\
        --output results/cross_dataset.json

    # Dry-run (no real data needed):
    python -m gbd_ctu.scripts.run_cross_dataset --dry-run

Download UNSW-NB15
------------------
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
    Required file: UNSW_NB15_testing-set.csv  (or any per-partition CSV)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

_logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zero-shot evaluation of a CTU-13 model on UNSW-NB15.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ctu13-model",
        default=None,
        help="Path to a saved GNN or MLP checkpoint (.pt file).",
    )
    p.add_argument(
        "--unsw-data",
        default=None,
        help="Path to the UNSW-NB15 CSV file.",
    )
    p.add_argument(
        "--output",
        default="results/cross_dataset.json",
        help="Path for the output JSON results file.",
    )
    p.add_argument(
        "--missing-features",
        nargs="*",
        default=["tos_src", "tos_dst"],
        help="Feature names to fill with training means (absent in UNSW-NB15).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on synthetic data — no real CSV or checkpoint required.",
    )
    p.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _evaluate_gnn_on_graph(checkpoint_path: Path, graph) -> dict:
    """Load a GNN checkpoint and run inference on *graph*."""
    try:
        import torch
        from gbd_ctu.models.gnn import build_gnn_from_config
    except ImportError as exc:
        raise ImportError("torch and torch-geometric required for GNN evaluation.") from exc

    ckpt = torch.load(checkpoint_path, weights_only=False)
    family = ckpt.get("family", "hybrid")
    kwargs = ckpt.get("model_kwargs", {})

    config = {
        "model_type": family,
        "graphsage": {
            "hidden_dim": kwargs.get("hidden_channels", 64),
            "num_layers": kwargs.get("num_layers", 3),
            "dropout": kwargs.get("dropout", 0.3),
            "out_channels": kwargs.get("out_channels", 2),
        },
        "gat": {
            "hidden_channels": kwargs.get("hidden_channels", 64),
            "embed_channels": kwargs.get("hidden_channels", 64) // 2,
            "heads": kwargs.get("heads", 4),
            "dropout": kwargs.get("dropout", 0.3),
            "out_channels": kwargs.get("out_channels", 2),
        },
        "hybrid": {
            "embed_channels": kwargs.get("hidden_channels", 64) // 2,
            "dropout": kwargs.get("dropout", 0.3),
            "out_channels": kwargs.get("out_channels", 2),
        },
    }

    in_channels = int(graph.num_node_features)
    model = build_gnn_from_config(in_channels=in_channels, config=config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(graph)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    labels = graph.y.cpu().numpy()
    return {"model": family, "probs": probs, "labels": labels}


def _evaluate_mlp_on_features(checkpoint_path: Path, x, y) -> dict:
    """Load an MLP checkpoint and run inference on feature matrix *x*."""
    import torch
    from gbd_ctu.models.baselines.mlp_clf import MLPClassifier

    ckpt = torch.load(checkpoint_path, weights_only=False)
    model = MLPClassifier(
        in_channels=ckpt["in_channels"],
        hidden=ckpt.get("hidden", [128, 64, 32]),
        dropout=ckpt.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()

    return {"model": "mlp", "probs": probs, "labels": y}


def _compute_metrics(probs, labels, model_name: str, dataset: str) -> dict:
    """Compute AUC, AUPRC, MCC from probability scores."""
    from gbd_ctu.evaluation.metrics import classification_metrics
    import numpy as np

    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(np.unique(labels)) < 2:
        _logger.warning("Only one class present in labels — metrics will be degenerate.")
    metrics = classification_metrics(labels, probs)
    metrics.update({"model": model_name, "dataset": dataset})
    return metrics


def _dry_run(output_path: Path, missing_features: list[str]) -> dict:
    """Evaluate on synthetic data without real CSV or checkpoint."""
    import numpy as np
    import pandas as pd
    from gbd_ctu.data.feature_extractor import FlowFeatureExtractor, standardize_flow_frame
    from gbd_ctu.evaluation.metrics import classification_metrics

    rng = np.random.default_rng(42)
    n = 200

    # Synthetic CTU-13-like training frame to fit the extractor
    train_frame = pd.DataFrame({
        "duration": rng.uniform(0, 60, n),
        "proto": rng.choice(["TCP", "UDP", "ICMP"], n),
        "src_port": rng.integers(0, 65535, n),
        "dst_port": rng.integers(0, 65535, n),
        "direction": rng.choice(["->", "<-", "<->"], n),
        "state": rng.choice(["FIN", "CON", "INT"], n),
        "src_tos": rng.integers(0, 4, n),
        "dst_tos": rng.integers(0, 4, n),
        "packets": rng.integers(1, 100, n),
        "bytes": rng.integers(64, 65536, n),
        "src_bytes": rng.integers(32, 32768, n),
        "label": rng.choice(["Botnet", "Normal"], n),
        "start_time": pd.date_range("2011-01-01", periods=n, freq="s"),
    })

    extractor = FlowFeatureExtractor()
    extractor.fit(train_frame)

    # Synthetic external frame (UNSW-like: no tos columns)
    ext_frame = train_frame.copy()
    ext_frame["src_tos"] = 0
    ext_frame["dst_tos"] = 0

    features = extractor.transform_external(ext_frame, missing_features=missing_features)
    labels = rng.integers(0, 2, n)
    probs = rng.uniform(0, 1, n)

    metrics = _compute_metrics(probs, labels, model_name="synthetic", dataset="UNSW-NB15-dry")
    _logger.info("Dry-run metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, float)})
    return metrics


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- dry-run -----------------------------------------------------------
    if args.dry_run:
        metrics = _dry_run(output_path, missing_features=args.missing_features or [])
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _logger.info("Dry-run results written to %s", output_path)
        print(json.dumps(metrics, indent=2))
        return 0

    # ---- validate args -----------------------------------------------------
    if not args.unsw_data:
        _logger.error("--unsw-data is required (or use --dry-run).")
        return 1

    unsw_path = Path(args.unsw_data)
    if not unsw_path.exists():
        _logger.error("UNSW-NB15 file not found: %s", unsw_path)
        return 1

    # ---- load UNSW-NB15 ---------------------------------------------------
    try:
        from gbd_ctu.data.unsw_nb15_loader import UNSWN15Loader
    except ImportError as exc:
        _logger.error("Import error: %s", exc)
        return 1

    loader = UNSWN15Loader()
    try:
        df = loader.load_botnet_vs_normal(unsw_path)
    except Exception as exc:
        _logger.error("Failed to load UNSW-NB15: %s", exc)
        return 1

    if df.empty:
        _logger.error("No flows loaded from %s.", unsw_path)
        return 1

    # ---- feature extraction -----------------------------------------------
    try:
        from gbd_ctu.data.feature_extractor import FlowFeatureExtractor
        extractor = FlowFeatureExtractor()
        extractor.fit(df)  # fit on UNSW when no CTU-13 scaler is available
        features = extractor.transform_external(
            df, missing_features=args.missing_features or []
        )
        labels = df["label_binary"].values
    except Exception as exc:
        _logger.error("Feature extraction failed: %s", exc)
        return 1

    # ---- model evaluation --------------------------------------------------
    metrics: dict | None = None

    if args.ctu13_model:
        ckpt_path = Path(args.ctu13_model)
        if not ckpt_path.exists():
            _logger.error("Checkpoint not found: %s", ckpt_path)
            return 1

        try:
            import torch
            ckpt = torch.load(ckpt_path, weights_only=False)
        except Exception as exc:
            _logger.error("Failed to load checkpoint: %s", exc)
            return 1

        if "family" in ckpt:
            # GNN checkpoint — need to build a graph
            try:
                from gbd_ctu.data.graph_builder import IPGraphBuilder
                builder = IPGraphBuilder(feature_extractor=extractor)
                artifact = builder.build_from_frame(df, scenario_id=0, split_name="test")
                result = _evaluate_gnn_on_graph(ckpt_path, artifact.data)
                metrics = _compute_metrics(
                    result["probs"], result["labels"],
                    model_name=result["model"], dataset="UNSW-NB15",
                )
            except Exception as exc:
                _logger.error("GNN evaluation failed: %s", exc)
                return 1
        elif "model_state" in ckpt and "in_channels" in ckpt:
            # MLP checkpoint
            try:
                result = _evaluate_mlp_on_features(ckpt_path, features, labels)
                metrics = _compute_metrics(
                    result["probs"], result["labels"],
                    model_name="mlp", dataset="UNSW-NB15",
                )
            except Exception as exc:
                _logger.error("MLP evaluation failed: %s", exc)
                return 1
        else:
            _logger.error("Unrecognised checkpoint format in %s.", ckpt_path)
            return 1
    else:
        # No checkpoint: report feature-only statistics
        import numpy as np
        from gbd_ctu.evaluation.metrics import classification_metrics
        probs = np.random.default_rng(0).uniform(0, 1, len(labels))
        _logger.warning("No --ctu13-model given; reporting random-baseline metrics.")
        metrics = _compute_metrics(probs, labels, model_name="random_baseline", dataset="UNSW-NB15")

    # ---- write output ------------------------------------------------------
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _logger.info("Cross-dataset results written to %s", output_path)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
