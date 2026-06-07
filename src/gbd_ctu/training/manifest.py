"""Experiment manifest: captures run configuration for reproducibility."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentManifest:
    """Immutable record of everything needed to reproduce an experiment.

    Attributes
    ----------
    experiment_name:
        Human-readable run identifier.
    seed:
        Global random seed used via ``seed_everything``.
    model_type:
        GNN family or baseline name (e.g. ``"graphsage"``, ``"random_forest"``).
    hyperparameters:
        Arbitrary key-value pairs (epochs, lr, hidden_channels, …).
    data:
        Provenance of input data (graph_dir, scenario_ids, …).
    environment:
        Auto-populated: Python version, platform, library versions.
    timestamp:
        ISO-8601 UTC timestamp of manifest creation.
    """

    experiment_name: str
    seed: int = 42
    model_type: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.environment:
            self.environment = _collect_environment()


def _collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for pkg in ("torch", "numpy", "pandas", "sklearn", "torch_geometric"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[pkg] = None
    return env


def save_manifest(manifest: ExperimentManifest, output_dir: str | Path) -> Path:
    """Serialise *manifest* to ``<output_dir>/manifest.json`` and return the path."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "manifest.json"
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return path
