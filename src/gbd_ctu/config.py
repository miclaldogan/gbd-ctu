"""GBD-CTU configuration loading.

This module loads package-bundled YAML configuration fragments and optional user
overrides. Inputs are optional YAML paths; output is a merged configuration
dictionary covering global, graph, model, and baseline settings.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


PACKAGE_CONFIG_FILES = ["base.yaml", "graph.yaml", "gnn.yaml", "baselines.yaml"]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_package_defaults() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for file_name in PACKAGE_CONFIG_FILES:
        with resources.files("gbd_ctu.configs").joinpath(file_name).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        merged = _deep_merge(merged, loaded)
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    defaults = _load_package_defaults()
    if config_path is None:
        return deepcopy(defaults)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    if not isinstance(user_config, dict):
        raise ValueError("Configuration root must be a mapping.")

    return _deep_merge(defaults, user_config)
