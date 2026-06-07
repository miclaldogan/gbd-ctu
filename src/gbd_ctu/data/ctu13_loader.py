"""GBD-CTU CTU-13 data loading.

This module loads CTU-13 bidirectional NetFlow files into normalized pandas
DataFrames. Inputs are scenario identifiers or flow-file paths; outputs are
canonical flow tables with numeric labels and scenario metadata suitable for
feature extraction and graph construction.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import pyshark
except ImportError:  # pragma: no cover - optional dependency
    pyshark = None

from gbd_ctu.data.utils import binary_label, normalize_column_name


LOGGER = logging.getLogger(__name__)
FLOW_FILE_PATTERNS: tuple[str, ...] = ("*.binetflow", "*.biargus", "*.csv")

SCENARIO_FAMILY: dict[int, str] = {
    1: "Neris",
    2: "Neris",
    3: "Rbot",
    4: "Rbot",
    5: "Virut",
    6: "Virut",
    7: "Menti",
    8: "Sogou",
    9: "Neris",
    10: "Murlo",
    11: "Neris",
    12: "NSIS.ay",
    13: "Virut",
}
PCAP_FILE_PATTERNS: tuple[str, ...] = ("*.pcap",)
RAW_FLOW_COLUMNS: tuple[str, ...] = (
    "start_time",
    "duration",
    "proto",
    "src_addr",
    "src_port",
    "direction",
    "dst_addr",
    "dst_port",
    "state",
    "src_tos",
    "dst_tos",
    "packets",
    "bytes",
    "src_bytes",
    "label",
)


def _scenario_capture_id(path: Path) -> int | None:
    match = re.search(r"(?:botnet|capture)[-_]?(\d+)", path.stem.lower())
    if match is None:
        return None
    return int(match.group(1))


def _normalize_flow_frame(frame: pd.DataFrame, scenario_name: str, scenario_id: int) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [normalize_column_name(column) for column in normalized.columns]
    defaults = {
        "start_time": pd.Timestamp("1970-01-01"),
        "duration": 0.0,
        "proto": "UNK",
        "src_addr": "0.0.0.0",
        "src_port": 0,
        "direction": "UNK",
        "dst_addr": "0.0.0.0",
        "dst_port": 0,
        "state": "UNK",
        "src_tos": 0,
        "dst_tos": 0,
        "packets": 0,
        "bytes": 0,
        "src_bytes": 0,
        "label": "Background",
    }
    for column, default in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default

    normalized["start_time"] = pd.to_datetime(normalized["start_time"], errors="coerce").fillna(
        pd.Timestamp("1970-01-01")
    )
    numeric_columns = [
        "duration",
        "src_port",
        "dst_port",
        "src_tos",
        "dst_tos",
        "packets",
        "bytes",
        "src_bytes",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0)
    for column in ["proto", "src_addr", "direction", "dst_addr", "state", "label"]:
        normalized[column] = normalized[column].astype(str).fillna("UNK")

    normalized["label_binary"] = normalized["label"].map(binary_label).astype(int)
    normalized["scenario_name"] = scenario_name
    normalized["scenario_id"] = int(scenario_id)
    return normalized[list(RAW_FLOW_COLUMNS) + ["label_binary", "scenario_name", "scenario_id"]]


def _read_flow_table(path: str | Path, scenario_id: int | None = None) -> pd.DataFrame:
    source_path = Path(path)
    frame = pd.read_csv(source_path, low_memory=False)
    capture_id = _scenario_capture_id(source_path)
    resolved_id = int(scenario_id if scenario_id is not None else capture_id or 0)
    return _normalize_flow_frame(frame, scenario_name=source_path.stem, scenario_id=resolved_id)


def discover_flow_files(data_root: str | Path, include_pcaps: bool = False) -> list[Path]:
    """Discover CTU-13 flow files under a dataset root."""

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    patterns: Iterable[str] = FLOW_FILE_PATTERNS + (PCAP_FILE_PATTERNS if include_pcaps else tuple())
    files = sorted({candidate for pattern in patterns for candidate in root.rglob(pattern)})
    if not files:
        raise FileNotFoundError(f"No CTU-13 flow files were found under {root}.")
    return files


def load_flow_table(path: str | Path) -> pd.DataFrame:
    """Load a CTU-13 flow-like file into a normalized canonical frame."""

    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix in {".binetflow", ".biargus", ".csv"}:
        return _read_flow_table(source_path)
    if suffix == ".pcap":
        return summarize_pcap(source_path)
    raise ValueError(f"Unsupported input format: {source_path.suffix}")


def summarize_pcap(path: str | Path) -> pd.DataFrame:
    """Summarize a PCAP into a coarse CTU-13-compatible flow table."""

    if pyshark is None:
        raise ImportError("pyshark is required to summarize PCAP inputs.")

    path = Path(path)
    records: list[dict[str, object]] = []
    capture = pyshark.FileCapture(str(path), keep_packets=False)
    try:
        for packet in capture:
            if not hasattr(packet, "ip"):
                continue
            protocol = str(getattr(packet, "transport_layer", "UNK")).upper()
            transport = getattr(packet, protocol.lower(), object())
            packet_length = int(getattr(packet, "length", 0) or 0)
            records.append(
                {
                    "start_time": pd.Timestamp(packet.sniff_time),
                    "duration": 0.0,
                    "proto": protocol,
                    "src_port": int(getattr(transport, "srcport", 0) or 0),
                    "dst_port": int(getattr(transport, "dstport", 0) or 0),
                    "src_addr": packet.ip.src,
                    "dst_addr": packet.ip.dst,
                    "state": "OBS",
                    "direction": "pcap",
                    "packets": 1,
                    "bytes": packet_length,
                    "src_bytes": packet_length,
                    "label": "Unlabeled",
                    "scenario": path.stem,
                }
            )
    finally:
        capture.close()

    frame = pd.DataFrame.from_records(records)
    return _normalize_flow_frame(frame, scenario_name=path.stem, scenario_id=_scenario_capture_id(path) or 0)


class CTU13Loader:
    """Scenario-aware loader for the CTU-13 bidirectional flow files."""

    def __init__(self, data_root: str | Path = "data/ctu13") -> None:
        self.data_root = Path(data_root)
        self.logger = LOGGER.getChild(self.__class__.__name__)

    def _scenario_map(self) -> dict[int, Path]:
        files = discover_flow_files(self.data_root, include_pcaps=False)
        scenario_map: dict[int, Path] = {}
        for ordinal, path in enumerate(files, start=1):
            scenario_map[ordinal] = path
        return scenario_map

    def available_scenarios(self) -> list[int]:
        """Return discovered scenario identifiers in deterministic order."""

        return sorted(self._scenario_map())

    def _resolve_scenario_path(self, scenario_id: int) -> Path:
        scenario_map = self._scenario_map()
        if scenario_id in scenario_map:
            return scenario_map[scenario_id]
        if 42 <= scenario_id <= 54 and (scenario_id - 41) in scenario_map:
            return scenario_map[scenario_id - 41]
        raise KeyError(f"Scenario {scenario_id} was not found under {self.data_root}")

    def _log_imbalance(self, scenario_id: int, frame: pd.DataFrame) -> None:
        positives = int(frame["label_binary"].sum())
        total = int(frame.shape[0])
        negatives = total - positives
        positive_rate = positives / total if total else 0.0
        imbalance_ratio = (negatives / positives) if positives else float("inf")
        self.logger.info(
            "Loaded scenario %s with %s flows: botnet=%s benign=%s positive_rate=%.4f imbalance_ratio=%s",
            scenario_id,
            total,
            positives,
            negatives,
            positive_rate,
            "inf" if imbalance_ratio == float("inf") else f"{imbalance_ratio:.4f}",
        )

    def load_scenario(self, scenario_id: int) -> pd.DataFrame:
        """Load a single CTU-13 scenario and log its class imbalance."""

        path = self._resolve_scenario_path(scenario_id)
        frame = _read_flow_table(path, scenario_id=scenario_id)
        self._log_imbalance(scenario_id, frame)
        return frame

    def load_all(self) -> dict[int, pd.DataFrame]:
        """Load all discovered CTU-13 scenarios indexed by scenario id."""

        loaded: dict[int, pd.DataFrame] = {}
        for scenario_id in self.available_scenarios():
            loaded[scenario_id] = self.load_scenario(scenario_id)
        return loaded
