"""GBD-CTU NetFlow feature extraction.

This module extracts a fixed 22-dimensional feature vector from CTU-13 NetFlow
records and applies StandardScaler normalization. Inputs are raw or normalized
flow DataFrames; outputs are normalized DataFrames and numpy arrays shaped
`(n_flows, 22)` suitable for graph edge features and classical baselines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None

from gbd_ctu.data.utils import binary_label, normalize_column_name, safe_divide


FLOW_FEATURE_COLUMNS: list[str] = [
    "duration",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "src_port_well_known",
    "src_port_registered",
    "src_port_dynamic",
    "dst_port_well_known",
    "dst_port_registered",
    "dst_port_dynamic",
    "direction_forward",
    "direction_reverse",
    "state_con",
    "state_other",
    "total_packets",
    "total_bytes",
    "src_bytes",
    "byte_per_packet",
    "packets_per_second",
    "bytes_per_second",
    "tos_src",
    "tos_dst",
]

EDGE_FEATURE_COLUMNS: list[str] = FLOW_FEATURE_COLUMNS.copy()

NODE_FEATURE_COLUMNS: list[str] = [
    "flow_count",
    "mean_duration",
    "total_bytes_sent",
    "total_bytes_recv",
    "unique_dst_count",
    "botnet_flow_ratio",
]


def _ensure_required_columns(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [normalize_column_name(column) for column in normalized.columns]
    defaults: dict[str, Any] = {
        "start_time": pd.Timestamp("1970-01-01"),
        "duration": 0.0,
        "proto": "UNK",
        "src_port": 0,
        "dst_port": 0,
        "src_addr": "0.0.0.0",
        "dst_addr": "0.0.0.0",
        "state": "UNK",
        "direction": "UNK",
        "src_tos": 0,
        "dst_tos": 0,
        "packets": 0,
        "bytes": 0,
        "src_bytes": 0,
        "label": "Background",
        "scenario": scenario,
    }
    for column, default in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default

    normalized["start_time"] = pd.to_datetime(normalized["start_time"], errors="coerce").fillna(
        pd.Timestamp("1970-01-01")
    )
    for numeric_column in [
        "duration",
        "src_port",
        "dst_port",
        "src_tos",
        "dst_tos",
        "packets",
        "bytes",
        "src_bytes",
    ]:
        normalized[numeric_column] = pd.to_numeric(normalized[numeric_column], errors="coerce").fillna(0)
    for categorical_column in ["proto", "state", "direction", "src_addr", "dst_addr", "label"]:
        normalized[categorical_column] = normalized[categorical_column].astype(str).fillna("UNK")
    normalized["scenario"] = normalized["scenario"].astype(str).fillna(scenario)
    return normalized


def _port_bucket(port: float) -> tuple[int, int, int]:
    if port <= 1023:
        return 1, 0, 0
    if port <= 49151:
        return 0, 1, 0
    return 0, 0, 1


def standardize_flow_frame(frame: pd.DataFrame, scenario: str | None = None) -> pd.DataFrame:
    """Normalize CTU-13 flows and derive the 22 fixed edge features."""

    scenario_name = scenario or str(frame.get("scenario", "unknown").iloc[0] if "scenario" in frame else "unknown")
    normalized = _ensure_required_columns(frame, scenario=scenario_name)
    normalized["label_binary"] = normalized["label"].map(binary_label).astype(int)

    protocol_series = normalized["proto"].str.upper()
    normalized["proto_tcp"] = protocol_series.eq("TCP").astype(int)
    normalized["proto_udp"] = protocol_series.eq("UDP").astype(int)
    normalized["proto_icmp"] = protocol_series.eq("ICMP").astype(int)

    src_port_buckets = normalized["src_port"].map(_port_bucket)
    dst_port_buckets = normalized["dst_port"].map(_port_bucket)
    normalized[["src_port_well_known", "src_port_registered", "src_port_dynamic"]] = pd.DataFrame(
        src_port_buckets.tolist(), index=normalized.index
    )
    normalized[["dst_port_well_known", "dst_port_registered", "dst_port_dynamic"]] = pd.DataFrame(
        dst_port_buckets.tolist(), index=normalized.index
    )

    direction_series = normalized["direction"].astype(str)
    normalized["direction_forward"] = direction_series.str.contains(r"->", regex=True).astype(int)
    normalized["direction_reverse"] = direction_series.str.contains(r"<-", regex=True).astype(int)

    state_series = normalized["state"].astype(str).str.upper()
    normalized["state_con"] = state_series.str.contains("CON|EST", regex=True).astype(int)
    normalized["state_other"] = 1 - normalized["state_con"]

    normalized["total_packets"] = normalized["packets"]
    normalized["total_bytes"] = normalized["bytes"]
    normalized["byte_per_packet"] = [
        safe_divide(bytes_value, packets_value)
        for bytes_value, packets_value in zip(normalized["bytes"], normalized["packets"])
    ]
    normalized["packets_per_second"] = [
        safe_divide(packets_value, duration_value)
        for packets_value, duration_value in zip(normalized["packets"], normalized["duration"])
    ]
    normalized["bytes_per_second"] = [
        safe_divide(bytes_value, duration_value)
        for bytes_value, duration_value in zip(normalized["bytes"], normalized["duration"])
    ]
    normalized["tos_src"] = normalized["src_tos"]
    normalized["tos_dst"] = normalized["dst_tos"]
    return normalized.sort_values("start_time").reset_index(drop=True)


class FlowFeatureExtractor:
    """Extract and StandardScale the 22 CTU-13 flow features.

    After :meth:`fit` the extractor stores :attr:`feature_means_` — the
    per-feature means of the *raw* (unscaled) training features.  These are
    used by :meth:`transform_external` to fill feature columns that cannot be
    derived from an external dataset's schema (e.g. ``tos_src`` / ``tos_dst``
    for UNSW-NB15 which lacks ``src_tos`` / ``dst_tos`` source columns).
    """

    def __init__(self) -> None:
        self.feature_names = FLOW_FEATURE_COLUMNS.copy()
        self.scaler = StandardScaler()
        self.is_fitted = False
        #: Per-feature means of the raw (unscaled) training features.
        #: Shape ``(22,)``.  Set by :meth:`fit` and :meth:`fit_transform`.
        self.feature_means_: np.ndarray | None = None

    def prepare_frame(self, frame: pd.DataFrame, scenario: str | None = None) -> pd.DataFrame:
        """Normalize a flow frame and derive the raw 22 feature columns."""

        return standardize_flow_frame(frame, scenario=scenario)

    def raw_transform(self, frame: pd.DataFrame, scenario: str | None = None) -> np.ndarray:
        """Convert a flow frame to an unscaled `(n_flows, 22)` array."""

        prepared = self.prepare_frame(frame, scenario=scenario)
        return prepared[self.feature_names].to_numpy(dtype=np.float32)

    def fit(self, frame: pd.DataFrame, scenario: str | None = None) -> "FlowFeatureExtractor":
        """Fit the StandardScaler on a training flow frame.

        Also stores :attr:`feature_means_` — column-wise means of the raw
        (unscaled) feature matrix — for later use in
        :meth:`transform_external`.
        """
        raw_matrix = self.raw_transform(frame, scenario=scenario)
        self.scaler.fit(raw_matrix)
        self.feature_means_ = raw_matrix.mean(axis=0).astype(np.float32)
        self.is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame, scenario: str | None = None) -> np.ndarray:
        """Transform a flow frame with the fitted StandardScaler."""

        if not self.is_fitted:
            raise RuntimeError("FlowFeatureExtractor must be fitted on training data before transform().")
        raw_matrix = self.raw_transform(frame, scenario=scenario)
        return self.scaler.transform(raw_matrix).astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame, scenario: str | None = None) -> np.ndarray:
        """Fit on a training flow frame and return normalized features."""

        raw_matrix = self.raw_transform(frame, scenario=scenario)
        normalized = self.scaler.fit_transform(raw_matrix).astype(np.float32)
        self.feature_means_ = raw_matrix.mean(axis=0).astype(np.float32)
        self.is_fitted = True
        return normalized

    def transform_external(
        self,
        frame: pd.DataFrame,
        missing_features: list[str] | None = None,
        scenario: str | None = None,
    ) -> np.ndarray:
        """Transform an external-dataset frame, filling missing features with training means.

        Unlike :meth:`transform`, this method replaces the raw feature values
        for *missing_features* with the corresponding column means from the
        CTU-13 training fit (:attr:`feature_means_`) instead of leaving them
        as zeros.  This prevents the StandardScaler from seeing out-of-range
        zeros for features that simply weren't present in the external schema.

        Parameters
        ----------
        frame:
            Flow DataFrame in CTU-13-compatible schema (e.g. as returned by
            :class:`~gbd_ctu.data.unsw_nb15_loader.UNSWN15Loader`).
        missing_features:
            Names from :data:`FLOW_FEATURE_COLUMNS` whose source columns are
            absent in *frame* and should be replaced with :attr:`feature_means_`.
            When ``None`` defaults to
            ``["tos_src", "tos_dst"]`` — the features absent in UNSW-NB15.
        scenario:
            Optional scenario label forwarded to :func:`standardize_flow_frame`.

        Returns
        -------
        np.ndarray of shape ``(N, 22)`` — scaled features.

        Raises
        ------
        RuntimeError
            If the extractor has not been fitted yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FlowFeatureExtractor must be fitted on training data before "
                "transform_external()."
            )
        if missing_features is None:
            missing_features = ["tos_src", "tos_dst"]

        raw_matrix = self.raw_transform(frame, scenario=scenario)

        if missing_features and self.feature_means_ is not None:
            for feat_name in missing_features:
                if feat_name not in self.feature_names:
                    continue
                idx = self.feature_names.index(feat_name)
                raw_matrix[:, idx] = self.feature_means_[idx]

        return self.scaler.transform(raw_matrix).astype(np.float32)

    def transform_splits(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame | None = None,
        test_frame: pd.DataFrame | None = None,
    ) -> dict[str, np.ndarray]:
        """Fit on train and transform optional validation/test frames."""

        outputs = {"train": self.fit_transform(train_frame)}
        if val_frame is not None:
            outputs["val"] = self.transform(val_frame)
        if test_frame is not None:
            outputs["test"] = self.transform(test_frame)
        return outputs


def build_feature_tensor(
    frame: pd.DataFrame,
    as_torch: bool = True,
    extractor: FlowFeatureExtractor | None = None,
):
    """Return a scaled flow feature tensor or numpy array for convenience."""

    active_extractor = extractor or FlowFeatureExtractor()
    feature_array = active_extractor.fit_transform(frame)
    if as_torch and torch is not None:
        return torch.tensor(feature_array, dtype=torch.float32)
    return feature_array
