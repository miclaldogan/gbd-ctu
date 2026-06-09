"""GBD-CTU NetFlow feature extraction.

This module extracts a fixed 35-dimensional feature vector from CTU-13 NetFlow
records and applies RobustScaler normalization. Inputs are raw or normalized
flow DataFrames; outputs are normalized DataFrames and numpy arrays shaped
`(n_flows, 35)` suitable for graph edge features and classical baselines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

try:
    import torch
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None

from gbd_ctu.data.utils import binary_label, normalize_column_name, safe_divide


FLOW_FEATURE_COLUMNS: list[str] = [
    # --- raw features (22) ---
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
    # --- log-transformed derived features (5) ---
    # Heavy-tailed flow metrics compress poorly for linear layers; log1p
    # makes them more Gaussian and helps both the skip-connection MLP path
    # and XGBoost/RF baselines that receive the same feature matrix.
    "log_duration",
    "log_total_bytes",
    "log_total_packets",
    "log_bytes_per_second",
    "log_packets_per_second",
    # --- temporal + asymmetry features (3) ---
    # Botnets beacon at irregular/off-peak hours and show asymmetric byte
    # counts (small outbound command, large inbound response or vice-versa).
    "hour_of_day",        # 0-23 normalised to [0,1]
    "is_nighttime",       # 1 if 23:00-06:00 (common botnet hours)
    "src_to_dst_ratio",   # src_bytes / total_bytes — flow direction asymmetry
    # --- IP-level behavioral features (5) ---
    # Group-by statistics per source IP across all flows in the scenario.
    # These encode scanning, beaconing, and activity-level signals that are
    # invisible from individual flow features alone.
    "src_ip_fan_out",          # log1p(unique dst IPs) — horizontal scanning
    "src_ip_flow_count",       # log1p(total flows from src IP) — activity level
    "src_ip_iat_mean",         # log1p(mean inter-arrival time, s) — periodicity
    "src_ip_iat_std",          # log1p(std of IAT) — low std = beaconing pattern
    "src_ip_dst_port_nunique", # log1p(unique dst ports) — port scanning
]

EDGE_FEATURE_COLUMNS: list[str] = FLOW_FEATURE_COLUMNS.copy()

NODE_FEATURE_COLUMNS: list[str] = [
    "flow_count",
    "mean_duration",
    "total_bytes_sent",
    "total_bytes_recv",
    "unique_dst_count",
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

    # Log-transformed derived features
    normalized["log_duration"] = np.log1p(normalized["duration"].clip(lower=0))
    normalized["log_total_bytes"] = np.log1p(normalized["total_bytes"].clip(lower=0))
    normalized["log_total_packets"] = np.log1p(normalized["total_packets"].clip(lower=0))
    normalized["log_bytes_per_second"] = np.log1p(normalized["bytes_per_second"].clip(lower=0))
    normalized["log_packets_per_second"] = np.log1p(normalized["packets_per_second"].clip(lower=0))

    # Temporal + asymmetry features
    hour = normalized["start_time"].dt.hour.fillna(0).astype(float)
    normalized["hour_of_day"] = hour / 23.0
    normalized["is_nighttime"] = ((hour >= 23) | (hour <= 6)).astype(float)
    normalized["src_to_dst_ratio"] = [
        safe_divide(s, b) for s, b in zip(normalized["src_bytes"], normalized["total_bytes"])
    ]

    # IP-level behavioral features — computed from cross-flow group stats.
    # Each flow is annotated with aggregated stats about its source IP's
    # behavior across the full scenario, capturing botnet signatures like
    # scanning (high fan-out), beaconing (low IAT std), and port sweeping.
    _grp = normalized.groupby("src_addr", sort=False)
    normalized["src_ip_fan_out"] = np.log1p(
        _grp["dst_addr"].transform("nunique").astype(float)
    )
    normalized["src_ip_flow_count"] = np.log1p(
        _grp["src_addr"].transform("count").astype(float)
    )
    normalized["src_ip_dst_port_nunique"] = np.log1p(
        normalized["src_addr"].map(
            _grp["dst_port"].nunique()
        ).fillna(0).astype(float)
    )
    # Inter-arrival time stats require sorting within each src group
    _iat_tmp = (
        normalized[["src_addr", "start_time"]]
        .copy()
        .sort_values(["src_addr", "start_time"])
    )
    _iat_tmp["_iat"] = (
        _iat_tmp.groupby("src_addr")["start_time"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
        .clip(lower=0)
    )
    _iat_stats = _iat_tmp.groupby("src_addr")["_iat"].agg(["mean", "std"]).fillna(0)
    normalized["src_ip_iat_mean"] = np.log1p(
        normalized["src_addr"].map(_iat_stats["mean"]).fillna(0).astype(float)
    )
    normalized["src_ip_iat_std"] = np.log1p(
        normalized["src_addr"].map(_iat_stats["std"]).fillna(0).astype(float)
    )

    return normalized.sort_values("start_time").reset_index(drop=True)


class FlowFeatureExtractor:
    """Extract and RobustScale the 35 CTU-13 flow features.

    RobustScaler (median + IQR) is used instead of StandardScaler because
    network traffic is heavily right-skewed — a single DDoS or exfiltration
    flow can be orders of magnitude larger than normal, pulling StandardScaler's
    mean and compressing the normal-traffic range.

    After :meth:`fit` the extractor stores :attr:`feature_means_` — the
    per-feature medians of the *raw* (unscaled) training features.  These are
    used by :meth:`transform_external` to fill feature columns that cannot be
    derived from an external dataset's schema (e.g. ``tos_src`` / ``tos_dst``
    for UNSW-NB15 which lacks ``src_tos`` / ``dst_tos`` source columns).
    """

    def __init__(self) -> None:
        self.feature_names = FLOW_FEATURE_COLUMNS.copy()
        self.scaler = RobustScaler()
        self.is_fitted = False
        #: Per-feature medians of the raw (unscaled) training features.
        #: Shape ``(35,)``.  Set by :meth:`fit` and :meth:`fit_transform`.
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
        self.feature_means_ = np.median(raw_matrix, axis=0).astype(np.float32)
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
        self.feature_means_ = np.median(raw_matrix, axis=0).astype(np.float32)
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
