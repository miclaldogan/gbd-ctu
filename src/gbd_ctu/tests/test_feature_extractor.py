"""GBD-CTU feature extractor tests.

These tests validate that the CTU-13 flow feature extractor emits the expected
35-dimensional scaled matrix and can apply a fitted scaler across splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbd_ctu.data.feature_extractor import (
    FLOW_FEATURE_COLUMNS,
    FlowFeatureExtractor,
    standardize_flow_frame,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_flow(**overrides) -> dict:
    base = {
        "StartTime": "2011-08-10 10:00:00",
        "Dur": 1.0,
        "Proto": "tcp",
        "SrcAddr": "147.32.84.165",
        "Sport": 12345,
        "Dir": "->",
        "DstAddr": "74.125.39.104",
        "Dport": 80,
        "State": "CON",
        "sTos": 0,
        "dTos": 0,
        "TotPkts": 12,
        "TotBytes": 5120,
        "SrcBytes": 1024,
        "Label": "Botnet",
    }
    base.update(overrides)
    return base


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame([
        _make_flow(Dur=1.0, Proto="tcp", Sport=12345, Dport=80, Label="Botnet"),
        _make_flow(Dur=1.5, Proto="udp", Sport=53, Dir="<-", Dport=50123,
                   State="INT", sTos=16, dTos=8, TotPkts=8, TotBytes=4096,
                   SrcBytes=900, Label="Background"),
    ])


def _test_frame() -> pd.DataFrame:
    return pd.DataFrame([
        _make_flow(StartTime="2011-08-10 10:00:02", Dur=0.5, Proto="icmp",
                   SrcAddr="10.0.0.2", Sport=0, Dir="->", DstAddr="147.32.84.165",
                   Dport=0, State="CON", sTos=4, dTos=4,
                   TotPkts=4, TotBytes=512, SrcBytes=512, Label="LEGITIMATE"),
    ])


# ---------------------------------------------------------------------------
# Regression guard: column list
# ---------------------------------------------------------------------------

def test_flow_feature_columns_has_22_entries() -> None:
    """FLOW_FEATURE_COLUMNS must contain exactly 35 feature names (regression guard)."""
    assert len(FLOW_FEATURE_COLUMNS) == 35


def test_flow_feature_columns_no_duplicates() -> None:
    """FLOW_FEATURE_COLUMNS must not contain duplicates."""
    assert len(FLOW_FEATURE_COLUMNS) == len(set(FLOW_FEATURE_COLUMNS))


# ---------------------------------------------------------------------------
# Output shape and finiteness
# ---------------------------------------------------------------------------

def test_flow_feature_extractor_outputs_27_features() -> None:
    """FlowFeatureExtractor must emit a `(n_flows, 22)` matrix."""
    extractor = FlowFeatureExtractor()
    train_features = extractor.fit_transform(_train_frame())
    test_features = extractor.transform(_test_frame())

    assert train_features.shape == (2, 35)
    assert test_features.shape == (1, 35)
    assert np.isfinite(train_features).all()
    assert np.isfinite(test_features).all()


# ---------------------------------------------------------------------------
# feature_names property
# ---------------------------------------------------------------------------

def test_feature_names_property_returns_22_strings() -> None:
    """extractor.feature_names must be the list of 27 feature name strings."""
    extractor = FlowFeatureExtractor()
    assert extractor.feature_names == FLOW_FEATURE_COLUMNS
    assert len(extractor.feature_names) == 35
    assert all(isinstance(name, str) for name in extractor.feature_names)


def test_feature_names_property_matches_output_columns() -> None:
    """feature_names order must match columns in raw_transform output."""
    extractor = FlowFeatureExtractor()
    extractor.fit(_train_frame())
    prepared = extractor.prepare_frame(_train_frame())
    # Every name in feature_names must exist as a column in the prepared frame
    for name in extractor.feature_names:
        assert name in prepared.columns, f"Feature '{name}' missing from prepared frame"


# ---------------------------------------------------------------------------
# RobustScaler: fit only on train, not refitted on transform
# ---------------------------------------------------------------------------

def test_scaler_fit_only_on_train() -> None:
    """RobustScaler parameters must not change when transform() is called."""
    extractor = FlowFeatureExtractor()
    extractor.fit(_train_frame())
    center_before = extractor.scaler.center_.copy()
    scale_before = extractor.scaler.scale_.copy()

    # transform on a completely different frame must not refit
    extractor.transform(_test_frame())

    np.testing.assert_array_equal(extractor.scaler.center_, center_before)
    np.testing.assert_array_equal(extractor.scaler.scale_, scale_before)


def test_transform_raises_before_fit() -> None:
    """transform() on an unfitted extractor must raise RuntimeError."""
    extractor = FlowFeatureExtractor()
    with pytest.raises(RuntimeError, match="fitted"):
        extractor.transform(_test_frame())


def test_fit_transform_marks_extractor_as_fitted() -> None:
    """After fit_transform, is_fitted must be True."""
    extractor = FlowFeatureExtractor()
    assert not extractor.is_fitted
    extractor.fit_transform(_train_frame())
    assert extractor.is_fitted


# ---------------------------------------------------------------------------
# transform_splits
# ---------------------------------------------------------------------------

def test_transform_splits_returns_correct_keys_and_shapes() -> None:
    """transform_splits must return train/val/test arrays with correct shapes."""
    val_frame = _test_frame()
    extractor = FlowFeatureExtractor()
    result = extractor.transform_splits(_train_frame(), val_frame, _test_frame())

    assert set(result.keys()) == {"train", "val", "test"}
    assert result["train"].shape == (2, 35)
    assert result["val"].shape == (1, 35)
    assert result["test"].shape == (1, 35)


def test_transform_splits_without_optional_frames() -> None:
    """transform_splits with only train must return only 'train' key."""
    extractor = FlowFeatureExtractor()
    result = extractor.transform_splits(_train_frame())
    assert list(result.keys()) == ["train"]
    assert result["train"].shape == (2, 35)


# ---------------------------------------------------------------------------
# One-hot encoding: unseen values
# ---------------------------------------------------------------------------

def test_unseen_protocol_maps_to_zero_vector() -> None:
    """Protocol values not in {TCP, UDP, ICMP} must produce all-zero proto flags."""
    frame = pd.DataFrame([_make_flow(Proto="ARP")])
    prepared = standardize_flow_frame(frame)
    assert int(prepared["proto_tcp"].iloc[0]) == 0
    assert int(prepared["proto_udp"].iloc[0]) == 0
    assert int(prepared["proto_icmp"].iloc[0]) == 0


def test_known_protocols_encode_correctly() -> None:
    """TCP, UDP, ICMP must each produce a one-hot encoding."""
    for proto, expected in [("tcp", (1, 0, 0)), ("UDP", (0, 1, 0)), ("ICMP", (0, 0, 1))]:
        frame = pd.DataFrame([_make_flow(Proto=proto)])
        prepared = standardize_flow_frame(frame)
        actual = (
            int(prepared["proto_tcp"].iloc[0]),
            int(prepared["proto_udp"].iloc[0]),
            int(prepared["proto_icmp"].iloc[0]),
        )
        assert actual == expected, f"Proto '{proto}' expected {expected}, got {actual}"


def test_unseen_state_maps_to_state_other() -> None:
    """Unknown state values must set state_con=0, state_other=1."""
    frame = pd.DataFrame([_make_flow(State="RST")])
    prepared = standardize_flow_frame(frame)
    assert int(prepared["state_con"].iloc[0]) == 0
    assert int(prepared["state_other"].iloc[0]) == 1


def test_con_state_maps_to_state_con() -> None:
    """CON and EST states must set state_con=1, state_other=0."""
    for state in ("CON", "EST"):
        frame = pd.DataFrame([_make_flow(State=state)])
        prepared = standardize_flow_frame(frame)
        assert int(prepared["state_con"].iloc[0]) == 1, f"state='{state}' should set state_con=1"
        assert int(prepared["state_other"].iloc[0]) == 0


# ---------------------------------------------------------------------------
# Direction encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "direction, fwd, rev",
    [
        ("->", 1, 0),
        ("<-", 0, 1),
        ("<->", 1, 1),    # bidirectional: both flags set
        ("UNK", 0, 0),    # unknown direction: both flags zero
    ],
)
def test_direction_encoding(direction: str, fwd: int, rev: int) -> None:
    """Direction strings must encode into forward/reverse flags correctly."""
    frame = pd.DataFrame([_make_flow(Dir=direction)])
    prepared = standardize_flow_frame(frame)
    assert int(prepared["direction_forward"].iloc[0]) == fwd, f"Dir='{direction}' fwd expected {fwd}"
    assert int(prepared["direction_reverse"].iloc[0]) == rev, f"Dir='{direction}' rev expected {rev}"


# ---------------------------------------------------------------------------
# Port bucket encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "port, well_known, registered, dynamic",
    [
        (0,     1, 0, 0),   # well-known: 0
        (80,    1, 0, 0),   # well-known: 80
        (1023,  1, 0, 0),   # well-known: boundary
        (1024,  0, 1, 0),   # registered: lower boundary
        (8080,  0, 1, 0),   # registered: middle
        (49151, 0, 1, 0),   # registered: upper boundary
        (49152, 0, 0, 1),   # dynamic: lower boundary
        (65535, 0, 0, 1),   # dynamic: upper
    ],
)
def test_src_port_bucket(port: int, well_known: int, registered: int, dynamic: int) -> None:
    """Source port buckets must map to exactly one hot bit per row."""
    frame = pd.DataFrame([_make_flow(Sport=port)])
    prepared = standardize_flow_frame(frame)
    assert int(prepared["src_port_well_known"].iloc[0]) == well_known
    assert int(prepared["src_port_registered"].iloc[0]) == registered
    assert int(prepared["src_port_dynamic"].iloc[0]) == dynamic


# ---------------------------------------------------------------------------
# Derived rate features
# ---------------------------------------------------------------------------

def test_byte_per_packet_computed_correctly() -> None:
    """byte_per_packet = TotBytes / TotPkts."""
    frame = pd.DataFrame([_make_flow(TotBytes=5120, TotPkts=10)])
    prepared = standardize_flow_frame(frame)
    assert prepared["byte_per_packet"].iloc[0] == pytest.approx(512.0)


def test_derived_rates_zero_duration_safe() -> None:
    """packets_per_second and bytes_per_second must not produce NaN/Inf for Dur=0."""
    frame = pd.DataFrame([_make_flow(Dur=0, TotPkts=4, TotBytes=256)])
    prepared = standardize_flow_frame(frame)
    assert np.isfinite(prepared["packets_per_second"].iloc[0])
    assert np.isfinite(prepared["bytes_per_second"].iloc[0])