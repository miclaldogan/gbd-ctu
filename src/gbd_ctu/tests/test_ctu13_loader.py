"""GBD-CTU CTU-13 loader tests.

These tests validate that CTU13Loader and its helpers correctly normalize raw
CTU-13 bidirectional flow data and map heterogeneous label strings to a clean
binary target column.  All tests use synthetic in-memory DataFrames and do not
require CTU-13 data on disk.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gbd_ctu.data.ctu13_loader import _normalize_flow_frame
from gbd_ctu.data.utils import binary_label


# ---------------------------------------------------------------------------
# binary_label helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        # canonical CTU-13 botnet labels
        ("Botnet", 1),
        ("BOTNET", 1),
        ("botnet", 1),
        ("Botnet V42.B3.killexe", 1),
        # legitimate/normal labels
        ("LEGITIMATE", 0),
        ("Legitimate", 0),
        ("Normal", 0),
        # background traffic
        ("Background", 0),
        ("background", 0),
        # empty / NaN
        ("", 0),
        ("nan", 0),
        # malware synonym
        ("malware", 1),
        ("Malware.something", 1),
    ],
)
def test_binary_label_mapping(raw: str, expected: int) -> None:
    """binary_label() must map botnet/malware → 1 and all others → 0."""
    assert binary_label(raw) == expected


# ---------------------------------------------------------------------------
# _normalize_flow_frame
# ---------------------------------------------------------------------------

def _make_raw_frame() -> pd.DataFrame:
    """Minimal CTU-13-style raw DataFrame with original column casing."""
    return pd.DataFrame(
        [
            {
                "StartTime": "2011-08-10 09:46:54",
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
                "Label": "Botnet V42.B3.killexe",
            },
            {
                "StartTime": "2011-08-10 09:46:55",
                "Dur": 0.5,
                "Proto": "udp",
                "SrcAddr": "8.8.8.8",
                "Sport": 53,
                "Dir": "<->",
                "DstAddr": "147.32.84.165",
                "Dport": 5000,
                "State": "CON",
                "sTos": 0,
                "dTos": 0,
                "TotPkts": 4,
                "TotBytes": 256,
                "SrcBytes": 128,
                "Label": "LEGITIMATE",
            },
            {
                "StartTime": "2011-08-10 09:46:56",
                "Dur": 0.1,
                "Proto": "icmp",
                "SrcAddr": "192.168.1.1",
                "Sport": 0,
                "Dir": "->",
                "DstAddr": "10.0.0.1",
                "Dport": 0,
                "State": "CON",
                "sTos": 0,
                "dTos": 0,
                "TotPkts": 1,
                "TotBytes": 64,
                "SrcBytes": 64,
                "Label": "Background",
            },
        ]
    )


def test_normalize_adds_label_binary() -> None:
    """_normalize_flow_frame must add a 'label_binary' column."""
    frame = _normalize_flow_frame(_make_raw_frame(), scenario_name="test", scenario_id=1)
    assert "label_binary" in frame.columns


def test_normalize_label_binary_values() -> None:
    """Botnet row → 1; non-botnet rows → 0."""
    frame = _normalize_flow_frame(_make_raw_frame(), scenario_name="test", scenario_id=1)
    assert list(frame["label_binary"]) == [1, 0, 0]


def test_normalize_label_binary_dtype() -> None:
    """label_binary must be integer dtype, not float or object."""
    frame = _normalize_flow_frame(_make_raw_frame(), scenario_name="test", scenario_id=1)
    assert frame["label_binary"].dtype.kind == "i"


def test_normalize_adds_scenario_columns() -> None:
    """scenario_name and scenario_id columns must be present and correct."""
    frame = _normalize_flow_frame(_make_raw_frame(), scenario_name="capture20110818", scenario_id=5)
    assert "scenario_name" in frame.columns
    assert "scenario_id" in frame.columns
    assert (frame["scenario_name"] == "capture20110818").all()
    assert (frame["scenario_id"] == 5).all()


def test_normalize_column_names_lowercased() -> None:
    """All column names in the output must be snake_case (no upper-case letters)."""
    frame = _normalize_flow_frame(_make_raw_frame(), scenario_name="test", scenario_id=1)
    for col in frame.columns:
        assert col == col.lower(), f"Column name '{col}' is not lower-case"


def test_normalize_numeric_columns_are_numeric() -> None:
    """Numeric fields must be numeric dtype after normalization."""
    frame = _normalize_flow_frame(_make_raw_frame(), scenario_name="test", scenario_id=1)
    for col in ("duration", "src_port", "dst_port", "packets", "bytes", "src_bytes"):
        assert pd.api.types.is_numeric_dtype(frame[col]), f"Column '{col}' is not numeric"


def test_normalize_missing_tos_columns_default_to_zero() -> None:
    """sTos / dTos columns that are absent must default to 0."""
    raw = _make_raw_frame().drop(columns=["sTos", "dTos"])
    frame = _normalize_flow_frame(raw, scenario_name="test", scenario_id=1)
    assert (frame["src_tos"] == 0).all()
    assert (frame["dst_tos"] == 0).all()


def test_normalize_only_0_and_1_in_label_binary() -> None:
    """label_binary must contain only 0 and 1."""
    frame = _normalize_flow_frame(_make_raw_frame(), scenario_name="test", scenario_id=1)
    assert set(frame["label_binary"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# CTU13Loader — class interface (no disk access required)
# ---------------------------------------------------------------------------

def test_normalize_flow_frame_large_imbalanced_sample() -> None:
    """Imbalance logging path: 100 rows, 3 botnet flows."""
    rows = [
        {
            "StartTime": "2011-08-10 10:00:00",
            "Dur": 0.1, "Proto": "tcp", "SrcAddr": "1.1.1.1",
            "Sport": i, "Dir": "->", "DstAddr": "2.2.2.2",
            "Dport": 80, "State": "CON", "sTos": 0, "dTos": 0,
            "TotPkts": 1, "TotBytes": 64, "SrcBytes": 32,
            "Label": "Botnet" if i < 3 else "Background",
        }
        for i in range(100)
    ]
    frame = _normalize_flow_frame(pd.DataFrame(rows), scenario_name="test", scenario_id=99)
    assert int(frame["label_binary"].sum()) == 3
    assert len(frame) == 100


# ---------------------------------------------------------------------------
# CTU13Loader.available_scenarios()
# ---------------------------------------------------------------------------

def _make_loader_with_fake_files(tmp_path):
    """Create a CTU13Loader pointing at a temp dir with synthetic .binetflow files."""
    from gbd_ctu.data.ctu13_loader import CTU13Loader

    # Write two minimal binetflow files so discover_flow_files finds them.
    header = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
    row = "2011-08-10 10:00:00,1.0,tcp,1.1.1.1,12345,->,2.2.2.2,80,CON,0,0,10,1024,512,Background\n"
    for fname in ("capture20110810.binetflow", "capture20110811.binetflow"):
        (tmp_path / fname).write_text(header + row, encoding="utf-8")

    return CTU13Loader(data_root=str(tmp_path))


def test_available_scenarios_returns_list(tmp_path) -> None:
    """available_scenarios() must return a plain Python list."""
    loader = _make_loader_with_fake_files(tmp_path)
    result = loader.available_scenarios()
    assert isinstance(result, list)


def test_available_scenarios_returns_ints(tmp_path) -> None:
    """Every element returned by available_scenarios() must be an integer."""
    loader = _make_loader_with_fake_files(tmp_path)
    for sid in loader.available_scenarios():
        assert isinstance(sid, int), f"Expected int, got {type(sid)}: {sid!r}"


def test_available_scenarios_count(tmp_path) -> None:
    """available_scenarios() must discover exactly as many scenarios as files."""
    loader = _make_loader_with_fake_files(tmp_path)
    assert len(loader.available_scenarios()) == 2


def test_available_scenarios_sorted(tmp_path) -> None:
    """available_scenarios() must return IDs in ascending sorted order."""
    loader = _make_loader_with_fake_files(tmp_path)
    ids = loader.available_scenarios()
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# CTU13Loader._log_imbalance()
# ---------------------------------------------------------------------------

def _make_loader(tmp_path):
    from gbd_ctu.data.ctu13_loader import CTU13Loader
    return CTU13Loader(data_root=str(tmp_path))


def _make_normalized_frame(n_botnet: int, n_background: int) -> pd.DataFrame:
    """Build a minimal normalized frame (already has label_binary)."""
    rows = []
    for i in range(n_botnet + n_background):
        rows.append({
            "start_time": "2011-08-10 10:00:00",
            "duration": 1.0, "proto": "tcp",
            "src_addr": "1.1.1.1", "src_port": i,
            "direction": "->", "dst_addr": "2.2.2.2", "dst_port": 80,
            "state": "CON", "src_tos": 0, "dst_tos": 0,
            "packets": 1, "bytes": 64, "src_bytes": 32,
            "label": "Botnet" if i < n_botnet else "Background",
            "label_binary": 1 if i < n_botnet else 0,
            "scenario_name": "test", "scenario_id": 1,
        })
    return pd.DataFrame(rows)


def test_log_imbalance_does_not_raise(tmp_path) -> None:
    """_log_imbalance must complete without raising for a typical imbalanced frame."""
    loader = _make_loader(tmp_path)
    frame = _make_normalized_frame(n_botnet=10, n_background=90)
    loader._log_imbalance(scenario_id=1, frame=frame)  # must not raise


def test_log_imbalance_balanced_frame_does_not_raise(tmp_path) -> None:
    """_log_imbalance must complete without raising even for a perfectly balanced frame."""
    loader = _make_loader(tmp_path)
    frame = _make_normalized_frame(n_botnet=50, n_background=50)
    loader._log_imbalance(scenario_id=1, frame=frame)  # must not raise


def test_log_imbalance_all_botnet_does_not_raise(tmp_path) -> None:
    """_log_imbalance must not raise when every row is botnet (zero negatives)."""
    loader = _make_loader(tmp_path)
    frame = _make_normalized_frame(n_botnet=10, n_background=0)
    loader._log_imbalance(scenario_id=99, frame=frame)  # must not raise


def test_log_imbalance_no_botnet_does_not_raise(tmp_path) -> None:
    """_log_imbalance must not raise when there are zero botnet flows (division guard)."""
    loader = _make_loader(tmp_path)
    frame = _make_normalized_frame(n_botnet=0, n_background=20)
    loader._log_imbalance(scenario_id=2, frame=frame)  # must not raise (inf ratio branch)

