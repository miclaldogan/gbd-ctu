"""GBD-CTU UNSW-NB15 data loader.

Loads UNSW-NB15 bidirectional NetFlow CSVs and remaps their columns to the
CTU-13-compatible schema expected by :class:`~gbd_ctu.data.feature_extractor.FlowFeatureExtractor`.

The UNSW-NB15 dataset (2015) uses a slightly different feature naming
convention.  This module handles the column remapping, binary-label
alignment, and derived column computation so that
``FlowFeatureExtractor.transform_external()`` can then produce the same
22-dimensional feature vector as for CTU-13.

References
----------
Moustafa & Slay, "UNSW-NB15: a comprehensive data set for network intrusion
detection systems (UNSW-NB15 network data set)", MilCIS 2015.

Download
--------
The dataset is available at:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Required file: ``UNSW_NB15_testing-set.csv`` (or any per-partition CSV).
Column names are case-insensitive; both the training and testing CSVs are
supported.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping: UNSW-NB15 name → CTU-13 name
# ---------------------------------------------------------------------------

#: Maps every known UNSW-NB15 column (lower-cased) to the CTU-13 schema.
#: Columns absent from this map are dropped.
UNSW_FEATURE_MAP: dict[str, str] = {
    # Network addresses
    "srcip": "src_addr",
    "dstip": "dst_addr",
    # Ports (two common spellings in different CSV versions)
    "sport": "src_port",
    "dsport": "dst_port",
    "dport": "dst_port",
    # Protocol / state
    "proto": "proto",
    "state": "state",
    # Duration
    "dur": "duration",
    # Byte counts
    "sbytes": "sbytes",        # source bytes — kept separate for summing
    "dbytes": "dbytes",        # dest bytes   — kept separate for summing
    # Packet counts
    "spkts": "spkts",          # source packets
    "dpkts": "dpkts",          # dest packets
    # Timing
    "stime": "start_time",
    "ltime": "end_time",
    # Labels
    "label": "label_binary",   # 0=normal, 1=attack
    "attack_cat": "label",     # textual attack category
}

#: FLOW_FEATURE_COLUMNS that are *not* derivable from UNSW-NB15 source columns.
#: These will be filled with training-set means by ``transform_external()``.
UNSW_MISSING_FLOW_FEATURES: list[str] = ["tos_src", "tos_dst"]

#: Source columns missing from UNSW (used for informational logging).
UNSW_ABSENT_SOURCE_COLS: list[str] = ["direction", "src_tos", "dst_tos"]


class UNSWN15Loader:
    """Load and remap UNSW-NB15 CSV files to the CTU-13 flow schema.

    The resulting DataFrame has the same column set as a
    ``CTU13Loader.load_scenario()`` output and can be passed directly to
    :meth:`~gbd_ctu.data.feature_extractor.FlowFeatureExtractor.transform_external`.

    Parameters
    ----------
    binary_label_col:
        Name of the UNSW column that holds the 0/1 label.  Default ``"label"``.
        Some UNSW-NB15 releases capitalise this as ``"Label"``.
    """

    def __init__(self, binary_label_col: str = "label") -> None:
        self._binary_label_col = binary_label_col.lower()

    def load(self, csv_path: str | Path) -> pd.DataFrame:
        """Load one UNSW-NB15 CSV and return a CTU-13-compatible DataFrame.

        Steps
        -----
        1. Read the CSV; lower-case all column names.
        2. Remap known columns via :data:`UNSW_FEATURE_MAP`.
        3. Derive ``bytes`` (total) = ``sbytes`` + ``dbytes``.
        4. Derive ``packets`` = ``spkts`` + ``dpkts``.
        5. Fill absent CTU-13 source columns with sensible defaults so that
           :func:`~gbd_ctu.data.feature_extractor.standardize_flow_frame`
           can compute all 22 features (missing ones are later replaced with
           training-set means via ``transform_external()``).

        Parameters
        ----------
        csv_path:
            Path to the UNSW-NB15 CSV file.

        Returns
        -------
        pd.DataFrame — flow table in CTU-13-compatible schema with at least:
        ``src_addr``, ``dst_addr``, ``src_port``, ``dst_port``, ``proto``,
        ``state``, ``duration``, ``src_bytes``, ``bytes``, ``packets``,
        ``label``, ``label_binary``, ``start_time``, ``scenario_id``,
        ``scenario_name``.

        Raises
        ------
        FileNotFoundError
            If *csv_path* does not exist.
        ValueError
            If the CSV has no recognisable columns from :data:`UNSW_FEATURE_MAP`.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"UNSW-NB15 file not found: {path}")

        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        _logger.info("Loaded UNSW-NB15 CSV: %s (%d rows, %d cols)", path.name, len(df), len(df.columns))

        recognized = [c for c in df.columns if c in UNSW_FEATURE_MAP]
        if not recognized:
            raise ValueError(
                f"No recognised UNSW-NB15 columns found in {path.name}. "
                f"Expected at least one of: {list(UNSW_FEATURE_MAP)}"
            )

        out = pd.DataFrame(index=df.index)

        # ---- remap known columns ----------------------------------------
        for unsw_col, ctu_col in UNSW_FEATURE_MAP.items():
            if unsw_col in df.columns:
                out[ctu_col] = df[unsw_col]

        # ---- derive total bytes / packets --------------------------------
        if "sbytes" in out.columns and "dbytes" in out.columns:
            s = pd.to_numeric(out["sbytes"], errors="coerce").fillna(0)
            d = pd.to_numeric(out["dbytes"], errors="coerce").fillna(0)
            out["bytes"] = (s + d).astype(np.int64)
            out.drop(columns=["sbytes", "dbytes"], inplace=True, errors="ignore")
        elif "sbytes" in out.columns:
            out["bytes"] = pd.to_numeric(out["sbytes"], errors="coerce").fillna(0).astype(np.int64)
            out["src_bytes"] = out["bytes"]
            out.drop(columns=["sbytes"], inplace=True, errors="ignore")

        if "spkts" in out.columns and "dpkts" in out.columns:
            sp = pd.to_numeric(out["spkts"], errors="coerce").fillna(0)
            dp = pd.to_numeric(out["dpkts"], errors="coerce").fillna(0)
            out["packets"] = (sp + dp).astype(np.int64)
            out.drop(columns=["spkts", "dpkts"], inplace=True, errors="ignore")
        elif "spkts" in out.columns:
            out["packets"] = pd.to_numeric(out["spkts"], errors="coerce").fillna(0).astype(np.int64)
            out.drop(columns=["spkts"], inplace=True, errors="ignore")

        # src_bytes must be present
        if "src_bytes" not in out.columns:
            out["src_bytes"] = out.get("bytes", pd.Series(0, index=out.index))

        # ---- start_time --------------------------------------------------
        if "start_time" in out.columns:
            out["start_time"] = pd.to_datetime(
                pd.to_numeric(out["start_time"], errors="coerce"),
                unit="s", errors="coerce",
            ).fillna(pd.Timestamp("2015-01-01"))
        else:
            out["start_time"] = pd.Timestamp("2015-01-01")

        # ---- numeric coercions ------------------------------------------
        for col in ("duration", "src_port", "dst_port", "src_bytes", "bytes", "packets"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

        # ---- label_binary -----------------------------------------------
        if "label_binary" not in out.columns:
            if self._binary_label_col in df.columns:
                out["label_binary"] = pd.to_numeric(
                    df[self._binary_label_col], errors="coerce"
                ).fillna(0).astype(int)
            else:
                _logger.warning("No binary label column found; setting label_binary=0.")
                out["label_binary"] = 0

        if "label" not in out.columns:
            out["label"] = out["label_binary"].map({0: "Normal", 1: "Attack"})

        # ---- fill absent CTU-13 source columns --------------------------
        # These will be overridden in transform_external() with feature_means_.
        _missing = []
        for col, default in [
            ("src_addr", "0.0.0.0"),
            ("dst_addr", "0.0.0.0"),
            ("proto", "UNK"),
            ("state", "UNK"),
            ("direction", "UNK"),   # not in UNSW → feature means will fix
            ("src_tos", 0),          # not in UNSW → feature means will fix
            ("dst_tos", 0),          # not in UNSW → feature means will fix
        ]:
            if col not in out.columns:
                out[col] = default
                _missing.append(col)

        if _missing:
            _logger.info(
                "UNSW-NB15 absent columns filled with defaults (use transform_external "
                "to replace with training means): %s", _missing
            )

        # ---- metadata ---------------------------------------------------
        out["scenario_id"] = 0
        out["scenario_name"] = "UNSW-NB15"

        _logger.info(
            "UNSW-NB15 loaded: %d flows | botnet=%.1f%%",
            len(out),
            100.0 * out["label_binary"].mean(),
        )
        return out.reset_index(drop=True)

    def load_botnet_vs_normal(self, csv_path: str | Path) -> pd.DataFrame:
        """Load UNSW-NB15 and keep only botnet (label=1) and normal (label=0) flows.

        Background / unknown flows are excluded so the evaluation mirrors the
        CTU-13 botnet-vs-normal protocol.
        """
        df = self.load(csv_path)
        mask = df["label_binary"].isin([0, 1])
        filtered = df[mask].reset_index(drop=True)
        _logger.info(
            "Kept %d/%d flows after botnet-vs-normal filter.",
            len(filtered), len(df),
        )
        return filtered
