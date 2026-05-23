"""GBD-CTU feature extractor tests.

These tests validate that the CTU-13 flow feature extractor emits the expected
22-dimensional scaled matrix and can apply a fitted scaler across splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbd_ctu.data.feature_extractor import FlowFeatureExtractor


def test_flow_feature_extractor_outputs_22_features() -> None:
    """FlowFeatureExtractor should emit a `(n_flows, 22)` matrix."""

    train_frame = pd.DataFrame(
        [
            {
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
            },
            {
                "StartTime": "2011-08-10 10:00:01",
                "Dur": 1.5,
                "Proto": "udp",
                "SrcAddr": "147.32.84.165",
                "Sport": 53,
                "Dir": "<-",
                "DstAddr": "8.8.8.8",
                "Dport": 50123,
                "State": "INT",
                "sTos": 16,
                "dTos": 8,
                "TotPkts": 8,
                "TotBytes": 4096,
                "SrcBytes": 900,
                "Label": "Background",
            },
        ]
    )
    test_frame = pd.DataFrame(
        [
            {
                "StartTime": "2011-08-10 10:00:02",
                "Dur": 0.5,
                "Proto": "icmp",
                "SrcAddr": "10.0.0.2",
                "Sport": 0,
                "Dir": "->",
                "DstAddr": "147.32.84.165",
                "Dport": 0,
                "State": "CON",
                "sTos": 4,
                "dTos": 4,
                "TotPkts": 4,
                "TotBytes": 512,
                "SrcBytes": 512,
                "Label": "LEGITIMATE",
            }
        ]
    )
    extractor = FlowFeatureExtractor()
    train_features = extractor.fit_transform(train_frame)
    test_features = extractor.transform(test_frame)

    assert train_features.shape == (2, 22)
    assert test_features.shape == (1, 22)
    assert np.isfinite(train_features).all()
    assert np.isfinite(test_features).all()