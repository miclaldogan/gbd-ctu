"""GBD-CTU model package.

This package contains graph neural architectures and classical ML baselines used
for CTU-13 botnet detection. Inputs are graph tensors or tabular matrices;
outputs are logits, probabilities, and fitted estimators.
"""

from gbd_ctu.models.gnn.gat import GATNodeClassifier
from gbd_ctu.models.gnn.graphsage import GraphSAGENodeClassifier
from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier

__all__ = [
    "GATNodeClassifier",
    "GraphSAGENodeClassifier",
    "GraphSageGATHybridClassifier",
]
