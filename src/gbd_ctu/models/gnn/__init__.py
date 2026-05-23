"""GBD-CTU graph neural network models.

This package provides GraphSAGE, GAT, and hybrid architectures for node-level
botnet classification on CTU-13 IP communication graphs.
"""

from gbd_ctu.models.gnn.gat import GATNodeClassifier
from gbd_ctu.models.gnn.graphsage import GraphSAGENodeClassifier
from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier

__all__ = ["GATNodeClassifier", "GraphSAGENodeClassifier", "GraphSageGATHybridClassifier"]
