# NeuralLogicGraph: lightweight graph builder that converts per-rule relationships
# into a graph structure for dashboard embedding (e.g., networkx json export).

from __future__ import annotations
from typing import Dict, Any, List, Tuple

import json
import torch
import torch.nn as nn


class NeuralLogicGraph(nn.Module):
    def __init__(self, rule_names: List[str]):
        super().__init__()
        self.rule_names = rule_names

    def forward(self, rule_corr: torch.Tensor) -> Dict[str, Any]:
        """
        rule_corr: [R, R] correlation or similarity matrix between rules
        returns a JSON-like graph with nodes and edges.
        """
        R = rule_corr.shape[0]
        nodes = [{"id": i, "name": self.rule_names[i]} for i in range(R)]
        edges = []
        for i in range(R):
            for j in range(i + 1, R):
                w = float(rule_corr[i, j].detach().cpu())
                if abs(w) > 0.05:
                    edges.append({"source": i, "target": j, "weight": w})
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def to_json(graph: Dict[str, Any]) -> str:
        return json.dumps(graph, indent=2)
