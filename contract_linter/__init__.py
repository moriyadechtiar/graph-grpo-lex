"""Contract Linter: structural checks and graph metrics for contract graphs.

Extracted from the GRAPH-GRPO-LEX Contract Linter

    from contract_linter import lint
    report = lint("examples/affiliate_clause_graph.json")
    print(report.summary())

Reference: M. Dechtiar, D. M. Katz, M. Sundaresan, S. Jaume, H. Wang,
"GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group
Relative Policy Optimization," IEEE ICDMW 2025, pp. 767-776.
"""
from .graph import EDGE_RULES, NODE_TYPES, ContractGraph, Edge, Node, load_graph
from .metrics import graph_metrics, leaves, orphans
from .checks import Finding, run_checks
from .slicing import context_slice, dependencies, dependents
from .scores import reference_free_scores
from .report import LintReport, lint

__version__ = "0.1.0"

__all__ = [
    "EDGE_RULES", "NODE_TYPES", "ContractGraph", "Edge", "Node", "load_graph",
    "graph_metrics", "orphans", "leaves",
    "Finding", "run_checks",
    "context_slice", "dependencies", "dependents",
    "reference_free_scores",
    "LintReport", "lint",
]
