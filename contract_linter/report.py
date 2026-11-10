"""One-call entry point: metrics plus findings."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List

from .checks import Finding, run_checks
from .graph import ContractGraph, GraphSource, load_graph
from .metrics import graph_metrics


@dataclass
class LintReport:
    graph: ContractGraph
    metrics: Dict[str, Any]
    findings: List[Finding]

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def counts(self) -> Dict[str, int]:
        return dict(Counter(f.check for f in self.findings))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.graph.contract_id,
            "metrics": self.metrics,
            "counts": self.counts(),
            "findings": [f.to_dict() for f in self.findings],
        }

    def summary(self, max_findings: int = 50) -> str:
        m = self.metrics
        lines = [
            f"Contract: {self.graph.contract_id or '(no id)'}",
            f"  nodes {m['nodes']}, edges {m['edges']}, density {m['density']}, "
            f"dependency depth {m['dependency_depth']}, k-core {m['k_core_k']}",
            f"  orphans {m['orphans']} ({m['orphan_ratio']:.0%}), leaves {m['leaves']} ({m['leaf_ratio']:.0%}), "
            f"articulation points {m['articulation_points']}",
            f"  findings: {len(self.errors)} error(s), {len(self.warnings)} warning(s), "
            f"{len(self.findings) - len(self.errors) - len(self.warnings)} info",
        ]
        for f in self.findings[:max_findings]:
            lines.append(f"  [{f.severity:<7}] {f.check}: {f.message}")
        if len(self.findings) > max_findings:
            lines.append(f"  ... {len(self.findings) - max_findings} more")
        return "\n".join(lines)


def lint(source: GraphSource, clause_level: bool = False) -> LintReport:
    """Compute graph metrics and run every check on a contract graph."""
    graph = load_graph(source)
    return LintReport(graph, graph_metrics(graph), run_checks(graph, clause_level=clause_level))
