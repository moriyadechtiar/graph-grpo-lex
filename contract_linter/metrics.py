"""Graph metrics from Section IV-A-3 of the GRAPH-GRPO-LEX paper.

``graph_metrics`` follows the ``metrics_for`` function of the
``ContractToGraph_complete_n=1_example`` notebook:

  density            edges divided by the maximum possible number of edges
  dependency_depth   length of the longest shortest path in the directed graph
  k_core_k           largest k for which a k-core exists (undirected view)
  orphans            nodes with no incoming edge
  leaves             nodes with no outgoing edge
  articulation       nodes whose removal disconnects the undirected graph
"""
from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx

from .graph import GraphSource, load_graph


def dependency_depth(g: nx.DiGraph) -> int:
    depth = 0
    for node in g.nodes():
        distances = nx.single_source_shortest_path_length(g, node)
        if distances:
            depth = max(depth, max(distances.values()))
    return depth


def orphans(source: GraphSource) -> List[str]:
    """Nodes with no incoming edge (the paper's definition of an orphan)."""
    g = load_graph(source).to_networkx()
    return sorted(n for n, d in g.in_degree() if d == 0)


def leaves(source: GraphSource) -> List[str]:
    """Nodes with no outgoing edge."""
    g = load_graph(source).to_networkx()
    return sorted(n for n, d in g.out_degree() if d == 0)


def graph_metrics(source: GraphSource, top: int = 5) -> Dict[str, Any]:
    graph = load_graph(source)
    g = graph.to_networkx()
    n, m = g.number_of_nodes(), g.number_of_edges()
    self_loops = nx.number_of_selfloops(g)

    if self_loops or n == 0:
        k_core_k = None
    else:
        k_core_k = max(nx.core_number(g.to_undirected()).values())

    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())
    orphan_count = sum(1 for v in g if in_deg[v] == 0)
    leaf_count = sum(1 for v in g if out_deg[v] == 0)
    articulation = sorted(nx.articulation_points(g.to_undirected())) if n else []

    def ranked(degrees: Dict[str, int]) -> List[Dict[str, Any]]:
        best = sorted(degrees.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return [{"node": node, "degree": deg} for node, deg in best if deg > 0]

    return {
        "contract_id": graph.contract_id,
        "nodes": n,
        "edges": m,
        "edge_records": len(graph.edges),
        "has_self_loops": bool(self_loops),
        "density": round(nx.density(g), 4) if n > 1 else 0.0,
        "dependency_depth": dependency_depth(g),
        "k_core_k": k_core_k,
        "orphans": orphan_count,
        "leaves": leaf_count,
        "orphan_ratio": round(orphan_count / n, 3) if n else 0.0,
        "leaf_ratio": round(leaf_count / n, 3) if n else 0.0,
        "articulation_points": len(articulation),
        "articulation_point_ids": articulation,
        "top_in_degree": ranked(in_deg),
        "top_out_degree": ranked(out_deg),
    }
