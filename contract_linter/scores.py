"""Reference-free quality scores in [0, 1] (higher is better).

These are the reference-free checks stored in the ContractGraphEval benchmark
under ``reference_free_judges``. They score a predicted graph on its own,
without a gold graph.

  unused_definitions     share of defined terms that the graph defines
  undefined_references   share of REFERENCES targets that are declared clauses
  duplicate_definitions  1 minus the share of defined-term names that repeat
  orphan_nodes           share of nodes that have at least one edge
  isolated_components    1 minus the share of small disconnected components
  domain_range_validity  share of edges whose endpoint types fit the edge type
  ontology_compliance    edge typing plus clause hierarchy (parent present,
                         IS_PART_OF edge present)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .checks import normalize_term
from .graph import EDGE_RULES, _unwrap

SMALL_COMPONENT_MAX_NODES = 8
SCORE_NAMES = ("unused_definitions", "undefined_references", "duplicate_definitions", "orphan_nodes",
               "isolated_components", "domain_range_validity", "ontology_compliance")

_ALLOWED_PAIRS = {
    ("CLAUSE", "DEFINED_TERM"): {"DEFINES", "USES"},
    ("CLAUSE", "VALUE"): {"CONTAINS"},
    ("CLAUSE", "PARTY"): {"MENTIONS_PARTY"},
    ("CLAUSE", "CLAUSE"): {"REFERENCES", "IS_PART_OF"},
}


def _src(e: Dict[str, Any]) -> str:
    return str(e.get("src") or e.get("source") or e.get("from"))


def _tgt(e: Dict[str, Any]) -> str:
    return str(e.get("tgt") or e.get("target") or e.get("to"))


def _unused_definitions(node_type, edges) -> float:
    terms = [nid for nid, t in node_type.items() if t == "DEFINED_TERM"]
    defined = {_tgt(e) for e in edges if e.get("type") == "DEFINES"}
    return 1.0 - sum(nid not in defined for nid in terms) / max(1, len(terms))


def _undefined_references(node_type, edges) -> float:
    targets = {_tgt(e) for e in edges if str(e.get("type", "")).upper() == "REFERENCES"}
    missing = [t for t in targets if node_type.get(t) != "CLAUSE"]
    return 1.0 - len(missing) / max(1, len(targets))


def _duplicate_definitions(nodes, node_type) -> float:
    by_name: Dict[str, List[str]] = defaultdict(list)
    for n in nodes:
        nid = str(n.get("id"))
        if node_type.get(nid) != "DEFINED_TERM":
            continue
        name = normalize_term(n.get("name") or n.get("label") or "")
        if name:
            by_name[name].append(nid)
    duplicated = sum(1 for ids in by_name.values() if len(ids) > 1)
    total = sum(1 for t in node_type.values() if t == "DEFINED_TERM")
    return 1.0 - duplicated / max(1, total)


def _orphan_nodes(node_type, edges) -> float:
    connected = {_src(e) for e in edges} | {_tgt(e) for e in edges}
    return 1.0 - len(set(node_type) - connected) / max(1, len(node_type))


def _isolated_components(node_type, edges) -> float:
    adj = defaultdict(set)
    for e in edges:
        s, t = _src(e), _tgt(e)
        adj[s].add(t)
        adj[t].add(s)
    visited, components = set(), []
    for nid in node_type:
        if nid in visited:
            continue
        comp, stack = [], [nid]
        visited.add(nid)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        components.append(comp)
    if len(components) <= 1:
        return 1.0
    isolated = [c for c in components[1:] if len(c) <= SMALL_COMPONENT_MAX_NODES]
    return 1.0 - len(isolated) / max(1, len(components) - 1)


def _domain_range_validity(node_type, edges) -> float:
    invalid = 0
    for e in edges:
        s, t = node_type.get(_src(e)), node_type.get(_tgt(e))
        if not s or not t:
            continue
        if (s, t) not in _ALLOWED_PAIRS or e.get("type") not in _ALLOWED_PAIRS[(s, t)]:
            invalid += 1
    return 1.0 - invalid / max(1, len(edges))


def _ontology_compliance(node_type, edges) -> float:
    invalid, part_of = 0, set()
    for e in edges:
        etype = str(e.get("type", e.get("edge_type", ""))).upper()
        s, t = _src(e), _tgt(e)
        rule = EDGE_RULES.get(etype)
        if node_type.get(s) is None or node_type.get(t) is None or rule is None \
                or (node_type.get(s), node_type.get(t)) != rule:
            invalid += 1
        if etype == "IS_PART_OF":
            part_of.add((s, t))
    with_parent = missing_parent = missing_edge = 0
    for nid, t in node_type.items():
        if t != "CLAUSE" or "." not in nid:
            continue
        with_parent += 1
        prefix = nid.rsplit(".", 1)[0]
        parents = [p for p in (prefix, f"ARTICLE {prefix}") if node_type.get(p) == "CLAUSE"]
        if not parents:
            missing_parent += 1
        elif not any((nid, p) in part_of for p in parents):
            missing_edge += 1
    expected = len(edges) + 2 * with_parent
    return max(0.0, min(1.0, 1.0 - (invalid + missing_parent + missing_edge) / max(1, expected)))


def reference_free_scores(graph: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Score a graph dictionary ({"nodes": [...], "edges": [...]}).

    A graph that is missing or not a dictionary, such as an unparseable model
    output, scores 0.0 on every check.
    """
    if not isinstance(graph, dict):
        return {name: 0.0 for name in SCORE_NAMES}
    graph = _unwrap(graph)
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    node_type = {str(n.get("id")): str(n.get("node_type", n.get("type", "")) or "").upper() for n in nodes}
    return {
        "unused_definitions": _unused_definitions(node_type, edges),
        "undefined_references": _undefined_references(node_type, edges),
        "duplicate_definitions": _duplicate_definitions(nodes, node_type),
        "orphan_nodes": _orphan_nodes(node_type, edges),
        "isolated_components": _isolated_components(node_type, edges),
        "domain_range_validity": _domain_range_validity(node_type, edges),
        "ontology_compliance": _ontology_compliance(node_type, edges),
    }
