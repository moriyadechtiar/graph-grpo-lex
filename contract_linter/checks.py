"""Lint checks: Each check returns ``Finding`` records with a severity:

  error    the graph contradicts the ontology or itself
  warning  a likely drafting or extraction problem
  info     a structural feature worth inspecting

A finding is a candidate for review, not a determination that the contract is
defective. An orphan clause or an unused definition can be intentional.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .graph import EDGE_RULES, ContractGraph, GraphSource, load_graph

SMALL_COMPONENT_MAX_NODES = 8

# Checks that need the whole contract. On a single-clause graph they fire on
# ordinary structure (a clause normally uses terms that are defined elsewhere).
CONTRACT_LEVEL_CHECKS = frozenset({
    "undefined_term", "unused_definition", "missing_parent_clause",
    "small_component", "articulation_point",
})


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    nodes: List[str] = field(default_factory=list)
    edge: Optional[Tuple[str, str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_term(label: Any) -> str:
    text = re.sub(r'["\u201c\u201d]', "", str(label if label is not None else ""))
    return text.strip().lower()


def _incoming(graph: ContractGraph, edge_type: str) -> Dict[str, List[str]]:
    table: Dict[str, List[str]] = defaultdict(list)
    for edge in graph.edges_of_type(edge_type):
        table[edge.tgt].append(edge.src)
    return table


def check_dangling_edges(graph: ContractGraph) -> List[Finding]:
    out = []
    for e in graph.edges:
        missing = [x for x in (e.src, e.tgt) if x not in graph.nodes]
        if missing:
            out.append(Finding("dangling_edge", "error",
                               f"{e.type} edge {e.src} -> {e.tgt} points to undeclared node(s): {', '.join(missing)}",
                               missing, (e.src, e.type, e.tgt)))
    return out


def check_ontology(graph: ContractGraph) -> List[Finding]:
    """Edge types must exist and connect the node types the ontology allows."""
    out = []
    for e in graph.edges:
        if e.src not in graph.nodes or e.tgt not in graph.nodes:
            continue  # reported as dangling_edge
        rule = EDGE_RULES.get(e.type)
        actual = (graph.nodes[e.src].type, graph.nodes[e.tgt].type)
        if rule is None:
            out.append(Finding("ontology_violation", "error",
                               f"Unknown edge type {e.type!r} on {e.src} -> {e.tgt}",
                               [e.src, e.tgt], (e.src, e.type, e.tgt)))
        elif actual != rule:
            out.append(Finding("ontology_violation", "error",
                               f"{e.type} must connect {rule[0]} -> {rule[1]}, found {actual[0]} -> {actual[1]} "
                               f"on {e.src} -> {e.tgt}", [e.src, e.tgt], (e.src, e.type, e.tgt)))
    return out


def check_definition_coverage(graph: ContractGraph) -> List[Finding]:
    """Undefined terms (used, never defined) and unused definitions (defined, never used)."""
    defined_by = _incoming(graph, "DEFINES")
    used_by = _incoming(graph, "USES")
    out = []
    for term in graph.ids_of_type("DEFINED_TERM"):
        label = graph.nodes[term].label
        if used_by.get(term) and not defined_by.get(term):
            users = sorted(set(used_by[term]))
            out.append(Finding("undefined_term", "warning",
                               f"Term {label!r} is used by {', '.join(users)} but no clause defines it",
                               [term] + users))
        if defined_by.get(term) and not used_by.get(term):
            out.append(Finding("unused_definition", "warning",
                               f"Term {label!r} is defined by {', '.join(sorted(set(defined_by[term])))} "
                               f"but never used", [term]))
    return out


def check_duplicate_definitions(graph: ContractGraph) -> List[Finding]:
    out = []
    defined_by = _incoming(graph, "DEFINES")
    for term, clauses in defined_by.items():
        distinct = sorted(set(clauses))
        if len(distinct) > 1 and term in graph.nodes:
            out.append(Finding("multiply_defined_term", "warning",
                               f"Term {graph.nodes[term].label!r} is defined by more than one clause: "
                               f"{', '.join(distinct)}", [term] + distinct))
    by_name: Dict[str, List[str]] = defaultdict(list)
    for term in graph.ids_of_type("DEFINED_TERM"):
        name = normalize_term(graph.nodes[term].label)
        if name:
            by_name[name].append(term)
    for name, ids in by_name.items():
        if len(ids) > 1:
            out.append(Finding("duplicate_term_name", "warning",
                               f"{len(ids)} term nodes share the name {name!r}: {', '.join(sorted(ids))}",
                               sorted(ids)))
    return out


def term_dependency_graph(graph: ContractGraph) -> nx.DiGraph:
    """Term A depends on term B when a clause that defines A also uses B."""
    defines: Dict[str, List[str]] = defaultdict(list)
    uses: Dict[str, List[str]] = defaultdict(list)
    for e in graph.edges_of_type("DEFINES"):
        defines[e.src].append(e.tgt)
    for e in graph.edges_of_type("USES"):
        uses[e.src].append(e.tgt)
    g = nx.DiGraph()
    g.add_nodes_from(graph.ids_of_type("DEFINED_TERM"))
    for clause, defined_terms in defines.items():
        for a in defined_terms:
            for b in uses.get(clause, []):
                if a != b:
                    g.add_edge(a, b, via=clause)
    return g


def check_cycles(graph: ContractGraph) -> List[Finding]:
    out = []
    for cycle in nx.simple_cycles(term_dependency_graph(graph)):
        names = [graph.nodes[t].label if t in graph.nodes else t for t in cycle]
        out.append(Finding("definition_cycle", "error",
                           "Circular definition: " + " -> ".join(names + names[:1]), list(cycle)))
    refs = nx.DiGraph()
    refs.add_edges_from((e.src, e.tgt) for e in graph.edges_of_type("REFERENCES"))
    for cycle in nx.simple_cycles(refs):
        out.append(Finding("reference_cycle", "info",
                           "Clauses reference each other in a loop: " + " -> ".join(cycle + cycle[:1]),
                           list(cycle)))
    return out


def check_hierarchy(graph: ContractGraph) -> List[Finding]:
    """A clause numbered 3.4 should have parent 3 (or ARTICLE 3) and an IS_PART_OF edge to it."""
    part_of = {(e.src, e.tgt) for e in graph.edges_of_type("IS_PART_OF")}
    out = []
    for clause in graph.ids_of_type("CLAUSE"):
        if "." not in clause:
            continue
        prefix = clause.rsplit(".", 1)[0]
        parents = [p for p in (prefix, f"ARTICLE {prefix}") if graph.node_type(p) == "CLAUSE"]
        if not parents:
            out.append(Finding("missing_parent_clause", "warning",
                               f"Clause {clause} has no parent clause {prefix} in the graph", [clause]))
        elif not any((clause, p) in part_of for p in parents):
            out.append(Finding("missing_is_part_of", "warning",
                               f"Clause {clause} has no IS_PART_OF edge to its parent {parents[0]}",
                               [clause, parents[0]]))
    return out


def check_connectivity(graph: ContractGraph) -> List[Finding]:
    g = graph.to_networkx()
    out = []
    for node, degree in g.degree():
        if degree == 0:
            out.append(Finding("isolated_node", "warning",
                               f"Node {node} has no edges", [node]))
    undirected = g.to_undirected()
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    for comp in components[1:]:
        if 1 < len(comp) <= SMALL_COMPONENT_MAX_NODES:
            out.append(Finding("small_component", "info",
                               f"{len(comp)} nodes are disconnected from the main graph: "
                               f"{', '.join(sorted(comp))}", sorted(comp)))
    for node in sorted(nx.articulation_points(undirected)):
        out.append(Finding("articulation_point", "info",
                           f"Node {node} is a single point of connection; removing it splits the graph",
                           [node]))
    return out


ALL_CHECKS = (
    check_dangling_edges, check_ontology, check_definition_coverage,
    check_duplicate_definitions, check_cycles, check_hierarchy, check_connectivity,
)

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


def run_checks(source: GraphSource, clause_level: bool = False) -> List[Finding]:
    """Run every check. Set ``clause_level=True`` for a single-clause graph to
    skip the checks that only make sense on a whole contract."""
    graph = load_graph(source)
    findings: List[Finding] = []
    for check in ALL_CHECKS:
        findings.extend(check(graph))
    if clause_level:
        findings = [f for f in findings if f.check not in CONTRACT_LEVEL_CHECKS]
    findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.check, f.nodes))
    return findings
