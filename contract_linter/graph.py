"""Loading and normalising contract graphs.

The expected format is the one used by the GRAPH-GRPO-LEX dataset and the
ContractGraphEval benchmark:

    {
      "contract_id": "...",
      "nodes": [{"id": "4.1", "node_type": "CLAUSE", "title": "4.1", "level": 2},
                {"id": "term:Affiliate", "node_type": "DEFINED_TERM", "name": "Affiliate"}],
      "edges": [{"src": "4.1", "tgt": "term:Affiliate", "type": "USES"}]
    }

Exports that use ``type`` for the node type, nest node fields under ``attrs``
(the contract-level graphs published as ``moriyad/contracts_graphs_sft``), or use
``source``/``target`` and ``from``/``to`` for edge endpoints, are accepted as well.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx

NODE_TYPES = ("CLAUSE", "PARTY", "DEFINED_TERM", "VALUE")

# Ontology: edge type -> (source node type, target node type)
EDGE_RULES = {
    "IS_PART_OF": ("CLAUSE", "CLAUSE"),
    "REFERENCES": ("CLAUSE", "CLAUSE"),
    "DEFINES": ("CLAUSE", "DEFINED_TERM"),
    "USES": ("CLAUSE", "DEFINED_TERM"),
    "MENTIONS_PARTY": ("CLAUSE", "PARTY"),
    "CONTAINS": ("CLAUSE", "VALUE"),
}


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    label: str
    attrs: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Edge:
    src: str
    tgt: str
    type: str


@dataclass
class ContractGraph:
    """A normalised contract graph. ``raw`` keeps the input dictionary."""

    contract_id: Optional[str]
    nodes: Dict[str, Node]
    edges: List[Edge]
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def node_type(self, node_id: str) -> Optional[str]:
        node = self.nodes.get(node_id)
        return node.type if node else None

    def ids_of_type(self, node_type: str) -> List[str]:
        return [n.id for n in self.nodes.values() if n.type == node_type]

    def edges_of_type(self, edge_type: str) -> List[Edge]:
        return [e for e in self.edges if e.type == edge_type]

    def to_networkx(self) -> nx.DiGraph:
        """Directed graph for topology metrics.

        Parallel edges between the same pair of nodes collapse into one edge
        whose ``etypes`` attribute lists every edge type. Endpoints that are
        referenced by an edge but not declared as nodes are added with
        ``ntype=None``.
        """
        g = nx.DiGraph()
        for node in self.nodes.values():
            g.add_node(node.id, ntype=node.type, label=node.label)
        for edge in self.edges:
            for endpoint in (edge.src, edge.tgt):
                if endpoint not in g:
                    g.add_node(endpoint, ntype=None, label=endpoint)
            if g.has_edge(edge.src, edge.tgt):
                g[edge.src][edge.tgt]["etypes"].add(edge.type)
            else:
                g.add_edge(edge.src, edge.tgt, etypes={edge.type})
        return g


def _first(d: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = d.get(key)
        if value is not None and value != "":
            return value
    return None


def _unwrap(obj: Dict[str, Any]) -> Dict[str, Any]:
    """ContractGraphEval predictions are stored as {"graph": {...}, "raw_text": ...}."""
    if "nodes" not in obj and isinstance(obj.get("graph"), dict):
        return obj["graph"]
    return obj


GraphSource = Union[ContractGraph, Dict[str, Any], str, Path]


def load_graph(source: GraphSource) -> ContractGraph:
    """Build a ContractGraph from a dict, a JSON string, or a path to a JSON file."""
    if isinstance(source, ContractGraph):
        return source
    if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith("{")):
        with open(source, encoding="utf-8") as fh:
            source = json.load(fh)
    elif isinstance(source, str):
        source = json.loads(source)
    if not isinstance(source, dict):
        raise TypeError(f"Cannot build a contract graph from {type(source).__name__}")

    raw = _unwrap(source)
    nodes: Dict[str, Node] = {}
    for item in raw.get("nodes") or []:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        node_id = str(item["id"])
        node_type = str(_first(item, "node_type", "type", "ntype") or "").upper()
        nested = item.get("attrs") if isinstance(item.get("attrs"), dict) else {}
        fields = {**nested, **{k: v for k, v in item.items() if k != "attrs"}}
        label = str(_first(fields, "name", "label", "title", "text") or node_id)
        attrs = {k: v for k, v in fields.items() if k not in ("id", "node_type", "type", "ntype")}
        nodes[node_id] = Node(node_id, node_type, label, attrs)

    edges: List[Edge] = []
    for item in raw.get("edges") or []:
        if not isinstance(item, dict):
            continue
        src = _first(item, "src", "source", "from")
        tgt = _first(item, "tgt", "target", "to")
        if src is None or tgt is None:
            continue
        edge_type = str(_first(item, "type", "edge_type", "etype", "relation") or "").upper()
        edges.append(Edge(str(src), str(tgt), edge_type))

    return ContractGraph(raw.get("contract_id"), nodes, edges, raw)
