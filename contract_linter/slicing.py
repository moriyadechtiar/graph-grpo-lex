"""Context slicing: which clauses bear on a given clause.

``dependencies`` follows what a clause relies on:
    clauses it REFERENCES, its parent clause (IS_PART_OF), and the clauses
    that DEFINE the terms it USES.

``dependents`` follows what relies on, or qualifies, a clause:
    clauses that REFERENCE it, its child clauses, and clauses that USE a term
    it DEFINES.

A retrieval policy that follows only ``dependencies`` misses a later clause that
conditions an earlier one ("Section 4.2: the permission in Section 4.1 applies
only if ..."), because that link points *into* the cited clause. ``dependents``
and ``context_slice(direction="both")`` include it.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

from .graph import ContractGraph, GraphSource, load_graph


def _neighbours(graph: ContractGraph, direction: str) -> Dict[str, List[tuple]]:
    """Map clause -> [(neighbour clause, reason)] for one direction."""
    defined_in: Dict[str, List[str]] = defaultdict(list)   # term -> clauses that define it
    used_in: Dict[str, List[str]] = defaultdict(list)      # term -> clauses that use it
    for e in graph.edges_of_type("DEFINES"):
        defined_in[e.tgt].append(e.src)
    for e in graph.edges_of_type("USES"):
        used_in[e.tgt].append(e.src)

    def term_name(term: str) -> str:
        return graph.nodes[term].label if term in graph.nodes else term

    table: Dict[str, List[tuple]] = defaultdict(list)
    for e in graph.edges:
        if e.type == "REFERENCES":
            if direction == "out":
                table[e.src].append((e.tgt, f"{e.src} references {e.tgt}"))
            else:
                table[e.tgt].append((e.src, f"{e.src} references {e.tgt}"))
        elif e.type == "IS_PART_OF":
            if direction == "out":
                table[e.src].append((e.tgt, f"{e.src} is part of {e.tgt}"))
            else:
                table[e.tgt].append((e.src, f"{e.src} is part of {e.tgt}"))
        elif e.type == "USES" and direction == "out":
            for definer in defined_in.get(e.tgt, []):
                if definer != e.src:
                    table[e.src].append((definer, f"{e.src} uses {term_name(e.tgt)!r}, defined in {definer}"))
        elif e.type == "DEFINES" and direction == "in":
            for user in used_in.get(e.tgt, []):
                if user != e.src:
                    table[e.src].append((user, f"{user} uses {term_name(e.tgt)!r}, defined in {e.src}"))
    return table


def _walk(graph: ContractGraph, start: str, direction: str, depth: int) -> Dict[str, Dict]:
    table = _neighbours(graph, direction)
    found: Dict[str, Dict] = {}
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, dist = queue.popleft()
        if dist == depth:
            continue
        for neighbour, reason in table.get(node, []):
            if neighbour == start:
                continue
            entry = found.setdefault(neighbour, {"distance": dist + 1, "via": []})
            if reason not in entry["via"] and entry["distance"] == dist + 1:
                entry["via"].append(reason)
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, dist + 1))
    return found


def dependencies(source: GraphSource, clause_id: str, depth: int = 1) -> Dict[str, Dict]:
    """Clauses that ``clause_id`` relies on, up to ``depth`` steps away."""
    return _walk(load_graph(source), str(clause_id), "out", depth)


def dependents(source: GraphSource, clause_id: str, depth: int = 1) -> Dict[str, Dict]:
    """Clauses that rely on, or qualify, ``clause_id``, up to ``depth`` steps away."""
    return _walk(load_graph(source), str(clause_id), "in", depth)


def context_slice(source: GraphSource, clause_id: str, direction: str = "both", depth: int = 1) -> Dict[str, Dict]:
    """Clauses to read together with ``clause_id``.

    direction: "out" (dependencies), "in" (dependents) or "both".
    Returns {clause: {"distance": k, "via": [reasons], "direction": "in"|"out"|"both"}}.
    """
    if direction not in ("in", "out", "both"):
        raise ValueError("direction must be 'in', 'out' or 'both'")
    graph = load_graph(source)
    result: Dict[str, Dict] = {}
    for way in (("out", "in") if direction == "both" else (direction,)):
        for clause, info in _walk(graph, str(clause_id), way, depth).items():
            if clause in result:
                result[clause]["via"].extend(r for r in info["via"] if r not in result[clause]["via"])
                result[clause]["distance"] = min(result[clause]["distance"], info["distance"])
                result[clause]["direction"] = "both"
            else:
                result[clause] = {"distance": info["distance"], "via": list(info["via"]), "direction": way}
    return dict(sorted(result.items(), key=lambda kv: (kv[1]["distance"], kv[0])))
