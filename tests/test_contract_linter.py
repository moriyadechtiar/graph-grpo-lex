"""Tests for the contract linter. Run with:  python -m pytest -q"""
import copy
import json
from pathlib import Path

import pytest

from contract_linter import (context_slice, dependencies, dependents, graph_metrics, lint,
                             load_graph, reference_free_scores, run_checks)
from contract_linter.__main__ import main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "affiliate_clause_graph.json"


@pytest.fixture
def base():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def checks(graph, **kw):
    return [f.check for f in run_checks(graph, **kw)]


# ---- loading -----------------------------------------------------------------

def test_load_from_path_dict_and_json_string(base):
    for source in (EXAMPLE, str(EXAMPLE), base, json.dumps(base)):
        g = load_graph(source)
        assert len(g.nodes) == 8 and len(g.edges) == 10
        assert g.node_type("term:Affiliate") == "DEFINED_TERM"


def test_load_accepts_alternative_keys_and_prediction_wrapper():
    wrapped = {"graph": {"nodes": [{"id": 1, "type": "clause"}, {"id": "t", "type": "defined_term", "label": "X"}],
                         "edges": [{"source": 1, "target": "t", "edge_type": "uses"}]}}
    g = load_graph(wrapped)
    assert g.node_type("1") == "CLAUSE"
    assert (g.edges[0].src, g.edges[0].tgt, g.edges[0].type) == ("1", "t", "USES")


def test_load_accepts_nested_attrs():
    g = load_graph({"nodes": [{"id": "para_1", "type": "CLAUSE", "attrs": {"title": "(a) Export", "level": 1}}],
                    "edges": []})
    assert g.nodes["para_1"].type == "CLAUSE"
    assert g.nodes["para_1"].label == "(a) Export"
    assert g.nodes["para_1"].attrs["level"] == 1


# ---- metrics -----------------------------------------------------------------

def test_metrics_on_example(base):
    m = graph_metrics(base)
    assert (m["nodes"], m["edges"]) == (8, 10)
    assert m["dependency_depth"] == 2            # 4.2 -> 4.1 -> 4
    assert m["orphans"] == 2                     # 1.2 and 4.2 have no incoming edge
    assert m["articulation_point_ids"] == ["1.2", "4.1"]
    assert m["has_self_loops"] is False


def test_metrics_on_empty_graph():
    m = graph_metrics({"nodes": [], "edges": []})
    assert m["nodes"] == 0 and m["density"] == 0.0 and m["k_core_k"] is None


# ---- checks ------------------------------------------------------------------

def test_example_has_one_undefined_term_and_no_errors(base):
    report = lint(base)
    assert report.errors == []
    undefined = [f for f in report.findings if f.check == "undefined_term"]
    assert len(undefined) == 1 and "Confidential Information" in undefined[0].message


def test_unused_definition(base):
    base["nodes"].append({"id": "term:Purpose", "node_type": "DEFINED_TERM", "name": "Purpose"})
    base["edges"].append({"src": "1.2", "tgt": "term:Purpose", "type": "DEFINES"})
    assert "unused_definition" in checks(base)


def test_dangling_edge_is_an_error(base):
    base["edges"].append({"src": "4.2", "tgt": "9.9", "type": "REFERENCES"})
    report = lint(base)
    assert [f.check for f in report.errors] == ["dangling_edge"]


def test_ontology_violation(base):
    base["edges"].append({"src": "4.1", "tgt": "party:Recipient", "type": "USES"})     # USES must target a term
    base["edges"].append({"src": "4.1", "tgt": "4.2", "type": "QUALIFIES"})            # unknown edge type
    assert checks(base).count("ontology_violation") == 2


def test_clause_to_clause_edges_are_valid(base):
    """IS_PART_OF and REFERENCES between clauses are allowed by the ontology."""
    assert "ontology_violation" not in checks(base)


def test_definition_cycle(base):
    # 1.2 defines Affiliate and uses Control; 1.3 defines Control and uses Affiliate.
    base["nodes"] += [{"id": "1.3", "node_type": "CLAUSE", "title": "1.3", "level": 2},
                      {"id": "term:Control", "node_type": "DEFINED_TERM", "name": "Control"}]
    base["edges"] += [{"src": "1.3", "tgt": "1", "type": "IS_PART_OF"},
                      {"src": "1.3", "tgt": "term:Control", "type": "DEFINES"},
                      {"src": "1.3", "tgt": "term:Affiliate", "type": "USES"},
                      {"src": "1.2", "tgt": "term:Control", "type": "USES"}]
    cycles = [f for f in run_checks(base) if f.check == "definition_cycle"]
    assert len(cycles) == 1 and cycles[0].severity == "error"
    assert set(cycles[0].nodes) == {"term:Affiliate", "term:Control"}


def test_reference_cycle_is_informational(base):
    base["edges"].append({"src": "4.1", "tgt": "4.2", "type": "REFERENCES"})
    found = [f for f in run_checks(base) if f.check == "reference_cycle"]
    assert len(found) == 1 and found[0].severity == "info"


def test_duplicate_definitions(base):
    base["edges"].append({"src": "4.1", "tgt": "term:Affiliate", "type": "DEFINES"})
    base["nodes"].append({"id": "term:affiliate ", "node_type": "DEFINED_TERM", "name": "\u201cAffiliate\u201d"})
    found = checks(base)
    assert "multiply_defined_term" in found and "duplicate_term_name" in found


def test_hierarchy_checks(base):
    base["nodes"].append({"id": "7.3", "node_type": "CLAUSE", "title": "7.3", "level": 2})
    assert "missing_parent_clause" in checks(base)
    no_edge = copy.deepcopy(base)
    no_edge["nodes"].append({"id": "7", "node_type": "CLAUSE", "title": "7", "level": 1})
    found = checks(no_edge)
    assert "missing_is_part_of" in found and "missing_parent_clause" not in found


def test_isolated_node_and_small_component(base):
    base["nodes"] += [{"id": "9", "node_type": "CLAUSE", "title": "9", "level": 1},
                      {"id": "10", "node_type": "CLAUSE", "title": "10", "level": 1},
                      {"id": "11", "node_type": "CLAUSE", "title": "11", "level": 1}]
    base["edges"].append({"src": "10", "tgt": "11", "type": "REFERENCES"})
    found = checks(base)
    assert "isolated_node" in found and "small_component" in found


def test_clause_level_mode_skips_whole_contract_checks(base):
    assert "undefined_term" in checks(base)
    assert "undefined_term" not in checks(base, clause_level=True)
    assert "articulation_point" not in checks(base, clause_level=True)


# ---- slicing -----------------------------------------------------------------

def test_outgoing_slice_misses_the_condition_and_incoming_slice_finds_it(base):
    """Clause 4.2 conditions clause 4.1. Following only what 4.1 points to never reaches 4.2."""
    out = dependencies(base, "4.1")
    assert set(out) == {"1.2", "4"}
    assert "4.2" not in out
    assert "4.2" in dependents(base, "4.1")
    both = context_slice(base, "4.1", direction="both")
    assert set(both) == {"1.2", "4", "4.2"}
    assert both["4.2"]["direction"] == "in"


def test_slice_depth(base):
    assert set(dependencies(base, "4.2", depth=1)) == {"4", "4.1", "1.2"}
    assert context_slice(base, "4.2", "out", depth=2)["4.1"]["distance"] == 1


def test_slice_rejects_bad_direction(base):
    with pytest.raises(ValueError):
        context_slice(base, "4.1", direction="sideways")


# ---- reference-free scores ---------------------------------------------------

def test_scores_on_clean_graph(base):
    scores = reference_free_scores(base)
    assert scores["domain_range_validity"] == 1.0
    assert scores["ontology_compliance"] == 1.0
    assert scores["orphan_nodes"] == 1.0
    assert scores["unused_definitions"] == 0.5        # one of two terms has a DEFINES edge


def test_undefined_references(base):
    assert reference_free_scores(base)["undefined_references"] == 1.0
    base["edges"].append({"src": "4.2", "tgt": "SCHEDULE 9", "type": "REFERENCES"})
    assert reference_free_scores(base)["undefined_references"] == 0.5


def test_duplicate_definitions_ignores_unnamed_nodes(base):
    base["nodes"].append({"id": "value:30 days", "node_type": "VALUE", "text": "30", "unit": "days"})
    base["edges"].append({"src": "4.1", "tgt": "value:30 days", "type": "CONTAINS"})
    assert reference_free_scores(base)["duplicate_definitions"] == 1.0


def test_scores_on_unparseable_prediction():
    assert set(reference_free_scores(None).values()) == {0.0}


# ---- command line ------------------------------------------------------------

def test_cli_exit_codes(tmp_path, base, capsys):
    assert main([str(EXAMPLE)]) == 0
    assert "undefined_term" in capsys.readouterr().out
    base["edges"].append({"src": "4.2", "tgt": "9.9", "type": "REFERENCES"})
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(base), encoding="utf-8")
    assert main([str(broken), "--json", "--scores", "--slice", "4.1"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["counts"]["dangling_edge"] == 1 and "4.2" in out["slice"]["context"]
