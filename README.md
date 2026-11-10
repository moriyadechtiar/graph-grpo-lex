# GRAPH-GRPO-LEX
**Contract Graph Modeling + Gated GRPO Training for LLMs**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20ICDMW%202025-blue)](https://doi.org/10.1109/ICDMW69685.2025.00092)
[![arXiv](https://img.shields.io/badge/arXiv-2511.06618-b31b1b)](https://arxiv.org/abs/2511.06618)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/moriyad/contract-clause-graphs)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Turn long, messy contracts into structured clause-level graphs—then optimize extraction with staged (gated) Group Relative Policy Optimization.

## Overview

- **Goal:** Convert contracts into **semantic graphs** (nodes: `CLAUSE`, `PARTY`, `DEFINED_TERM`, `VALUE`; edges: `IS_PART_OF`, `DEFINES`, `USES`, `REFERENCES`, `MENTIONS_PARTY`, `CONTAINS`), then analyze them with a **contract linter** built from graph metrics.
- **Approach:** SFT a base LLM on clause→minigraph pairs, then **reinforcement-train with GRPO** using a **gated reward curriculum** to stabilize learning and boost F1.
- **What’s here:** the **`contract_linter`** Python package (importable, tested, with a command line), the notebooks for dataset creation, SFT, gated-GRPO training and evaluation, and the annotator test used to validate the labels.

## Quick start

```bash
git clone https://github.com/moriyadechtiar/graph-grpo-lex.git
cd graph-grpo-lex
pip install -r requirements.txt          # networkx only
python -m contract_linter examples/affiliate_clause_graph.json --slice 4.1
```

```
Contract: EXAMPLE-NDA (constructed illustration)
  nodes 8, edges 10, density 0.1786, dependency depth 2, k-core 2
  orphans 2 (25%), leaves 5 (62%), articulation points 2
  findings: 0 error(s), 1 warning(s), 2 info
  [warning] undefined_term: Term 'Confidential Information' is used by 4.1 but no clause defines it
  [info   ] articulation_point: Node 1.2 is a single point of connection; removing it splits the graph
  [info   ] articulation_point: Node 4.1 is a single point of connection; removing it splits the graph
Context for clause 4.1 (direction=both, depth=1):
  1.2  [out, distance 1]  4.1 uses 'Affiliate', defined in 1.2
  4  [out, distance 1]  4.1 is part of 4
  4.2  [in, distance 1]  4.2 references 4.1
```

From Python:

```python
from contract_linter import lint, context_slice, reference_free_scores

report = lint("examples/affiliate_clause_graph.json")
report.metrics            # density, dependency depth, k-core, orphans, leaves, articulation points
report.findings           # list of Finding(check, severity, message, nodes, edge)

context_slice(report.graph, "4.1", direction="both")    # clauses to read together with 4.1
reference_free_scores(report.graph.raw)                 # seven scores in [0, 1], no gold graph needed
```

The command exits with status 1 when the graph has error-level findings, so it can run in CI. Tests: `python -m pytest -q`.

## Contract linter

### Checks

| Check | Severity | Meaning |
|---|---|---|
| `dangling_edge` | error | An edge points to a node that is not declared |
| `ontology_violation` | error | Unknown edge type, or an edge between node types the ontology does not allow |
| `definition_cycle` | error | Term A is defined using term B, and B using A |
| `undefined_term` | warning | A term is used but no clause defines it |
| `unused_definition` | warning | A term is defined but never used |
| `multiply_defined_term`, `duplicate_term_name` | warning | A term is defined by several clauses, or two term nodes share a name |
| `missing_parent_clause`, `missing_is_part_of` | warning | Clause 3.4 has no parent clause 3, or no `IS_PART_OF` edge to it |
| `isolated_node` | warning | A node has no edges |
| `small_component` | info | A few nodes are disconnected from the main graph |
| `reference_cycle` | info | Clauses reference each other in a loop |
| `articulation_point` | info | A node whose removal splits the graph |

A finding is a candidate for review, not a determination that the contract is defective. For a graph of a single clause, pass `--clause-level` to skip the checks that need the whole contract.

### Graph metrics

| Metric | Definition | Reading |
|---|---|---|
| Density | Edges ÷ maximum possible edges | Overall interconnection |
| Dependency depth | Longest shortest path in the directed graph | How far a reader must follow links |
| k-core | Largest k with a non-empty k-core | The most interdependent core |
| Orphans, leaves | Nodes with no incoming, or no outgoing, edge | Completeness and terminal consequences |
| Articulation points | Nodes whose removal disconnects the graph | Single points of failure |
| Top in-degree, out-degree | Most referenced and most referencing nodes | Review priority |

### Context slicing

`dependencies(graph, clause)` follows what a clause relies on: clauses it references, its parent, and the clauses that define the terms it uses. `dependents(graph, clause)` follows the other direction: clauses that reference it, its children, and clauses that use a term it defines.

In the example above, clause 4.2 conditions the permission in clause 4.1. A retrieval step that starts at 4.1 and follows only its outgoing links reaches the definition in 1.2 but never reaches 4.2. `context_slice(..., direction="both")` includes it.

### Reference-free scores

`reference_free_scores` computes seven quality scores for a predicted graph without a gold graph. They are the reference-free checks used in the [ContractGraphEval](https://github.com/moriyadechtiar/ensemble-graph-eval) benchmark. To score a file of predictions:

```bash
python examples/score_predictions.py predictions.jsonl
```

## Data

The dataset is on Hugging Face: [`moriyad/contract-clause-graphs`](https://huggingface.co/datasets/moriyad/contract-clause-graphs).

- 2,599 clause→graph pairs from 42 commercial agreements in CUAD.
- Train, validation and test splits hold different contracts.
- Graphs were labelled by GPT-5 and validated against three human annotators with the alternative annotator test (`alt_test_llm_judge.ipynb`).

Results are reported in the [paper](https://doi.org/10.1109/ICDMW69685.2025.00092).

## Reward design

- **Structure score** → valid/near-valid JSON
- **Strict/Fuzzy F1** → node/edge extraction accuracy
- **Embedding similarity** → semantic closeness
- **Gated curriculum** → introduce signals in stages to reduce reward noise & collapse.

## Repository structure

```
graph-grpo-lex/
├── contract_linter/    linter package: loading, metrics, checks, slicing, scores, command line
├── tests/              unit tests
├── examples/           example graph and a script that scores a file of predictions
└── *.ipynb             dataset creation, labelling, annotator test, SFT, gated GRPO, contract-level demo
```

The notebooks were written for Google Colab and read from Google Drive paths.

## Citation

```bibtex
@inproceedings{dechtiar2025graphgrpolex,
  title     = {GRAPH-GRPO-LEX: Contract Graph Modeling and Reinforcement Learning with Group Relative Policy Optimization},
  author    = {Dechtiar, Moriya and Katz, Daniel Martin and Sundaresan, Mari and Jaume, Sylvain and Wang, Hongming},
  booktitle = {2025 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages     = {767--776},
  year      = {2025},
  doi       = {10.1109/ICDMW69685.2025.00092}
}
```

## Related repositories

| Repository | Paper | Contents |
|---|---|---|
| [ContractSmellsAutoLabelDetection](https://github.com/moriyadechtiar/ContractSmellsAutoLabelDetection) | *Machine Learning with Applications* 20 (2025) 100639 | Contract-smell labeling and detection |
| [ensemble-graph-eval](https://github.com/moriyadechtiar/ensemble-graph-eval) | *Procedia Computer Science* 275 (2026) 513–522 | ContractGraphEval benchmark and multi-judge evaluation |

## License

Code: MIT. See [`LICENSE`](LICENSE). The underlying contracts come from [CUAD](https://www.atticusprojectai.org/cuad) (The Atticus Project, CC BY 4.0).

## Contact

Moriya Dechtiar, moriya.dechtiar@alumni.harvard.edu. Issues and pull requests are welcome.
