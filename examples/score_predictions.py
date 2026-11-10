#!/usr/bin/env python3
"""Score a file of predicted graphs with the reference-free checks.

    python examples/score_predictions.py predictions.jsonl
    python examples/score_predictions.py ContractGraphEval_v1.1.jsonl --graph prediction.graph

Each line of the input is a JSON object. ``--graph`` is the dotted path to the
graph inside it (default: the object itself, or ``prediction.graph`` when
present). Prints the mean of each check and the samples with the lowest scores.
"""
import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contract_linter import reference_free_scores  # noqa: E402


def read(record, dotted):
    value = record
    for key in dotted.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def main():
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)      # exit quietly when the output is piped to head or less
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="JSON Lines file, one object per line")
    ap.add_argument("--graph", help="dotted path to the graph inside each object")
    ap.add_argument("--worst", type=int, default=5, help="how many low-scoring samples to list")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    results = []
    for i, row in enumerate(rows):
        if args.graph:
            graph = read(row, args.graph)
        else:
            graph = read(row, "prediction.graph") if isinstance(row.get("prediction"), dict) else row
        scores = reference_free_scores(graph)
        results.append((sum(scores.values()) / len(scores), row.get("sample_id") or row.get("id") or i, scores))

    names = list(results[0][2])
    print(f"{len(results)} graphs scored\n")
    print(f"{'check':<24}{'mean':>8}")
    for name in names:
        print(f"{name:<24}{sum(r[2][name] for r in results) / len(results):>8.3f}")
    print(f"\nLowest-scoring {min(args.worst, len(results))}:")
    for mean, ident, scores in sorted(results, key=lambda r: r[0])[: args.worst]:
        low = min(scores, key=scores.get)
        print(f"  {mean:.3f}  weakest: {low} ({scores[low]:.2f})  {str(ident)[-60:]}")


if __name__ == "__main__":
    main()
