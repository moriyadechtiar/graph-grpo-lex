"""Command line:  python -m contract_linter GRAPH.json [options]

Exit status is 1 when the graph has error-level findings, otherwise 0.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys

from .report import lint
from .scores import reference_free_scores
from .slicing import context_slice


def main(argv=None) -> int:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)      # exit quietly when the output is piped to head or less
    ap = argparse.ArgumentParser(prog="contract_linter", description="Lint a contract graph.")
    ap.add_argument("graph", help="path to a graph JSON file ({'nodes': [...], 'edges': [...]})")
    ap.add_argument("--clause-level", action="store_true",
                    help="the graph covers a single clause; skip whole-contract checks")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--scores", action="store_true", help="also print reference-free scores in [0, 1]")
    ap.add_argument("--slice", metavar="CLAUSE", help="list the clauses to read together with CLAUSE")
    ap.add_argument("--direction", choices=("in", "out", "both"), default="both")
    ap.add_argument("--depth", type=int, default=1)
    args = ap.parse_args(argv)

    report = lint(args.graph, clause_level=args.clause_level)
    out = report.to_dict()
    if args.scores:
        out["reference_free_scores"] = reference_free_scores(report.graph.raw)
    if args.slice:
        out["slice"] = {"clause": args.slice, "direction": args.direction, "depth": args.depth,
                        "context": context_slice(report.graph, args.slice, args.direction, args.depth)}

    if args.json:
        print(json.dumps(out, indent=2, default=list))
    else:
        print(report.summary())
        if args.scores:
            print("Reference-free scores:")
            for name, value in out["reference_free_scores"].items():
                print(f"  {name:<24}{value:.3f}")
        if args.slice:
            print(f"Context for clause {args.slice} (direction={args.direction}, depth={args.depth}):")
            for clause, info in out["slice"]["context"].items():
                print(f"  {clause}  [{info['direction']}, distance {info['distance']}]  " + "; ".join(info["via"]))
            if not out["slice"]["context"]:
                print("  (none)")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
