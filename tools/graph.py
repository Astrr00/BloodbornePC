# SPDX-License-Identifier: GPL-3.0-or-later
"""Rebuild the local code knowledge graph (graphify-out/, gitignored) for src/, tests/ and tools/.

Structural (AST) extraction only: no LLM and no API key. Generated recompiler output (build*/) is never scanned.
Usage: python tools/graph.py   then   graphify query "<question>"
"""
import json
from pathlib import Path

from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.detect import detect
from graphify.export import to_json
from graphify.extract import collect_files, extract
from graphify.report import generate


def main():
    root = Path(__file__).resolve().parent.parent
    out = root / "graphify-out"
    out.mkdir(exist_ok=True)
    files = [f for d in ("src", "tests", "tools") for f in collect_files(root / d)]
    result = extract(files, cache_root=root, parallel=False)  # parallel=False: no multiprocessing guard needed on Windows
    result.setdefault("hyperedges", [])
    g = build_from_json(result, root=str(root), directed=True)
    comm = cluster(g)
    # deterministic community names: dominant file + the best-connected node
    labels = {}
    for cid, members in comm.items():
        nodes = [g.nodes[n] for n in members]
        top = max(members, key=lambda n: g.degree(n))
        names = [Path(n.get("source_file") or "?").name for n in nodes]
        labels[cid] = f"{max(set(names), key=names.count)}: {str(g.nodes[top].get('label', top)).strip('()')[:28]}"
    if not to_json(g, comm, str(out / "graph.json"), community_labels=labels, force=True):
        raise SystemExit("graph.json not written")
    q = suggest_questions(g, comm, labels)
    report = generate(g, comm, score_all(g, comm), labels, god_nodes(g), surprising_connections(g, comm), detect(root / "src"),
                      {"input": 0, "output": 0}, str(root), suggested_questions=q)
    (out / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, {len(comm)} communities -> {out}")


if __name__ == "__main__":
    main()
