#!/usr/bin/env python3
"""Benchmark graph coloring algorithms on exported JSON graphs."""
import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple


def greedy_coloring(adj: List[List[int]]) -> List[int]:
    n = len(adj)
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    colors = [-1] * n
    for u in order:
        used = {colors[v] for v in adj[u] if colors[v] != -1}
        c = 0
        while c in used:
            c += 1
        colors[u] = c
    return colors


def dsatur_coloring(adj: List[List[int]]) -> List[int]:
    n = len(adj)
    colors = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    degrees = [len(adj[i]) for i in range(n)]

    for _ in range(n):
        # pick uncolored vertex with max saturation, tie-break by degree
        best = None
        for i in range(n):
            if colors[i] != -1:
                continue
            sat = len(neighbor_colors[i])
            if best is None:
                best = i
            else:
                best_sat = len(neighbor_colors[best])
                if sat > best_sat or (sat == best_sat and degrees[i] > degrees[best]):
                    best = i
        if best is None:
            break
        used = neighbor_colors[best]
        c = 0
        while c in used:
            c += 1
        colors[best] = c
        for v in adj[best]:
            if colors[v] == -1:
                neighbor_colors[v].add(c)
    return colors


def _sat_coloring_impl(adj: List[List[int]], k: int) -> Optional[List[int]]:
    try:
        from pysat.formula import CNF
        from pysat.solvers import Solver
    except Exception:
        return None

    n = len(adj)

    def var(v: int, c: int) -> int:
        return v * k + c + 1

    cnf = CNF()
    # at least one color per vertex
    for v in range(n):
        cnf.append([var(v, c) for c in range(k)])
        # at most one color
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                cnf.append([-var(v, c1), -var(v, c2)])

    # edge constraints
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for c in range(k):
                cnf.append([-var(u, c), -var(v, c)])

    with Solver(name="glucose3", bootstrap_with=cnf) as solver:
        if not solver.solve():
            return None
        model = solver.get_model()

    colors = [-1] * n
    for v in range(n):
        for c in range(k):
            lit = var(v, c)
            if lit in model:
                colors[v] = c
                break
    return colors


def sat_coloring(adj: List[List[int]], max_k: Optional[int] = None, max_vertices: int = 200) -> Optional[List[int]]:
    n = len(adj)
    if n == 0:
        return []
    if n > max_vertices:
        return None
    greedy_colors = greedy_coloring(adj)
    ub = max_k or (max(greedy_colors) + 1)
    for k in range(1, ub + 1):
        colors = _sat_coloring_impl(adj, k)
        if colors is not None:
            return colors
    return None


def measure(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    elapsed = time.perf_counter() - start
    return result, elapsed


def color_count(colors: Optional[List[int]]) -> Optional[int]:
    if colors is None:
        return None
    if not colors:
        return 0
    return max(colors) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark coloring algorithms on exported graphs")
    parser.add_argument("--input", default="coloring_graphs", help="Directory with JSON graph exports")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of graphs processed")
    parser.add_argument("--sat-max-vertices", type=int, default=200, help="Skip SAT for graphs larger than this")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return 1

    files = sorted(input_dir.glob("*.json"))
    if args.limit is not None:
        files = files[: args.limit]

    rows = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        adj = data.get("adjacency_list") or []
        graph_id = data.get("id", path.stem)
        dwa_type = data.get("dwa_type", "unknown")
        height = data.get("height", -1)
        n = data.get("num_vertices", len(adj))

        greedy_colors, greedy_time = measure(greedy_coloring, adj)
        dsatur_colors, dsatur_time = measure(dsatur_coloring, adj)
        sat_colors, sat_time = measure(sat_coloring, adj, None, args.sat_max_vertices)

        rows.append(
            {
                "id": graph_id,
                "dwa_type": dwa_type,
                "height": height,
                "n": n,
                "greedy_k": color_count(greedy_colors),
                "greedy_ms": greedy_time * 1000,
                "dsatur_k": color_count(dsatur_colors),
                "dsatur_ms": dsatur_time * 1000,
                "sat_k": color_count(sat_colors),
                "sat_ms": sat_time * 1000 if sat_colors is not None else None,
            }
        )

    header = (
        "id", "dwa_type", "height", "n",
        "greedy_k", "greedy_ms",
        "dsatur_k", "dsatur_ms",
        "sat_k", "sat_ms",
    )
    print("\t".join(header))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["id"]),
                    str(row["dwa_type"]),
                    str(row["height"]),
                    str(row["n"]),
                    str(row["greedy_k"]),
                    f"{row['greedy_ms']:.3f}",
                    str(row["dsatur_k"]),
                    f"{row['dsatur_ms']:.3f}",
                    str(row["sat_k"]) if row["sat_k"] is not None else "-",
                    f"{row['sat_ms']:.3f}" if row["sat_ms"] is not None else "-",
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
