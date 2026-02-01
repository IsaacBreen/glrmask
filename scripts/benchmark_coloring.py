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


def rlf_coloring(adj: List[List[int]]) -> List[int]:
    """
    Recursive Largest First graph coloring.
    1. Select uncolored vertex with max degree as seed
    2. Greedily add vertices to current color class that:
       - Are not adjacent to any in current class
       - Among candidates, prefer those with most colored neighbors
    3. Repeat until all colored
    """
    n = len(adj)
    colors = [-1] * n
    neighbors = [set(adj[i]) for i in range(n)]
    remaining_degree = [len(adj[i]) for i in range(n)]
    colored_neighbor_count = [0] * n
    current_color = 0
    remaining = set(range(n))

    while remaining:
        # Start new color class with highest-degree uncolored vertex
        seed = max(remaining, key=lambda v: remaining_degree[v])
        color_class = {seed}
        colors[seed] = current_color
        remaining.remove(seed)
        for u in adj[seed]:
            if colors[u] == -1:
                colored_neighbor_count[u] += 1
                remaining_degree[u] -= 1

        # Greedily add compatible vertices
        candidates = remaining - neighbors[seed]
        while candidates:
            # Pick candidate with most already-colored neighbors (max saturation)
            best = max(candidates, key=lambda v: colored_neighbor_count[v])
            color_class.add(best)
            colors[best] = current_color
            remaining.remove(best)
            for u in adj[best]:
                if colors[u] == -1:
                    colored_neighbor_count[u] += 1
                    remaining_degree[u] -= 1
            candidates -= neighbors[best]
            candidates.discard(best)

        current_color += 1

    return colors


def validate_coloring(adj: List[List[int]], colors: List[int]) -> bool:
    for u, neighbors in enumerate(adj):
        for v in neighbors:
            if colors[u] == colors[v]:
                raise ValueError(f"Invalid: {u} and {v} adjacent with same color {colors[u]}")
    return True


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

    files = sorted(input_dir.rglob("*.json"))
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
        rlf_colors, rlf_time = measure(rlf_coloring, adj)
        sat_colors, sat_time = measure(sat_coloring, adj, None, args.sat_max_vertices)

        validate_coloring(adj, greedy_colors)
        validate_coloring(adj, dsatur_colors)
        validate_coloring(adj, rlf_colors)
        if sat_colors is not None:
            validate_coloring(adj, sat_colors)

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
                "rlf_k": color_count(rlf_colors),
                "rlf_ms": rlf_time * 1000,
                "sat_k": color_count(sat_colors),
                "sat_ms": sat_time * 1000 if sat_colors is not None else None,
            }
        )

    header = (
        "id", "dwa_type", "height", "n",
        "greedy_k", "greedy_ms",
        "dsatur_k", "dsatur_ms",
        "rlf_k", "rlf_ms",
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
                    str(row["rlf_k"]),
                    f"{row['rlf_ms']:.3f}",
                    str(row["sat_k"]) if row["sat_k"] is not None else "-",
                    f"{row['sat_ms']:.3f}" if row["sat_ms"] is not None else "-",
                ]
            )
        )

    # Summary by DWA type
    from collections import defaultdict

    by_type = defaultdict(list)
    for row in rows:
        by_type[row["dwa_type"]].append(row)

    print("\n# Summary by DWA type")
    print("dwa_type\tcount\tavg_n\tavg_greedy_ms\tavg_dsatur_ms\tavg_rlf_ms")
    for dtype in sorted(by_type.keys()):
        group = by_type[dtype]
        count = len(group)
        avg_n = sum(r["n"] for r in group) / count
        avg_greedy = sum(r["greedy_ms"] for r in group) / count
        avg_dsatur = sum(r["dsatur_ms"] for r in group) / count
        avg_rlf = sum(r["rlf_ms"] for r in group) / count
        print(
            f"{dtype}\t{count}\t{avg_n:.1f}\t{avg_greedy:.3f}\t{avg_dsatur:.3f}\t{avg_rlf:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
