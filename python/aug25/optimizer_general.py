from typing import Any, Dict, List, Tuple, Optional, Iterable
from common_interface import GraphProvider, RangeSet

class CompEdge:
    __slots__ = ("pop", "sid", "dest", "bv")
    def __init__(self, pop: int, sid: Optional[int], dest: int, bv: RangeSet):
        self.pop = pop
        self.sid = sid
        self.dest = dest
        self.bv = bv

class CompressedTrie:
    def __init__(self, roots: Dict[int,int], ends: List[bool], edges: List[List[CompEdge]]):
        self.roots = roots
        self.ends = ends
        self.edges = edges
        self.node_count = len(ends)
        self.edge_count = sum(len(e) for e in edges)

def optimize(provider: GraphProvider, roots_map: Dict[int, int]) -> CompressedTrie:
    """
    Optimizes a graph provided via a common interface using bisimulation-style quotienting.
    Note: This requires the provider to expose its raw arena for full graph inspection,
    which is a limitation of the per-token `iter_edges` interface for this kind of optimization.
    We assume the provider has an `.arena` attribute (like `Precompute2`).
    """
    if not hasattr(provider, 'arena'):
        print("Warning: optimizer_general.py requires a provider with an .arena attribute. Skipping optimization.")
        all_nodes = list(roots_map.values()) # A guess
        ends = [provider.is_end(u) for u in all_nodes]
        edges = [[] for _ in all_nodes]
        return CompressedTrie(roots_map, ends, edges)

    arena = provider.arena
    print("Optimizing graph using bisimulation-style quotient...")

    node_ids_sorted = sorted(int(i) for i in arena.keys())
    dense_of: Dict[int, int] = {old: idx for idx, old in enumerate(node_ids_sorted)}
    N: int = len(node_ids_sorted)

    ends: List[bool] = [False] * N
    raw_edges: List[List[Tuple[int, Optional[int], int, Tuple[Tuple[int, int], ...]]]] = [[] for _ in range(N)]

    for old_idx, node in arena.items():
        u = dense_of[int(old_idx)]
        ends[u] = bool((node.get("value", {}) or {}).get("end", False))
        for (p, s), dest_map in node.get("children", []):
            for v_old, edge_bv in dest_map:
                if int(v_old) not in dense_of: continue
                v = dense_of[int(v_old)]
                raw_edges[u].append((p, s, v, edge_bv.intervals))

    prev_class: List[int] = [1 if e else 0 for e in ends]

    def aggregate_edges_for_node(u: int, cls: List[int]) -> List[Tuple[int, Optional[int], int, Tuple[Tuple[int, int], ...]]]:
        aggr: Dict[Tuple[int, Optional[int], int], List[Tuple[int, int]]] = {}
        for (p, s, v, intervals) in raw_edges[u]:
            dcls = cls[v]
            key = (p, s, dcls)
            if key not in aggr: aggr[key] = []
            aggr[key].extend(intervals)

        items: List[Tuple[int, Optional[int], int, Tuple[Tuple[int, int], ...]]] = []
        for (p, s, dcls), ranges in aggr.items():
            merged = RangeSet._merge_unsorted(ranges)
            items.append((p, s, dcls, tuple(merged)))
        items.sort()
        return items

    for it in range(40):
        sig_to_id: Dict[Tuple[bool, Tuple], int] = {}
        new_class: List[int] = [0] * N
        next_id = 0
        changes = 0

        for u in range(N):
            items = aggregate_edges_for_node(u, prev_class)
            sig = (ends[u], tuple(items))
            cid = sig_to_id.get(sig)
            if cid is None:
                cid = next_id
                sig_to_id[sig] = cid
                next_id += 1
            new_class[u] = cid
            if new_class[u] != prev_class[u]:
                changes += 1

        prev_class = new_class
        if changes == 0:
            break

    num_classes = max(prev_class) + 1 if prev_class else 0
    q_ends: List[bool] = [False] * num_classes
    q_edges: List[List[CompEdge]] = [[] for _ in range(num_classes)]
    class_built: List[bool] = [False] * num_classes

    q_roots: Dict[int, int] = {}
    for sid, old_root in roots_map.items():
        if int(old_root) in dense_of:
            q_roots[int(sid)] = prev_class[dense_of[int(old_root)]]

    for u in range(N):
        cid = prev_class[u]
        if class_built[cid]:
            continue
        class_built[cid] = True
        q_ends[cid] = ends[u]

        items = aggregate_edges_for_node(u, prev_class)
        out: List[CompEdge] = []
        for (p, s, dcls, intervals) in items:
            rs = RangeSet(tuple(intervals))
            out.append(CompEdge(pop=p, sid=s, dest=dcls, bv=rs))
        q_edges[cid] = out

    return CompressedTrie(roots=q_roots, ends=q_ends, edges=q_edges)
