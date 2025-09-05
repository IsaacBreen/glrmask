import json
from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict
from bisect import bisect_left, bisect_right
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
# tqdm is intentionally not used here to avoid overhead


def _intervals_key(rs: RangeSet) -> Tuple[Tuple[int, int], ...]:
    # Canonical immutable key for a RangeSet's intervals
    return tuple((int(a), int(b)) for a, b in (rs.intervals or []))


class _BitsetCache:
    """
    Interns ffi.Bitset objects keyed by a canonicalized tuple of intervals.
    Avoids reconstructing identical Bitsets repeatedly.
    """
    __slots__ = ("_cache",)

    def __init__(self):
        self._cache: Dict[Tuple[Tuple[int, int], ...], ffi.Bitset] = {}

    def get(self, intervals_key: Tuple[Tuple[int, int], ...]) -> Optional[ffi.Bitset]:
        if not intervals_key:
            return None
        bs = self._cache.get(intervals_key)
        if bs is None:
            bs = ffi.Bitset.from_ranges(list(intervals_key))
            self._cache[intervals_key] = bs
        return bs


class Model(GraphProvider):
    """
    Highly optimized precompute3 model.

    Key optimizations versus the baseline:
    - Remove debug prints and huge merge depths.
    - Normalize and store children once, preserving RangeSet for equivalence checks.
    - Build a compiled, per-node, per-pop structure grouping destinations by state-interval filters and LLM BVs.
    - Call gss_popn_collect at most once per (node, pop) step and reuse those results.
    - For each (pop, state-filter, llm-BV) group, merge parents once and prune once, then share the pruned child across all dest merges.
    - Cache Bitset conversion for LLM BVs.

    Why precompute2 seemed faster than the original precompute3:
    - precompute2 exploded state intervals to single SIDs at build time; runtime filtering was just equality checks.
    - precompute3 filtered peeks by scanning interval lists per destination (O(P*I*D)), created/pruned child GSS per destination,
      and even printed stats and used enormous merge depths.
    This version of precompute3 eliminates those overheads while keeping a compact graph representation.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        self.max_depth: Dict[int, int] = {}
        self._bv_cache = _BitsetCache()

        # Normalize BVs to RangeSet; store stateIDBV as list of (s,e). Record max_depth.
        # Additionally build a compiled per-pop view for fast get_mask().
        #
        # Keep a copy of the normalized children for iter_edges ("children_raw_p3") to ensure correctness checks work.
        # Compiled structure layout:
        #   node["_compiled_by_pop"] = {
        #       pop:int -> [
        #           (
        #             llm_bitset: ffi.Bitset|None,
        #             # Dests grouped by identical state filter (tuple of (a,b) intervals) or None for epsilon (all states)
        #             [ (state_key: Optional[Tuple[(int,int), ...]], dest_indices: List[int]), ... ]
        #           ),
        #           ...
        #       ]
        #   }
        for uid, node in self.arena.items():
            uid_int = int(uid)
            # Record node max_depth (if absent, assume 0)
            try:
                self.max_depth[uid_int] = int(node.get("max_depth", 0))
            except Exception:
                self.max_depth[uid_int] = 0

            ch = node.get("children") or []
            normalized_children: List[Tuple[Tuple[int, RangeSet], List[Tuple[int, List[Tuple[int, int]]]]]] = []
            compiled_by_pop: Dict[int, Dict[Tuple[Tuple[int, int], ...], Dict[Optional[Tuple[Tuple[int, int], ...]], List[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            # Normalize JSON and build compiled groupings
            for edge_key, dest_map in ch:
                pop, llm_bv_json = edge_key
                pop = int(pop)
                llm_rs = RangeSet.from_json(llm_bv_json)
                # Normalize dests
                newdm: List[Tuple[int, List[Tuple[int, int]]]] = []
                for dest_idx, state_bv in dest_map:
                    dest_idx = int(dest_idx)
                    state_ranges = [(int(a), int(b)) for a, b in (state_bv or [])]
                    newdm.append((dest_idx, state_ranges))
                normalized_children.append(((pop, llm_rs), newdm))

                # Build compiled groups:
                # group by pop -> llm_key -> state_key (None means epsilon/all) -> list of dests
                llm_key = _intervals_key(llm_rs)
                for dest_idx, state_ranges in newdm:
                    state_key: Optional[Tuple[Tuple[int, int], ...]]
                    if not state_ranges:
                        state_key = None
                    else:
                        state_key = tuple(state_ranges)
                    compiled_by_pop[pop][llm_key][state_key].append(dest_idx)

            # Store normalized (for iter_edges correctness) and compiled structures (for fast get_mask)
            node["children_raw_p3"] = normalized_children
            # Freeze compiled structure to lists of tuples for fast iteration
            frozen_compiled: Dict[int, List[Tuple[Optional[ffi.Bitset], List[Tuple[Optional[Tuple[Tuple[int, int], ...]], List[int]]]]]] = {}
            for pop, llm_groups in compiled_by_pop.items():
                groups_list = []
                for llm_key, state_map in llm_groups.items():
                    bitset = self._bv_cache.get(llm_key)
                    # list of (state_key, dest_indices)
                    state_items = [(sk, list(dests)) for sk, dests in state_map.items()]
                    groups_list.append((bitset, state_items))
                frozen_compiled[int(pop)] = groups_list
            node["_compiled_by_pop"] = frozen_compiled

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data['precomputed3']
        arena_json = data['trie3_god']
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        return Model(roots_map, arena)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        # For equivalence checking, we must "explode" the state_bv into individual
        # state IDs to match the GraphProvider interface expected by the checker.
        # This is not used by the performance-critical get_mask() method.
        for (pop, llm_rs), dests in self.arena.get(node, {}).get("children_raw_p3") or []:
            if llm_rs.contains(token):
                for dest_idx, state_bv_ranges in dests:
                    if not state_bv_ranges:  # Epsilon transition on GSS stack
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv_ranges:
                            a = int(start)
                            b = int(end)
                            if b < a:
                                a, b = b, a
                            for sid in range(a, b + 1):
                                yield (int(pop), sid, int(dest_idx))

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        # Local bindings for speed
        roots_map = self.roots_map
        max_depth = self.max_depth
        arena = self.arena
        gss_merge = ffi.gss_merge_many_with_depth
        gss_popn_collect = ffi.gss_popn_collect
        gss_allow = ffi.gss_allow_only_llm_tokens_and_prune

        final_mask = ffi.Bitset.zeros()
        values: Dict[int, ffi.GSSNode] = {}
        stopped: set[int] = set()
        # depth -> set(node_idx); and a heap-like list of current depths
        todo: Dict[int, set[int]] = defaultdict(set)
        depths_heap: List[int] = []

        def enqueue(d: int, node_idx: int):
            bucket = todo[d]
            if not bucket:
                depths_heap.append(d)
            bucket.add(node_idx)

        def pop_min_bucket() -> Tuple[int, set[int]]:
            idx = depths_heap.index(min(depths_heap))
            d = depths_heap.pop(idx)
            nodes = todo.pop(d)
            return d, nodes

        # Seed
        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            r = int(root_idx)
            if r in values:
                merged = gss_merge([values[r], gss.clone_node()], 1)
                if merged.ptr() != values[r].ptr():
                    values[r] = merged
            else:
                values[r] = gss.clone_node()
            enqueue(max_depth.get(r, 0), r)

        while todo:
            _, node_indices = pop_min_bucket()

            for node_idx in list(node_indices):
                if node_idx in stopped:
                    continue

                agg = values.pop(node_idx, None)
                if agg is None:
                    continue

                if self.is_end(node_idx):
                    final_mask = final_mask.union(agg.allowed_llm_tokens())

                if not agg.is_ok():
                    stopped.add(node_idx)
                    continue

                node = arena.get(node_idx, {})
                compiled_by_pop = node.get("_compiled_by_pop") or {}
                if not compiled_by_pop:
                    continue

                # For each pop-group, collect peeks once and reuse
                for pop, llm_groups in compiled_by_pop.items():
                    peeks = gss_popn_collect(agg, int(pop))
                    if not peeks:
                        continue

                    # Build "all parents" and index peeks by sid
                    # Note: multiple parents may share the same sid; we keep all of them.
                    all_parents = [p for _, p in peeks]
                    if not all_parents:
                        continue
                    sid_to_parents: DefaultDict[int, List[ffi.GSSNode]] = defaultdict(list)
                    for sid_val, parent_node in peeks:
                        sid_to_parents[int(sid_val)].append(parent_node)
                    # Sorted list of unique sids to quickly collect by interval via bisect
                    unique_sids = sorted(sid_to_parents.keys())

                    # Helper: collect parents whose sids lie in any of the intervals in state_key
                    def collect_by_state_key(state_key: Optional[Tuple[Tuple[int, int], ...]]) -> List[ffi.GSSNode]:
                        if state_key is None:
                            return all_parents
                        # For each interval, find the slice in unique_sids and collect
                        matched: List[ffi.GSSNode] = []
                        for a, b in state_key:
                            if b < a:
                                a, b = b, a
                            lo = bisect_left(unique_sids, a)
                            hi = bisect_right(unique_sids, b)
                            for idx in range(lo, hi):
                                sid = unique_sids[idx]
                                matched.extend(sid_to_parents[sid])
                        return matched

                    # Process each llm-group once: for each state_key-group, merge+prune once, then reuse across all dests
                    for llm_bitset, state_items in llm_groups:
                        for state_key, dests in state_items:
                            if not dests:
                                continue
                            matched = collect_by_state_key(state_key)
                            if not matched:
                                continue

                            child_gss = gss_merge(matched, 1)
                            if not child_gss.is_ok():
                                continue

                            pruned_child = child_gss
                            if llm_bitset is not None:
                                pruned_child = child_gss.clone_node()
                                gss_allow(pruned_child, llm_bitset)
                                if not pruned_child.is_ok():
                                    continue

                            # Merge the pruned child into all destinations
                            for d in dests:
                                d = int(d)
                                if d in values:
                                    combined = gss_merge([values[d], pruned_child], 1)
                                    if combined.ptr() == values[d].ptr():
                                        continue
                                    values[d] = combined
                                else:
                                    values[d] = pruned_child
                                enqueue(self.max_depth.get(d, 0), d)

        return final_mask
