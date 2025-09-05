import json
from typing import Dict, List, Tuple, Optional, DefaultDict, Iterable
from collections import defaultdict
from bisect import bisect_left, bisect_right
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
# tqdm is intentionally not used (to avoid overhead)


def _intervals_key(rs: RangeSet) -> Tuple[Tuple[int, int], ...]:
    # Canonical immutable key for a RangeSet's intervals
    return tuple((int(a), int(b)) for a, b in (rs.intervals or []))


class _BitsetCache:
    """
    Interns ffi.Bitset objects keyed by a canonicalized tuple of intervals.
    This avoids repeatedly constructing identical Bitsets for the same BV.
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
    Optimized precompute2-style model that:
    - Converts the precompute3 graph to a precompute2-like structure (edges keyed by (pop, sid_opt)).
    - Pre-aggregates destination edges and unions LLM token bit-vectors at build time.
    - Builds a per-node, per-pop compiled view to minimize repeated GSS operations during get_mask().
    - Caches Bitset conversions of LLM token RangeSets.
    - Avoids repeated gss_popn_collect calls for the same pop in one step and shares pruned child GSS across multiple destinations.

    Why precompute2 was much faster than precompute3 before:
    - precompute3 filtered peeks by scanning state-interval lists per destination (O(P*I*D)), where P=peeks, I=intervals, D=destinations,
      and created/pruned child GSS per destination.
    - precompute2 exploded state intervals into single SIDs up front and then only matched peeks by equality (O(P) per (pop, sid) key),
      plus it aggregated LLM bit-vectors per destination, reducing runtime overhead.
    - precompute3 also had expensive debug prints, used very large merge depths, and repeated Bitset construction.
    This implementation keeps the p2 data model but further optimizes runtime by grouping work per-pop and per-LLM BV.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        self.roots_map = dict((int(s), int(r)) for s, r in roots_map)
        self.arena = arena
        self.max_state_id: int = int(max_state_id)
        self.max_depth: Dict[int, int] = {}
        self._bv_cache = _BitsetCache()

        # Convert precompute3 graph structure to precompute2-like structure, but do it correctly and efficiently.
        #
        # Target normalized layout stored for compatibility (used by iter_edges equivalence checks):
        #   node["children_raw_p2"] = [((pop:int, sid_opt:Optional[int]), [(dest:int, llm_rs:RangeSet), ...]), ...]
        #
        # Additionally, build a compiled structure for fast get_mask():
        #   node["_compiled_by_pop"] = {
        #       pop:int -> {
        #           # sid_opt=None and sid_opt=int have separate groups
        #           None: { llm_key: {"bitset": ffi.Bitset|None, "dests": [int, ...]} },
        #           sid:int: { llm_key: {"bitset": ffi.Bitset|None, "dests": [int, ...]} },
        #       }
        #   }
        #
        # The compiled structure groups destinations by identical LLM token BVs per (pop, sid_opt).
        # At runtime we:
        #   - call gss_popn_collect once per pop,
        #   - build matched parents list once per sid_opt (either all peeks or filtered by equality),
        #   - prune once per LLM group (reuse for all dests in that group).
        for uid, n in self.arena.items():
            uid = int(uid)

            # Record node max_depth (if absent, assume 0)
            try:
                self.max_depth[uid] = int(n.get("max_depth", 0))
            except Exception:
                self.max_depth[uid] = 0

            p3_children = n.get("children") or []

            # Aggregate into precompute2 format: (pop, sid_opt) -> {dest -> RangeSet(llm tokens)}
            p2_children_agg: DefaultDict[Tuple[int, Optional[int]], Dict[int, RangeSet]] = defaultdict(dict)

            for edge_key, dest_map in p3_children:
                pop, llm_bv_json = edge_key
                pop = int(pop)
                llm_rs = RangeSet.from_json(llm_bv_json)
                if llm_rs.is_empty():
                    # No LLM tokens allowed along this edge => no-op at runtime
                    continue

                for dest_idx, state_bv_ranges in dest_map:
                    dest_idx = int(dest_idx)
                    ranges = state_bv_ranges or []
                    if not ranges:
                        # Epsilon on tokenizer-state (accept any sid)
                        key = (pop, None)
                        cur = p2_children_agg[key].get(dest_idx)
                        p2_children_agg[key][dest_idx] = llm_rs if cur is None else cur.union(llm_rs)
                    else:
                        # Expand (start,end) inclusive range into individual SIDs
                        # Note: This can be large, but it is precisely why precompute2 was faster:
                        # it avoids interval checks during runtime by pre-splitting.
                        for start, end in ranges:
                            a = int(start)
                            b = int(end)
                            if b < a:
                                a, b = b, a
                            # enumerate sids in [a,b]
                            for sid in range(a, b + 1):
                                key = (pop, sid)
                                cur = p2_children_agg[key].get(dest_idx)
                                p2_children_agg[key][dest_idx] = llm_rs if cur is None else cur.union(llm_rs)

            # Convert aggregated map to final list format for iter_edges and compile it for get_mask.
            children_raw_p2: List[Tuple[Tuple[int, Optional[int]], List[Tuple[int, RangeSet]]]] = []
            compiled_by_pop: Dict[int, Dict[Optional[int], Dict[Tuple[Tuple[int, int], ...], Dict[str, object]]]] = defaultdict(lambda: defaultdict(dict))

            for (pop, sid_opt), dests_map in p2_children_agg.items():
                # Store raw for iter_edges
                raw_list = list(dests_map.items())  # [(dest, RangeSet), ...]
                children_raw_p2.append(((pop, sid_opt), raw_list))

                # Compile groups per LLM bv
                group = compiled_by_pop[pop][sid_opt]
                for d, llm_rs in dests_map.items():
                    llm_key = _intervals_key(llm_rs)
                    entry = group.get(llm_key)
                    if entry is None:
                        entry = {"bitset": self._bv_cache.get(llm_key), "dests": []}
                        group[llm_key] = entry
                    entry["dests"].append(int(d))

            n["children_raw_p2"] = children_raw_p2
            # Freeze compiled structure to regular dicts for faster lookups
            n["_compiled_by_pop"] = {int(k): {sidk: {lk: {"bitset": v["bitset"], "dests": list(v["dests"])}
                                                     for lk, v in sidv.items()}
                                              for sidk, sidv in v.items()}
                                     for k, v in compiled_by_pop.items()}

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        # This model uses the precompute3 graph as input and compiles it to precompute2-like.
        roots_map = data['precomputed3']
        arena_json = data['trie3_god']
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        max_state_id = int(max(dict(data['parser']['stage_7_table']).keys()))
        return Model(roots_map, arena, max_state_id)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        # Use the raw precompute2-style children for equivalence checking.
        for (pop, sid_opt), dests in self.arena.get(node, {}).get("children_raw_p2") or []:
            for dest, rs in dests:
                if rs.contains(token):
                    yield (int(pop), sid_opt, int(dest))

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        # Local bindings for speed
        roots_map = self.roots_map
        max_depth = self.max_depth
        arena = self.arena
        bv_cache_get = self._bv_cache.get
        gss_merge = ffi.gss_merge_many_with_depth
        gss_popn_collect = ffi.gss_popn_collect
        gss_allow = ffi.gss_allow_only_llm_tokens_and_prune

        final_mask = ffi.Bitset.zeros()
        values: Dict[int, ffi.GSSNode] = {}
        stopped: set[int] = set()
        # depth -> set(node_idx); we also maintain a min-heap of depths
        todo: Dict[int, set[int]] = defaultdict(set)
        depths_heap: List[int] = []

        def enqueue(d: int, node_idx: int):
            bucket = todo[d]
            if not bucket:
                # First time this depth is active; add to heap
                depths_heap.append(d)
            bucket.add(node_idx)

        def pop_min_bucket() -> Tuple[int, set[int]]:
            # Find min depth without O(n) scanning
            idx = depths_heap.index(min(depths_heap))
            d = depths_heap.pop(idx)
            nodes = todo.pop(d)
            return d, nodes

        # Seed: map tokenizer state -> corresponding trie root and merge clones into values
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

        # Main loop
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

                # Process edges grouped by 'pop' so we only pop once per pop value
                for pop, sid_groups in compiled_by_pop.items():
                    peeks = gss_popn_collect(agg, int(pop))
                    if not peeks:
                        continue

                    # Build 'all' parents and mapping sid -> [parents]
                    all_parents = [p for _, p in peeks]
                    if not all_parents:
                        continue
                    sid_to_parents: DefaultDict[int, List[ffi.GSSNode]] = defaultdict(list)
                    for sid_val, parent_node in peeks:
                        sid_to_parents[int(sid_val)].append(parent_node)

                    # Helper to get matched parents for this sid group
                    def matched_for_sid_opt(sid_opt_val: Optional[int]) -> List[ffi.GSSNode]:
                        if sid_opt_val is None:
                            return all_parents
                        lst = sid_to_parents.get(int(sid_opt_val))
                        return lst if lst else []

                    # For each sid group, prune once per distinct LLM BV and reuse across all its dests
                    for sid_opt, llm_groups in sid_groups.items():
                        matched = matched_for_sid_opt(sid_opt)
                        if not matched:
                            continue

                        # Merge matched parents once per sid group; we will clone+prune per-LLM below
                        merged_child = gss_merge(matched, 1)
                        if not merged_child.is_ok():
                            continue

                        for llm_key, entry in llm_groups.items():
                            dests: List[int] = entry["dests"]
                            if not dests:
                                continue

                            # Prepare pruned child for this LLM BV
                            child_gss_for_llm = merged_child
                            llm_bitset = entry["bitset"]
                            if llm_bitset is not None:
                                child_gss_for_llm = merged_child.clone_node()
                                gss_allow(child_gss_for_llm, llm_bitset)
                                if not child_gss_for_llm.is_ok():
                                    continue

                            # Share the pruned child across all dest merges (merge doesn't mutate inputs)
                            for d in dests:
                                d = int(d)
                                if d in values:
                                    combined = gss_merge([values[d], child_gss_for_llm], 1)
                                    if combined.ptr() == values[d].ptr():
                                        continue
                                    values[d] = combined
                                else:
                                    values[d] = child_gss_for_llm
                                enqueue(max_depth.get(d, 0), d)

        return final_mask
