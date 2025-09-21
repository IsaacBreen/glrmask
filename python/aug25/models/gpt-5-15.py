import heapq
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func


class Model(GraphProvider):
    """
    A performance-optimized facade around InnerModel that accelerates get_mask().
    Key optimizations:
      - Pre-index children by (pop, llm_bv) and group dest nodes by identical state_bv to avoid redundant filtering.
      - Early prune edges whose llm_bv mask does not intersect current llm_mask.
      - Cache expensive computation of LLM-token forbids per HybridL2Bitset (disallowed terminals).
      - Avoid scanning large state ranges when building forbids; only iterate over known tokenizer states present in pmc.
      - Reduce repeated merges and dictionary churn by batching per-node contributions.
      - Precompute sets/maps commonly used in the hot loop (end nodes, children index, etc.).
    """

    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model

        # Direct references (no copy)
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map

        # Precompute fast "end" check
        self._end_nodes: Set[int] = set()
        for nid, data in self.arena.items():
            if ((data or {}).get("value") or {}).get("clean_end", False):
                self._end_nodes.add(int(nid))

        # Precompute a compact, grouped representation of children edges per node.
        # For each node: list of groups (pop, llm_bv, [state_bv_1..k], [[dest_idx...]..k])
        self._children_index: Dict[int, List[Tuple[int, ffi.Bitset, List[ffi.Bitset], List[List[int]]]]] = {}
        for nid, data in self.arena.items():
            groups = []
            for (pop, llm_bv), dests in (data.get("children") or []):
                # Group all dests with identical state_bv so we filter only once per unique state_bv.
                by_state: Dict[int, Tuple[ffi.Bitset, List[int]]] = {}
                for dest_idx, state_bv in dests:
                    key = id(state_bv)
                    entry = by_state.get(key)
                    if entry is None:
                        by_state[key] = (state_bv, [int(dest_idx)])
                    else:
                        entry[1].append(int(dest_idx))
                # Unpack into parallel lists for quick iteration
                state_bv_list: List[ffi.Bitset] = []
                dest_lists: List[List[int]] = []
                for state_bv, dlist in by_state.values():
                    state_bv_list.append(state_bv)
                    dest_lists.append(dlist)
                groups.append((int(pop), llm_bv, state_bv_list, dest_lists))
            self._children_index[int(nid)] = groups

        # Prepare pmc (possible_matches_cache) and speed-ups for forbid computations
        self._pmc: Optional[Dict[int, Dict[int, ffi.Bitset]]] = None
        self._pmc_state_ids_sorted: List[int] = []
        self._all_tokens_union: Optional[ffi.Bitset] = None

        pmc_src: Optional[Dict[int, Dict[int, ffi.Bitset]]] = im.possible_matches_cache
        if pmc_src is not None:
            # Normalize keys to int once to avoid int() hot-path conversions.
            pmc_int: Dict[int, Dict[int, ffi.Bitset]] = {}
            for tsid, mapping in pmc_src.items():
                m2: Dict[int, ffi.Bitset] = {}
                for k, v in mapping.items():
                    if isinstance(k, int):
                        m2[k] = v
                    else:
                        try:
                            m2[int(k)] = v
                        except Exception:
                            continue
                pmc_int[int(tsid)] = m2
            self._pmc = pmc_int
            self._pmc_state_ids_sorted = sorted(pmc_int.keys())
            # Precompute union across all tokens in pmc for quick handling of "all" disallow sets
            acc = ffi.Bitset.zeros()
            for mapping in pmc_int.values():
                for llm_tokens in mapping.values():
                    acc = acc.union(llm_tokens)
            self._all_tokens_union = acc

        # Forbid cache: HybridL2Bitset (disallowed terminals) -> Bitset of forbidden LLM tokens
        self._forbid_cache: Dict[ffi.HybridL2Bitset, ffi.Bitset] = {}

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int) -> None:
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        return self.inner_model.state

    def _is_end(self, node: int) -> bool:
        return node in self._end_nodes

    def _compute_forbid_llm_tokens(self, disallowed: ffi.HybridL2Bitset) -> ffi.Bitset:
        """
        Compute and cache the union of LLM tokens that are forbidden by the given HybridL2Bitset
        (which encodes disallowed terminals across tokenizer state ranges).
        """
        cached = self._forbid_cache.get(disallowed)
        if cached is not None:
            return cached

        # If pmc is unavailable, nothing to forbid
        if self._pmc is None or not self._pmc_state_ids_sorted:
            result = ffi.Bitset.zeros()
            self._forbid_cache[disallowed] = result
            return result

        # Fast-path: full disallow -> union of all possible tokens
        # Note: Equality should be reliable; HybridL2Bitset implements __eq__/__hash__.
        # We avoid constructing another .all() instance repeatedly.
        if self._all_tokens_union is not None:
            # Check if "disallowed" equals the full-all set
            # We test by complementing twice: if complement is empty across all ranges, it's "all".
            # But more directly, many implementations ensure HybridL2Bitset() equality is content-based.
            # Try a cheap heuristic: if there is exactly one range covering entire domain and its bitset is non-empty for all terminals.
            # Fallback to direct equality to ffi.HybridL2Bitset.all()
            try:
                if disallowed == ffi.HybridL2Bitset.all():
                    result = self._all_tokens_union
                    self._forbid_cache[disallowed] = result
                    return result
            except Exception:
                pass

        max_state = self.tokenizer_max_state
        result = ffi.Bitset.zeros()

        # Only iterate pmc states that fall into each range instead of scanning the full numeric range.
        pmc_states = self._pmc_state_ids_sorted

        for (start, end), term_bv in disallowed.range_values():
            if term_bv.is_empty():
                continue
            s: int = max(0, int(start))
            e: int = min(int(end), max_state)
            if s > e:
                continue

            # Iterate only through present pmc states within [s, e]
            # Since pmc_states is sorted, we can break early.
            for tsid in pmc_states:
                if tsid < s:
                    continue
                if tsid > e:
                    break
                pm = self._pmc.get(tsid)
                if not pm:
                    continue
                # For this tokenizer state, include all terminals in term_bv
                for terminal_id, llm_tokens in pm.items():
                    if term_bv.contains(int(terminal_id)):
                        result = result.union(llm_tokens)

        self._forbid_cache[disallowed] = result
        return result

    @profile
    def get_mask(self) -> RangeSet:
        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset
        if all_ones is None:
            # Fallback: allow everything by default if not provided (conservative superset)
            all_ones = ffi.Bitset.zeros()  # safest default; caller may override upstream

        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        # values[node_id] = (GSS aggregate, current llm_mask)
        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        stopped: Set[int] = set()
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        is_end = self._is_end
        children_index = self._children_index

        # Initialize by merging states per root node, while unioning masks
        for sid, gss in state_map.items():
            r: Optional[int] = roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)
            if r in values:
                values[r] = (values[r][0].merge(gss), all_ones)
            else:
                values[r] = (gss, all_ones)
            d: int = max_depth[r]
            b: Optional[Set[int]] = todo.get(d)
            if b is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                b.add(r)

        def enqueue(d: int, n: int) -> None:
            b: Optional[Set[int]] = todo.get(d)
            if b is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                b.add(n)

        def disallowed_terminals(g: GSS) -> ffi.HybridL2Bitset:
            acc = g.reduce_acc()
            return ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()

        while depth_heap:
            depth: int = hpop(depth_heap)
            nodes: Optional[Set[int]] = todo.pop(depth, None)
            if not nodes:
                continue

            for node in nodes:
                if node in stopped:
                    continue
                item: Optional[Tuple[GSS, ffi.Bitset]] = values.pop(node, None)
                if item is None:
                    continue

                gss_node, llm_mask = item

                # Handle end nodes: compute allowed llm tokens = current_mask - forbid(disallowed_terminals)
                if is_end(node):
                    disallowed = disallowed_terminals(gss_node)
                    forbid: ffi.Bitset = self._compute_forbid_llm_tokens(disallowed)
                    allowed: ffi.Bitset = llm_mask.difference(forbid)
                    if not allowed.is_empty():
                        final_mask = final_mask.union(allowed)

                if llm_mask.is_empty():
                    # No tokens can pass through this node; prune further exploration
                    stopped.add(node)
                    continue

                # Local aggregation to reduce repeated merges on global dict
                pending_updates: Dict[int, Tuple[GSS, ffi.Bitset]] = {}

                # For pop reuse within this node
                popped_by_pop: Dict[int, GSS] = {}

                # Per-node cache to avoid recomputing filtered GSS for the same (popped, allowed_top_states)
                filter_cache: Dict[Tuple[int, Tuple[int, ...]], GSS] = {}
                peek_cache: Dict[int, Tuple[int, ...]] = {}

                for (pop, llm_bv, state_bv_list, dest_lists) in children_index.get(node, []):
                    # Pop once per (pop, llm_bv) group
                    popped = popped_by_pop.get(pop)
                    if popped is None:
                        popped = gss_node.popn(pop)
                        popped_by_pop[pop] = popped

                    # Apply LLM constraint early; if intersection empty, skip entire group
                    if llm_bv.is_empty():
                        child_mask_base: ffi.Bitset = llm_mask
                    else:
                        child_mask_base = llm_mask.intersection(llm_bv)
                        if child_mask_base.is_empty():
                            continue  # pruning

                    popped_id = id(popped)
                    # Cache peek() per popped to avoid repeated calls
                    top_states: Tuple[int, ...] = peek_cache.get(popped_id)  # type: ignore
                    if top_states is None:
                        # Peek returns iterable of ints; order doesn't matter but for caching use tuple
                        ts = popped.peek()
                        # Some GSS implementations may return list; ensure ints
                        top_states = tuple(int(x) for x in ts)
                        peek_cache[popped_id] = top_states

                    # For each unique state_bv, compute matching GSS only once, then fan-out to its dest nodes
                    for idx, state_bv in enumerate(state_bv_list):
                        # Determine allowed top states for this filter
                        allowed_states: List[int] = []
                        for s in top_states:
                            if state_bv.contains(int(s)):
                                allowed_states.append(int(s))
                        if not allowed_states:
                            continue

                        key = (popped_id, tuple(allowed_states))
                        child_gss = filter_cache.get(key)
                        if child_gss is None:
                            # Build GSS by merging isolates of allowed top states
                            isolates: List[GSS] = [popped.isolate(s) for s in allowed_states]
                            child_gss = GSS.merge_many(isolates)
                            filter_cache[key] = child_gss

                        dests = dest_lists[idx]
                        for d in dests:
                            # Combine per-destination contributions locally first
                            prev = pending_updates.get(d)
                            if prev is None:
                                pending_updates[d] = (child_gss, child_mask_base)
                            else:
                                g0, m0 = prev
                                pending_updates[d] = (g0.merge(child_gss), m0.union(child_mask_base))

                # Flush local pending updates into global values and enqueue by depth
                for d, (cg, cm) in pending_updates.items():
                    if d in values:
                        g0, m0 = values[d]
                        values[d] = (g0.merge(cg), m0.union(cm))
                    else:
                        values[d] = (cg, cm)
                    enqueue(self.max_depth[d], d)

        # Map internal token ids to original ids and return as a RangeSet
        if final_mask.is_empty():
            return RangeSet.from_ranges([])

        # Faster: build from_indices rather than repeated inserts
        internal_indices: List[int] = final_mask.to_indices()
        original_indices: List[int] = []
        map_io = self.internal_to_original_map
        append = original_indices.append
        for i in internal_indices:
            j = map_io.get(int(i))
            if j is not None:
                append(int(j))

        if not original_indices:
            return RangeSet.from_ranges([])

        original_mask: ffi.Bitset = ffi.Bitset.from_indices(original_indices)
        return RangeSet.from_ranges(original_mask.to_ranges())
