import bisect
import heapq
from typing import Dict, List, Optional, Set, Tuple

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
    A faster get_mask implementation focused on:
    - Avoiding exponential blowup in traversal by using a monotone worklist (fixpoint) algorithm.
    - Reducing nested-loop overhead by precompiling graph adjacency and filtering empty transitions.
    - Dramatically speeding up forbidden-token computation by:
        * Iterating only over relevant tokenizer states using a sorted index and binary search.
        * Iterating over just the actually disallowed terminals per (range, bv) via bv.to_indices().
        * Avoiding repeated int() conversions for terminal ids in possible_matches_cache.
    - Avoiding redundant reprocessing by only scheduling nodes when their accumulated
      GSS or LLM-mask state changes.
    """

    def __init__(self, inner_model: InnerModel):
        # Hold onto the inner model and exported structures
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model

        # Core data from the precomputed model
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth

        # Caches and tokenization bounds
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map

        # Build an optimized representation of the graph and caches that accelerates get_mask.

        # 1) Precompute end-node flags for fast checks
        self._is_end: Dict[int, bool] = {}
        for node_id, node_obj in self.arena.items():
            clean_end = bool(((node_obj or {}).get("value") or {}).get("clean_end", False))
            self._is_end[int(node_id)] = clean_end

        # 2) Precompile adjacency: keep grouping by (pop, llm_bv) but filter out empty state_bv transitions.
        #    children_grouped[node] = List[ Tuple[pop:int, llm_bv:ffi.Bitset, dests: List[Tuple[dest_idx:int, state_bv:ffi.Bitset]]] ]
        self._children_grouped: Dict[int, List[Tuple[int, ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]] = {}
        for node_id, node_obj in self.arena.items():
            node_children = (node_obj or {}).get("children") or []
            grouped: List[Tuple[int, ffi.Bitset, List[Tuple[int, ffi.Bitset]]]] = []
            for edge_info, dests in node_children:
                pop, llm_bv = edge_info
                # filter out empty state_bv dests early
                filtered_dests: List[Tuple[int, ffi.Bitset]] = [(int(d), sbv) for (d, sbv) in (dests or []) if not sbv.is_empty()]
                if filtered_dests:
                    grouped.append((int(pop), llm_bv, filtered_dests))
            if grouped:
                self._children_grouped[int(node_id)] = grouped

        # 3) Optimize possible_matches_cache (pmc):
        #    - Convert terminal keys from strings to ints once.
        #    - Build a sorted index of tokenizer state ids (tsid) that actually appear.
        # Assumption (as per existing code): pmc is present (not a bottleneck). Still handle None defensively.
        self._pmc_int: Optional[Dict[int, Dict[int, ffi.Bitset]]] = None
        self._pmc_tsid_keys_sorted: Optional[List[int]] = None
        if self.possible_matches_cache:
            pmc_int: Dict[int, Dict[int, ffi.Bitset]] = {}
            for tsid_raw, term_map in self.possible_matches_cache.items():
                tsid = int(tsid_raw)
                # Convert terminal ids to ints
                new_term_map: Dict[int, ffi.Bitset] = {}
                for term_id_raw, llm_tokens in (term_map or {}).items():
                    # term ids might be stored as strings; normalize to int once
                    term_id = int(term_id_raw)
                    new_term_map[term_id] = llm_tokens
                if new_term_map:
                    pmc_int[tsid] = new_term_map
            self._pmc_int = pmc_int
            self._pmc_tsid_keys_sorted = sorted(pmc_int.keys())

        # 4) Pre-grab max_state bound and all-ones mask
        self._max_state: int = self.tokenizer_max_state
        self._all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int) -> None:
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        return self._is_end.get(int(node), False)

    def _tsid_keys_in_range(self, start: int, end: int) -> List[int]:
        """
        Return the subset of tokenizer state ids with precomputed matches that lie in [start, end].
        Leverages binary search over a pre-sorted index of tsids.
        """
        if not self._pmc_tsid_keys_sorted:
            return []
        keys = self._pmc_tsid_keys_sorted
        lo = bisect.bisect_left(keys, start)
        hi = bisect.bisect_right(keys, end)
        if lo >= hi:
            return []
        return keys[lo:hi]

    def _compute_forbid_tokens(self, g: GSS) -> ffi.Bitset:
        """
        Compute the union of forbidden LLM tokens for the given GSS, using:
        - HybridL2Bitset of disallowed terminals across tokenizer state ranges.
        - A pre-indexed, integer-keyed possible_matches_cache.
        This implementation iterates only over relevant tokenizer states and disallowed terminals.
        """
        forbid: ffi.Bitset = ffi.Bitset.zeros()
        pmc = self._pmc_int
        if pmc is None:
            # Defensive: if pmc missing, nothing to forbid using grammar-terminal matching.
            return forbid

        # Determine disallowed terminals across tokenizer state ranges
        acc = g.reduce_acc()
        disallowed = ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()

        max_state = self._max_state

        for (start, end), bv in disallowed.range_values():
            if bv.is_empty():
                continue
            # Clip to tokenizer max_state
            a = int(start)
            b = int(end)
            if a > max_state:
                continue
            if b > max_state:
                b = max_state
            if a > b:
                continue

            # Fetch only the tsids in [a, b] that actually have possible matches
            tsids = self._tsid_keys_in_range(a, b)
            if not tsids:
                continue

            # Enumerate only the actually disallowed terminal ids for this range
            term_ids: List[int] = bv.to_indices()
            if not term_ids:
                continue

            # Iterate tsids outer, term_ids inner to benefit from dict locality for pm lookups
            for tsid in tsids:
                pm = pmc.get(tsid)
                if not pm:
                    continue
                for tid in term_ids:
                    tokens = pm.get(int(tid))
                    if tokens is not None and not tokens.is_empty():
                        forbid = forbid.union(tokens)

        return forbid

    @profile
    def get_mask(self) -> RangeSet:
        """
        Faster implementation based on a monotone worklist algorithm:
        - Aggregate per-node (GSS, llm_mask) state.
        - Schedule downstream propagation only when per-node state increases.
        - Use precompiled adjacency and optimized forbidden-token computation at end nodes.
        """
        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self._all_ones
        if all_ones is None:
            # Defensive fallback: if not provided, start with zeros -> no tokens allowed.
            all_ones = ffi.Bitset.zeros()

        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        # Aggregate state: node -> (GSS, llm_mask)
        agg: Dict[int, Tuple[GSS, ffi.Bitset]] = {}

        # Initialize worklist with roots for all current parser states
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth

        # Use a heap ordered by depth to approximately follow a topological-like order
        # for better cache behavior. Each item is (depth, node_id).
        heap: List[Tuple[int, int]] = []
        in_queue: Set[int] = set()

        def push_node(n: int) -> None:
            if n not in in_queue:
                in_queue.add(n)
                d = max_depth.get(int(n), 0)
                heapq.heappush(heap, (int(d), int(n)))

        # Merge per-root GSS, and attach an all-ones mask
        for sid, gss in state_map.items():
            r = roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)
            if r in agg:
                g0, m0 = agg[r]
                # Merge GSS and retain full mask
                agg[r] = (g0.merge(gss), m0)
            else:
                agg[r] = (gss, all_ones)
            push_node(r)

        # Shortcut: nothing to process
        if not heap:
            return RangeSet.from_ranges(final_mask.to_ranges())

        # Access to precompiled adjacency and helpers
        children_grouped = self._children_grouped
        is_end = self.is_end

        while heap:
            _, node = heapq.heappop(heap)
            in_queue.discard(node)

            item = agg.get(node)
            if not item:
                # No accumulated state at this node
                continue

            gss_node, llm_mask = item

            # 1) If this node is an end-node, compute allowed tokens and accumulate into final_mask
            if is_end(node) and not llm_mask.is_empty():
                forbid = self._compute_forbid_tokens(gss_node)
                if not forbid.is_empty():
                    allowed = llm_mask.difference(forbid)
                else:
                    allowed = llm_mask
                if not allowed.is_empty():
                    final_mask = final_mask.union(allowed)

            # 2) If mask is empty, no need to propagate further
            if llm_mask.is_empty():
                continue

            # 3) Propagate to children
            groups = children_grouped.get(node, [])
            if not groups:
                continue

            for pop, llm_bv, dests in groups:
                # Compute popped GSS once per (pop, llm_bv) group
                popped: GSS = gss_node.popn(pop)

                # Compute child_mask once per group
                child_mask: ffi.Bitset = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)
                if child_mask.is_empty():
                    # Intersection killed the mask; no tokens pass through this group
                    continue

                # Lazily compute isolates per state from popped.peek()
                iso_by_state: Optional[Dict[int, GSS]] = None
                # We'll only build the mapping if a dest needs it (i.e., non-empty state_bv).
                # state_bv empties were already filtered out in __init__.

                for dest_idx, state_bv in dests:
                    # Build iso_by_state lazily
                    if iso_by_state is None:
                        peek_states: List[int] = [int(s) for s in popped.peek()]
                        if not peek_states:
                            # No states on stack -> no match for any dests
                            break
                        iso_by_state = {}
                        for s in peek_states:
                            iso_by_state[s] = popped.isolate(s)

                    # Build matched list by iterating only over states in the bitset
                    matched_list: List[GSS] = []
                    for s in state_bv.to_indices():
                        gs = iso_by_state.get(int(s))
                        if gs is not None:
                            matched_list.append(gs)
                    if not matched_list:
                        continue

                    child_gss: GSS = GSS.merge_many(matched_list)

                    d = int(dest_idx)
                    prev = agg.get(d)
                    if prev is None:
                        agg[d] = (child_gss, child_mask)
                        push_node(d)
                    else:
                        g0, m0 = prev
                        g_new = g0.merge(child_gss)
                        m_new = m0.union(child_mask)

                        # Schedule only if something actually changed.
                        # We try to avoid reprocessing when neither GSS nor mask grew.
                        # For mask change detection, use symmetric_difference emptiness.
                        mask_changed = not m_new.symmetric_difference(m0).is_empty()
                        # For GSS, we opportunistically check identity (persistent impls often return self if unchanged).
                        # If identity not preserved by merge, we conservatively assume change to ensure correctness.
                        gss_changed = (g_new is not g0)

                        if mask_changed or gss_changed:
                            agg[d] = (g_new, m_new)
                            push_node(d)

        # Translate internal token ids back to original ids
        original_mask: ffi.Bitset = ffi.Bitset.zeros()
        itomap = self.internal_to_original_map
        for i in final_mask.to_indices():
            oi = itomap.get(int(i))
            if oi is not None:
                original_mask.insert(int(oi))

        return RangeSet.from_ranges(original_mask.to_ranges())
