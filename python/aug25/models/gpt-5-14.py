import heapq
import _sep1 as ffi
from typing import Dict, List, Set, Tuple, Optional

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel
from ..common_interface import GraphProvider, RangeSet

try:
    profile
except NameError:
    def profile(func): return func


class Model(GraphProvider):
    """
    An optimized model focused on making get_mask fast.

    Key changes vs the baseline get_mask implementation:
    - Compute "allowed" tokens directly instead of building a large "forbid" bitset
      (which is typically much bigger). This dramatically reduces the amount of
      union work when the grammar at a given point allows only a small set of terminals.
    - Heavy caching:
        * Cache, for a given acceptance (HybridL2Bitset of terminals), the union of LLM
          tokens allowed across all tokenizer states. This prevents recomputing the large
          union repeatedly when many end nodes share the same acceptance.
        * Cache, for a specific tokenizer state (tsid) and a terminal-set Bitset,
          the union of LLM tokens for terminals in that set at that state. This converts
          repeated O(|keys(pmc[tsid])|) loops into O(1) cache lookups after the first time.
        * Cache the to_indices() result for terminal bitsets we repeatedly examine,
          only when we need it (adaptive strategy).
    - Adaptive strategy for computing union over terminals:
        * If the number of terminals in a terminal-set Bitset is small (relative to the
          number of keys in pmc[tsid]), iterate the allowed terminals and do fast dict
          lookups.
        * Otherwise, iterate pmc[tsid] keys and test membership with Bitset.contains().

    Assumptions (as requested):
    - _sep1 (ffi) and LeveledGSS are not bottlenecks; we focus on algorithmic + cache improvements.
    - We do not modify precompute3_model_pure_python.py. This class wraps it.
    """

    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model

        # Direct references from InnerModel
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map

        # Preprocess possible_matches_cache into fast structures:
        # - _pmc_map_by_state[tsid]: dict[int -> ffi.Bitset] where key is terminal_id (int)
        # - _pmc_items_by_state[tsid]: list[(terminal_id (int), ffi.Bitset)] for fast iteration
        self._pmc_map_by_state: Dict[int, Dict[int, ffi.Bitset]] = {}
        self._pmc_items_by_state: Dict[int, List[Tuple[int, ffi.Bitset]]] = {}
        if self.possible_matches_cache:
            for tsid_k, mapping in self.possible_matches_cache.items():
                tsid = int(tsid_k)
                if not mapping:
                    self._pmc_map_by_state[tsid] = {}
                    self._pmc_items_by_state[tsid] = []
                    continue
                m: Dict[int, ffi.Bitset] = {}
                items: List[Tuple[int, ffi.Bitset]] = []
                for term_k, llm_bs in mapping.items():
                    # Keys in pmc may be str or int; normalize to int once here
                    try:
                        term_id = int(term_k)
                    except Exception:
                        # Ignore malformed keys
                        continue
                    m[term_id] = llm_bs
                    items.append((term_id, llm_bs))
                self._pmc_map_by_state[tsid] = m
                self._pmc_items_by_state[tsid] = items

        # Cache for: (tsid, terminals_bitset_json) -> union LLM tokens bitset at that state over that terminal-set
        self._union_by_tsid_termset_key: Dict[Tuple[int, str], ffi.Bitset] = {}
        # Cache for: terminals_bitset_json -> list[int] (indices of terminals). Built on demand.
        self._termset_indices_cache: Dict[str, List[int]] = {}
        # Cache for: HybridL2Bitset (terminals_union) -> global union LLM tokens (across all tsid) for that acceptance
        self._acc_allowed_llm_cache: Dict[ffi.HybridL2Bitset, ffi.Bitset] = {}

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        return Model(InnerModel.from_json_string(s))

    def commit(self, token_id: int) -> None:
        self.inner_model.commit(token_id)

    @property
    def state(self) -> Dict[int, GSS]:
        return self.inner_model.state

    def is_end(self, node: int) -> bool:
        return bool(((self.arena.get(node) or {}).get("value") or {}).get("clean_end", False))

    # -------------- Internal helpers for fast "allowed" computation --------------

    def _get_termset_indices(self, termset: ffi.Bitset) -> List[int]:
        """
        Returns all terminal ids set in 'termset' as a list, using a cache keyed by JSON string.
        Only computed if needed (adaptive).
        """
        key = termset.to_json_string()
        cached = self._termset_indices_cache.get(key)
        if cached is not None:
            return cached
        idxs = termset.to_indices()
        self._termset_indices_cache[key] = idxs
        return idxs

    def _union_for_tsid_and_termset(self, tsid: int, termset: ffi.Bitset) -> ffi.Bitset:
        """
        For a specific tokenizer state (tsid) and a terminal-set Bitset, return the union of LLM tokens
        across all terminals in that set at that state. Uses adaptive strategy and caches the result.
        """
        items = self._pmc_items_by_state.get(tsid)
        if not items:
            return ffi.Bitset.zeros()

        key = (tsid, termset.to_json_string())
        cached = self._union_by_tsid_termset_key.get(key)
        if cached is not None:
            return cached

        pm = self._pmc_map_by_state[tsid]
        res = ffi.Bitset.zeros()

        # Adaptive strategy: choose whether to iterate allowed terminals or pmc keys
        try:
            termset_size = len(termset)
        except Exception:
            termset_size = None

        pm_size = len(items)

        if termset_size is not None and termset_size <= (pm_size // 2 + 2):
            # Iterate allowed terminals (fewer)
            indices = self._get_termset_indices(termset)
            for t in indices:
                bs = pm.get(t)
                if bs:
                    res = res.union(bs)
        else:
            # Iterate pmc keys and test membership
            contains = termset.contains
            for t, bs in items:
                if contains(int(t)):
                    res = res.union(bs)

        self._union_by_tsid_termset_key[key] = res
        return res

    def _allowed_llm_for_acceptance(self, terminals_union: ffi.HybridL2Bitset) -> ffi.Bitset:
        """
        Compute the union of allowed LLM tokens across all tokenizer states described in 'terminals_union'.
        Cached per terminals_union (HybridL2Bitset implements __hash__/__eq__).
        """
        cached = self._acc_allowed_llm_cache.get(terminals_union)
        if cached is not None:
            return cached

        res = ffi.Bitset.zeros()
        max_state = self.tokenizer_max_state

        for (start, end), termset in terminals_union.range_values():
            if termset.is_empty():
                continue
            lo = int(start)
            hi = int(end)
            if lo > max_state:
                continue
            if hi > max_state:
                hi = max_state
            for tsid in range(lo, hi + 1):
                part = self._union_for_tsid_and_termset(tsid, termset)
                if not part.is_empty():
                    res = res.union(part)

        self._acc_allowed_llm_cache[terminals_union] = res
        return res

    # -------------- Main algorithm --------------

    @profile
    def get_mask(self) -> RangeSet:
        """
        Compute the final allowed LLM-token mask by exploring the arena graph with GSS merging.
        Performance improvements:
        - Compute "allowed" tokens directly at end nodes using cached union lookups.
        - Reuse depth scheduling and node-value merging from the baseline, but with
          substantially less expensive end-node token aggregation.
        """
        state_map: Dict[int, GSS] = self.state
        all_ones: Optional[ffi.Bitset] = self.all_internal_llm_tokens_bitset
        if all_ones is None:
            # Fallback: if for some reason it's None, assume empty (no allowed tokens).
            all_ones = ffi.Bitset.zeros()

        final_mask: ffi.Bitset = ffi.Bitset.zeros()

        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        stopped: Set[int] = set()
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end

        max_state: int = self.tokenizer_max_state

        # Seed frontier by roots (merge GSS by same root-id, union masks)
        for sid, gss in state_map.items():
            r = roots_map.get(int(sid))
            if r is None:
                continue
            r = int(r)
            if r in values:
                values[r] = (values[r][0].merge(gss), all_ones)
            else:
                values[r] = (gss, all_ones)
            d = int(max_depth[r])
            bag = todo.get(d)
            if bag is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                bag.add(r)

        def enqueue(depth: int, node_id: int) -> None:
            bag = todo.get(depth)
            if bag is None:
                todo[depth] = {node_id}
                hp(depth_heap, depth)
            else:
                bag.add(node_id)

        while depth_heap:
            depth: int = hpop(depth_heap)
            nodes: Optional[Set[int]] = todo.pop(depth, None)
            if not nodes:
                continue

            for node in nodes:
                if node in stopped:
                    continue
                item = values.pop(node, None)
                if item is None:
                    continue
                gss_node, llm_mask = item

                # If node is an end, compute allowed tokens directly (fast path).
                if is_end(node):
                    acc = gss_node.reduce_acc()
                    # If there is no acceptance, no tokens are allowed at this node
                    if acc is not None:
                        allowed_global = self._allowed_llm_for_acceptance(acc.terminals_union)
                        if not allowed_global.is_empty():
                            allowed_here = llm_mask if allowed_global.is_empty() else llm_mask.intersection(allowed_global)
                            if not allowed_here.is_empty():
                                final_mask = final_mask.union(allowed_here)

                # If nothing remains allowed in the mask, we can stop exploring this node
                if llm_mask.is_empty():
                    stopped.add(node)
                    continue

                # Expand children
                children = (arena.get(node, {}).get("children") or [])
                for (pop, llm_bv), dests in children:
                    popped: GSS = gss_node.popn(pop)

                    # Compute the LLM-mask contribution for this transition once
                    child_mask_base: ffi.Bitset = llm_mask if llm_bv.is_empty() else llm_mask.intersection(llm_bv)
                    if child_mask_base.is_empty():
                        # Intersection kills the path; skip all dests for this (pop, llm_bv)
                        continue

                    top_states = popped.peek()
                    if not top_states:
                        continue

                    # For each destination, filter by state_bv and merge matched isolates
                    for dest_idx, state_bv in dests:
                        if state_bv.is_empty():
                            # No states allowed for this destination
                            continue

                        matched: List[GSS] = [popped.isolate(s) for s in top_states if state_bv.contains(int(s))]
                        if not matched:
                            continue

                        child_gss: GSS = GSS.merge_many(matched)
                        d = int(dest_idx)
                        if d in values:
                            g0, m0 = values[d]
                            # Merge GSS and union LLM-mask at the destination
                            values[d] = (g0.merge(child_gss), m0.union(child_mask_base))
                        else:
                            values[d] = (child_gss, child_mask_base)
                        enqueue(int(max_depth[d]), d)

        # Convert internal-token mask back to original tokenizer ids
        original_mask: ffi.Bitset = ffi.Bitset.zeros()
        for i in final_mask.to_indices():
            orig = self.internal_to_original_map.get(int(i))
            if orig is not None:
                original_mask.insert(int(orig))

        return RangeSet.from_ranges(original_mask.to_ranges())
