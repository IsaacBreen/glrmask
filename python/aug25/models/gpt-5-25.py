"""
Fast get_mask implementation with aggressive algorithmic and micro-optimizations.

Key improvements over the baseline:
- Zero stats/metrics overhead.
- Batch merge: defer all GSS merges per node until the node is actually processed, replacing
  thousands of incremental merges with a single merge_many.
- Pop/peek caching: compute popn and peek once per (GSS, pop) edge-block and reuse it for
  all its dests.
- Global LLM-mask intersection cache: reuse acc.llm_mask ∩ edge_llm_bv results across all
  apply_and_prune calls, regardless of the GSS instance.
- Optional per-(GSS, heads) isolate_many memoization to reuse subgraphs when identical head
  subsets are requested multiple times for the same popped GSS.

Usage:
- Import this module and call `patch_models()` once at startup to monkey-patch both
  python/aug25/models/precompute3_model_pure_python and
  python/aug25/models/precompute3_model_pure_python_with_stats Model.get_mask with the fast version.

Or:
- Call fast_get_mask(model) directly with a model instance (from either module).
"""

from __future__ import annotations

import sys
import heapq
import collections
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any

# We rely only on the interfaces exposed by the existing modules
# RangeSet and PyAcc will be taken from the model's module to ensure type-compat.
try:
    from python.aug25.common_interface import RangeSet  # type: ignore
except Exception:
    # Fallback if import paths differ in the environment
    from ..aug25.common_interface import RangeSet  # type: ignore


class _FastMaskComputer:
    """
    A per-call object that computes get_mask(self.model) with heavy caching and
    better batching to minimize Python overhead and GSS churn.
    """

    __slots__ = (
        "m",
        "PyAcc",  # The PyAcc class for this model's module (compatible type)
        "all_ones",
        "roots_map",
        "max_depth",
        "arena",
        "pmc",
        "internal_to_original_map",
        "state_map",
        "is_end",
        # Caches
        "_popn_cache",
        "_peek_cache",
        "_isolate_cache",
        "_acc_mask_intersection_cache",
        "_apply_cache",
    )

    def __init__(self, model: Any):
        self.m = model
        # Resolve the correct PyAcc class for the incoming model instance.
        mod = sys.modules.get(model.__class__.__module__)
        if mod is None or not hasattr(mod, "PyAcc"):
            raise RuntimeError(
                f"Unable to resolve PyAcc type from model module {model.__class__.__module__}"
            )
        self.PyAcc = getattr(mod, "PyAcc")
        self.all_ones: Optional[RangeSet] = model.all_internal_llm_tokens_bitset
        self.roots_map: Dict[int, int] = model.roots_map
        self.max_depth: Dict[int, int] = model.max_depth
        self.arena: Dict[int, dict] = model.arena
        self.pmc: Dict[int, Dict[int, RangeSet]] = model.possible_matches_cache or {}
        self.internal_to_original_map: Dict[int, int] = model.internal_to_original_map
        self.state_map = model.state
        self.is_end = model.is_end

        # Caches for this single computation (discarded at the end)
        # Keyed by object id to avoid holding references that could prevent GC (and to avoid
        # needing __hash__/__eq__ guarantees for GSS).
        self._popn_cache: Dict[Tuple[int, int], Any] = {}  # (id(gss), pop) -> popped_gss
        self._peek_cache: Dict[int, Tuple[int, ...]] = {}  # id(gss) -> tuple(head_ids)
        self._isolate_cache: Dict[Tuple[int, Tuple[int, ...]], Any] = {}  # (id(gss), heads) -> gss
        # Cache for acc.llm_mask ∩ llm_bv -> None or resulting mask
        self._acc_mask_intersection_cache: Dict[Tuple[RangeSet, RangeSet], Optional[RangeSet]] = {}
        # Optional cache for apply_and_prune across identical (GSS, llm_bv) calls
        self._apply_cache: Dict[Tuple[int, RangeSet], Any] = {}

    # ---------- Low-level cached wrappers over hot GSS operations ----------

    def _popn(self, gss: Any, pop: int) -> Any:
        key = (id(gss), int(pop))
        cached = self._popn_cache.get(key)
        if cached is not None:
            return cached
        res = gss.popn(pop)
        self._popn_cache[key] = res
        return res

    def _peek(self, gss: Any) -> Tuple[int, ...]:
        key = id(gss)
        cached = self._peek_cache.get(key)
        if cached is not None:
            return cached
        res = tuple(gss.peek())  # tuple for hashing/reuse
        self._peek_cache[key] = res
        return res

    def _isolate_many(self, gss: Any, heads: Iterable[int]) -> Any:
        # canonical tuple key so repeated requests for same heads hit
        ht = tuple(sorted(heads))
        key = (id(gss), ht)
        cached = self._isolate_cache.get(key)
        if cached is not None:
            return cached
        res = gss.isolate_many(ht)
        self._isolate_cache[key] = res
        return res

    # ---------- Accumulator mask intersection cache ----------

    def _intersect_acc_mask(self, acc: Any, edge_llm_bv: RangeSet) -> Optional[Any]:
        """
        Return a possibly new PyAcc whose llm_mask was intersected with edge_llm_bv,
        or None if it becomes empty.
        Caches per (acc.llm_mask, edge_llm_bv) to avoid recomputing RangeSet ops.
        """
        # We cache on the two RangeSets to maximize reuse across GSSs and nodes.
        key = (acc.llm_mask, edge_llm_bv)
        cached_mask = self._acc_mask_intersection_cache.get(key)
        if cached_mask is not None or key in self._acc_mask_intersection_cache:
            # cached_mask is either None (meaning it pruned) or a RangeSet
            if cached_mask is None:
                return None
            # Rebuild a PyAcc with the caller's terminals_union (keeps identity semantics)
            return self.PyAcc(terminals_union=acc.terminals_union, llm_mask=cached_mask)

        new_mask = acc.llm_mask.intersection(edge_llm_bv)
        if new_mask.is_empty():
            self._acc_mask_intersection_cache[key] = None
            return None

        # Store resulting mask to reuse
        self._acc_mask_intersection_cache[key] = new_mask
        return self.PyAcc(terminals_union=acc.terminals_union, llm_mask=new_mask)

    def _apply_and_prune(self, gss: Any, llm_bv: RangeSet) -> Any:
        """
        Apply the edge mask to all accumulators inside gss, pruning empty ones.
        Cached per (id(gss), llm_bv) since it's common to apply the same mask to the same
        intermediate GSS multiple times (especially within the same node).
        """
        key = (id(gss), llm_bv)
        cached = self._apply_cache.get(key)
        if cached is not None:
            return cached

        # Function that uses the global acc-mask cache above
        def f(acc: Any) -> Optional[Any]:
            return self._intersect_acc_mask(acc, llm_bv)

        res = gss.apply_and_prune(f)
        self._apply_cache[key] = res
        return res

    # ---------- Initialization: derive allowed LLM mask per-acc upfront ----------

    def _initialize_acc_factory(self, max_state: Optional[int]) -> Any:
        all_ones = self.all_ones
        pmc = self.pmc
        PyAcc = self.PyAcc

        # Pre-bind methods for speed
        RangeSet_empty = RangeSet.empty
        RangeSet_from_indices = RangeSet.from_indices

        def initialize_acc(acc: Any) -> Any:
            # Compute disallowed LLM mask by mapping disallowed terminals via pmc,
            # then allowed_mask = all_ones \ disallowed.
            disallowed_map: Dict[int, RangeSet] = acc.terminals_union

            if not disallowed_map:
                # Fast path: nothing to disallow
                allowed_mask = (all_ones if all_ones is not None else RangeSet_empty())
                return PyAcc(terminals_union={}, llm_mask=allowed_mask)

            disallowed_llm_mask = RangeSet_empty()
            # Iterate only tokenizer states we have mapping for
            for tsid, disallowed_terms in disallowed_map.items():
                # It is possible tokenizer states be out of bounds; skip if no pmc
                mapping = pmc.get(tsid)
                if not mapping:
                    continue

                # Iterate terminal ids. Typically few; union their mapped llm-sets.
                for term_id in disallowed_terms.to_indices():
                    mapped = mapping.get(term_id)
                    if mapped is not None:
                        disallowed_llm_mask = disallowed_llm_mask.union(mapped)

            allowed_mask = (all_ones if all_ones is not None else RangeSet_empty()).difference(disallowed_llm_mask)
            return PyAcc(terminals_union={}, llm_mask=allowed_mask)

        return initialize_acc

    # ---------- Main algorithm ----------

    def compute(self) -> RangeSet:
        """
        Optimized, allocation- and Python-overhead-aware traversal.
        """
        # Abbreviations
        state_map: Dict[int, Any] = self.state_map
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end

        # Compute initial per-acc allowed_mask and seed queue.
        initialize_acc = self._initialize_acc_factory(getattr(self.m, "tokenizer_max_state", None))
        apply_memo: Dict[Any, Any] = {}  # Memoize per-acc init across shared acc instances

        # We'll batch incoming GSS per node as lists and merge only when processing the node.
        pending_values: Dict[int, List[Any]] = collections.defaultdict(list)
        # A normal priority queue over depth, but node enqueues are deduplicated per depth using a set.
        depth_buckets: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []
        hp, hpop = heapq.heappush, heapq.heappop

        # Seed
        for sid, gss in state_map.items():
            root_node = roots_map[int(sid)]
            # Initialize acc masks once per unique acc
            gss_init = gss.apply(initialize_acc, apply_memo)
            pending_values[root_node].append(gss_init)

            d = max_depth[root_node]
            bucket = depth_buckets.get(d)
            if bucket is None:
                depth_buckets[d] = {root_node}
                hp(depth_heap, d)
            else:
                bucket.add(root_node)

        # Utility to enqueue node once
        def enqueue(node_id: int) -> None:
            d = max_depth[node_id]
            bucket = depth_buckets.get(d)
            if bucket is None:
                depth_buckets[d] = {node_id}
                hp(depth_heap, d)
            else:
                bucket.add(node_id)

        # Final mask over internal LLM token ids
        final_mask = RangeSet.empty()

        # Main traversal by monotonically non-increasing depth (topologically valid)
        while depth_heap:
            depth = hpop(depth_heap)
            nodes = depth_buckets.pop(depth, None)
            if not nodes:
                continue

            while nodes:
                node = nodes.pop()

                # Merge all incoming GSSs for this node in one shot
                gss_list = pending_values.pop(node, None)
                if not gss_list:
                    continue
                gss_node = gss_list[0] if len(gss_list) == 1 else gss_list[0].merge_many(gss_list)

                # End node: reduce and collect mask
                if is_end(node):
                    reduced_acc = gss_node.reduce_acc()
                    if reduced_acc:
                        final_mask = final_mask.union(reduced_acc.llm_mask)

                # Traverse outgoing edges
                children = arena.get(node, {}).get("children") or []
                for (pop, llm_bv), dests in children:
                    # Compute popped once per edge-block
                    popped = self._popn(gss_node, int(pop))
                    if popped.is_empty():
                        continue

                    # Heads after pop, computed once for the whole dest-block
                    heads_after_pop = self._peek(popped)
                    if not heads_after_pop:
                        continue

                    # For each dest, filter heads and produce a child GSS (lazy; no merge yet)
                    # We also reuse the popped GSS isolate results when the same head subset occurs.
                    append_to_node = pending_values.setdefault
                    for dest_idx, state_bv in dests:
                        # Filter: keep only heads present in the destination state set
                        # heads list is small (typically <= ~30), so this loop is cheap.
                        kept_heads = [h for h in heads_after_pop if state_bv.contains(h)]
                        if not kept_heads:
                            continue

                        child_gss = self._isolate_many(popped, kept_heads)
                        if child_gss.is_empty():
                            continue

                        # Apply edge LLM mask to all accs, pruning empty accs
                        child_gss = self._apply_and_prune(child_gss, llm_bv)
                        if child_gss.is_empty():
                            continue

                        dnode = int(dest_idx)
                        append_to_node(dnode, []).append(child_gss)
                        enqueue(dnode)

        # Convert internal indices to original ids
        orig_ids: List[int] = []
        for i in final_mask.to_indices():
            mapped = self.internal_to_original_map.get(i)
            if mapped is not None:
                orig_ids.append(mapped)

        return RangeSet.from_indices(orig_ids)


def fast_get_mask(model: Any) -> RangeSet:
    """
    Compute the LLM token mask for either:
      - python.aug25.models.precompute3_model_pure_python.Model
      - python.aug25.models.precompute3_model_pure_python_with_stats.Model
    The logic is independent of stats/metrics and uses optimized traversal.
    """
    return _FastMaskComputer(model).compute()


def _make_bound_fast_get_mask(model_module) -> Any:
    """
    Create a descriptor-compatible bound method for Model.get_mask monkey-patching.
    """

    def _bound(self):
        return fast_get_mask(self)

    _bound.__name__ = "get_mask"
    _bound.__qualname__ = f"{model_module.__name__}.Model.get_mask"
    return _bound


def patch_models() -> None:
    """
    Monkey-patch both Model.get_mask implementations (with and without stats) to use the fast path.
    Safe to call multiple times.
    """
    # Patch pure python model
    try:
        m1 = sys.modules.get("python.aug25.models.precompute3_model_pure_python")
        if m1 is None:
            import importlib

            m1 = importlib.import_module("python.aug25.models.precompute3_model_pure_python")
        m1.Model.get_mask = _make_bound_fast_get_mask(m1)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Patch stats model, if present
    try:
        m2 = sys.modules.get("python.aug25.models.precompute3_model_pure_python_with_stats")
        if m2 is None:
            import importlib

            m2 = importlib.import_module("python.aug25.models.precompute3_model_pure_python_with_stats")
        m2.Model.get_mask = _make_bound_fast_get_mask(m2)  # type: ignore[attr-defined]
    except Exception:
        pass


__all__ = ["fast_get_mask", "patch_models"]
