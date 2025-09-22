"""
High-performance get_mask override for the Precompute3 models.

What this does
--------------
- Monkey-patches both:
  - python.aug25.models.precompute3_model_pure_python.Model.get_mask
  - python.aug25.models.precompute3_model_pure_python_with_stats.Model.get_mask
  to an optimized, stats-free implementation.

Key optimizations
-----------------
1) Batch merges: Accumulate child GSS contributions per node and defer to a single
   GSS.merge_many when the node is actually processed. This removes thousands of incremental merges.

2) Group by "pop" per node: Compute gss_node.popn(pop) once per distinct pop value and reuse for all edges with that pop.
   This significantly reduces popn() calls.

3) Apply mask once per (pop, llm_bv) edge block: Instead of applying llm_bv intersection per dest,
   apply it once on the popped GSS for the whole edge block, then isolate subsets per dest.
   This reduces apply_and_prune calls by roughly the average "dests per edge" factor.

4) Replace repeated contains() calls with bitset intersections:
   For each edge block, compute masked_heads set once and use RangeSet intersection with each dest's state_bv;
   then convert to indices for a single isolate_many. This removes N*contains overhead.

Usage
-----
Just import this module once early in your program (or run it as a script) to patch the models:

    import gpt_5_19  # (this file)
    gpt_5_19.install()  # apply the patch

The optimization is transparent to the rest of your code.
"""

from typing import Dict, List, Tuple, Optional, Iterable, Any, Set
import sys

# Optional: If your runtime doesn't have these modules available at import time,
# the install() function catches ImportError and patches only what's available.
try:
    from python.aug25.models import precompute3_model_pure_python as m_pure
except Exception:  # noqa: BLE001 - best effort import
    m_pure = None  # type: ignore

try:
    from python.aug25.models import precompute3_model_pure_python_with_stats as m_stats
except Exception:  # noqa: BLE001 - best effort import
    m_stats = None  # type: ignore

try:
    from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
except Exception as e:  # noqa: BLE001
    # If your environment doesn't have this import path, re-raise with a clearer message.
    raise ImportError("LeveledGSS (GSS) implementation not found. Ensure python.gss_tester.implementations.leveled_impl is importable.") from e


def _make_optimized_get_mask(m) -> Any:
    """
    Create an optimized get_mask bound to a particular module (m) that provides:
      - m.PyAcc dataclass
      - Model instances with:
        .state: Dict[int, GSS]
        .roots_map: Dict[int, int]
        .max_depth: Dict[int, int]
        .arena: Dict[int, dict] with children as [((pop:int, llm_bv:RangeSet), [(dest_idx, state_bv:RangeSet), ...]), ...]
        .is_end(node_id: int) -> bool
        .possible_matches_cache: Optional[Dict[int, Dict[int, RangeSet]]]
        .tokenizer_max_state: int
        .all_internal_llm_tokens_bitset: RangeSet
        .internal_to_original_map: Dict[int, int]
    """

    PyAcc = m.PyAcc

    def optimized_get_mask(self) -> Any:
        """
        Optimized, stats-free get_mask.

        Core algorithm:
        - Initialize per-acc llm_mask once from disallowed terminals -> allowed LLM tokens.
        - Traverse the trie nodes by their max_depth buckets.
        - For each node:
            - Merge pending GSS contributions once (deferred merge).
            - For each distinct pop in outgoing edges:
                - popped = gss_node.popn(pop)
                - For each (llm_bv, dests) in this pop-group:
                    - masked = popped.apply_and_prune(mask_intersect_fn(llm_bv))   # once!
                    - masked_heads_rs = RS.from_indices(masked.peek())
                    - For each (dest_idx, state_bv):
                        - keep_rs = masked_heads_rs.intersection(state_bv)
                        - if keep_rs non-empty:
                            - child_gss = masked.isolate_many(keep_rs.to_indices())
                            - append child_gss to pending_values[dest_idx]
                            - enqueue by max_depth[dest_idx]
        - At end nodes, reduce_acc and union the llm_mask into final.
        - Convert internal to original ids and return.
        """
        # Pull commonly used fields locally for speed
        state_map: Dict[int, GSS] = self.state
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end

        # RangeSet class (FFIRangeSet) via instance type; avoids imports
        rs_cls = type(self.all_internal_llm_tokens_bitset)
        RS_empty = rs_cls.empty  # staticmethod on class
        RS_from_indices = rs_cls.from_indices  # staticmethod on class

        all_ones = self.all_internal_llm_tokens_bitset  # RangeSet universe

        # Prepare final mask
        final_mask = RS_empty()

        # Seed: Initialize llm_mask per-acc and enqueue roots
        # Convert disallowed terminals -> disallowed llm bitset -> allowed set (all_ones \ disallowed)
        pmc: Dict[int, Dict[int, Any]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        def initialize_acc(acc: PyAcc) -> PyAcc:
            disallowed_llm_mask = RS_empty()
            disallowed_map = acc.terminals_union  # Dict[tokenizer_state_id -> RangeSet of disallowed terminal ids]
            # For each tokenizer state id with disallowed terminals, union their mapped LLM tokens
            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]
                # to_indices: FFI returns Python list of ints; small in practice
                for terminal_id in disallowed_terminals.to_indices():
                    bs = terminals_to_llm.get(terminal_id)
                    if bs is not None:
                        disallowed_llm_mask = disallowed_llm_mask.union(bs)

            allowed_mask = all_ones.difference(disallowed_llm_mask)
            # Consume the terminals map (no longer needed after llm_mask derived)
            return PyAcc(terminals_union={}, llm_mask=allowed_mask)

        # Pending contributions per node; defer merges for fewer, bigger merges
        pending_values: Dict[int, List[GSS]] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        from heapq import heappush, heappop
        hp, hpop = heappush, heappop

        # Initialize with tokenizer states mapped to trie roots
        apply_memo: Dict[PyAcc, PyAcc] = {}
        for sid, gss in state_map.items():
            root_node = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)
            lst = pending_values.get(root_node)
            if lst is None:
                pending_values[root_node] = [gss_initialized]
            else:
                lst.append(gss_initialized)
            d = max_depth[root_node]
            bucket = todo.get(d)
            if bucket is None:
                todo[d] = {root_node}
                hp(depth_heap, d)
            else:
                bucket.add(root_node)

        # Small helper: Enqueue a node by depth
        def enqueue(d: int, n: int) -> None:
            bucket = todo.get(d)
            if bucket is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                bucket.add(n)

        # Main traversal by depth buckets
        while depth_heap:
            depth = hpop(depth_heap)
            bucket = todo.get(depth)
            if not bucket:
                # can happen if emptied later
                todo.pop(depth, None)
                continue

            while bucket:
                node = bucket.pop()

                gss_list = pending_values.pop(node, None)
                if not gss_list:
                    # No contributions to this node (already processed/merged elsewhere)
                    continue

                # Merge contributions only once per node pop
                gss_node: GSS = GSS.merge_many(gss_list)

                # End-node aggregation: reduce and union llm_mask
                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
                        final_mask = final_mask.union(reduced_acc.llm_mask)

                # Traverse edges
                node_data = arena.get(node)
                if not node_data:
                    continue
                children = node_data.get("children") or []
                if not children:
                    continue

                # Group edges by pop to reuse popn()
                edges_by_pop: Dict[int, List[Tuple[Any, List[Tuple[int, Any]]]]] = {}  # pop -> list of (llm_bv, dests)
                for edge_key, dests in children:
                    pop_val, llm_bv = edge_key  # pop:int, llm_bv:RangeSet
                    lst = edges_by_pop.get(pop_val)
                    if lst is None:
                        edges_by_pop[pop_val] = [(llm_bv, dests)]
                    else:
                        lst.append((llm_bv, dests))

                for pop_val, edge_blocks in edges_by_pop.items():
                    # pop once per distinct pop
                    popped: GSS = gss_node if pop_val == 0 else gss_node.popn(pop_val)
                    if popped.is_empty():
                        continue

                    # For each (llm_bv, dests) in this pop-group:
                    # - Apply llm mask once to popped (apply_and_prune)
                    # - Compute masked heads once
                    for llm_bv, dests in edge_blocks:
                        # Apply-and-prune once per edge block (pop, llm_bv)
                        def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                            new_mask = acc.llm_mask.intersection(llm_bv)
                            if new_mask.is_empty():
                                return None
                            return PyAcc(
                                terminals_union=acc.terminals_union,
                                llm_mask=new_mask
                            )

                        masked: GSS = popped.apply_and_prune(intersect_and_prune)
                        if masked.is_empty():
                            continue

                        # Compute masked head set once, convert to RangeSet for fast intersection with dest's state_bv
                        masked_heads = masked.peek()
                        if not masked_heads:
                            continue
                        masked_heads_rs = RS_from_indices(list(masked_heads))

                        # Now route to destinations; isolate subsets by intersecting head set with the state_bv
                        for dest_idx, state_bv in dests:
                            keep_rs = masked_heads_rs.intersection(state_bv)
                            if keep_rs.is_empty():
                                continue

                            keep_states = keep_rs.to_indices()
                            if not keep_states:
                                continue

                            child_gss: GSS = masked.isolate_many(keep_states)
                            if child_gss.is_empty():
                                continue

                            lst = pending_values.get(dest_idx)
                            if lst is None:
                                pending_values[dest_idx] = [child_gss]
                            else:
                                lst.append(child_gss)

                            enqueue(max_depth[dest_idx], dest_idx)

            # Remove bucket for this depth
            todo.pop(depth, None)

        # Convert internal indices to original IDs
        original_indices: List[int] = []
        for i in final_mask.to_indices():
            mapped = self.internal_to_original_map.get(i)
            if mapped is not None:
                original_indices.append(mapped)

        return RS_from_indices(original_indices)

    return optimized_get_mask


def install() -> None:
    """
    Apply the optimized get_mask to both precompute3 models (pure_python and with_stats)
    if they are present in the environment.
    """
    patched = False
    if m_pure is not None:
        optimized_get_mask = _make_optimized_get_mask(m_pure)
        m_pure.Model.get_mask = optimized_get_mask  # type: ignore[attr-defined]
        patched = True
    if m_stats is not None:
        optimized_get_mask = _make_optimized_get_mask(m_stats)
        m_stats.Model.get_mask = optimized_get_mask  # type: ignore[attr-defined]
        patched = True

    if not patched:
        raise ImportError(
            "Neither python.aug25.models.precompute3_model_pure_python nor "
            "python.aug25.models.precompute3_model_pure_python_with_stats could be imported. "
            "Ensure your PYTHONPATH includes the project root before calling install()."
        )


# Auto-install on import if desired. Comment out if you want manual control.
# install()

if __name__ == "__main__":
    # If run as a script, install immediately.
    install()
    print("Optimized get_mask installed for available Precompute3 models.")
