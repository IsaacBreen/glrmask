import heapq
from typing import Dict, List, Set, Optional, Iterable, Any, Tuple
import time
import collections

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel, PyAcc
from ..common_interface import GraphProvider, RangeSet
from ..range_set.rangeset_stats import reset_rangeset_stats, print_rangeset_stats


class Model(GraphProvider):
    """
    Optimized get_mask:
    - Hoist popped.peek() out of the inner dest loop (was called per-destination).
    - Fast path: when all peeked states are kept by state_bv, avoid isolate_many by reusing 'popped'.
    - Reuse a single acc_memo dictionary for all apply_and_prune calls within an edge-group
      so that identical PyAcc transformations are memoized across many destinations sharing the same llm_bv.
    - Avoid repeated attribute lookups in tight loops by binding locals.
    """

    def __init__(self, inner_model: InnerModel):
        self.inner_model: InnerModel = inner_model
        im: InnerModel = self.inner_model
        self.arena: Dict[int, dict] = im.arena
        self.roots_map: Dict[int, int] = im.roots_map
        self.max_depth: Dict[int, int] = im.max_depth
        self.possible_matches_cache: Optional[Dict[int, Dict[int, RangeSet]]] = im.possible_matches_cache
        self.tokenizer_max_state: int = im.tokenizer.max_state()
        self.all_internal_llm_tokens_bitset: Optional[RangeSet] = im.all_internal_llm_tokens_bitset
        self.internal_to_original_map: Dict[int, int] = im.internal_to_original_map
        self.all_terminals_bitset: Optional[RangeSet] = im.all_terminals_bitset

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

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Optimizations:
        - Initialize per-accumulator LLM mask once (complement of forbidden-from-terminals).
        - Hoist popped.peek() outside of destination loop to avoid redundant calls.
        - Use a single acc_memo per edge-group (pop, llm_bv) so that PyAcc transforms are reused.
        - Short-circuit isolate_many when all peeked states are kept for a destination.
        """
        # Reset global RangeSet stats/metrics for a fresh run
        reset_rangeset_stats()

        state_map: Dict[int, GSS] = self.state

        t0 = time.time()
        stats = collections.defaultdict(float)

        all_ones: Optional[RangeSet] = self.all_internal_llm_tokens_bitset
        final_mask: RangeSet = RangeSet.empty()

        # Additional histograms and peaks
        depth_hist = collections.Counter()       # depth -> nodes processed
        dest_len_hist = collections.Counter()    # len(dests) per edge-group -> count
        stats['values_peak'] = 0
        stats['depth_heap_peak'] = 0
        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask
        values: Dict[int, GSS] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, RangeSet]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        # Seed: Initialize llm_mask in each GSS, consume terminals_union, and enqueue roots.
        t_init_start = time.time()

        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask = RangeSet.empty()
            disallowed_map = acc.terminals_union
            stats['init_acc_calls'] += 1
            for tsid, disallowed_terminals in disallowed_map.items():
                stats['init_disallowed_terminals_total'] += len(disallowed_terminals)

            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    stats['init_pmc_misses'] += 1
                    continue
                stats['init_pmc_hits'] += 1
                terminals_to_llm = pmc[tsid]
                # Expand terminals just once per accumulator. Counts are small, this is fine.
                for terminal_id in disallowed_terminals.to_indices():
                    llm_mask = terminals_to_llm.get(terminal_id)
                    if llm_mask is not None:
                        disallowed_llm_mask = disallowed_llm_mask.union(llm_mask)

            if all_ones is not None:
                if disallowed_llm_mask.is_empty():
                    allowed_mask = all_ones
                else:
                    allowed_mask = all_ones.difference(disallowed_llm_mask)
            else:
                # Degenerate case: no known tokens -> empty mask
                allowed_mask = RangeSet.empty()

            if allowed_mask.is_empty():
                stats['init_allowed_mask_empty'] += 1
            return PyAcc(
                terminals_union={},  # consume
                llm_mask=allowed_mask,
            )

        apply_memo: Dict[PyAcc, PyAcc] = {}
        for sid, gss in state_map.items():
            r: int = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)
            if r in values:
                values[r] = values[r].merge(gss_initialized)
                stats['init_gss_merges'] += 1
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                bucket.add(r)
        stats['t_init'] = time.time() - t_init_start
        stats['init_apply_memo_size'] = len(apply_memo)
        stats['init_gss_count'] = len(state_map)

        def enqueue(d: int, n: int) -> None:
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                bucket.add(n)

        # Main loop
        t_main_loop_start = time.time()
        while depth_heap:
            depth: int = hpop(depth_heap)
            while todo[depth]:
                node: int = todo[depth].pop()
                gss_node: GSS = values.pop(node)
                stats['nodes_processed'] += 1
                depth_hist[depth] += 1
                stats['values_peak'] = max(stats['values_peak'], len(values))
                stats['depth_heap_peak'] = max(stats['depth_heap_peak'], len(depth_heap))

                # End-node handling: just union the allowed LLM tokens
                if is_end(node):
                    stats['end_nodes'] += 1
                    t_reduce_start = time.time()
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
                        stats['reduce_acc_nonempty'] += 1
                        final_mask = final_mask.union(reduced_acc.llm_mask)
                    stats['t_reduce_acc'] += time.time() - t_reduce_start
                    stats['reduce_acc_calls'] += 1

                # Traverse edges and propagate masks
                edges = arena.get(node, {}).get("children") or []
                stats['edges_traversed'] += len(edges)  # number of edge-groups

                for (pop, llm_bv), dests in edges:
                    stats['llm_bv_total_len'] += len(llm_bv)
                    dest_len_hist[len(dests)] += 1
                    stats['edges_destinations'] += len(dests)
                    stats['popn_calls'] += 1

                    # Pop once per edge-group
                    t_popn_start = time.time()
                    popped: GSS = gss_node.popn(pop)
                    stats['t_popn'] += time.time() - t_popn_start
                    if popped.is_empty():
                        stats['edges_pruned_empty_after_popn'] += 1
                        continue

                    # Hoist peek out of dest loop (critical optimization)
                    peek_payload = popped.peek()
                    # Normalize to a small, indexable Python list for fast iteration
                    # Try to avoid extra conversions if it is already a small list/tuple.
                    if hasattr(peek_payload, "to_indices"):
                        # It's likely a RangeSet; extract once
                        peeked_sids: List[int] = list(peek_payload.to_indices())
                    elif isinstance(peek_payload, (list, tuple)):
                        peeked_sids = list(peek_payload)
                    else:
                        # Fallback: materialize to a list
                        peeked_sids = list(peek_payload)

                    len_peeked = len(peeked_sids)
                    if len_peeked == 0:
                        stats['edges_pruned_by_state_filter'] += len(dests)
                        stats['dests_processed'] += len(dests)
                        continue

                    # Reuse the same acc-memo across all destinations of this edge-group.
                    acc_memo_group: Dict[PyAcc, Optional[PyAcc]] = {}

                    # Prepare the prune function once per edge-group (shares llm_bv and acc_memo).
                    def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                        stats['apply_prune_acc_visited'] += 1
                        cached = acc_memo_group.get(acc)
                        if cached is not None or acc in acc_memo_group:
                            stats['acc_memo_hits'] += 1
                            return cached
                        stats['acc_memo_misses'] += 1
                        new_mask = acc.llm_mask.intersection(llm_bv)
                        if new_mask.is_empty():
                            result = None
                            stats['apply_prune_acc_pruned'] += 1
                        else:
                            result = PyAcc(
                                terminals_union=acc.terminals_union,
                                llm_mask=new_mask
                            )
                        acc_memo_group[acc] = result
                        return result

                    for dest_idx, state_bv in dests:
                        stats['dests_processed'] += 1

                        # Filter peeked sids by state_bv using a tiny Python loop (len_peeked is usually small).
                        t_intersect_start = time.time()
                        contains = state_bv.contains  # bind for speed
                        stats['sid_candidates_total'] += len_peeked
                        # Build filtered list
                        if len_peeked == 1:
                            sid0 = peeked_sids[0]
                            values_to_keep = [sid0] if contains(sid0) else []
                        elif len_peeked == 2:
                            a, b = peeked_sids
                            tmp = []
                            if contains(a):
                                tmp.append(a)
                            if contains(b):
                                tmp.append(b)
                            values_to_keep = tmp
                        else:
                            # General case
                            values_to_keep = [sid for sid in peeked_sids if contains(sid)]

                        kept_len = len(values_to_keep)
                        stats['sid_kept_total'] += kept_len
                        stats['t_sid_intersection'] += time.time() - t_intersect_start

                        if kept_len == 0:
                            stats['edges_pruned_by_state_filter'] += 1
                            continue

                        # Isolate only if not all peeked sids are kept
                        stats['isolate_many_calls'] += 1
                        if kept_len == len_peeked:
                            child_gss: GSS = popped
                        else:
                            t_isolate_start = time.time()
                            child_gss = popped.isolate_many(values_to_keep)
                            stats['t_isolate_many'] += time.time() - t_isolate_start

                        if child_gss.is_empty():
                            stats['edges_pruned_empty_after_isolate'] += 1
                            continue

                        # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv (reusing acc memo)
                        stats['apply_prune_calls'] += 1
                        t_apply_prune_start = time.time()
                        child_gss = child_gss.apply_and_prune(intersect_and_prune)
                        stats['acc_memo_total_entries'] += len(acc_memo_group)
                        stats['acc_memo_max_size'] = max(stats.get('acc_memo_max_size', 0), len(acc_memo_group))
                        stats['t_apply_prune'] += time.time() - t_apply_prune_start
                        if child_gss.is_empty():
                            stats['edges_pruned_after_apply'] += 1
                            continue

                        d: int = int(dest_idx)
                        if d in values:
                            stats['gss_merges'] += 1
                            t_merge_start = time.time()
                            values[d] = values[d].merge(child_gss)
                            stats['t_gss_merge'] += time.time() - t_merge_start
                        else:
                            values[d] = child_gss
                        stats['edges_enqueued'] += 1
                        enqueue(max_depth[d], d)

            todo.pop(depth)
        stats['t_main_loop'] = time.time() - t_main_loop_start

        # Convert internal mask back to original IDs
        t_final_convert_start = time.time()
        original_indices: List[int] = []
        stats['final_mask_internal_len'] = len(final_mask)
        for i in final_mask.to_indices():
            mapped = self.internal_to_original_map.get(i)
            if mapped is not None:
                original_indices.append(mapped)
        stats['t_final_convert'] = time.time() - t_final_convert_start
        stats['final_mask_original_len'] = len(original_indices)

        t_total = time.time() - t0
        stats['t_total'] = t_total
        print("\n--- get_mask stats ---")
        for k, v in sorted(stats.items()):
            if k.startswith('t_'):
                if t_total > 1e-6:
                    print(f"{k:<25}: {v:8.4f}s ({v/t_total*100:5.1f}%)")
                else:
                    print(f"{k:<25}: {v:8.4f}s")
            else:
                print(f"{k:<25}: {int(v)}")
        print("----------------------\n")

        # Derived metrics and histograms for deeper insight
        print("\n--- get_mask derived metrics ---")
        if stats.get('popn_calls', 0):
            pr_empty_popn = stats.get('edges_pruned_empty_after_popn', 0) / max(1, stats['popn_calls'])
            print(f"pruned_after_popn_rate       : {pr_empty_popn:8.4f}")
        if stats.get('dests_processed', 0):
            pr_after_apply = stats.get('edges_pruned_after_apply', 0) / max(1, stats['dests_processed'])
            pr_by_state    = stats.get('edges_pruned_by_state_filter', 0) / max(1, stats['dests_processed'])
            pr_empty_iso   = stats.get('edges_pruned_empty_after_isolate', 0) / max(1, stats['dests_processed'])
            print(f"pruned_after_apply_rate      : {pr_after_apply:8.4f}")
            print(f"pruned_by_state_filter_rate  : {pr_by_state:8.4f}")
            print(f"pruned_empty_after_isolate   : {pr_empty_iso:8.4f}")
        if stats.get('sid_candidates_total', 0):
            keep_rate = stats.get('sid_kept_total', 0) / max(1, stats['sid_candidates_total'])
            print(f"sid_keep_rate                : {keep_rate:8.4f}")
        if stats.get('apply_prune_acc_visited', 0):
            acc_prune_rate = stats.get('apply_prune_acc_pruned', 0) / max(1, stats['apply_prune_acc_visited'])
            memo_hit_rate  = stats.get('acc_memo_hits', 0) / max(1, stats['acc_memo_hits'] + stats.get('acc_memo_misses', 0))
            print(f"apply_prune_acc_prune_rate   : {acc_prune_rate:8.4f}")
            print(f"acc_memo_hit_rate            : {memo_hit_rate:8.4f}")
        if stats.get('edges_traversed', 0):
            avg_dests_per_group = stats.get('edges_destinations', 0) / max(1, stats['edges_traversed'])
            print(f"avg_dests_per_edge_group     : {avg_dests_per_group:8.4f}")
        if stats.get('edges_traversed', 0):
            avg_llm_bv_len = stats.get('llm_bv_total_len', 0) / max(1, stats['edges_traversed'])
            print(f"avg_llm_bv_cardinality       : {avg_llm_bv_len:8.4f}")
        print(f"values_map_peak_size         : {int(stats.get('values_peak', 0))}")
        print(f"depth_heap_peak_size         : {int(stats.get('depth_heap_peak', 0))}")
        print("-------------------------------\n")

        # Histograms
        print("--- get_mask distributions ---")
        if depth_hist:
            print("Depth histogram (nodes processed):")
            for d, cnt in sorted(depth_hist.items()):
                print(f"  depth[{d}]: {cnt}")
        if dest_len_hist:
            print("Edge-group destination count histogram:")
            for ln, cnt in sorted(dest_len_hist.items()):
                print(f"  dests_len[{ln}]: {cnt}")
        print("-------------------------------\n")

        print_rangeset_stats()
        return RangeSet.from_indices(original_indices)
