import heapq
from typing import Dict, List, Set, Optional
import time
import collections

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from .precompute3_model_pure_python import Model as InnerModel, PyAcc
from ..common_interface import GraphProvider, RangeSet


class Model(GraphProvider):
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

        Changes for get_mask_only:
        - Initialize a per-accumulator LLM mask (PyAcc.llm_mask) BEFORE traversal by computing
          the forbidden terminals -> forbidden LLM tokens and taking the complement.
        - Consume terminals_union (set to HybridL2Bitset.all()) after initialization.
        - As we traverse edges, intersect llm_mask with the edge's LLM bitset using apply.
        - At end nodes, simply reduce acc over the GSS and union the llm_mask into the final.
        """
        state_map: Dict[int, GSS] = self.state

        t0 = time.time()
        stats = collections.defaultdict(float)

        all_ones: Optional[RangeSet] = self.all_internal_llm_tokens_bitset
        final_mask: RangeSet = RangeSet.empty()

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
            disallowed_map = dict(acc.terminals_union)

            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]
                for terminal_id in disallowed_terminals.to_indices():
                    if terminal_id in terminals_to_llm:
                        disallowed_llm_mask = disallowed_llm_mask.union(
                            terminals_to_llm[terminal_id]
                        )

            allowed_mask = (all_ones if all_ones is not None else RangeSet.empty()).difference(disallowed_llm_mask)
            return PyAcc(
                terminals_union=tuple(),  # consume
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

                # End-node handling: just union the allowed LLM tokens
                if is_end(node):
                    stats['end_nodes'] += 1
                    t_reduce_start = time.time()
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
                        final_mask = final_mask.union(reduced_acc.llm_mask)
                    stats['t_reduce_acc'] += time.time() - t_reduce_start
                    stats['reduce_acc_calls'] += 1

                # Traverse edges and propagate masks
                edges = arena.get(node, {}).get("children") or []
                stats['edges_traversed'] += len(edges)
                for (pop, llm_bv), dests in edges:
                    stats['popn_calls'] += 1
                    t_popn_start = time.time()
                    popped: GSS = gss_node.popn(pop)
                    stats['t_popn'] += time.time() - t_popn_start
                    if popped.is_empty():
                        continue

                    for dest_idx, state_bv in dests:
                        stats['dests_processed'] += 1
                        if state_bv.is_empty():
                            continue

                        # values_to_keep = [s for s in popped.peek() if state_bv.contains(s)]
                        t_intersect_start = time.time()
                        sid_vals = RangeSet.from_indices(popped.peek())
                        values_to_keep_rs = sid_vals.intersection(state_bv)
                        stats['t_sid_intersection'] += time.time() - t_intersect_start
                        values_to_keep = values_to_keep_rs.to_indices()

                        if not values_to_keep:
                            continue

                        stats['isolate_many_calls'] += 1
                        t_isolate_start = time.time()
                        child_gss: GSS = popped.isolate_many(values_to_keep)
                        stats['t_isolate_many'] += time.time() - t_isolate_start
                        if child_gss.is_empty():
                            continue

                        # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv
                        if not llm_bv.is_empty():
                            stats['apply_prune_calls'] += 1
                            t_apply_prune_start = time.time()
                            acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}

                            def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                                if acc in acc_memo:
                                    return acc_memo[acc]
                                new_mask = acc.llm_mask.intersection(llm_bv)
                                if new_mask.is_empty():
                                    result = None
                                else:
                                    result = PyAcc(
                                        terminals_union=acc.terminals_union,
                                        llm_mask=new_mask
                                    )
                                acc_memo[acc] = result
                                return result

                            child_gss = child_gss.apply_and_prune(intersect_and_prune)
                            stats['t_apply_prune'] += time.time() - t_apply_prune_start
                            if child_gss.is_empty():
                                continue

                        d: int = int(dest_idx)
                        if d in values:
                            stats['gss_merges'] += 1
                            t_merge_start = time.time()
                            existing_gss = values[d]
                            new_gss = child_gss
                            merged_gss = existing_gss.merge(new_gss)
                            values[d] = merged_gss
                            stats['t_gss_merge'] += time.time() - t_merge_start
                        else:
                            values[d] = child_gss
                        enqueue(max_depth[d], d)

            todo.pop(depth)
        stats['t_main_loop'] = time.time() - t_main_loop_start


        # Convert internal mask back to original IDs
        t_final_convert_start = time.time()
        original_indices: List[int] = []
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_indices.append(self.internal_to_original_map[i])
        stats['t_final_convert'] = time.time() - t_final_convert_start

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
        return RangeSet.from_indices(original_indices)
