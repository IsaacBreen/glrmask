import heapq
from typing import Dict, List, Set, Optional, Tuple

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
# from python.gss_tester.implementations.leveled_impl_cpp import Leveled_impl_cppGSS as GSS
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

        all_ones: Optional[RangeSet] = self.all_internal_llm_tokens_bitset
        final_mask: RangeSet = RangeSet.empty()

        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask
        values: Dict[int, GSS] = {}
        depth_heap: List[Tuple[int, int]] = []  # Stores (-depth, node_id)
        enqueued_nodes: Set = set()

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, RangeSet]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        # Seed: Initialize llm_mask in each GSS, consume terminals_union, and enqueue roots.
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask = RangeSet.empty()
            disallowed_map = acc.terminals_union

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
                terminals_union={},  # consume
                llm_mask=allowed_mask,
            )

        for sid, gss in state_map.items():
            r: int = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc)
            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            if r not in enqueued_nodes:
                enqueued_nodes.add(r)
                hp(depth_heap, (-d, r))

        def enqueue(d: int, n: int) -> None:
            if n not in enqueued_nodes:
                enqueued_nodes.add(n)
                hp(depth_heap, (-d, n))

        # Main loop
        while depth_heap:
            neg_depth, node = hpop(depth_heap)
            gss_node: GSS = values.pop(node)
            enqueued_nodes.remove(node)

            # End-node handling: just union the allowed LLM tokens
            if is_end(node):
                reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                if reduced_acc:
                    final_mask = final_mask.union(reduced_acc.llm_mask)

            # Zombie traversal avoidance
            node_llm_bv_union = arena.get(node, {}).get("llm_bv_union", RangeSet.empty())
            potential_new_tokens = node_llm_bv_union.difference(final_mask)
            if potential_new_tokens.is_empty():
                continue

            gss_mask_acc = gss_node.reduce_acc()
            if gss_mask_acc and gss_mask_acc.llm_mask.intersection(potential_new_tokens).is_empty():
                continue

            # Traverse edges and propagate masks
            edges = arena.get(node, {}).get("children") or []
            for (pop, llm_bv), dests in edges:
                llm_bv = llm_bv.difference(final_mask)
                if llm_bv.is_empty():
                    continue

                popped: GSS = gss_node.popn(pop)
                if popped.is_empty():
                    continue

                # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv
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

                popped = popped.apply_and_prune(intersect_and_prune)
                if popped.is_empty():
                    continue

                reduced = popped.reduce_acc()
                if not reduced or reduced.is_empty():
                    continue

                for dest_idx, state_bv in dests:
                    peeked = popped.peek()
                    values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]

                    if not values_to_keep:
                        continue

                    child_gss: GSS = popped.isolate_many(values_to_keep)
                    if child_gss.is_empty():
                        continue

                    reduced_child = child_gss.reduce_acc()
                    if not reduced_child or reduced_child.is_empty():
                        continue

                    d: int = int(dest_idx)
                    if d in values:
                        values[d] = values[d].merge(child_gss)
                    else:
                        values[d] = child_gss
                    enqueue(max_depth[d], d)


        # Convert internal mask back to original IDs
        original_indices: List[int] = []
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_indices.append(self.internal_to_original_map[i])

        return RangeSet.from_indices(original_indices)