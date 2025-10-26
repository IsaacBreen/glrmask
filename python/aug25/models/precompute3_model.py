import json
import os
import time
import heapq
import collections
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm

from python.gss_tester.implementations.reference_impl import ReferenceGSS as GSS


@dataclass(frozen=True, eq=False)
class PyAcc:
    llm_mask: ffi.Bitset
    terminals: ffi.HybridL2Bitset

    def merge(self, other: 'PyAcc') -> 'PyAcc':
        return PyAcc(
            llm_mask=self.llm_mask.union(other.llm_mask),
            terminals=self.terminals.union(other.terminals),
        )


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation).
    Normalizes input arena by converting JSON bitsets into ffi.Bitset instances
    and provides graph traversal and mask computation interfaces.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.id_to_token: Dict[int, bytes] = {}
        self.max_depth: Dict[int, int] = {}
        self.internal_to_original_map: Dict[int, List[int]] = {}
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = None
        self.debug_logging = os.environ.get("RUST_LOG") == "debug"

        # Normalize arena children bitsets and cache max_depth
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        for uid, node in tqdm(
            self.arena.items(),
            desc="Normalizing precompute3 BVs",
            total=len(self.arena),
        ):
            uid_int = int(uid)
            try:
                md = node.get("max_depth", 0)
                self.max_depth[uid_int] = int(md)
            except Exception:
                self.max_depth[uid_int] = 0

            children = node.get("children") or []
            if not children:
                node["children"] = []
                continue

            new_children = []
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv = bs_from_json(dumps(llm_bv_json))

                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv = bs_from_json(dumps(state_bv_json))
                    new_dest_map.append((int(dest_idx), state_bv))

                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children

    @staticmethod
    def gss_from_ffi_node(gss_node: ffi.GSSNode) -> GSS:
        flat = gss_node.flatten()
        stacks = []
        for stack, (llm_mask, terminals) in flat:
            stacks.append((stack, PyAcc(llm_mask, terminals)))
        return GSS.from_stacks(stacks)

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        if model.debug_logging:
            print(model.constraint.dump_precomputed3())
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        model.possible_matches_cache = model.constraint.possible_matches()
        if model.debug_logging:
            print("model.possible_matches_cache", model.possible_matches_cache)

        vocab = data.get('precompute3_vocab') or data.get('precompute2_vocab') or data.get('precompute_vocab')
        if vocab:
            model.internal_to_original_map = {int(k): v for k, v in vocab['internal_to_original']}
        else:
            # Fallback for old format: one-to-one mapping
            i2o_map_one_to_one = model.constraint.internal_to_original_map()
            model.internal_to_original_map = {k: [v] for k, v in i2o_map_one_to_one.items()}

        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        Only used by equivalence checking; not performance-critical.
        """
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():  # Epsilon on GSS stack
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end + 1):
                                yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        state_map: Dict[int, ffi.GSSNode] = self.constraint_state.get_state_map()

        if self.debug_logging:
            print("states in get_mask:")
            for k, v in state_map.items():
                print(f"state {k}: gss_ptr={v.ptr()} flat={self.gss_from_ffi_node(v)}")


        # t0 = time.time()

        final_mask = ffi.Bitset.zeros()

        # node_idx -> GSSNode
        values: Dict[int, ffi.GSSNode] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)


        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        if self.debug_logging:
            print("\n--- Seeding work queue ---")
        for sid, gss in state_map.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)
            
            if self.debug_logging:
                print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss.ptr()}")

            gss = gss.clone_node()
            ffi.gss_prune_llm_tokens_by_disallowed_terminals(gss, self.possible_matches_cache)

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss = existing
                if self.debug_logging:
                    print(f"  - MERGING into root {root_idx}:")
                    print(f"    - Existing GSS: ptr={existing_gss.ptr()} flat={self.gss_from_ffi_node(existing_gss)}")
                    print(f"    - New GSS:      ptr={gss.ptr()} flat={self.gss_from_ffi_node(gss)}")

                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss], 1)
                values[root_idx] = merged_gss
            else:
                values[root_idx] = gss

            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        if self.debug_logging:
            print("--- After Seeding ---")
            for node_idx, gss in values.items():
                print(f"Node {node_idx}: gss_ptr={gss.ptr()} flat={self.gss_from_ffi_node(gss)}")


        # Main scheduler

        # Helper to enqueue a node at a given depth
        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        heappop = heapq.heappop
        arena = self.arena
        is_end = self.is_end

        if self.debug_logging:
            print("\n--- Main loop ---")
        iter_count = 0
        while True:
            iter_count += 1
            # Pop the smallest depth bucket (skip stale heap entries)
            node_indices: Optional[set[int]] = None
            current_depth = -1
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                if self.debug_logging:
                    print(f"[{iter_count}] Loop finished: no more nodes to process.")
                break  # nothing left to process

            if self.debug_logging:
                print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")

            # Process all nodes in this depth bucket
            for node_idx in node_indices:
                if node_idx in stopped:
                    if self.debug_logging:
                        print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                    continue

                gss_node = values.pop(node_idx)
                if gss_node is None:
                    if self.debug_logging:
                        print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                if self.debug_logging:
                    print(f"  - Node {node_idx}: Popped gss_ptr={gss_node.ptr()} flat={self.gss_from_ffi_node(gss_node)}")

                # End-node handling
                if is_end(node_idx):
                    if self.debug_logging:
                        print(f"    - END NODE found")
                        print(f"      - final_mask before: {final_mask.to_ranges()}")
                        print(self.constraint.state_with_nodes([(0, gss_node)]))

                    # Clone the GSS node to avoid modifying it in place if it's shared
                    gss_node_copy = gss_node.clone_node()

                    # Get the final allowed tokens from the pruned GSS
                    final_allowed_tokens = gss_node_copy.allowed_llm_tokens()

                    if self.debug_logging:
                        print(f"Allowed LLM tokens from pruned GSS: {final_allowed_tokens.to_ranges()}")

                    final_mask = final_mask.union(final_allowed_tokens)
                    if self.debug_logging:
                        print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if not gss_node.is_alive():
                    stopped.add(node_idx)
                    if self.debug_logging:
                        print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                if not children:
                    if self.debug_logging:
                        print(f"    - No children for node {node_idx}")
                for (pop, llm_bv), dests in children:
                    if self.debug_logging:
                        print(f"    - Edge: pop={pop}, llm_bv={llm_bv.to_ranges()}")
                    # Collect all pops from GSS parents
                    peeks = ffi.gss_popn_collect(gss_node, pop)
                    if self.debug_logging:
                        print(f"      - Found {len(peeks)} peeks from GSS set")
                    if not peeks:
                        continue

                    for dest_idx, state_bv in dests:
                        if self.debug_logging:
                            print(f"      - Dest: idx={dest_idx}, state_bv={state_bv.to_ranges()}")
                        # Filter peeks by destination state bitset
                        matched = []
                        if not state_bv.is_empty():
                            contains = state_bv.contains
                            for sid_val, parent_node in peeks:
                                if contains(sid_val):
                                    if self.debug_logging:
                                        print(f"        - Matched parent state {sid_val} in dest state_bv, node={self.gss_from_ffi_node(parent_node)}")
                                    matched.append(parent_node)
                        if not matched:
                            if self.debug_logging:
                                print(f"        - No matched parent GSS nodes")
                            continue
                        if self.debug_logging:
                            print(f"        - Matched {len(matched)} parent GSS nodes")

                        # Merge matched parent GSS nodes
                        child_gss_node = ffi.gss_merge_many_with_depth(matched, 1)
                        if self.debug_logging:
                            print(f"        - Child GSS: ptr={child_gss_node.ptr()} flat={self.gss_from_ffi_node(child_gss_node)}")

                        # Apply edge's LLM token mask to the new GSS node
                        ffi.gss_allow_only_llm_tokens_and_prune(child_gss_node, llm_bv)

                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss = existing
                            merged_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss_node], 1)
                            values[d] = merged_gss
                            if self.debug_logging:
                                print(f"        - MERGING with {self.gss_from_ffi_node(existing_gss)}\n...and {self.gss_from_ffi_node(child_gss_node)}")
                                print(f"        - Merged GSS: ptr={merged_gss.ptr()} flat={self.gss_from_ffi_node(merged_gss)}")
                                print(f"        - Full structure: Existing: {self.constraint.state_with_nodes([(0, existing_gss)])}")
                                print(f"                          New:      {self.constraint.state_with_nodes([(0, child_gss_node)])}")
                                print(f"        - Full structure: Merged:   {self.constraint.state_with_nodes([(0, merged_gss)])}")
                                print(f"        - Enqueue {d}: UPDATING gss_ptr={merged_gss.ptr()}")
                        else:
                            values[d] = child_gss_node
                            if self.debug_logging:
                                print(f"        - Enqueue new node {self.gss_from_ffi_node(child_gss_node)} at idx {d}")
                                print(f"        - Full structure: {self.constraint.state_with_nodes([(0, child_gss_node)])}")

                        enqueue(max_depth[d], d)

        if self.debug_logging:
            print("final internal mask:", final_mask.to_ranges())

        # original_mask = ffi.Bitset.zeros()
        original_mask = set()
        for i in final_mask.to_indices():
            original_mask.update(self.internal_to_original_map[i])

        x = RangeSet.from_indices(original_mask)
        if self.debug_logging:
            print("original mask:", x)
        return x
