import json
import time
import heapq
import collections
from typing import Dict, List, Tuple, Optional

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm


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
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
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
        print("\n--- get_mask START ---")
        print("GSS at start of get_mask:")
        print(self.constraint_state)
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        print(f"Filtered state_to_gss: { {k: v.ptr() for k, v in state_to_gss.items()} }")
        t0 = time.time()

        final_mask = ffi.Bitset.zeros()

        # node_idx -> (GSSNode, Bitset)
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)


        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss_clone], 1)
                values[root_idx] = (merged_gss, existing_mask.union(new_mask))
            else:
                values[root_idx] = (gss_clone, new_mask)

            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

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
                print(f"[{iter_count}] Loop finished: no more nodes to process.")
                break  # nothing left to process

            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")

            # Process all nodes in this depth bucket
            for node_idx in node_indices:
                if node_idx in stopped:
                    print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                gss_node, llm_mask = item
                print(f"  - Node {node_idx}: Popped gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                # End-node handling
                if is_end(node_idx):
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")

                    # Correct logic: intersect propagated mask with GSS active tokens
                    gss_active_tokens = gss_node.allowed_llm_tokens()

                    tokens_to_add = llm_mask.intersection(gss_active_tokens)

                    print(f"      - llm_mask (propagated): {llm_mask.to_ranges()}")
                    print(f"      - gss_active_tokens (from GSS): {gss_active_tokens.to_ranges()}")
                    print(f"      - tokens_to_add (intersection): {tokens_to_add.to_ranges()}")

                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if not gss_node.is_alive():
                    stopped.add(node_idx)
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                # if not children:
                #     print(f"    - No children for node {node_idx}")
                for (pop, llm_bv), dests in children:
                    print(f"    - Edge: pop={pop}, llm_bv={llm_bv.to_ranges()}")
                    # Collect all pops from GSS parents
                    peeks = gss_node.popn_fast(pop)
                    print(f"      - Found {len(peeks)} peeks from GSS set")
                    if not peeks:
                        continue

                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        print(f"      - Dest: idx={dest_idx}, state_bv={state_bv.to_ranges()}")
                        # Filter peeks by destination state bitset
                        matched = []
                        if not state_bv.is_empty():
                            contains = state_bv.contains
                            for sid_val, parent_node in peeks:
                                if contains(sid_val):
                                    matched.append(parent_node)
                        print(f"        - Matched {len(matched)} parent GSS nodes")
                        if not matched:
                            continue

                        # Merge matched parent GSS nodes
                        child_gss_node = ffi.gss_merge_many_with_depth(matched, 1)

                        # Compute child mask (intersection with llm_bv when present)
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)
                        print(f"        - Child mask: {child_llm_mask.to_ranges()}")

                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            merged_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss_node], 1)
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (merged_gss, combined_mask)
                            print(f"        - Enqueue {d}: UPDATING gss_ptr={merged_gss.ptr()}, mask={combined_mask.to_ranges()}")
                        else:
                            values[d] = (child_gss_node, child_llm_mask)
                            print(f"        - Enqueue {d}: CREATING gss_ptr={child_gss_node.ptr()}, mask={child_llm_mask.to_ranges()}")

                        enqueue(max_depth[d], d)

        original_mask = self.constraint.internal_bv_to_original(final_mask)
        return RangeSet.from_ranges(original_mask.to_ranges())
