import json
from typing import Dict, List, Tuple, Optional
import time
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from tqdm.auto import tqdm

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.roots_map = {int(s): int(r) for s, r in roots_map}
        self.arena = arena

        # Pre-compute frequently accessed data
        self.max_depth = []
        self.is_end_node = []
        self.node_children = []
        self.node_id_to_idx = {}

        # First pass: create index mapping
        for uid in arena.keys():
            uid_int = int(uid)
            self.node_id_to_idx[uid_int] = len(self.node_id_to_idx)

        # Allocate arrays
        num_nodes = len(self.node_id_to_idx)
        self.max_depth = [0] * num_nodes
        self.is_end_node = [False] * num_nodes
        self.node_children = [None] * num_nodes

        # Second pass: populate arrays and normalize BVs
        for uid, node in tqdm(arena.items(), desc="Normalizing precompute3 BVs", total=len(arena)):
            uid_int = int(uid)
            idx = self.node_id_to_idx[uid_int]

            # Store max_depth
            self.max_depth[idx] = int(node.get("max_depth", 0))

            # Pre-compute is_end
            self.is_end_node[idx] = bool((node.get("value") or {}).get("clean_end", False))

            # Process and store children
            ch = node.get("children") or []
            if ch:
                processed_children = []
                for edge_key, dest_map in ch:
                    pop, llm_bv_json = edge_key
                    llm_bv = ffi.Bitset.from_json_string(json.dumps(llm_bv_json))

                    processed_dests = []
                    for dest_idx, state_bv_json in dest_map:
                        state_bv = ffi.Bitset.from_json_string(json.dumps(state_bv_json))
                        dest_idx_mapped = self.node_id_to_idx[int(dest_idx)]
                        processed_dests.append((dest_idx_mapped, state_bv))

                    processed_children.append(((int(pop), llm_bv), processed_dests))

                self.node_children[idx] = processed_children
                node["children"] = processed_children
            else:
                self.node_children[idx] = []
                node["children"] = []

        # Pre-compute depth buckets for faster scheduling
        self.depth_buckets = {}
        for idx in range(num_nodes):
            depth = self.max_depth[idx]
            if depth not in self.depth_buckets:
                self.depth_buckets[depth] = []
            self.depth_buckets[depth].append(idx)
        self.sorted_depths = sorted(self.depth_buckets.keys())

    @staticmethod
    def from_json_string(s: str) -> "Model":
        data = json.loads(s)
        roots_map = data['precomputed3']
        arena_json = data['trie3_god']
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        idx = self.node_id_to_idx.get(node)
        return self.is_end_node[idx] if idx is not None else False

    def iter_edges(self, node: int, token: int):
        idx = self.node_id_to_idx.get(node)
        if idx is None:
            return

        for (pop, llm_bv), dests in self.node_children[idx]:
            if llm_bv.contains(token):
                for dest_idx_mapped, state_bv in dests:
                    # Convert back to original node ID for interface
                    dest_node = None
                    for orig_id, mapped_idx in self.node_id_to_idx.items():
                        if mapped_idx == dest_idx_mapped:
                            dest_node = orig_id
                            break

                    if dest_node is not None:
                        if state_bv.is_empty():
                            yield (int(pop), None, dest_node)
                        else:
                            for start, end in state_bv.to_ranges():
                                for sid in range(start, end):
                                    yield (int(pop), sid, dest_node)

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        print("\n--- get_mask START ---")
        print(self.constraint_state)
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        print(f"Filtered state_to_gss: { {k: v.ptr() for k, v in state_to_gss.items()} }")

        t0 = time.time()
        final_mask = ffi.Bitset.zeros()

        # Use arrays instead of dicts where possible
        num_nodes = len(self.node_id_to_idx)

        # Pre-allocate arrays for node state
        node_gss_nodes = [None] * num_nodes
        node_masks = [None] * num_nodes
        node_active = [False] * num_nodes
        node_stopped = [False] * num_nodes

        # Seed initial nodes
        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_id = self.roots_map.get(int(sid))
            if root_id is None:
                continue

            root_idx = self.node_id_to_idx.get(root_id)
            if root_idx is None:
                continue

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            if node_gss_nodes[root_idx] is None:
                node_gss_nodes[root_idx] = gss_clone
                node_masks[root_idx] = new_mask
            else:
                existing_gss = node_gss_nodes[root_idx]
                existing_mask = node_masks[root_idx]
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                node_gss_nodes[root_idx] = ffi.gss_merge_many_with_depth([node_gss_nodes[root_idx], gss_clone], 1)
                node_masks[root_idx] = node_masks[root_idx].union(new_mask)
                print(f"      - Merged result: gss_ptr={node_gss_nodes[root_idx].ptr()}, mask={node_masks[root_idx].to_ranges()}")

            node_active[root_idx] = True

        # Process by depth (but using pre-computed buckets)
        print("\n--- Main loop ---")
        iter_count = 0
        for depth in self.sorted_depths:
            iter_count += 1
            has_active = False

            # Check if any nodes at this depth are active
            for idx in self.depth_buckets[depth]:
                if node_active[idx]:
                    has_active = True
                    break

            print(f"\n[{iter_count}] Processing depth={depth}, active={has_active}")
            if not has_active:
                continue

            # Process all active nodes at this depth
            for idx in self.depth_buckets[depth]:
                if not node_active[idx] or node_stopped[idx]:
                    if node_active[idx]:
                        print(f"  - Node {idx}: SKIPPING (already stopped)")
                    continue

                gss_node = node_gss_nodes[idx]
                llm_mask = node_masks[idx]

                if gss_node is None:
                    continue
                print(f"  - PROCESS: node_ptr={idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                # Clear this node's state after processing
                node_active[idx] = False
                node_gss_nodes[idx] = None
                node_masks[idx] = None

                # Check if end node
                if self.is_end_node[idx]:
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")
                    gss_active_tokens = gss_node.allowed_llm_tokens()
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if not gss_node.is_alive():
                    node_stopped[idx] = True
                    print(f"    - STOPPING node {idx} (GSS not alive)")
                    continue

                # Process children
                children = self.node_children[idx]
                if not children:
                    continue

                for (pop, llm_bv), dests in children:
                    print(f"    - Edge: pop={pop}, llm_bv={llm_bv.to_ranges()}")
                    # Batch collect all popn results
                    all_peeks = gss_node.popn_fast(pop)

                    if not all_peeks:
                        continue

                    # Process destinations
                    for dest_idx, state_bv in dests:
                        print(f"      - Dest: idx={dest_idx}, state_bv={state_bv.to_ranges()}")
                        # Fast path for empty state_bv
                        if state_bv.is_empty():
                            continue

                        # Filter peeks by state bitset
                        matched_parents = []
                        for sid_val, parent_node in all_peeks:
                            if state_bv.contains(sid_val) and parent_node:
                                matched_parents.append(parent_node)

                        print(f"        - Matched {len(matched_parents)} parent GSS nodes")
                        if not matched_parents:
                            continue

                        # Calculate child mask
                        child_llm_mask = llm_mask
                        if not llm_bv.is_empty():
                            child_llm_mask = child_llm_mask.intersection(llm_bv)
                        print(f"        - Child mask: {child_llm_mask.to_ranges()}")

                        child_gss = ffi.gss_merge_many_with_depth(matched_parents, 1)
                        if not child_gss.is_alive():
                            continue

                        # Update destination node
                        if node_gss_nodes[dest_idx] is None:
                            node_gss_nodes[dest_idx] = child_gss
                            node_masks[dest_idx] = child_llm_mask
                            print(f"        - Enqueue {dest_idx}: CREATING gss_ptr={child_gss.ptr()}, mask={child_llm_mask.to_ranges()}")
                        else:
                            # Merge with existing
                            existing_gss = node_gss_nodes[dest_idx]
                            existing_mask = node_masks[dest_idx]
                            print(f"        - Enqueue {dest_idx}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={child_llm_mask.to_ranges()}")
                            node_gss_nodes[dest_idx] = ffi.gss_merge_many_with_depth([node_gss_nodes[dest_idx], child_gss], 1)
                            node_masks[dest_idx] = node_masks[dest_idx].union(child_llm_mask)
                            print(f"          - Merged result: gss_ptr={node_gss_nodes[dest_idx].ptr()}, mask={node_masks[dest_idx].to_ranges()}")

                        node_active[dest_idx] = True

        print(f"\n--- get_mask END (took {time.time() - t0:.4f}s) ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
