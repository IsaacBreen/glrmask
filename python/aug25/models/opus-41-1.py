import json
from typing import Dict, List, Tuple, Optional
import time
from ..common_interface import GraphProvider
import _sep1 as ffi
from tqdm.auto import tqdm

class Model(GraphProvider):
    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
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
            self.is_end_node[idx] = bool((node.get("value") or {}).get("end", False))

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
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data['precomputed3']
        arena_json = data['trie3_god']
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        return Model(roots_map, arena)

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

    def get_mask(self, state_to_gss: Dict[int, ffi.GSSNode]) -> ffi.Bitset:
        final_mask = ffi.Bitset.zeros()

        # Use arrays instead of dicts where possible
        num_nodes = len(self.node_id_to_idx)

        # Pre-allocate arrays for node state
        node_gss_lists = [None] * num_nodes
        node_masks = [None] * num_nodes
        node_active = [False] * num_nodes
        node_stopped = [False] * num_nodes

        # Seed initial nodes
        for sid, gss in state_to_gss.items():
            root_id = self.roots_map.get(int(sid))
            if root_id is None:
                continue

            root_idx = self.node_id_to_idx.get(root_id)
            if root_idx is None:
                continue

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()

            if node_gss_lists[root_idx] is None:
                node_gss_lists[root_idx] = [gss_clone]
                node_masks[root_idx] = new_mask
            else:
                node_gss_lists[root_idx].append(gss_clone)
                node_masks[root_idx] = node_masks[root_idx].union(new_mask)

            node_active[root_idx] = True

        # Process by depth (but using pre-computed buckets)
        for depth in self.sorted_depths:
            has_active = False

            # Check if any nodes at this depth are active
            for idx in self.depth_buckets[depth]:
                if node_active[idx]:
                    has_active = True
                    break

            if not has_active:
                continue

            # Process all active nodes at this depth
            for idx in self.depth_buckets[depth]:
                if not node_active[idx] or node_stopped[idx]:
                    continue

                gss_list = node_gss_lists[idx]
                llm_mask = node_masks[idx]

                if gss_list is None:
                    continue

                # Clear this node's state after processing
                node_active[idx] = False
                node_gss_lists[idx] = None
                node_masks[idx] = None

                # Check if end node
                if self.is_end_node[idx]:
                    final_mask = final_mask.union(llm_mask)

                # Filter GSS nodes
                ok_gss = [g for g in gss_list if g]
                if not ok_gss:
                    node_stopped[idx] = True
                    continue

                # Process children
                children = self.node_children[idx]
                if not children:
                    continue

                for (pop, llm_bv), dests in children:
                    # Batch collect all popn results
                    all_peeks = []
                    for gss_node in ok_gss:
                        all_peeks.extend(gss_node.popn_fast(pop))

                    if not all_peeks:
                        continue

                    # Process destinations
                    for dest_idx, state_bv in dests:
                        # Fast path for empty state_bv
                        if state_bv.is_empty():
                            continue

                        # Filter peeks by state bitset
                        matched_parents = []
                        for sid_val, parent_node in all_peeks:
                            if state_bv.contains(sid_val) and parent_node:
                                matched_parents.append(parent_node)

                        if not matched_parents:
                            continue

                        # Calculate child mask
                        child_llm_mask = llm_mask
                        if not llm_bv.is_empty():
                            child_llm_mask = child_llm_mask.intersection(llm_bv)

                        # Update destination node
                        if node_gss_lists[dest_idx] is None:
                            node_gss_lists[dest_idx] = matched_parents[:]
                            node_masks[dest_idx] = child_llm_mask
                        else:
                            # Merge with existing
                            existing_set = set(id(g) for g in node_gss_lists[dest_idx])
                            new_parents = [p for p in matched_parents if id(p) not in existing_set]
                            if new_parents:
                                node_gss_lists[dest_idx].extend(new_parents)
                            node_masks[dest_idx] = node_masks[dest_idx].union(child_llm_mask)

                        node_active[dest_idx] = True

        return final_mask