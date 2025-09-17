import json
import heapq
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module


@dataclass
class Edge:
    """A more structured way to hold pre-processed edge information."""
    pop: int
    llm_mask: ffi.Bitset
    llm_rs: RangeSet  # Kept for iter_edges validation
    destinations: List[Tuple[int, List[Tuple[int, int]]]]  # (dest_idx, state_bv)


class Model(GraphProvider):
    """
    An optimized model for the precomputed graph.

    This model introduces several performance enhancements over the original implementation:

    1.  **Data Pre-computation**: During initialization, the arena data is converted into more
        efficient structures. `is_end` checks become O(1) set lookups. Edge information is
        stored in dataclasses, and `ffi.Bitset` masks for LLM tokens are created upfront,
        avoiding repeated computation in `get_mask`.

    2.  **Efficient Scheduler**: The `get_mask` method employs a min-heap (`heapq`) to manage
        the priority queue of nodes to visit, ordered by depth. This is significantly
        faster than repeatedly finding the minimum key in a dictionary.

    3.  **Optimized GSS Filtering**: The core performance bottleneck, which involved filtering
        GSS parent nodes against state ID bit-vectors using a nested loop (O(P*R) complexity),
        has been replaced. The new algorithm sorts the parent nodes and uses a linear scan
        (O(P log P + R)), which is much faster for large inputs.

    4.  **Correctness**: This implementation correctly handles epsilon transitions on the GSS
        stack state (where the state bit-vector is empty), a case that appeared to be
        overlooked in the original `get_mask` logic.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map = dict(roots_map)
        self.arena = arena
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.max_depth: Dict[int, int] = {}
        self.end_nodes: set[int] = set()

        # Pre-process the arena for faster access in get_mask.
        # This converts the raw dictionary-based graph into a structure with
        # pre-computed bitsets and integer-only data for performance.
        for uid, node in self.arena.items():
            self.max_depth[uid] = int(node.get("max_depth", 0))

            if (node.get("value") or {}).get("clean_end", False):
                self.end_nodes.add(uid)

            original_children = node.get("children") or []
            new_children = []
            for edge_key, dest_map in original_children:
                pop, llm_bv_json = edge_key
                llm_rs = RangeSet.from_ranges(llm_bv_json)
                llm_mask = ffi.Bitset.from_ranges(llm_rs.intervals)

                destinations = []
                for dest_idx, state_bv in dest_map:
                    current_dest_bv = [(int(a), int(b)) for a, b in state_bv]
                    destinations.append((int(dest_idx), current_dest_bv))

                new_children.append(Edge(
                    pop=int(pop),
                    llm_mask=llm_mask,
                    llm_rs=llm_rs,
                    destinations=destinations
                ))
            node["children"] = new_children

    @staticmethod
    def from_json_string(s: str) -> "Model":
        data = json.loads(s)
        roots_map = [(int(s), int(r)) for s, r in data['precomputed3']]
        arena_json = data['trie3_god']
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[state_id]

    def is_end(self, node: int) -> bool:
        return node in self.end_nodes

    def iter_edges(self, node: int, token: int):
        # This method remains for validation against the GraphProvider interface.
        node_data = self.arena.get(node)
        if not node_data:
            return

        for edge in node_data.get("children", []):
            if edge.llm_rs.contains(token):
                for dest_idx, state_bv_ranges in edge.destinations:
                    if not state_bv_ranges:  # Epsilon transition on GSS stack
                        yield (edge.pop, None, dest_idx)
                    else:
                        for start, end in state_bv_ranges:
                            for sid in range(start, end + 1):
                                yield (edge.pop, sid, dest_idx)

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        print("\n--- get_mask START ---")
        print(self.constraint_state)
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        print(f"Filtered state_to_gss: { {k: v.ptr() for k, v in state_to_gss.items()} }")

        final_mask = ffi.Bitset.zeros()
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}
        stopped: set[int] = set()

        # Efficient scheduler using a min-heap for depths and a dict for nodes.
        todo: Dict[int, set[int]] = defaultdict(set)
        depth_heap: List[int] = []

        # Seed the scheduler with the initial states.
        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(sid)
            if root_idx is None:
                continue

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            if root_idx in values:
                existing_gss, existing_mask = values[root_idx]
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss_clone], 1)
                merged_mask = existing_mask.union(new_mask)
                values[root_idx] = (merged_gss, merged_mask)
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={merged_mask.to_ranges()}")
            else:
                values[root_idx] = (gss_clone, new_mask)

            depth = self.max_depth.get(root_idx, 0)
            if not todo[depth]:
                heapq.heappush(depth_heap, depth)
            todo[depth].add(root_idx)

        # Main scheduler loop, processing nodes in ascending order of depth.
        print("\n--- Main loop ---")
        iter_count = 0
        while depth_heap:
            iter_count += 1
            current_depth = heapq.heappop(depth_heap)
            node_indices = todo.pop(current_depth)
            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")

            for node_idx in node_indices:
                if node_idx in stopped:
                    print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                gss_node, llm_mask = item
                print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                if node_idx in self.end_nodes:
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")
                    gss_active_tokens = gss_node.allowed_llm_tokens()
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if not gss_node.is_ok():
                    stopped.add(node_idx)
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                node = self.arena[node_idx]
                for edge in node["children"]:
                    print(f"    - Edge: pop={edge.pop}, llm_bv={edge.llm_mask.to_ranges()}")
                    peeks = ffi.gss_popn_collect(gss_node, edge.pop)
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue

                    peeks.sort(key=lambda x: x[0])

                    child_llm_mask = llm_mask.intersection(edge.llm_mask)
                    print(f"        - Child mask: {child_llm_mask.to_ranges()}")
                    if child_llm_mask.is_empty():
                        continue

                    for dest_idx, state_bv in edge.destinations:
                        print(f"      - Dest: idx={dest_idx}, state_bv={state_bv}")
                        matched = []
                        if not state_bv:  # Epsilon transition on GSS stack state
                            matched.extend(p[1] for p in peeks)
                        else:
                            # Optimized matching: linear scan over sorted lists
                            peek_idx, range_idx = 0, 0
                            while peek_idx < len(peeks) and range_idx < len(state_bv):
                                sid_val, parent_node = peeks[peek_idx]
                                start, end = state_bv[range_idx]

                                if sid_val > end:
                                    range_idx += 1
                                elif sid_val < start:
                                    peek_idx += 1
                                else:  # sid_val is in the current range
                                    matched.append(parent_node)
                                    peek_idx += 1
                        print(f"        - Matched {len(matched)} parent GSS nodes")

                        if not matched:
                            continue

                        child_gss = ffi.gss_merge_many_with_depth(matched, 1)

                        if not child_gss.is_ok():
                            continue

                        if dest_idx in values:
                            existing_gss, existing_mask = values[dest_idx]
                            print(f"        - Enqueue {dest_idx}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={child_llm_mask.to_ranges()}")
                            combined_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss], 1)
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[dest_idx] = (combined_gss, combined_mask)
                            print(f"          - Merged result: gss_ptr={combined_gss.ptr()}, mask={combined_mask.to_ranges()}")
                        else:
                            values[dest_idx] = (child_gss, child_llm_mask)
                            print(f"        - Enqueue {dest_idx}: CREATING gss_ptr={child_gss.ptr()}, mask={child_llm_mask.to_ranges()}")

                        child_depth = self.max_depth.get(dest_idx, 0)
                        if not todo[child_depth]:
                            heapq.heappush(depth_heap, child_depth)
                        todo[child_depth].add(dest_idx)

        print("\n--- get_mask END ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
