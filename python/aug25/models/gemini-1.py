import json
import heapq
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
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.max_depth: Dict[int, int] = {}
        self.end_nodes: set[int] = set()

        # Pre-process the arena for faster access in get_mask.
        # This converts the raw dictionary-based graph into a structure with
        # pre-computed bitsets and integer-only data for performance.
        for uid, node in self.arena.items():
            self.max_depth[uid] = int(node.get("max_depth", 0))

            if (node.get("value") or {}).get("end", False):
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
        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(constraint)
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
        state_to_gss = self.constraint_state.get_state_map()
        final_mask = ffi.Bitset.zeros()
        values: Dict[int, ffi.GSSNode] = {}
        stopped: set[int] = set()

        # Efficient scheduler using a min-heap for depths and a dict for nodes.
        todo: Dict[int, set[int]] = defaultdict(set)
        depth_heap: List[int] = []

        # Seed the scheduler with the initial states.
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(sid)
            if root_idx is None:
                continue

            if root_idx in values:
                existing = values[root_idx]
                merged = ffi.gss_merge_many_with_depth([existing, gss.clone_node()], 9999999)
                if merged.ptr() == existing.ptr():
                    continue
                values[root_idx] = merged
            else:
                values[root_idx] = gss.clone_node()

            depth = self.max_depth.get(root_idx, 0)
            if not todo[depth]:
                heapq.heappush(depth_heap, depth)
            todo[depth].add(root_idx)

        # Main scheduler loop, processing nodes in ascending order of depth.
        while depth_heap:
            current_depth = heapq.heappop(depth_heap)
            node_indices = todo.pop(current_depth)

            for node_idx in node_indices:
                if node_idx in stopped:
                    continue

                agg = values.pop(node_idx, None)
                if agg is None:
                    continue

                if node_idx in self.end_nodes:
                    final_mask = final_mask.union(agg.allowed_llm_tokens())

                if not agg.is_ok():
                    stopped.add(node_idx)
                    continue

                node = self.arena[node_idx]
                for edge in node["children"]:
                    peeks = ffi.gss_popn_collect(agg, edge.pop)
                    if not peeks:
                        continue

                    peeks.sort(key=lambda x: x[0])

                    for dest_idx, state_bv in edge.destinations:
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

                        if not matched:
                            continue

                        child_gss = ffi.gss_merge_many_with_depth(matched, 1)
                        ffi.gss_allow_only_llm_tokens_and_prune(child_gss, edge.llm_mask)

                        if not child_gss.is_ok():
                            continue

                        if dest_idx in values:
                            combined = ffi.gss_merge_many_with_depth([values[dest_idx], child_gss], 1)
                            if combined.ptr() == values[dest_idx].ptr():
                                continue
                            values[dest_idx] = combined
                        else:
                            values[dest_idx] = child_gss

                        child_depth = self.max_depth.get(dest_idx, 0)
                        if not todo[child_depth]:
                            heapq.heappush(depth_heap, child_depth)
                        todo[child_depth].add(dest_idx)

        return RangeSet.from_ranges(final_mask.to_ranges())

