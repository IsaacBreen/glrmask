import json
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from tqdm import tqdm

# --- Data Structures for the Optimized Graph ---

class _PopGroup:
    """
    Represents all transitions from a single node for a specific pop count.
    This structure is optimized for fast lookups during get_mask.
    """
    __slots__ = ("sid_to_arcs", "eps_arcs")

    def __init__(
        self,
        sid_to_arcs: Dict[int, List[Tuple[int, Optional[ffi.Bitset]]]],
        eps_arcs: List[Tuple[int, Optional[ffi.Bitset]]],
    ):
        # Maps a tokenizer state ID (sid) to a list of possible transitions.
        # Each arc is (destination_node_index, llm_bitset_or_None).
        # None for llm_bitset means no token restriction.
        self.sid_to_arcs = sid_to_arcs

        # A list of transitions that are not dependent on the SID (epsilon transitions).
        self.eps_arcs = eps_arcs

class _NodeData:
    """
    A preprocessed, optimized representation of a node in the trie.
    """
    __slots__ = ("is_end", "groups", "max_depth")

    def __init__(self, is_end: bool, groups: Dict[int, _PopGroup], max_depth: int):
        self.is_end = is_end  # Does this node represent an end state?
        self.groups = groups  # Maps pop_count -> _PopGroup
        self.max_depth = max_depth

# --- The Model Implementation ---

class Model(GraphProvider):
    """
    A highly optimized model (gemini-5) that synthesizes the best features
    of previous top-performing models (gemini-4, gpt-5-9).

    Key Optimizations:
    1.  **Aggressive Graph Preprocessing**: During initialization, the graph is
        transformed into a structure optimized for `get_mask`. All parallel
        edges are merged by unioning their respective bitsets. Transitions are
        then grouped by pop count and indexed by tokenizer state ID (SID),
        enabling direct, O(1) lookups for valid transitions. This is the core
        strategy from gpt-5-9 and gemini-4.

    2.  **Optimized `get_mask` Algorithm**:
        - A depth-based scheduler prioritizes nodes to explore the graph efficiently.
        - **Intersection Caching**: Within the processing of a single node,
          the results of `llm_mask.intersection(edge_llm_bv)` are cached to
          avoid redundant bitset operations, a key optimization from gpt-5-9.
        - **Precise State Tracking**: A robust check prevents re-processing nodes
          if their state (both GSS parent set and LLM mask) has not changed,
          avoiding redundant computations. This improves upon the logic in
          some previous models.
        - **Efficient GSS Parent Grouping**: GSS parents are grouped by SID only
          once per pop, minimizing redundant work when multiple arcs share the
          same SID condition.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.nodes: Dict[int, _NodeData] = {}
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.max_depth: Dict[int, int] = {} # For the scheduler

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # Preprocess the entire arena into our optimized format.
        for uid, node_def in tqdm(
            arena.items(),
            desc="Optimizing Graph (gemini-5)",
            total=len(arena),
        ):
            uid_int = int(uid)
            self.max_depth[uid_int] = int(node_def.get("max_depth", 0))
            is_end = bool((node_def.get("value") or {}).get("clean_end", False))

            children = node_def.get("children") or []
            if not children:
                self.nodes[uid_int] = _NodeData(is_end, {}, self.max_depth[uid_int])
                continue

            # --- Step 1: Aggregate all transitions by pop count ---
            # This intermediate structure helps merge LLM bitsets for identical
            # (pop, sid, dest) or (pop, epsilon, dest) transitions.
            pop_aggregator = defaultdict(lambda: {"sid_map": defaultdict(lambda: defaultdict(ffi.Bitset.zeros)), "eps_map": defaultdict(ffi.Bitset.zeros)})

            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                pop = int(pop)
                llm_bv = bs_from_json(dumps(llm_bv_json))

                agg = pop_aggregator[pop]
                
                for dest_idx, state_bv_json in dest_map:
                    dest_idx = int(dest_idx)
                    state_bv = bs_from_json(dumps(state_bv_json))

                    if state_bv.is_empty():
                        # Epsilon transition (applies to all SIDs)
                        agg["eps_map"][dest_idx] = agg["eps_map"][dest_idx].union(llm_bv)
                    else:
                        # SID-specific transitions
                        sid_map_for_pop = agg["sid_map"]
                        for start, end in state_bv.to_ranges():
                            end = min(end, max_state_id + 1)
                            for sid in range(start, end):
                                sid_map_for_pop[sid][dest_idx] = sid_map_for_pop[sid][dest_idx].union(llm_bv)

            # --- Step 2: Convert aggregated data into final, compact _PopGroup structures ---
            final_groups: Dict[int, _PopGroup] = {}
            for pop, agg in pop_aggregator.items():
                # Convert sid_map from dict-of-dicts to dict-of-lists
                sid_to_arcs = {
                    sid: [(dest, None if bv.is_empty() else bv) for dest, bv in dest_map.items()]
                    for sid, dest_map in agg["sid_map"].items()
                }
                # Convert eps_map
                eps_arcs = [(dest, None if bv.is_empty() else bv) for dest, bv in agg["eps_map"].items()]
                
                final_groups[pop] = _PopGroup(sid_to_arcs, eps_arcs)

            self.nodes[uid_int] = _NodeData(is_end, final_groups, self.max_depth[uid_int])

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        max_state_id = int(max(dict(data['parser']['stage_7_table']).keys()))
        model = Model(roots_map, arena, max_state_id)
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        node_data = self.nodes.get(node)
        return node_data.is_end if node_data else False

    def iter_edges(self, node: int, token: int):
        """
        Explodes the optimized graph structure for validation purposes.
        Not used in the performance-critical path.
        """
        node_data = self.nodes.get(node)
        if not node_data:
            return

        for pop, group in node_data.groups.items():
            # SID-specific arcs
            for sid, arcs in group.sid_to_arcs.items():
                for dest_idx, llm_bv in arcs:
                    if llm_bv is None or llm_bv.contains(token):
                        yield (pop, sid, dest_idx)
            # Epsilon arcs
            for dest_idx, llm_bv in group.eps_arcs:
                if llm_bv is None or llm_bv.contains(token):
                    yield (pop, None, dest_idx)

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        print("\n--- get_mask START ---")
        state_to_gss = self.constraint_state.filtered_state_gss_map()

        final_mask = ffi.Bitset.zeros()
        # {node_idx: ({gss_parents}, llm_mask)}
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}
        # {depth: {node_idx, ...}}
        todo: Dict[int, Set[int]] = defaultdict(set)
        # min-heap of depths to visit
        depth_heap: List[int] = []

        # --- Seeding Phase ---
        print("\n--- Seeding work queue ---")
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(sid)
            if root_idx is None:
                continue

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            if new_mask.is_empty():
                continue
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")

            existing = values.get(root_idx)
            if existing:
                existing_gss, existing_mask = existing
                print(f"    - MERGE: gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                merged_gss = ffi.gss_merge_many_with_depth([existing_gss, gss_clone], 1)
                values[root_idx] = (merged_gss, existing_mask.union(new_mask))
                print(f"      - Merged result: gss_ptr={merged_gss.ptr()}, mask={values[root_idx][1].to_ranges()}")
            else:
                values[root_idx] = (gss_clone, new_mask)
                depth = self.max_depth.get(root_idx, 0)
                if not todo[depth]:
                    heapq.heappush(depth_heap, depth)
                todo[depth].add(root_idx)

        # --- Main Scheduler Loop ---
        print("\n--- Main loop ---")
        iter_count = 0
        while depth_heap:
            iter_count += 1
            current_depth = heapq.heappop(depth_heap)
            node_indices = todo.pop(current_depth, set())
            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")
            if not node_indices:
                continue

            for node_idx in node_indices:
                item = values.pop(node_idx, None)
                if not item:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                gss_node, llm_mask = item
                print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={gss_node.ptr()}, mask={llm_mask.to_ranges()}")

                node_data = self.nodes.get(node_idx)
                if not node_data:
                    continue

                if node_data.is_end:
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")
                    gss_active_tokens = gss_node.allowed_llm_tokens()
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if not gss_node.is_alive():
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                # --- Process Transitions for the Current Node ---
                for pop, group in node_data.groups.items():
                    print(f"    - Edge group: pop={pop}")
                    peeks = gss_node.popn_fast(pop)
                    print(f"      - Found {len(peeks)} peeks from GSS")
                    if not peeks:
                        continue

                    next_gss: Dict[int, List[ffi.GSSNode]] = defaultdict(list)
                    next_mask: Dict[int, ffi.Bitset] = defaultdict(ffi.Bitset.zeros)
                    
                    # Cache for llm_mask.intersection(llm_bv) results
                    inter_cache: Dict[Optional[int], ffi.Bitset] = {None: llm_mask}

                    # Process epsilon transitions (not dependent on SID)
                    if group.eps_arcs:
                        all_parents = [p for _, p in peeks]
                        if all_parents:
                            for dest_idx, llm_bv in group.eps_arcs:
                                llm_bv_id = id(llm_bv) if llm_bv is not None else None
                                child_mask = inter_cache.get(llm_bv_id)
                                if child_mask is None:
                                    child_mask = llm_mask.intersection(llm_bv)
                                    inter_cache[llm_bv_id] = child_mask
                                
                                if not child_mask.is_empty():
                                    next_gss[dest_idx].extend(all_parents)
                                    next_mask[dest_idx] = next_mask[dest_idx].union(child_mask)

                    # Group GSS parents by SID for efficient lookup.
                    sid_to_parents = defaultdict(list)
                    for sid, parent_gss in peeks:
                        sid_to_parents[sid].append(parent_gss)

                    # Process SID-specific transitions
                    for sid, parents in sid_to_parents.items():
                        arcs = group.sid_to_arcs.get(sid)
                        if not arcs:
                            continue
                        for dest_idx, llm_bv in arcs:
                            llm_bv_id = id(llm_bv) if llm_bv is not None else None
                            child_mask = inter_cache.get(llm_bv_id)
                            if child_mask is None:
                                child_mask = llm_mask.intersection(llm_bv)
                                inter_cache[llm_bv_id] = child_mask

                            if not child_mask.is_empty():
                                next_gss[dest_idx].extend(parents)
                                next_mask[dest_idx] = next_mask[dest_idx].union(child_mask)

                    # --- Flush accumulated children to the main queue ---
                    for dest_idx, parents_list in next_gss.items():
                        print(f"      - Dest: idx={dest_idx}")
                        print(f"        - Matched {len(parents_list)} parent GSS nodes")
                        if not parents_list:
                            continue
                        child_gss = ffi.gss_merge_many_with_depth(parents_list, 1)
                        if not child_gss.is_alive():
                            continue

                        child_llm_mask = next_mask[dest_idx]
                        
                        existing = values.get(dest_idx)
                        if existing:
                            existing_gss, existing_mask = existing
                            print(f"        - Enqueue {dest_idx}: MERGING gss1_ptr={existing_gss.ptr()}, mask1={existing_mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={child_llm_mask.to_ranges()}")
                            merged_gss = ffi.gss_merge_many_with_depth([existing_gss, child_gss], 1)
                            new_mask = existing_mask.union(child_llm_mask)
                            if merged_gss.ptr() == existing_gss.ptr() and new_mask == existing_mask:
                                continue
                            values[dest_idx] = (merged_gss, new_mask)
                            print(f"          - Merged result: gss_ptr={merged_gss.ptr()}, mask={new_mask.to_ranges()}")
                        else:
                            values[dest_idx] = (child_gss, child_llm_mask)
                            print(f"        - Enqueue {dest_idx}: CREATING gss_ptr={child_gss.ptr()}, mask={child_llm_mask.to_ranges()}")

                        depth = self.max_depth.get(dest_idx, 0)
                        if not todo[depth]:
                            heapq.heappush(depth_heap, depth)
                        todo[depth].add(dest_idx)
                        
        print("\n--- get_mask END ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
