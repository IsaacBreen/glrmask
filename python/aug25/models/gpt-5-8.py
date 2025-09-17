import json
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterable, Set
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # compiled module (Bitset, GSSNode, etc.)


class PopIndex:
    """
    Per-node, per-pop precomputed dispatch index to eliminate O(dests * peeks) filtering.

    - wild: transitions that apply to all tokenizer states (state_bv.is_empty()).
            Mapping: dest_idx -> llm_token_filter
            Where llm_token_filter is:
              - None: means pass-through (no token filter)
              - ffi.Bitset: a concrete token filter
    - sid_map: transitions for specific tokenizer states.
            Mapping: sid -> List[(dest_idx, llm_token_filter)]
            Where llm_token_filter is as above.
    """
    __slots__ = ("wild", "sid_map")

    def __init__(self):
        self.wild: Dict[int, Optional[ffi.Bitset]] = {}
        self.sid_map: Dict[int, List[Tuple[int, Optional[ffi.Bitset]]]] = {}


class Model(GraphProvider):
    """
    Optimized model that builds a dispatch index per node:
      pop -> { wildcard-dests (state-agnostic), per-state dests }
    to avoid repeatedly scanning and filtering destination state bitsets.

    The public interface (GraphProvider) is preserved:
      - from_json_string
      - get_root
      - is_end
      - iter_edges
      - get_mask
    iter_edges uses the original normalized children list to preserve
    equivalence checking semantics. get_mask uses the optimized index.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        # Store arena as int keyed
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.arena: Dict[int, dict] = {int(k): v for k, v in arena.items()}

        # Core per-node caches
        self.max_depth: Dict[int, int] = {}
        self.end_flags: Dict[int, bool] = {}
        self.children: Dict[int, List[Tuple[Tuple[int, ffi.Bitset], List[Tuple[int, ffi.Bitset]]]]] = {}
        self.pop_index_by_node: Dict[int, Dict[int, PopIndex]] = {}

        bs_from_json = ffi.Bitset.from_json_string
        dumps = json.dumps

        # Normalize, cache, and build indexes
        for uid, node in self.arena.items():
            uid_int = int(uid)

            # Cache end flags and max_depth
            self.end_flags[uid_int] = bool((node.get("value") or {}).get("end", False))
            try:
                self.max_depth[uid_int] = int(node.get("max_depth", 0))
            except Exception:
                self.max_depth[uid_int] = 0

            # Normalize children: convert JSON bitsets into ffi.Bitset
            children_json = node.get("children") or []
            if not children_json:
                self.children[uid_int] = []
                self.pop_index_by_node[uid_int] = {}
                continue

            normalized_children: List[
                Tuple[Tuple[int, ffi.Bitset], List[Tuple[int, ffi.Bitset]]]
            ] = []

            for edge_key, dest_map in children_json:
                pop_val, llm_bv_json = edge_key
                pop_val = int(pop_val)
                llm_bv = bs_from_json(dumps(llm_bv_json))

                new_dest_map: List[Tuple[int, ffi.Bitset]] = []
                for dest_idx, state_bv_json in dest_map:
                    dest_idx_int = int(dest_idx)
                    state_bv = bs_from_json(dumps(state_bv_json))
                    new_dest_map.append((dest_idx_int, state_bv))

                normalized_children.append(((pop_val, llm_bv), new_dest_map))

            self.children[uid_int] = normalized_children
            self.pop_index_by_node[uid_int] = self._build_pop_index(normalized_children)

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
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return self.end_flags.get(int(node), False)

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        Only used by equivalence checking; not performance-critical.
        """
        children = self.children.get(node, []) or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():  # Epsilon on GSS stack / no state filter
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            end = min(end, len(state_bv))
                            for sid in range(start, end):
                                yield (int(pop), sid, int(dest_idx))

    # --------------------
    # Internal helpers
    # --------------------

    @staticmethod
    def _merge_token_filters(existing: Optional[ffi.Bitset], new_bv: Optional[ffi.Bitset]) -> Optional[ffi.Bitset]:
        """
        Merge two token filters:
          - None means pass-through (no filter)
          - ffi.Bitset means concrete filter
        Union semantics:
          - None U anything = None
          - Bitset U None = None
          - Bitset U Bitset = union of two bitsets
          - None with None = None
        """
        if existing is None or new_bv is None:
            return None
        return existing.union(new_bv)

    def _build_pop_index(
        self,
        children: List[Tuple[Tuple[int, ffi.Bitset], List[Tuple[int, ffi.Bitset]]]],
    ) -> Dict[int, PopIndex]:
        """
        Build the per-pop dispatch index for a node.

        For each pop value:
          - Collect wildcard transitions (state_bv.is_empty()) into a dest->tokens map
          - For state-specific transitions:
              For each sid in the dest's state bitset, add dest->tokens for that sid
          - Where tokens are unioned across edges (llm_bv is unioned; llm_bv.is_empty() means pass-through)
        """
        # Group by pop
        pops: Dict[int, List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]] = defaultdict(list)
        for (pop_val, llm_bv), dests in children:
            pops[int(pop_val)].append((llm_bv, dests))

        result: Dict[int, PopIndex] = {}

        for pop_val, groups in pops.items():
            pop_index = PopIndex()
            # Temporary building structures
            wild_map: Dict[int, Optional[ffi.Bitset]] = {}
            # sid -> (dest -> token_filter)
            sid_map_temp: Dict[int, Dict[int, Optional[ffi.Bitset]]] = {}

            for llm_bv, dests in groups:
                # Interpret llm_bv.is_empty() as "no filter" / pass-through
                llm_filter: Optional[ffi.Bitset] = None if llm_bv.is_empty() else llm_bv

                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():
                        # Wildcard over states: applies to all sids
                        prev = wild_map.get(dest_idx)
                        merged = self._merge_token_filters(prev, llm_filter)
                        wild_map[dest_idx] = merged
                    else:
                        # Enumerate state ranges; states domain is typically small (e.g., ~1000)
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end):
                                dmap = sid_map_temp.get(sid)
                                if dmap is None:
                                    dmap = {}
                                    sid_map_temp[sid] = dmap
                                prev = dmap.get(dest_idx)
                                merged = self._merge_token_filters(prev, llm_filter)
                                dmap[dest_idx] = merged

            # Freeze sid_map into lists for faster iteration and less overhead
            sid_map_final: Dict[int, List[Tuple[int, Optional[ffi.Bitset]]]] = {}
            for sid, dmap in sid_map_temp.items():
                # Convert dict -> list of (dest_idx, filter), no need to sort
                sid_map_final[sid] = list(dmap.items())

            pop_index.wild = wild_map
            pop_index.sid_map = sid_map_final
            result[pop_val] = pop_index

        return result

    # --------------------
    # Fast get_mask
    # --------------------

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. Uses the per-pop dispatch index to avoid repeated O(dests * peeks) filters.
        """
        state_to_gss = self.constraint_state.get_state_map()
        final_mask = ffi.Bitset.zeros()

        # Node -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[Set[ffi.GSSNode], ffi.Bitset]] = {}

        stopped: Set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, Set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths

        heappush = heapq.heappush
        heappop = heapq.heappop

        # Seed: map tokenizer states and their filtered GSS to trie roots
        roots_map = self.roots_map
        max_depth = self.max_depth
        values_get = values.get

        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()

            existing = values_get(root_idx)
            if existing is not None:
                gss_set, existing_mask = existing
                gss_set.add(gss_clone)
                values[root_idx] = (gss_set, existing_mask.union(new_mask))
            else:
                values[root_idx] = ({gss_clone}, new_mask)

            depth = max_depth.get(root_idx, 0)
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        # Main scheduler
        arena_children = self.children
        pop_index_by_node = self.pop_index_by_node
        max_depth_get = max_depth.get
        is_end = self.is_end

        def enqueue(depth_val: int, node_idx: int) -> None:
            bucket = todo.get(depth_val)
            if bucket is None:
                todo[depth_val] = {node_idx}
                heappush(depth_heap, depth_val)
            else:
                bucket.add(node_idx)

        while True:
            # Pop the smallest depth bucket (skip stale heap entries)
            node_indices: Optional[Set[int]] = None
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                break  # nothing left to process

            # Process all nodes in this depth bucket
            for node_idx in node_indices:
                if node_idx in stopped:
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    continue
                gss_set, llm_mask = item

                # End-node handling
                if is_end(node_idx):
                    final_mask = final_mask.union(llm_mask)

                if not gss_set:
                    stopped.add(node_idx)
                    continue

                # Use precomputed pop dispatch for this node
                pop_index_map = pop_index_by_node.get(node_idx)
                if not pop_index_map:
                    continue  # no outgoing transitions

                # For each pop value from this node, dispatch peeks by state quickly
                for pop_val, pop_index in pop_index_map.items():
                    # Collect all peeks from GSS parents for this pop
                    sid_to_parents: Dict[int, List[ffi.GSSNode]] = defaultdict(list)
                    all_parents: Set[ffi.GSSNode] = set()
                    for g in gss_set:
                        # popn_fast returns List[(sid_val, parent_node)]
                        peeks = g.popn_fast(pop_val)
                        if not peeks:
                            continue
                        for sid_val, parent_node in peeks:
                            sid_to_parents[sid_val].append(parent_node)
                            all_parents.add(parent_node)

                    if not sid_to_parents and not all_parents:
                        continue

                    # Aggregate: dest -> set(GSSNode), and union of token filters
                    dest_gss_nodes: Dict[int, Set[ffi.GSSNode]] = {}
                    dest_token_filters: Dict[int, Optional[ffi.Bitset]] = {}

                    # Wildcard transitions (apply to all sids)
                    wild = pop_index.wild
                    if wild and all_parents:
                        for dest_idx, tok_filter in wild.items():
                            s = dest_gss_nodes.get(dest_idx)
                            if s is None:
                                s = set()
                                dest_gss_nodes[dest_idx] = s
                            s.update(all_parents)
                            prev_tf = dest_token_filters.get(dest_idx)
                            dest_token_filters[dest_idx] = self._merge_token_filters(prev_tf, tok_filter)

                    # State-specific transitions
                    sid_map = pop_index.sid_map
                    if sid_map:
                        for sid_val, parents_list in sid_to_parents.items():
                            dests = sid_map.get(sid_val)
                            if not dests:
                                continue
                            for dest_idx, tok_filter in dests:
                                s = dest_gss_nodes.get(dest_idx)
                                if s is None:
                                    s = set()
                                    dest_gss_nodes[dest_idx] = s
                                s.update(parents_list)
                                prev_tf = dest_token_filters.get(dest_idx)
                                dest_token_filters[dest_idx] = self._merge_token_filters(prev_tf, tok_filter)

                    if not dest_gss_nodes:
                        continue

                    # Merge into values map and enqueue children
                    for d, gss_children in dest_gss_nodes.items():
                        # Compute child mask: if tok_filter is None => pass-through, else intersect
                        tok_filter = dest_token_filters.get(d)
                        child_llm_mask = llm_mask if tok_filter is None else llm_mask.intersection(tok_filter)

                        existing = values.get(d)
                        if existing is not None:
                            existing_gss_set, existing_mask = existing
                            old_len = len(existing_gss_set)
                            existing_gss_set.update(gss_children)
                            if len(existing_gss_set) != old_len:
                                combined_mask = existing_mask.union(child_llm_mask)
                                values[d] = (existing_gss_set, combined_mask)
                                enqueue(max_depth_get(d, 0), d)
                            else:
                                # Set did not change; still update mask (theoretically could add new tokens)
                                combined_mask = existing_mask.union(child_llm_mask)
                                values[d] = (existing_gss_set, combined_mask)
                                # Depth bucket remains untouched if set unchanged
                        else:
                            values[d] = (set(gss_children), child_llm_mask)
                            enqueue(max_depth_get(d, 0), d)

        original_mask = self.constraint.internal_bv_to_original(final_mask)
        return RangeSet.from_ranges(original_mask.to_ranges())
