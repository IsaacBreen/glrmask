import json
import heapq
from typing import Dict, List, Tuple, Optional, DefaultDict

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # compiled module with Bitset and GSSNode
from tqdm.auto import tqdm


class Model(GraphProvider):
    """
    Optimized precomputed trie model (generation 5.7).

    Key changes vs precompute3_model:
    - Build a per-node, per-pop index mapping tokenizer state -> [(dest_idx, llm_bv or None)]
      so we can jump directly from a GSS pop to the relevant destinations without scanning
      all dest bitsets.
    - Aggregate contributions over all matched states per (node, pop, dest) before updating
      the scheduler's working set to minimize repeated unions and enqueue operations.
    - Preserve the external interface and iter_edges semantics for equivalence checking.

    Semantics around llm_bv:
    - If an edge group carries an empty llm_bv (bv.is_empty() == True) it is treated as "no filter"
      (i.e., child mask = current llm mask). When multiple groups contribute to a dest/state, the
      resulting behavior is: if any group is "no filter", the union is "no filter"; otherwise we
      intersect with union of all contributing llm_bv.
    """

    # Entry type for fast pop index:
    # For a given node and pop, for a specific tokenizer state (sid),
    # the value is a list of (dest_idx, llm_bv_or_none) where:
    #   - llm_bv_or_none is ffi.Bitset if constrained, or None if unconstrained.
    _StateEntries = Dict[int, List[Tuple[int, Optional[ffi.Bitset]]]]

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict], max_state_id: int):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.max_depth: Dict[int, int] = {}

        # children normalized as in precompute3_model, plus build pop-index for fast lookup
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # node_id -> (pop -> state -> list[(dest_idx, bv or None)])
        self._pop_index: Dict[int, Dict[int, Model._StateEntries]] = {}

        for uid, node in tqdm(
            self.arena.items(),
            desc="Normalizing gpt-5-7 arena and building pop index",
            total=len(self.arena),
        ):
            uid_int = int(uid)

            # Cache max_depth
            try:
                md = node.get("max_depth", 0)
                self.max_depth[uid_int] = int(md)
            except Exception:
                self.max_depth[uid_int] = 0

            # Normalize children into ffi bitsets (keep original structure for iter_edges)
            children = node.get("children") or []
            if not children:
                node["children"] = []
                self._pop_index[uid_int] = {}
                continue

            new_children = []
            # Build fast index holder for this node
            pop_to_state_entries: Dict[int, Model._StateEntries] = {}

            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                pop = int(pop)
                llm_bv = bs_from_json(dumps(llm_bv_json))

                # "Unconstrained" llm filter is represented by empty bitset as per existing impl
                llm_unconstrained = llm_bv.is_empty()

                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    dest_idx = int(dest_idx)
                    state_bv = bs_from_json(dumps(state_bv_json))
                    new_dest_map.append((dest_idx, state_bv))

                    # Build fast pop index for this (node, pop)
                    # Skip epsilon-on-GSS transitions (empty state_bv) here: the existing get_mask
                    # ignores them (handled only by iter_edges in precompute3_model).
                    if not state_bv.is_empty():
                        # lazily allocate per-pop map
                        s_map = pop_to_state_entries.get(pop)
                        if s_map is None:
                            s_map = {}
                            pop_to_state_entries[pop] = s_map

                        # Fan-out to states in this bitset
                        # to_ranges() yields disjoint intervals [start, end)
                        for start, end in state_bv.to_ranges():
                            # Note: states are small (~1000); this is a fast loop and allows
                            # direct sid lookup during get_mask without per-dest contains() calls.
                            end = min(end, max_state_id + 1)
                            for sid in range(start, end):
                                entries = s_map.get(sid)
                                if entries is None:
                                    entries = []
                                    s_map[sid] = entries
                                # Store None for unconstrained (means "no filter")
                                entries.append((dest_idx, None if llm_unconstrained else llm_bv))

                new_children.append(((pop, llm_bv), new_dest_map))

            node["children"] = new_children
            self._pop_index[uid_int] = pop_to_state_entries

    @staticmethod
    def from_json_string(s: str) -> "Model":
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        max_state_id = int(max(dict(data['parser']['stage_7_table']).keys()))
        model = Model(roots_map, arena, max_state_id)
        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(constraint)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("end", False))

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        Only used by equivalence checking; not performance-critical.

        This uses the normalized arena children (same as precompute3_model).
        """
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():  # Epsilon on GSS stack
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end):
                                yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This version uses the per-pop, per-state fast index to avoid
        scanning all destination bitsets on every transition.
        """
        state_to_gss = self.constraint_state.get_state_to_gss_map()
        Bitset = ffi.Bitset

        final_mask: ffi.Bitset = Bitset.zeros()

        # node_idx -> (set(GSSNode), Bitset)
        values: Dict[int, Tuple[set, ffi.Bitset]] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)

        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        for sid, gss in state_to_gss.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            # Note: clone_node and allowed_llm_tokens provided by ffi; fast operations
            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()

            existing = values.get(root_idx)
            if existing is not None:
                gss_set, existing_mask = existing
                gss_set.add(gss_clone)
                # Union allowed tokens into the node-level mask
                values[root_idx] = (gss_set, existing_mask.union(new_mask))
            else:
                values[root_idx] = ({gss_clone}, new_mask)

            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        heappop = heapq.heappop
        arena = self.arena
        is_end = self.is_end
        pop_index_all = self._pop_index

        while True:
            # Pop the smallest depth bucket (skip stale heap entries)
            node_indices: Optional[set[int]] = None
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

                # Fast dispatch via precomputed pop-index for this node
                pop_index = pop_index_all.get(node_idx)
                if not pop_index:
                    continue  # no outgoing transitions

                # For each pop group, we collect peeks once and map sids directly to dests
                for pop, state_to_entries in pop_index.items():
                    # Collect all pops from GSS parents
                    peeks = []
                    append_peek = peeks.append
                    for g in gss_set:
                        # popn_fast(pop) -> List[(sid_val, parent_gss_node)]
                        for pair in g.popn_fast(pop):
                            append_peek(pair)

                    if not peeks:
                        continue

                    # Bucket peeks by sid to dedupe membership lookups and aggregate parents
                    sid_to_parents: Dict[int, List] = {}
                    for sid_val, parent_node in peeks:
                        lst = sid_to_parents.get(sid_val)
                        if lst is None:
                            lst = [parent_node]
                            sid_to_parents[sid_val] = lst
                        else:
                            lst.append(parent_node)

                    # Aggregate per-destination contributions across all matched sids
                    # dest_idx -> (parents_list, any_unconstrained, union_llm_bv or None if none seen yet)
                    dest_parents: Dict[int, List] = {}
                    dest_any_unconstrained: Dict[int, bool] = {}
                    dest_llm_union: Dict[int, ffi.Bitset] = {}

                    for sid_val, parents in sid_to_parents.items():
                        entries = state_to_entries.get(sid_val)
                        if not entries:
                            continue
                        # entries: List[(dest_idx, bv_or_none)]
                        for dest_idx, bv in entries:
                            # Accumulate parents per dest
                            plist = dest_parents.get(dest_idx)
                            if plist is None:
                                dest_parents[dest_idx] = parents.copy()
                            else:
                                plist.extend(parents)

                            if bv is None:
                                # Unconstrained filter present for this dest at this state
                                dest_any_unconstrained[dest_idx] = True
                            else:
                                # Union llm_bv across matched states for this dest
                                existing_bv = dest_llm_union.get(dest_idx)
                                if existing_bv is None:
                                    dest_llm_union[dest_idx] = bv
                                else:
                                    dest_llm_union[dest_idx] = existing_bv.union(bv)

                    if not dest_parents:
                        continue

                    # Merge into scheduler state per dest
                    enqueue = self._enqueue_helper(todo, depth_heap)
                    for d, child_nodes in dest_parents.items():
                        # Compute child mask for this dest
                        if dest_any_unconstrained.get(d, False):
                            child_llm_mask = llm_mask
                        else:
                            bv_union = dest_llm_union.get(d)
                            # If no union bits (i.e., no constrained contributions), skip mask-only update
                            if bv_union is None:
                                # No tokens allowed through (unless unconstrained, which we already handled)
                                # We still carry gss parents below; mask will remain as-is if node already exists.
                                child_llm_mask = Bitset.zeros()
                            else:
                                child_llm_mask = llm_mask.intersection(bv_union)

                        existing = values.get(d)
                        if existing is not None:
                            existing_gss_set, existing_mask = existing
                            old_len = len(existing_gss_set)
                            # Note: 'child_nodes' contains unique parents per (pop, node); safe to update set
                            existing_gss_set.update(child_nodes)

                            # Merge masks unconditionally; correctness requires new tokens to propagate
                            new_mask = existing_mask.union(child_llm_mask)
                            values[d] = (existing_gss_set, new_mask)

                            # Re-enqueue if either set size grew or new_mask is different (not just identity)
                            if len(existing_gss_set) != old_len or new_mask is not existing_mask:
                                enqueue(max_depth[d], d)
                        else:
                            # Initialize from scratch
                            # Convert child_nodes to set once
                            values[d] = (set(child_nodes), child_llm_mask)
                            enqueue(max_depth[d], d)

        return RangeSet.from_ranges(final_mask.to_ranges())

    @staticmethod
    def _enqueue_helper(todo: Dict[int, set], depth_heap: List[int]):
        heappush = heapq.heappush

        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        return enqueue

