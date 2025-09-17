import json
from typing import Dict, List, Tuple, Optional, Iterable
from collections import defaultdict

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # compiled module (provides Bitset and GSS primitives)


class _BitsetInterner:
    """
    Interns Bitsets by their JSON representation to reuse identical bitsets across the arena.
    This reduces both memory footprint and speeds up equality-by-identity checks and caches.
    """
    __slots__ = ("_map",)

    def __init__(self) -> None:
        self._map: Dict[str, ffi.Bitset] = {}

    def load(self, json_obj) -> ffi.Bitset:
        # Serialize using stable formatting to maximize cache hits.
        key = json.dumps(json_obj, sort_keys=True, separators=(",", ":"))
        bs = self._map.get(key)
        if bs is None:
            bs = ffi.Bitset.from_json_string(key)
            self._map[key] = bs
        return bs


class Model(GraphProvider):
    """
    Extremely optimized precompute3 model following the get_mask3 idea from constraint.rs:
      - Carry an "allowed LLM token bitset" along the traversal, instead of calling
        gss_allow_only_llm_tokens_and_prune at every edge.
      - Compute peeks (gss_popn_collect) once per (node, pop) bucket, then reuse for all edges
        that share that pop.
      - Merge matched parents once per (node, pop, state_bv) and reuse that merged GSS for all
        destinations that share that state_bv.
      - Intern bitsets for llm_bv and state_bv to maximize object identity reuse, enabling cheap
        dictionary caches keyed by id(bitset).
      - Depth-based scheduler with "values" accumulator per node, de-duplicating work between arrivals.

    Interface is fully compatible with the provided GraphProvider.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Roots map from tokenizer state id -> trie node id
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

        # Global interner for all bitsets seen in the arena
        interner = _BitsetInterner()

        # Node data store
        # Each node_idx -> {
        #   "end": bool,
        #   "max_depth": int,
        #   "pops": {
        #       pop: [
        #           {
        #               "llm_bv": ffi.Bitset,
        #               "groups": { id(state_bv): (state_bv, [dest_idx, ...]), ... }
        #           },
        #           ...
        #       ]
        #   }
        # }
        self.nodes: Dict[int, dict] = {}
        self.max_depth: Dict[int, int] = {}

        # Normalize and compress the arena in one pass
        for uid, node in arena.items():
            node_idx = int(uid)
            value = node.get("value") or {}
            is_end = bool(value.get("end", False))
            max_depth = int(node.get("max_depth", 0) or 0)
            self.max_depth[node_idx] = max_depth

            pops: Dict[int, List[dict]] = defaultdict(list)

            # children is a list of [edge_key, dest_map], where:
            #   edge_key = (pop, llm_bv_json)
            #   dest_map = list of [dest_idx, state_bv_json]
            #
            # We convert llm_bv_json and state_bv_json to interned ffi.Bitset,
            # and group destinations that share the same state_bv (by id) for each edge.
            children: Iterable = node.get("children") or []
            for edge_key, dest_map in children:
                pop_raw, llm_bv_json = edge_key
                pop = int(pop_raw)
                llm_bv = interner.load(llm_bv_json)

                # Group dests for this (pop, llm_bv) edge by state_bv identity
                groups: Dict[int, Tuple[ffi.Bitset, List[int]]] = {}
                for dest_idx_raw, state_bv_json in dest_map:
                    dest_idx = int(dest_idx_raw)
                    state_bv = interner.load(state_bv_json)
                    key = id(state_bv)
                    entry = groups.get(key)
                    if entry is None:
                        groups[key] = (state_bv, [dest_idx])
                    else:
                        entry[1].append(dest_idx)

                pops[pop].append(
                    {
                        "llm_bv": llm_bv,
                        "groups": groups,  # { id(state_bv): (state_bv, [dest_idx, ...]) }
                    }
                )

            self.nodes[node_idx] = {
                "end": is_end,
                "max_depth": max_depth,
                "pops": dict(pops),
            }

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
        info = self.nodes.get(int(node))
        if info is None:
            return False
        return bool(info.get("end", False))

    def iter_edges(self, node: int, token: int):
        """
        Slow-path (not used in performance-critical get_mask) to validate consistency via the checker.
        We "explode" the state bitset to individual state IDs to satisfy the checker interface.
        """
        node_info = self.nodes.get(int(node))
        if not node_info:
            return
        pops = node_info.get("pops", {})
        for pop, edges in pops.items():
            for edge in edges:
                llm_bv = edge["llm_bv"]
                if not llm_bv.contains(int(token)):
                    continue
                for (_key, (state_bv, dests)) in edge["groups"].items():
                    if state_bv.is_empty():  # epsilon pop over GSS states
                        for dest_idx in dests:
                            yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            # to_ranges() yields half-open [start, end)
                            for sid in range(start, end):
                                for dest_idx in dests:
                                    yield (int(pop), int(sid), int(dest_idx))

    # --------- Bitset helpers ---------
    @staticmethod
    def _bv_union(a: ffi.Bitset, b: ffi.Bitset) -> ffi.Bitset:
        # Returns a new bitset (a ∪ b)
        return a.union(b)

    @staticmethod
    def _bv_intersect(a: ffi.Bitset, b: ffi.Bitset) -> ffi.Bitset:
        # Prefer "intersection" name; fallback to "intersect" if necessary.
        if hasattr(a, "intersection"):
            return a.intersection(b)
        elif hasattr(a, "intersect"):
            return a.intersect(b)
        # Last resort (should not happen): filter via ranges (slow).
        # This fallback is rarely exercised and only present for safety.
        out = ffi.Bitset.zeros()
        for start, end in a.to_ranges():
            for sid in range(start, end):
                if b.contains(sid):
                    out = out.union(ffi.Bitset.from_json_string(json.dumps({"ranges": [[sid, sid + 1]]})))
        return out

    # --------- Core routine ---------
    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Fast mask computation:
          - Values accumulator per node holds (GSSNode, allowed_bv).
          - allowed_bv flows along edges as intersection with each edge's llm_bv.
          - When reaching end nodes, we add (gss.allowed_llm_tokens() ∩ allowed_bv) to final mask.
          - No token-pruning over GSS on every edge; we only restrict via carried allowed_bv.
        """
        state_to_gss = self.constraint_state.get_state_map()
        final_mask = ffi.Bitset.zeros()

        # values[node_idx] = (aggregated_gss, allowed_bv)
        values: Dict[int, Tuple[ffi.GSSNode, ffi.Bitset]] = {}

        # Nodes that determined to stop (no need to revisit)
        stopped: set[int] = set()

        # Depth scheduler: depth -> set(node_idx)
        todo: Dict[int, set[int]] = defaultdict(set)

        # Seed from tokenizer states
        for sid, gss in state_to_gss.items():
            root_idx = self.roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            # Initial allowed_bv for this root is what current GSS allows
            allowed_bv = gss.allowed_llm_tokens()

            if root_idx in values:
                prev_gss, prev_allowed = values[root_idx]
                merged_gss = ffi.gss_merge_many_with_depth([prev_gss, gss.clone_node()], 999999999)
                merged_allowed = self._bv_union(prev_allowed, allowed_bv)
                values[root_idx] = (merged_gss, merged_allowed)
            else:
                values[root_idx] = (gss.clone_node(), allowed_bv)

            depth = self.max_depth.get(root_idx, 0)
            todo[depth].add(root_idx)

        # Main scheduler loop
        while todo:
            current_depth = min(todo.keys())
            node_indices = todo.pop(current_depth)

            for node_idx in list(node_indices):
                if node_idx in stopped:
                    continue

                agg: Optional[Tuple[ffi.GSSNode, ffi.Bitset]] = values.pop(node_idx, None)
                if agg is None:
                    continue

                agg_gss, allowed_bv = agg

                # End-node contribution
                if self.is_end(node_idx):
                    end_tokens = agg_gss.allowed_llm_tokens()
                    end_tokens = self._bv_intersect(end_tokens, allowed_bv)
                    if not end_tokens.is_empty():
                        final_mask = final_mask.union(end_tokens)

                keep_going = agg_gss.is_ok() and not allowed_bv.is_empty()
                if not keep_going:
                    stopped.add(node_idx)
                    continue

                node_info = self.nodes.get(node_idx)
                if not node_info:
                    continue

                pops = node_info.get("pops", {})

                # For each pop-value bucket: compute peeks once, cache per state_bv
                for pop, edges in pops.items():
                    peeks = ffi.gss_popn_collect(agg_gss, int(pop))
                    if not peeks:
                        continue

                    # Cache matches and merged GSS per state_bv within this (node_idx, pop) processing
                    matched_cache: Dict[int, List[ffi.GSSNode]] = {}
                    merged_cache: Dict[int, ffi.GSSNode] = {}

                    # Process each edge under this pop
                    for edge in edges:
                        llm_bv = edge["llm_bv"]
                        # Propagate allowed_bv by intersecting with this edge's llm_bv
                        new_allowed_bv = self._bv_intersect(allowed_bv, llm_bv)
                        if new_allowed_bv.is_empty():
                            # Nothing can pass through this edge
                            continue

                        # For each (state_bv -> dest_ids) group within this edge
                        for state_key, (state_bv, dest_ids) in edge["groups"].items():
                            # Build matched parents for this state_bv (once per state_bv per pop)
                            matched = matched_cache.get(state_key)
                            if matched is None:
                                if state_bv.is_empty():
                                    # Empty state_bv means epsilon over GSS states - match all parents
                                    matched = [parent for (_sid, parent) in peeks]
                                else:
                                    matched = []
                                    for sid_val, parent in peeks:
                                        if state_bv.contains(sid_val):
                                            matched.append(parent)
                                matched_cache[state_key] = matched

                            if not matched:
                                continue

                            # Merge matched parents (once per state_bv per pop)
                            child_gss = merged_cache.get(state_key)
                            if child_gss is None:
                                child_gss = ffi.gss_merge_many_with_depth(matched, 1)
                                merged_cache[state_key] = child_gss

                            if not child_gss.is_ok():
                                continue

                            # Propagate to each destination, merging values and union-ing allowed_bv
                            for dest_idx in dest_ids:
                                d = int(dest_idx)
                                prev = values.get(d)
                                if prev is None:
                                    values[d] = (child_gss, new_allowed_bv)
                                else:
                                    prev_gss, prev_allowed = prev
                                    combined = ffi.gss_merge_many_with_depth([prev_gss, child_gss], 1)
                                    combined_allowed = self._bv_union(prev_allowed, new_allowed_bv)
                                    values[d] = (combined, combined_allowed)

                                child_depth = self.max_depth.get(d, 0)
                                todo[child_depth].add(d)

        original_mask = self.constraint.internal_bv_to_original(final_mask)
        return RangeSet.from_ranges(original_mask.to_ranges())
