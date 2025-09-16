import json
from typing import Dict, List, Tuple, Optional, Iterable
from collections import deque
from dataclasses import dataclass
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi


@dataclass
class _NodeState:
    # Aggregated state per trie node during get_mask()
    gss_set: set  # set[ffi.GSSNode]
    mask: ffi.Bitset
    mask_hash: int
    in_queue: bool = False


def _bitset_fingerprint(bs: ffi.Bitset) -> int:
    # Compute a stable fingerprint for a Bitset by hashing its ranges.
    # This lets us detect whether a union actually changed the mask without relying on internal APIs.
    rngs = bs.to_ranges()
    # Convert to ints and hash the tuple of ranges
    return hash(tuple((int(a), int(b)) for a, b in rngs))


class Model(GraphProvider):
    """
    Optimized graph model with a high-performance get_mask().

    Key improvements over the baseline:
    - Children grouped by 'pop' to compute GSS popn() once per pop value and reuse across all edges for that pop.
    - For each pop group, we materialize sid->parents only once and reuse across all destination filters.
    - Epsilon state transitions (empty state bitset) are handled correctly and efficiently.
    - Avoid repeated work by:
        - Skipping propagation when the parent's llm mask has no intersection with the edge's llm bitset.
        - Caching the node's gss_set and only recomputing peeks when that set grows.
        - Re-enqueueing children only when their gss_set grows or their mask actually changes (tested via a fingerprint).
    - Uses a simple fixpoint work queue (deque) without per-iteration min() over depth buckets.
    - Deduplicates Bitset instances from JSON via a small parse-time cache to improve memory locality.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Map tokenizer state -> trie root
        self.roots_map: Dict[int, int] = dict((int(s), int(r)) for s, r in roots_map)
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

        # Per-node "end" flags
        self.end_flags: Dict[int, bool] = {}

        # Children grouped by pop:
        # node_id -> { pop (int): List[(llm_bv: Bitset, dests: List[(dest_idx: int, state_bv: Bitset)])] }
        self.children_by_pop: Dict[int, Dict[int, List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]]] = {}

        # Deduplicate Bitset instances during parse
        bitset_cache: Dict[str, ffi.Bitset] = {}

        def parse_bv(obj) -> ffi.Bitset:
            s = json.dumps(obj, separators=(",", ":"))
            bs = bitset_cache.get(s)
            if bs is None:
                bs = ffi.Bitset.from_json_string(s)
                bitset_cache[s] = bs
            return bs

        # Normalize arena nodes
        for uid_raw, node in arena.items():
            uid = int(uid_raw)
            val = node.get("value") or {}
            self.end_flags[uid] = bool(val.get("end", False))

            # Collect children by pop
            pop_map: Dict[int, List[Tuple[ffi.Bitset, List[Tuple[int, ffi.Bitset]]]]] = {}
            ch = node.get("children") or []
            for edge_key, dest_map in ch:
                pop_raw, llm_bv_json = edge_key
                pop = int(pop_raw)
                llm_bv = parse_bv(llm_bv_json)

                dests: List[Tuple[int, ffi.Bitset]] = []
                for dest_idx_raw, state_bv_json in dest_map:
                    dest_idx = int(dest_idx_raw)
                    state_bv = parse_bv(state_bv_json)
                    dests.append((dest_idx, state_bv))

                pop_map.setdefault(pop, []).append((llm_bv, dests))

            self.children_by_pop[uid] = pop_map

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)
        model.constraint_state = ffi.GrammarConstraintState.from_json_string(s)
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return self.end_flags.get(int(node), False)

    def iter_edges(self, node: int, token: int):
        # For equivalence checking, we explode the state bitset in per-token edges,
        # mirroring the baseline implementation and respecting the llm_bv.contains(token) filter.
        node = int(node)
        token = int(token)
        pop_map = self.children_by_pop.get(node)
        if not pop_map:
            return
        for pop, groups in pop_map.items():
            for llm_bv, dests in groups:
                if llm_bv.contains(token):
                    for dest_idx, state_bv in dests:
                        if state_bv.is_empty():
                            # Epsilon transition on the tokenizer state bitset
                            yield (int(pop), None, int(dest_idx))
                        else:
                            for start, end in state_bv.to_ranges():
                                for sid in range(int(start), int(end)):
                                    yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        state_to_gss = self.constraint_state.get_state_to_gss_map()
        # Final mask to return
        final_mask = ffi.Bitset.zeros()

        # Per-node aggregation
        values: Dict[int, _NodeState] = {}

        # Work queue for fixpoint propagation
        q: deque[int] = deque()

        # Seed: map each tokenizer state to its trie root, aggregate GSS clones and llm masks
        for sid_raw, gss in state_to_gss.items():
            sid = int(sid_raw)
            root_idx = self.roots_map.get(sid)
            if root_idx is None:
                continue
            root = int(root_idx)

            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            new_hash = _bitset_fingerprint(new_mask)

            st = values.get(root)
            if st is None:
                st = _NodeState(gss_set={gss_clone}, mask=new_mask, mask_hash=new_hash, in_queue=True)
                values[root] = st
                q.append(root)
            else:
                # Merge GSS set
                before = len(st.gss_set)
                st.gss_set.add(gss_clone)
                gss_changed = len(st.gss_set) != before

                # Merge mask
                merged_mask = st.mask.union(new_mask)
                merged_hash = _bitset_fingerprint(merged_mask)
                mask_changed = merged_hash != st.mask_hash
                if mask_changed:
                    st.mask = merged_mask
                    st.mask_hash = merged_hash

                if (gss_changed or mask_changed) and not st.in_queue:
                    st.in_queue = True
                    q.append(root)

        # Main propagation loop (fixpoint)
        while q:
            node_idx = q.popleft()
            st = values.get(node_idx)
            if st is None:
                # Node got cleared somehow; skip
                continue
            st.in_queue = False

            # If end node, accumulate its mask
            if self.end_flags.get(node_idx, False):
                final_mask = final_mask.union(st.mask)

            # If node has no viable GSS nodes, skip propagation
            # We check viability on the current gss_set.
            gss_ok_list = [g for g in st.gss_set if g]
            if not gss_ok_list:
                continue

            pop_map = self.children_by_pop.get(node_idx)
            if not pop_map:
                continue

            # For each unique pop under this node, collect peeks once and reuse for all llm groups for this pop.
            for pop, groups in pop_map.items():
                # Precompute child masks for groups, and track if any group is relevant
                child_masks: List[ffi.Bitset] = []
                any_relevant = False
                parent_mask = st.mask
                for llm_bv, _dests in groups:
                    # Edge-level llm filter, compute child mask only when non-empty intersection
                    if llm_bv.is_empty():
                        cm = parent_mask
                    else:
                        cm = parent_mask.intersection(llm_bv)
                    child_masks.append(cm)
                    if not cm.is_empty():
                        any_relevant = True

                if not any_relevant:
                    # Parent mask doesn't enable any tokens for this pop; skip expensive popn() calls
                    continue

                # Compute peeks for this pop
                # sid -> list[parent_nodes]
                sid_to_parents: Dict[int, List[ffi.GSSNode]] = {}
                pop_n = int(pop)
                for gss_node in gss_ok_list:
                    for sid_val, parent_node in gss_node.popn_fast(pop_n):
                        sid = int(sid_val)
                        lst = sid_to_parents.get(sid)
                        if lst is None:
                            sid_to_parents[sid] = [parent_node]
                        else:
                            lst.append(parent_node)

                if not sid_to_parents:
                    # No possible transitions for this pop
                    continue

                # Precompute OK parent nodes and filtered per-sid ok lists
                ok_nodes_set: set = set()
                for lst in sid_to_parents.values():
                    for pn in lst:
                        ok_nodes_set.add(pn)

                if not ok_nodes_set:
                    continue

                # Build per-sid ok lists for quick updates
                sid_keys = list(sid_to_parents.keys())
                sid_to_parents_ok: Dict[int, List[ffi.GSSNode]] = {}
                for sid in sid_keys:
                    src_list = sid_to_parents[sid]
                    # filter by ok set
                    dst_list = [pn for pn in src_list if pn in ok_nodes_set]
                    if dst_list:
                        sid_to_parents_ok[sid] = dst_list
                if not sid_to_parents_ok:
                    continue

                # Prepare an "all parents ok" set for epsilon state transitions
                # Note: ok_nodes_set already has unique nodes.
                all_parents_ok_set = ok_nodes_set

                # Now fan out to each (llm_bv group, dests)
                for (llm_bv, dests), child_mask in zip(groups, child_masks):
                    if child_mask.is_empty():
                        continue  # nothing to propagate for this group

                    for dest_idx, state_bv in dests:
                        d = int(dest_idx)
                        dst_state = values.get(d)
                        if dst_state is None:
                            # Lazily construct with the mask now; GSS set will be updated below
                            dst_mask = child_mask
                            dst_state = _NodeState(gss_set=set(), mask=dst_mask, mask_hash=_bitset_fingerprint(dst_mask), in_queue=False)
                            values[d] = dst_state

                        # 1) Update GSS for destination (filtered by state_bv)
                        gss_before = len(dst_state.gss_set)
                        if state_bv.is_empty():
                            # Epsilon on tokenizer state: all ok parents qualify
                            dst_state.gss_set.update(all_parents_ok_set)
                        else:
                            # Only parents whose sid is in state_bv
                            for sid in sid_to_parents_ok.keys():
                                if state_bv.contains(sid):
                                    dst_state.gss_set.update(sid_to_parents_ok[sid])
                        gss_changed = len(dst_state.gss_set) != gss_before

                        # 2) Update llm mask for destination
                        merged_mask = dst_state.mask.union(child_mask)
                        merged_hash = _bitset_fingerprint(merged_mask)
                        mask_changed = merged_hash != dst_state.mask_hash
                        if mask_changed:
                            dst_state.mask = merged_mask
                            dst_state.mask_hash = merged_hash

                        # 3) Enqueue destination if anything changed
                        if (gss_changed or mask_changed) and not dst_state.in_queue:
                            dst_state.in_queue = True
                            q.append(d)

        return RangeSet.from_ranges(final_mask.to_ranges())
