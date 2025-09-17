import json
from typing import Dict, List, Tuple, Optional, Iterable
import time
from collections import deque
from dataclasses import dataclass
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi


@dataclass
class _NodeState:
    # Aggregated state per trie node during get_mask()
    gss_node: ffi.GSSNode
    mask: ffi.Bitset # set[ffi.GSSNode]
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
        self.constraint: Optional[ffi.GrammarConstraint] = None
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
            self.end_flags[uid] = bool(val.get("clean_end", False))

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
        model.constraint = ffi.GrammarConstraint.from_json_string(s)
        model.constraint_state = ffi.GrammarConstraintState(model.constraint)
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
        print("\n--- get_mask START ---")
        print(self.constraint_state)
        state_to_gss = self.constraint_state.filtered_state_gss_map()
        print(f"Filtered state_to_gss: { {k: v.ptr() for k, v in state_to_gss.items()} }")

        t0 = time.time()
        # Final mask to return
        # Final mask to return
        final_mask = ffi.Bitset.zeros()

        # Per-node aggregation
        values: Dict[int, _NodeState] = {}

        # Work queue for fixpoint propagation
        q: deque[int] = deque()

        # Seed: map each tokenizer state to its trie root, aggregate GSS clones and llm masks
        print("\n--- Seeding work queue ---")
        for sid_raw, gss in state_to_gss.items():
            sid = int(sid_raw)
            root_idx = self.roots_map.get(sid)
            if root_idx is None:
                continue
            root = int(root_idx)


            gss_clone = gss.clone_node()
            new_mask = gss_clone.allowed_llm_tokens()
            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_ptr={gss_clone.ptr()}, mask={new_mask.to_ranges()}")
            new_hash = _bitset_fingerprint(new_mask)

            st = values.get(root)
            if st is None:
                st = _NodeState(gss_node=gss_clone, mask=new_mask, mask_hash=new_hash, in_queue=True)
                values[root] = st
                q.append(root)
            else:
                print(f"    - MERGE: gss1_ptr={st.gss_node.ptr()}, mask1={st.mask.to_ranges()} WITH gss2_ptr={gss_clone.ptr()}, mask2={new_mask.to_ranges()}")
                # Merge GSS set
                merged_gss = ffi.gss_merge_many_with_depth([st.gss_node, gss_clone], 1)
                gss_changed = merged_gss.ptr() != st.gss_node.ptr()
                if gss_changed:
                    st.gss_node = merged_gss

                # Merge mask
                merged_mask = st.mask.union(new_mask)
                merged_hash = _bitset_fingerprint(merged_mask)
                mask_changed = merged_hash != st.mask_hash
                if mask_changed:
                    st.mask = merged_mask
                    st.mask_hash = merged_hash
                print(f"      - Merged result: gss_ptr={st.gss_node.ptr()}, mask={st.mask.to_ranges()}")

                if (gss_changed or mask_changed) and not st.in_queue:
                    st.in_queue = True
                    q.append(root)

        # Main propagation loop (fixpoint)
        print("\n--- Main loop ---")

        # Main propagation loop (fixpoint)
        print("\n--- Main loop ---")
        iter_count = 0
        while q:
            iter_count += 1
            node_idx = q.popleft()
            print(f"\n[{iter_count}] Processing node={node_idx}")
            st = values.get(node_idx)
            if st is None:
                print(f"  - Node {node_idx}: SKIPPING (no value)")
                # Node got cleared somehow; skip
                continue
            st.in_queue = False
            print(f"  - PROCESS: node_ptr={node_idx}, gss_ptr={st.gss_node.ptr()}, mask={st.mask.to_ranges()}")

            # If end node, accumulate its mask

            # If end node, accumulate its mask
            if self.end_flags.get(node_idx, False):
                print(f"    - END NODE found. Updating final_mask.")
                print(f"      - final_mask before: {final_mask.to_ranges()}")
                gss_active_tokens = st.gss_node.allowed_llm_tokens()
                tokens_to_add = st.mask.intersection(gss_active_tokens)
                print(f"      - glr_active_tokens to union: {tokens_to_add.to_ranges()}")
                final_mask = final_mask.union(tokens_to_add)
                print(f"      - final_mask after:  {final_mask.to_ranges()}")

            # If node has no viable GSS nodes, skip propagation
            if not st.gss_node.is_alive():
                print(f"    - STOPPING node {node_idx} (GSS not alive)")
                continue

            pop_map = self.children_by_pop.get(node_idx)
            if not pop_map:
                continue

            # For each unique pop under this node, collect peeks once and reuse for all llm groups for this pop.

            # For each unique pop under this node, collect peeks once and reuse for all llm groups for this pop.
            for pop, groups in pop_map.items():
                print(f"    - Edge group: pop={pop}")
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
                # Compute peeks for this pop
                # sid -> list[parent_nodes]
                sid_to_parents: Dict[int, List[ffi.GSSNode]] = {}
                peeks = st.gss_node.popn_fast(int(pop))
                print(f"      - Found {len(peeks)} peeks from GSS")
                if not peeks:
                    # No possible transitions for this pop
                    continue

                for sid_val, parent_node in peeks:
                    sid = int(sid_val)
                    lst = sid_to_parents.get(sid)
                    if lst is None:
                        sid_to_parents[sid] = [parent_node]
                    else:
                        lst.append(parent_node)

                # Now fan out to each (llm_bv group, dests)

                # Now fan out to each (llm_bv group, dests)
                for (llm_bv, dests), child_mask in zip(groups, child_masks):
                    print(f"    - Edge: llm_bv={llm_bv.to_ranges()}")
                    print(f"      - Child mask: {child_mask.to_ranges()}")
                    if child_mask.is_empty():
                        continue  # nothing to propagate for this group

                    # This logic is complex; let's simplify by merging parents per destination

                    # This logic is complex; let's simplify by merging parents per destination
                    dest_to_parents = defaultdict(list)


                    all_parents_list = [p for _, p in peeks]

                    for dest_idx, state_bv in dests:
                        print(f"      - Dest: idx={dest_idx}, state_bv={state_bv.to_ranges()}")
                        d = int(dest_idx)
                        dst_state = values.get(d)
                        if dst_state is None:
                            # Lazily construct with the mask now; GSS set will be updated below
                            dst_mask = child_mask
                            # Placeholder GSS node, will be replaced
                            dst_state = _NodeState(gss_node=ffi.gss_merge_many_with_depth([], 1), mask=dst_mask, mask_hash=_bitset_fingerprint(dst_mask), in_queue=False)
                            values[d] = dst_state

                        parents_for_dest = []
                        if state_bv.is_empty():
                            parents_for_dest.extend(all_parents_list)
                        else:
                            for sid, parents in sid_to_parents.items():
                                if state_bv.contains(sid):
                                    parents_for_dest.extend(parents)

                        print(f"        - Matched {len(parents_for_dest)} parent GSS nodes")
                        if not parents_for_dest:
                            continue

                        # 1) Update GSS for destination
                        print(f"        - Enqueue {d}: MERGING gss1_ptr={dst_state.gss_node.ptr()}, mask1={dst_state.mask.to_ranges()} WITH gss2_ptr={child_gss.ptr()}, mask2={child_mask.to_ranges()}")
                        if not dst_state.gss_node.is_alive(): # Was placeholder
                            merged_gss = child_gss
                        else:
                            merged_gss = ffi.gss_merge_many_with_depth([dst_state.gss_node, child_gss], 1)

                        gss_changed = merged_gss.ptr() != dst_state.gss_node.ptr()
                        dst_state.gss_node = merged_gss

                        # 2) Update llm mask for destination
                        merged_mask = dst_state.mask.union(child_mask)
                        merged_hash = _bitset_fingerprint(merged_mask)
                        mask_changed = merged_hash != dst_state.mask_hash
                        if mask_changed:
                            dst_state.mask = merged_mask
                            dst_state.mask_hash = merged_hash
                        print(f"          - Merged result: gss_ptr={dst_state.gss_node.ptr()}, mask={dst_state.mask.to_ranges()}")

                        # 3) Enqueue destination if anything changed
                        if (gss_changed or mask_changed) and not dst_state.in_queue:
                            dst_state.in_queue = True
                            q.append(d)

        print(f"\n--- get_mask END (took {time.time() - t0:.4f}s) ---")
        print(f"Final mask internal: {final_mask.to_ranges()}")
        original_mask = self.constraint.internal_bv_to_original(final_mask)
        print(f"Final mask mapped: {original_mask.to_ranges()}")
        return RangeSet.from_ranges(original_mask.to_ranges())
