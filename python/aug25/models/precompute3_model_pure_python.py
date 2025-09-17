import json
import time
import heapq
import collections
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm
from gss_tester.fast_impl import FastGSS, _Node as PyGSSNodeInternal


@dataclass(frozen=True)
class PyAcc:
    terminals_union: ffi.HybridL2Bitset


def merge_acc(acc1: PyAcc, acc2: PyAcc) -> PyAcc:
    return PyAcc(terminals_union=acc1.terminals_union.union(acc2.terminals_union))


def convert_rust_gss_to_python_gss(rust_gss_node: ffi.GSSNode) -> FastGSS:
    memo_nodes: Dict[int, PyGSSNodeInternal] = {}
    child_to_parents: Dict[PyGSSNodeInternal, Set[Tuple[int, PyGSSNodeInternal]]] = {}
    q = collections.deque([rust_gss_node])
    visited_rust_ptrs = {rust_gss_node.ptr()}
    while q:
        rust_node = q.popleft()
        py_acc = PyAcc(terminals_union=rust_node.local_acc_terminals_union())
        py_node = PyGSSNodeInternal(acc=py_acc, depth=rust_node.depth())
        memo_nodes[rust_node.ptr()] = py_node
        for _, pred_rust_node in rust_node.predecessors():
            if pred_rust_node.ptr() not in visited_rust_ptrs:
                visited_rust_ptrs.add(pred_rust_node.ptr())
                q.append(pred_rust_node)
    q = collections.deque([rust_gss_node])
    visited_rust_ptrs = {rust_gss_node.ptr()}
    while q:
        rust_node = q.popleft()
        py_node = memo_nodes[rust_node.ptr()]
        for state_id, pred_rust_node in rust_node.predecessors():
            py_pred_node = memo_nodes[pred_rust_node.ptr()]
            child_to_parents.setdefault(py_node, set()).add((state_id, py_pred_node))
            if pred_rust_node.ptr() not in visited_rust_ptrs:
                visited_rust_ptrs.add(pred_rust_node.ptr())
                q.append(pred_rust_node)
    py_head_node = memo_nodes[rust_gss_node.ptr()]
    def acc_factory():
        return PyAcc(terminals_union=ffi.HybridL2Bitset.all())
    py_root_node = next((node for node in memo_nodes.values() if node.depth == 0), PyGSSNodeInternal(acc=acc_factory(), depth=0))
    return FastGSS(heads=frozenset([py_head_node]), acc_default_factory=acc_factory, root=py_root_node, child_to_parents=child_to_parents, path_cache={})


def popn_fast_py(gss: FastGSS, n: int) -> List[Tuple[int, FastGSS]]:
    current_heads = gss._heads
    for _ in range(n):
        next_heads = set()
        for head in current_heads:
            if head in gss._child_to_parents:
                for _, parent in gss._child_to_parents[head]:
                    next_heads.add(parent)
        current_heads = next_heads
        if not current_heads:
            return []
    result = []
    for head in current_heads:
        if head in gss._child_to_parents:
            for state_id, parent_node in gss._child_to_parents[head]:
                isolated_gss = FastGSS(heads=frozenset([parent_node]), acc_default_factory=gss._acc_default_factory, root=gss._root, child_to_parents=gss._child_to_parents, path_cache=gss._path_cache)
                result.append((state_id, isolated_gss))
    return result


def get_disallowed_terminals_py(gss: FastGSS) -> ffi.HybridL2Bitset:
    if not gss._heads:
        return ffi.HybridL2Bitset.all().complement()
    it_heads = iter(gss._heads)
    try:
        first_head = next(it_heads)
        allowed_terminals = first_head.acc.terminals_union
        for head in it_heads:
            allowed_terminals = allowed_terminals.union(head.acc.terminals_union)
    except StopIteration:
        return ffi.HybridL2Bitset.all().complement()
    return allowed_terminals.complement()


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation).
    Normalizes input arena by converting JSON bitsets into ffi.Bitset instances
    and provides graph traversal and mask computation interfaces.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Map tokenizer state -> trie root node
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.constraint: Optional[ffi.GrammarConstraint] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None
        self.id_to_token: Dict[int, bytes] = {}
        self.max_depth: Dict[int, int] = {}
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = None

        # Normalize arena children bitsets and cache max_depth
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        for uid, node in tqdm(
            self.arena.items(),
            desc="Normalizing precompute3 BVs",
            total=len(self.arena),
        ):
            uid_int = int(uid)
            try:
                md = node.get("max_depth", 0)
                self.max_depth[uid_int] = int(md)
            except Exception:
                self.max_depth[uid_int] = 0

            children = node.get("children") or []
            if not children:
                node["children"] = []
                continue

            new_children = []
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv = bs_from_json(dumps(llm_bv_json))

                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv = bs_from_json(dumps(state_bv_json))
                    new_dest_map.append((int(dest_idx), state_bv))

                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children

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
        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        model.possible_matches_cache = model.constraint.possible_matches()
        return model

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        Only used by equivalence checking; not performance-critical.
        """
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():  # Epsilon on GSS stack
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end + 1):
                                yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        self.constraint_state.commit(token_id)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        print("\n--- get_mask START ---")
        print("GSS at start of get_mask:")
        rust_state_map = self.constraint_state.get_state_map()
        state_map = {sid: convert_rust_gss_to_python_gss(gss) for sid, gss in rust_state_map.items()}

        all_ones_mask = self.constraint.all_internal_llm_tokens_bitset()

        t0 = time.time()

        final_mask = ffi.Bitset.zeros()

        # node_idx -> (FastGSS, Bitset)
        values: Dict[int, Tuple[FastGSS, ffi.Bitset]] = {}

        stopped: set[int] = set()  # nodes that stopped (no gss parents)
        todo: Dict[int, set[int]] = {}  # depth -> set(node_idx)
        depth_heap: List[int] = []  # min-heap of depths (may contain duplicates)

        # Seed: map tokenizer states and their filtered GSS to trie roots
        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        print("\n--- Seeding work queue ---")
        for sid, gss in state_map.items():
            new_mask = all_ones_mask
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_heads={[h.id for h in gss._heads]}, mask={new_mask.to_ranges()}")

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                merged_gss = FastGSS.merge([existing_gss, gss], merge_acc)
                values[root_idx] = (merged_gss, existing_mask.union(new_mask))
            else:
                values[root_idx] = (gss, new_mask)

            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        # Main scheduler

        # Helper to enqueue a node at a given depth
        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        heappop = heapq.heappop
        arena = self.arena
        is_end = self.is_end

        print("\n--- Main loop ---")
        iter_count = 0
        while True:
            iter_count += 1
            # Pop the smallest depth bucket (skip stale heap entries)
            node_indices: Optional[set[int]] = None
            current_depth = -1
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                print(f"[{iter_count}] Loop finished: no more nodes to process.")
                break  # nothing left to process

            print(f"\n[{iter_count}] Processing depth={current_depth}, nodes={node_indices}")

            # Process all nodes in this depth bucket
            for node_idx in node_indices:
                if node_idx in stopped:
                    print(f"  - Node {node_idx}: SKIPPING (already stopped)")
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    print(f"  - Node {node_idx}: SKIPPING (no value)")
                    continue
                gss_node, llm_mask = item
                print(f"  - Node {node_idx}: Popped gss_heads={[h.id for h in gss_node._heads]}, mask={llm_mask.to_ranges()}")

                # End-node handling
                if is_end(node_idx):
                    print(f"    - END NODE found. Updating final_mask.")
                    print(f"      - final_mask before: {final_mask.to_ranges()}")

                    # Calculate forbidden_llm_tokens based on GSS's disallowed terminals
                    forbidden_llm_tokens = ffi.Bitset.zeros()
                    disallowed_terminals_l2 = get_disallowed_terminals_py(gss_node)
                    possible_matches = self.possible_matches_cache

                    for (start, end), disallowed_bv in disallowed_terminals_l2.range_values():
                        if disallowed_bv.is_empty():
                            continue

                        for tsid in range(start, end + 1):
                            possible_matches_for_state = possible_matches.get(tsid)
                            if not possible_matches_for_state:
                                continue

                            for terminal_id_str, llm_tokens_for_terminal in possible_matches_for_state.items():
                                terminal_id = int(terminal_id_str)
                                if disallowed_bv.contains(terminal_id):
                                    forbidden_llm_tokens = forbidden_llm_tokens.union(llm_tokens_for_terminal)

                    gss_active_tokens = all_ones_mask
                    glr_active_tokens = llm_mask.intersection(gss_active_tokens)
                    final_allowed_tokens = glr_active_tokens.difference(forbidden_llm_tokens)
                    tokens_to_add = final_allowed_tokens

                    print(f"      - llm_mask (propagated): {llm_mask.to_ranges()}")
                    print(f"      - gss_active_tokens (from GSS): {gss_active_tokens.to_ranges()}")
                    print(f"      - tokens_to_add (intersection): {tokens_to_add.to_ranges()}")

                    final_mask = final_mask.union(tokens_to_add)
                    print(f"      - final_mask after:  {final_mask.to_ranges()}")

                if llm_mask.is_empty():
                    stopped.add(node_idx)
                    print(f"    - STOPPING node {node_idx} (GSS not alive)")
                    continue

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                # if not children:
                #     print(f"    - No children for node {node_idx}")
                for (pop, llm_bv), dests in children:
                    print(f"    - Edge: pop={pop}, llm_bv={llm_bv.to_ranges()}")
                    # Collect all pops from GSS parents
                    peeks = popn_fast_py(gss_node, pop)
                    print(f"      - Found {len(peeks)} peeks from GSS set")
                    if not peeks:
                        continue

                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        print(f"      - Dest: idx={dest_idx}, state_bv={state_bv.to_ranges()}")
                        # Filter peeks by destination state bitset
                        matched = []
                        if not state_bv.is_empty():
                            contains = state_bv.contains
                            for sid_val, parent_node in peeks:
                                if contains(sid_val):
                                    matched.append(parent_node)
                        print(f"        - Matched {len(matched)} parent GSS nodes")
                        if not matched:
                            continue

                        # Merge matched parent GSS nodes
                        child_gss_node = FastGSS.merge(matched, merge_acc)

                        # Compute child mask (intersection with llm_bv when present)
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)
                        print(f"        - Child mask: {child_llm_mask.to_ranges()}")

                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            merged_gss = FastGSS.merge([existing_gss, child_gss_node], merge_acc)
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (merged_gss, combined_mask)
                            print(f"        - Enqueue {d}: UPDATING gss_heads={[h.id for h in merged_gss._heads]}, mask={combined_mask.to_ranges()}")
                        else:
                            values[d] = (child_gss_node, child_llm_mask)
                            print(f"        - Enqueue {d}: CREATING gss_heads={[h.id for h in child_gss_node._heads]}, mask={child_llm_mask.to_ranges()}")

                        enqueue(max_depth[d], d)

        original_mask = self.constraint.internal_bv_to_original(final_mask)
        temp = RangeSet.from_ranges(original_mask.to_ranges())
        print(f"\n--- get_mask END ---")
        print(f"Final mask: {temp.to_ranges()}")
        return temp
