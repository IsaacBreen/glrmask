import json
import time
import heapq
import collections
from typing import Dict, List, Tuple, Optional, Set

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm

from gss_tester.fast_impl import FastGSS, _Node


def merge_acc(acc1, acc2):
    term1, llm1 = acc1
    term2, llm2 = acc2
    return (term1.union(term2), llm1.union(llm2))


def rust_gss_to_fast_gss(rust_gss_head: ffi.GSSNode) -> FastGSS:
    memo = {}  # rust_ptr -> python _Node
    child_to_parents = {}

    q = collections.deque([rust_gss_head])
    visited_ptrs = {rust_gss_head.ptr()}

    while q:
        rust_node = q.popleft()
        ptr = rust_node.ptr()

        if ptr not in memo:
            memo[ptr] = _Node(
                acc=(rust_node.local_acc_terminals_union(), rust_node.local_acc_llm_tokens_union()),
                depth=rust_node.depth()
            )
        py_node = memo[ptr]

        if not rust_node.is_root():
            parents = set()
            for state_id, pred_rust_node in rust_node.predecessors():
                pred_ptr = pred_rust_node.ptr()
                if pred_ptr not in memo:
                    memo[pred_ptr] = _Node(
                        acc=(pred_rust_node.local_acc_terminals_union(), pred_rust_node.local_acc_llm_tokens_union()),
                        depth=pred_rust_node.depth()
                    )
                py_pred_node = memo[pred_ptr]
                parents.add((state_id, py_pred_node))

                if pred_ptr not in visited_ptrs:
                    q.append(pred_rust_node)
                    visited_ptrs.add(pred_ptr)

            child_to_parents[py_node] = parents

    py_head = memo[rust_gss_head.ptr()]
    py_root = next((node for node in memo.values() if node.depth == 0), None)
    if py_root is None:
        raise ValueError("Could not find root of GSS graph")

    return FastGSS(
        heads=frozenset([py_head]),
        acc_default_factory=lambda: py_root.acc,
        root=py_root,
        child_to_parents=child_to_parents,
        path_cache={}
    )


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
            break

    nodes_at_n_heads = current_heads

    peeks = []
    for node in nodes_at_n_heads:
        if node in gss._child_to_parents:
            for value, parent in gss._child_to_parents[node]:
                isolated_parent_gss = FastGSS(
                    heads=frozenset([parent]),
                    acc_default_factory=gss._acc_default_factory,
                    root=gss._root,
                    child_to_parents=gss._child_to_parents,
                    path_cache=gss._path_cache
                )
                peeks.append((value, isolated_parent_gss))
    return peeks


def get_allowed_llm_tokens(gss: FastGSS) -> ffi.Bitset:
    q = collections.deque(list(gss._heads))
    visited = set()
    root_accs_llm = []

    while q:
        node = q.popleft()
        if node in visited:
            continue
        visited.add(node)

        if node.depth == 0:
            root_accs_llm.append(node.acc[1])

        if node in gss._child_to_parents:
            for _, parent in gss._child_to_parents[node]:
                q.append(parent)

    if not root_accs_llm:
        aggregated_llm = ffi.Bitset.zeros()
    else:
        aggregated_llm = root_accs_llm[0]
        for i in range(1, len(root_accs_llm)):
            aggregated_llm = aggregated_llm.union(root_accs_llm[i])

    final_llm = ffi.Bitset.zeros()
    for head in gss._heads:
        local_llm = head.acc[1]
        final_llm = final_llm.union(local_llm.intersection(aggregated_llm))

    return final_llm


def is_alive_py(gss: FastGSS) -> bool:
    return not get_allowed_llm_tokens(gss).is_empty()


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
        t0 = time.time()

        final_mask = ffi.Bitset.zeros()
        values: Dict[int, Tuple[FastGSS, ffi.Bitset]] = {}
        stopped: Set[int] = set()
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        heappush = heapq.heappush
        roots_map = self.roots_map
        max_depth = self.max_depth

        state_to_gss_and_mask = self.constraint_state.filtered_state_gss_map()

        py_gss_map = {}
        for sid, (gss, new_mask) in state_to_gss_and_mask.items():
            py_gss_map[sid] = (rust_gss_to_fast_gss(gss), new_mask)

        for sid, (gss, new_mask) in py_gss_map.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                merged_gss = FastGSS.merge([existing_gss, gss], merge_acc)
                values[root_idx] = (merged_gss, existing_mask.union(new_mask))
            else:
                values[root_idx] = (gss, new_mask)

            depth = max_depth[root_idx]
            if depth not in todo:
                todo[depth] = set()
                heappush(depth_heap, depth)
            todo[depth].add(root_idx)

        def enqueue(depth: int, node_idx: int) -> None:
            if depth not in todo:
                todo[depth] = set()
                heappush(depth_heap, depth)
            todo[depth].add(node_idx)

        heappop = heapq.heappop
        arena = self.arena
        is_end = self.is_end

        iter_count = 0
        while True:
            iter_count += 1
            node_indices: Optional[set[int]] = None
            current_depth = -1
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                break

            for node_idx in node_indices:
                if node_idx in stopped:
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    continue
                gss_node, llm_mask = item

                if is_end(node_idx):
                    gss_active_tokens = get_allowed_llm_tokens(gss_node)
                    tokens_to_add = llm_mask.intersection(gss_active_tokens)
                    final_mask = final_mask.union(tokens_to_add)

                if not is_alive_py(gss_node):
                    stopped.add(node_idx)
                    continue

                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                for (pop, llm_bv), dests in children:
                    peeks = popn_fast_py(gss_node, pop)
                    if not peeks:
                        continue

                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        matched = []
                        if not state_bv.is_empty():
                            contains = state_bv.contains
                            for sid_val, parent_node in peeks:
                                if contains(sid_val):
                                    matched.append(parent_node)
                        if not matched:
                            continue

                        child_gss_node = FastGSS.merge(matched, merge_acc)
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)

                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            merged_gss = FastGSS.merge([existing_gss, child_gss_node], merge_acc)
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (merged_gss, combined_mask)
                        else:
                            values[d] = (child_gss_node, child_llm_mask)

                        enqueue(max_depth[d], d)

        original_mask = self.constraint.internal_bv_to_original(final_mask)
        return RangeSet.from_ranges(original_mask.to_ranges())
