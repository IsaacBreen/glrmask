import json
import heapq
import collections
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm
from gss_tester.fast_impl import FastGSS, _Node as PyGSSNodeInternal

@dataclass(frozen=True)
class Reduce:
    nonterminal_id: int
    len: int
    production_ids: Tuple[int, ...]

@dataclass(frozen=True)
class Split:
    shift: Optional[int]
    reduces: Dict[int, Dict[int, Tuple[int, ...]]] # len -> nt_id -> pids

# Action can be a Shift (int), Reduce, or Split
Action = Union[int, Reduce, Split]

@dataclass
class Row:
    actions: Dict[int, Action] = field(default_factory=dict) # terminal_id -> Action
    gotos: Dict[int, int] = field(default_factory=dict) # nonterminal_id -> state_id

@dataclass
class ParserTable:
    start_state_id: int
    table: Dict[int, Row]

@dataclass(frozen=True)
class PyAcc:
    terminals_union: ffi.HybridL2Bitset

    def __hash__(self):
        return hash(self.terminals_union)

    def to_json_serializable(self):
        return {"terminals_union": self.terminals_union.to_json_serializable()}

    def __repr__(self):
        return f"PyAcc({self.terminals_union})"


def merge_acc(acc1: PyAcc, acc2: PyAcc) -> PyAcc:
    return PyAcc(terminals_union=acc1.terminals_union.union(acc2.terminals_union))

def get_disallowed_terminals_py(gss: FastGSS) -> ffi.HybridL2Bitset:
    merged_acc = gss.get_acc(merge_acc)
    return merged_acc.terminals_union.complement()

def popn_fast_py(gss: FastGSS, n: int) -> FastGSS:
    result = gss
    for _ in range(n):
        result = result.pop()
    return result

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
        self.id_to_token: Dict[int, bytes] = {}
        self.max_depth: Dict[int, int] = {}
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = None
        self.tokenizer: Optional[ffi.Regex] = None
        self.parser_table: Optional[ParserTable] = None
        self.state: Dict[int, FastGSS] = {}
        self.internal_to_original_map: Dict[int, int] = {}
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = None
        self.constraint_state: Optional[ffi.GrammarConstraintState] = None

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

        # Load tokenizer and parser table from the full constraint JSON
        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.tokenizer = constraint.tokenizer()
        model.tokenizer_initial_state = model.tokenizer.initial_state_id()

        parser_data = data['parser']
        table_data = parser_data['stage_7_table']
        start_state_id = parser_data['start_state_id']
        py_table = {}
        for state_id_str, row_data in table_data:
            state_id = int(state_id_str)
            py_row = Row()
            # Parse actions from 'shifts_and_reduces_full'
            for term_id_str, action_data in row_data['shifts_and_reduces_full']:
                term_id = int(term_id_str)
                variant = action_data['variant']
                if variant == 'Shift':
                    py_row.actions[term_id] = action_data['state_id']
                elif variant == 'Reduce':
                    pids = tuple(sorted(action_data['production_ids']))
                    py_row.actions[term_id] = Reduce(action_data['nonterminal_id'], action_data['len'], pids)
                elif variant == 'Split':
                    shift = action_data['shift']
                    reduces = {}
                    for len_str, nts_data in action_data['reduces']:
                        len_int = int(len_str)
                        nts = {}
                        for nt_id_str, pids in nts_data:
                            nt_id_int = int(nt_id_str)
                            nts[nt_id_int] = tuple(sorted(pids))
                        reduces[len_int] = nts
                    py_row.actions[term_id] = Split(shift, reduces)
            # Parse gotos
            for nt_id_str, goto_data in row_data['gotos']:
                nt_id = int(nt_id_str)
                if goto_data['state_id'] is not None:
                    py_row.gotos[nt_id] = goto_data['state_id']
            py_table[state_id] = py_row
        model.parser_table = ParserTable(start_state_id, py_table)

        def acc_factory():
            return PyAcc(terminals_union=ffi.HybridL2Bitset.all())
        initial_gss = FastGSS.initial(acc_factory).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        model.possible_matches_cache = constraint.possible_matches()
        model.internal_to_original_map = constraint.internal_to_original_map()
        model.all_internal_llm_tokens_bitset = constraint.all_internal_llm_tokens_bitset()

        model.constraint_state = ffi.GrammarConstraintState(constraint)

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
        token_bytes = self.id_to_token.get(token_id)
        if not token_bytes:
            self.state = {}
            return

        new_states: Dict[int, List[FastGSS]] = collections.defaultdict(list)

        q = collections.deque()
        for tokenizer_sid, gss in self.state.items():
            q.append((0, tokenizer_sid, gss)) # offset, tokenizer_state, gss

        visited_q_items = set()

        while q:
            offset, tokenizer_sid, gss = q.popleft()

            # GSS is not hashable, use its serializable form for visited check
            q_item = (offset, tokenizer_sid, gss)
            if q_item in visited_q_items:
                continue
            visited_q_items.add(q_item)

            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tokenizer_sid)

            for terminal_id, width in matches:
                gss_after_step = self._process_token(gss, terminal_id)
                if any(h is not gss_after_step._root for h in gss_after_step._heads):
                    # After a grammar token is consumed, disallow it from being matched again immediately
                    # if the tokenizer could produce it from its new state.
                    disallowed_terminals = ffi.HybridL2Bitset.new()
                    if end_state is not None:
                        terminals_accessible = self.tokenizer.tokens_accessible_from_state(end_state)
                        if terminal_id in terminals_accessible:
                            disallowed_bv = ffi.Bitset.from_indices([terminal_id])
                            disallowed_terminals.insert_l2_bitset(end_state, disallowed_bv)
                    
                    def disallow_func(acc: PyAcc) -> PyAcc:
                        return PyAcc(terminals_union=acc.terminals_union.__sub__(disallowed_terminals))
                    
                    final_gss = gss_after_step.apply_to_all(disallow_func)

                    new_offset = offset + width
                    next_tokenizer_sid = self.tokenizer_initial_state
                    if new_offset == len(token_bytes):
                        new_states[next_tokenizer_sid].append(final_gss)
                    else:
                        q.append((new_offset, next_tokenizer_sid, final_gss))

            if end_state is not None:
                new_states[end_state].append(gss)

        merged_states = {
            sid: FastGSS.merge(gss_list, merge_acc)
            for sid, gss_list in new_states.items()
            if gss_list
        }

        self.state = merged_states

    def _process_token(self, gss: FastGSS, terminal_id: int) -> FastGSS:
        heads_by_state: Dict[int, List[PyGSSNodeInternal]] = collections.defaultdict(list)
        for head in gss._heads:
            for state_id in gss.peek_from_head(head):
                heads_by_state[state_id].append(head)

        shifted_gsses = []
        reductions_to_do: Dict[Reduce, List[FastGSS]] = collections.defaultdict(list)

        for state_id, heads in heads_by_state.items():
            state_gss = FastGSS(frozenset(heads), gss._acc_default_factory, gss._root, gss._child_to_parents, gss._path_cache)
            row = self.parser_table.table.get(state_id)
            if not row: continue
            action = row.actions.get(terminal_id)
            if not action: continue

            def handle_shift(shift_to_state_id, gss_to_shift):
                shifted_gsses.append(gss_to_shift.push(shift_to_state_id))

            def handle_reduce(reduce_action, gss_to_reduce):
                popped_gss = gss_to_reduce.popn(reduce_action.len)
                if any(h is not popped_gss._root for h in popped_gss._heads):
                    reductions_to_do[reduce_action].append(popped_gss)

            if isinstance(action, int):
                handle_shift(action, state_gss)
            elif isinstance(action, Reduce):
                handle_reduce(action, state_gss)
            elif isinstance(action, Split):
                if action.shift is not None:
                    handle_shift(action.shift, state_gss)
                for length, nts in action.reduces.items():
                    for nt_id, pids in nts.items():
                        handle_reduce(Reduce(nt_id, length, pids), state_gss)

        for reduce_action, gss_list in reductions_to_do.items():
            merged_popped_gss = FastGSS.merge(gss_list, merge_acc)
            for from_state_id in merged_popped_gss.peek():
                goto_state_id = self.parser_table.table[from_state_id].gotos.get(reduce_action.nonterminal_id)
                if goto_state_id is not None:
                    shifted_gsses.append(merged_popped_gss.isolate(from_state_id).push(goto_state_id))

        return FastGSS.merge(shifted_gsses, merge_acc) if shifted_gsses else FastGSS.initial(gss._acc_default_factory)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """

        print("\n--- get_mask START ---")
        print("GSS at start of get_mask:")
        state_map = self.state

        expected_state_map = {sid: convert_rust_gss_to_python_gss(rust_gss) for sid, rust_gss in self.constraint_state.get_state_map().items()}

        print(f"state_map:")
        for sid, gss in state_map.items():
            print(gss)

        print(f"expected_state_map:")
        for sid, gss in expected_state_map.items():
            print(gss)

        assert state_map == expected_state_map, f"state_map != expected_state_map: {state_map.keys() ^ expected_state_map.keys()}"

        all_ones_mask = self.all_internal_llm_tokens_bitset

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

            print(f"  SEED: sid={sid}, root_idx={root_idx}, gss_heads={[h.id for h in gss._heads]}, mask={new_mask}")

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
                print(f"  - Node {node_idx}: Popped gss_heads={[h.id for h in gss_node._heads]}, mask={llm_mask}")

                # End-node handling
                if is_end(node_idx):
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
                    if not final_allowed_tokens.is_empty():
                        before_len = final_mask.len()
                        final_mask = final_mask.union(final_allowed_tokens)
                        after_len = final_mask.len()
                        if after_len > before_len:
                            print(f"    - END NODE. final_mask len: {before_len} -> {after_len} (+{after_len - before_len}) with tokens {final_allowed_tokens}")

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
                    print(f"    - Edge: pop={pop}, llm_bv={llm_bv}")
                    # Collect all pops from GSS parents
                    popped = popn_fast_py(gss_node, pop)

                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        # Filter peeks by destination state bitset
                        matched = []
                        if not state_bv.is_empty():
                            for sid_val in popped.peek():
                                if state_bv.contains(sid_val):
                                    matched.append(popped.isolate(sid_val))
                        if not matched:
                            continue

                        # Merge matched parent GSS nodes
                        child_gss_node = FastGSS.merge(matched, merge_acc)

                        # Compute child mask (intersection with llm_bv when present)
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)

                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            merged_gss = FastGSS.merge([existing_gss, child_gss_node], merge_acc)
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (merged_gss, combined_mask)
                            print(f"      - Dest: idx={d}, state_bv={state_bv}, matched={len(matched)}, child_mask={child_llm_mask}")
                            print(f"        -> UPDATING gss_heads={[h.id for h in merged_gss._heads]}, mask={combined_mask}")
                        else:
                            values[d] = (child_gss_node, child_llm_mask)
                            print(f"      - Dest: idx={d}, state_bv={state_bv}, matched={len(matched)}, child_mask={child_llm_mask}")
                            print(f"        -> CREATING gss_heads={[h.id for h in child_gss_node._heads]}, mask={child_llm_mask}")

                        enqueue(max_depth[d], d)

        original_mask = ffi.Bitset.zeros()
        for internal_id in final_mask.to_indices():
            if internal_id in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[internal_id])
        temp = RangeSet.from_ranges(original_mask.to_ranges())
        print(f"\n--- get_mask END ---")
        print(f"Final mask: {temp.to_ranges()}")
        return temp
