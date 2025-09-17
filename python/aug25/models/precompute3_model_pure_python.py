import json
import time
import heapq
import collections
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm
from gss_tester.fast_impl import FastGSS, _Node as PyGSSNodeInternal


@dataclass(frozen=True)
class Shift:
    state_id: int

@dataclass(frozen=True)
class Reduce:
    nonterminal_id: int
    len: int
    production_ids: Tuple[int, ...]

@dataclass(frozen=True)
class Split:
    shift: Optional[int]
    reduces: Tuple[Reduce, ...]

Action = Union[Shift, Reduce, Split]

@dataclass(frozen=True)
class Goto:
    state_id: Optional[int]
    accept: bool

@dataclass
class Row:
    actions: Dict[int, Action]  # terminal_id -> Action
    gotos: Dict[int, Goto]      # nonterminal_id -> Goto


@dataclass(frozen=True)
class PyAcc:
    terminals_union: ffi.HybridL2Bitset


def merge_acc(acc1: PyAcc, acc2: PyAcc) -> PyAcc:
    return PyAcc(terminals_union=acc1.terminals_union.union(acc2.terminals_union))

def popn_fast_py(gss: FastGSS, n: int) -> FastGSS:
    for _ in range(n):
        gss = gss.pop()
    return gss


def get_disallowed_terminals_py(gss: FastGSS) -> ffi.HybridL2Bitset:
    merged_acc = gss.get_acc(merge_acc)
    return merged_acc.terminals_union.complement()


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
        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        model.possible_matches_cache = model.constraint.possible_matches()

        # Load parse table
        glr_parser = model.constraint.glr_parser()
        raw_table_data = glr_parser.get_parse_table()
        model.start_state_id = raw_table_data['start_state_id']
        model.parse_table = model._parse_table_data(raw_table_data['table'])

        # Initialize state_map
        def acc_factory():
            return PyAcc(terminals_union=ffi.HybridL2Bitset.all())
        initial_gss = FastGSS.initial(acc_factory).push(model.start_state_id)
        model.state_map = {0: initial_gss} # tokenizer state 0 -> initial GSS

        return model

    def _parse_table_data(self, raw_table: Dict) -> Dict[int, Row]:
        parsed_table = {}
        for state_id_str, raw_row in raw_table.items():
            state_id = int(state_id_str)
            
            # Parse actions
            actions = {}
            for term_id_str, raw_action in raw_row['shifts_and_reduces'].items():
                term_id = int(term_id_str)
                action_type = raw_action[0]
                if action_type == 'shift':
                    actions[term_id] = Shift(state_id=raw_action[1])
                elif action_type == 'reduce':
                    _, nt_id, length, pids = raw_action
                    actions[term_id] = Reduce(nonterminal_id=nt_id, len=length, production_ids=tuple(pids))
                elif action_type == 'split':
                    _, shift_id, reduces_dict = raw_action
                    reduces = []
                    for length, nts_dict in reduces_dict.items():
                        for nt_id, pids in nts_dict.items():
                            reduces.append(Reduce(nonterminal_id=nt_id, len=length, production_ids=tuple(pids)))
                    actions[term_id] = Split(shift=shift_id, reduces=tuple(reduces))
            
            # Parse gotos
            gotos = {}
            for nt_id_str, raw_goto in raw_row['gotos'].items():
                nt_id = int(nt_id_str)
                goto_state_id, accept = raw_goto
                gotos[nt_id] = Goto(state_id=goto_state_id, accept=accept)

            parsed_table[state_id] = Row(actions=actions, gotos=gotos)
        return parsed_table

    def _handle_reduce(self, gss_after_pop: FastGSS, reduce: Reduce) -> FastGSS:
        gss_after_full_pop = gss_after_pop
        for _ in range(reduce.len - 1):
            gss_after_full_pop = gss_after_full_pop.pop()
            
        goto_gsses = []
        for top_state_id in gss_after_full_pop.peek():
            gss_with_top_state = gss_after_full_pop.isolate(top_state_id)
            
            row = self.parse_table.get(top_state_id)
            if not row: continue
            
            goto = row.gotos.get(reduce.nonterminal_id)
            if not goto or goto.state_id is None: continue
            
            goto_gsses.append(gss_with_top_state.push(goto.state_id))
            
        if not goto_gsses:
            return FastGSS.initial(gss_after_pop._acc_default_factory)
        return FastGSS.merge(goto_gsses, merge_acc)


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
        token_bytes = self.id_to_token.get(token_id)
        if not token_bytes:
            self.state_map = {}
            return

        final_states: Dict[int, List[FastGSS]] = collections.defaultdict(list)
        queue_map: Dict[int, Dict[int, List[FastGSS]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        
        for sid, gss in self.state_map.items():
            queue_map[0][sid].append(gss)

        sorted_offsets = [0] if 0 in queue_map else []

        while sorted_offsets:
            offset = sorted_offsets.pop(0)
            states_at_offset = queue_map.pop(offset)

            for sid, gss_list in states_at_offset.items():
                gss = FastGSS.merge(gss_list, merge_acc)

                end_state, matches = self.constraint.tokenizer().execute_from_state(token_bytes[offset:], sid)

                for terminal_id, width in matches:
                    new_gss = self._process_terminal(gss, terminal_id)
                    if any(h is not new_gss._root for h in new_gss._heads):
                        new_offset = offset + width
                        next_sid = 0  # tokenizer resets
                        
                        if new_offset == len(token_bytes):
                            final_states[next_sid].append(new_gss)
                        else:
                            if new_offset not in queue_map and new_offset not in [o for o, _ in enumerate(sorted_offsets)]:
                                sorted_offsets.append(new_offset)
                                sorted_offsets.sort()
                            queue_map[new_offset][next_sid].append(new_gss)
                
                if end_state is not None:
                    final_states[end_state].append(gss)

        self.state_map = {
            sid: FastGSS.merge(gss_list, merge_acc)
            for sid, gss_list in final_states.items()
            if gss_list
        }

    def _process_terminal(self, gss: FastGSS, terminal_id: int) -> FastGSS:
        shifted_gsses, reduced_gsses = [], []
        for top_state_id in gss.peek():
            gss_for_state = gss.isolate(top_state_id)
            if not any(h is not gss_for_state._root for h in gss_for_state._heads): continue
            
            popped_gss = gss_for_state.pop()
            row = self.parse_table.get(top_state_id)
            if not row or terminal_id not in row.actions: continue
            action = row.actions[terminal_id]

            if isinstance(action, (Shift, Split)) and (shift_to := getattr(action, 'shift', action.state_id)) is not None:
                shifted_gsses.append(popped_gss.push(shift_to))
            if isinstance(action, (Reduce, Split)):
                reductions = (action,) if isinstance(action, Reduce) else action.reduces
                for r in reductions: reduced_gsses.append(self._handle_reduce(popped_gss, r))

        return FastGSS.merge(shifted_gsses + reduced_gsses, merge_acc) if shifted_gsses or reduced_gsses else FastGSS.initial(gss._acc_default_factory)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        print("\n--- get_mask START ---")
        state_map = self.state_map

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
                    popped = popn_fast_py(gss_node, pop)

                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        print(f"      - Dest: idx={dest_idx}, state_bv={state_bv.to_ranges()}")
                        # Filter peeks by destination state bitset
                        matched = []
                        if not state_bv.is_empty():
                            for sid_val in popped.peek():
                                if state_bv.contains(sid_val):
                                    matched.append(popped.isolate(sid_val))
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
