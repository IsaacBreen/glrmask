from __future__ import annotations
import json
import heapq
import collections
import os
import time
from typing import Dict, List, Tuple, Optional, Union, Callable, Iterable, Set, Type
from dataclasses import dataclass, field
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi


@dataclass(frozen=True)
class Reduce:
    nonterminal_id: int
    len: int
    production_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Split:
    shift: Optional[int]
    reduces: Dict[int, Dict[int, Tuple[int, ...]]]  # len -> nt_id -> pids


Action = Union[int, Reduce, Split]  # Shift (int), Reduce, or Split


@dataclass
class Row:
    actions: Dict[int, Action] = field(default_factory=dict)  # terminal_id -> Action
    gotos: Dict[int, int] = field(default_factory=dict)  # nonterminal_id -> state_id


@dataclass
class ParserTable:
    start_state_id: int
    table: Dict[int, Row]


@dataclass(frozen=True)
class PyAcc:
    terminals_union: ffi.HybridL2Bitset

    def __hash__(self) -> int:
        return hash(self.terminals_union)

    def merge(self, other: "PyAcc") -> "PyAcc":
        return PyAcc(self.terminals_union.union(other.terminals_union))


# GSS implementation

@dataclass(frozen=True, eq=False)
class _StackNode:
    prev: Optional["_StackNode"]
    value: int

    def __repr__(self) -> str:
        return f"_StackNode(value={self.value!r}, prev_id={id(self.prev) if self.prev else None})"


class _NodeFactory:
    def __init__(self) -> None:
        self._table: Dict[Tuple[Optional[_StackNode], int], _StackNode] = {}

    def get(self, prev: Optional[_StackNode], value: int) -> _StackNode:
        key = (prev, value)
        node = self._table.get(key)
        if node is None:
            node = _StackNode(prev, value)
            self._table[key] = node
        return node


class GSS:
    def __init__(
        self,
        node_factory: Optional[_NodeFactory] = None,
        heads: Optional[Dict[_StackNode, List[PyAcc]]] = None,
        empty_accs: Optional[List[PyAcc]] = None,
    ) -> None:
        self._factory: _NodeFactory = node_factory if node_factory is not None else _NodeFactory()
        self._heads: Dict[_StackNode, List[PyAcc]] = heads if heads is not None else {}
        self._empty_accs: List[PyAcc] = empty_accs if empty_accs is not None else []

    @classmethod
    def from_stacks(cls: Type["GSS"], stacks: List[Tuple[List[int], PyAcc]]) -> "GSS":
        factory = _NodeFactory()
        heads: Dict[_StackNode, List[PyAcc]] = {}
        empty_accs: List[PyAcc] = []
        for vals, acc in stacks:
            if not vals:
                empty_accs.append(acc)
                continue
            prev: Optional[_StackNode] = None
            for v in vals:
                prev = factory.get(prev, v)
            heads.setdefault(prev, []).append(acc)
        return cls(factory, heads, empty_accs)

    def _clone_with(
        self,
        heads: Optional[Dict[_StackNode, List[PyAcc]]] = None,
        empty_accs: Optional[List[PyAcc]] = None,
    ) -> "GSS":
        return GSS(
            self._factory,
            heads if heads is not None else dict(self._heads),
            empty_accs if empty_accs is not None else list(self._empty_accs),
        )

    def push(self, value: int) -> "GSS":
        new_heads: Dict[_StackNode, List[PyAcc]] = {}
        for head, accs in self._heads.items():
            new_heads.setdefault(self._factory.get(head, value), []).extend(accs)
        if self._empty_accs:
            new_heads.setdefault(self._factory.get(None, value), []).extend(self._empty_accs)
        return self._clone_with(heads=new_heads, empty_accs=[])

    def pop(self) -> "GSS":
        new_heads: Dict[_StackNode, List[PyAcc]] = {}
        new_empty: List[PyAcc] = []
        for head, accs in self._heads.items():
            if head.prev is None:
                new_empty.extend(accs)
            else:
                new_heads.setdefault(head.prev, []).extend(accs)
        return self._clone_with(heads=new_heads, empty_accs=new_empty)

    def popn(self, n: int) -> "GSS":
        gss = self
        for _ in range(n):
            gss = gss.pop()
        return gss

    def is_empty(self) -> bool:
        return not (self._empty_accs or self._heads)

    def isolate(self, value: Optional[int]) -> "GSS":
        if value is None:
            return self._clone_with(heads={}, empty_accs=list(self._empty_accs))
        return self._clone_with(
            heads={h: list(a) for h, a in self._heads.items() if h.value == value},
            empty_accs=[],
        )

    def apply(self, func: Callable[[PyAcc], PyAcc]) -> "GSS":
        return self._clone_with(
            heads={h: [func(a) for a in accs] for h, accs in self._heads.items()},
            empty_accs=[func(a) for a in self._empty_accs],
        )

    def prune(self, predicate: Callable[[PyAcc], bool]) -> "GSS":
        new_heads = {}
        for head, accs in self._heads.items():
            kept = [a for a in accs if predicate(a)]
            if kept:
                new_heads[head] = kept
        return self._clone_with(heads=new_heads, empty_accs=[a for a in self._empty_accs if predicate(a)])

    def peek(self) -> Set[int]:
        return {h.value for h, accs in self._heads.items() if accs}

    def reduce_acc(self) -> Optional[PyAcc]:
        combined: Optional[PyAcc] = None
        for accs in self._heads.values():
            for a in accs:
                combined = a if combined is None else combined.merge(a)
        for a in self._empty_accs:
            combined = a if combined is None else combined.merge(a)
        return combined

    @staticmethod
    def merge(gss_list: Iterable["GSS"]) -> "GSS":
        merged: Dict[Tuple[int, ...], PyAcc] = {}
        for gss in gss_list:
            for head, accs in gss._heads.items():
                vals: List[int] = []
                cur: Optional[_StackNode] = head
                while cur is not None:
                    vals.append(cur.value)
                    cur = cur.prev
                key = tuple(reversed(vals))
                for a in accs:
                    merged[key] = merged[key].merge(a) if key in merged else a
            for a in gss._empty_accs:
                key = ()
                merged[key] = merged[key].merge(a) if key in merged else a
        if not merged:
            return GSS()
        return GSS.from_stacks([(list(k), v) for k, v in merged.items()])


def get_disallowed_terminals_py(gss: GSS) -> ffi.HybridL2Bitset:
    acc = gss.reduce_acc()
    return ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation), simplified and concise.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.id_to_token: Dict[int, bytes] = {}
        self.max_depth: Dict[int, int] = {}
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = None
        self.tokenizer: Optional[ffi.Regex] = None
        self.glr_parser: Optional[ffi.GLRParser] = None
        self.ignore_terminal_id: Optional[int] = None
        self.parser_table: Optional[ParserTable] = None
        self.state: Dict[int, GSS] = {}
        self.internal_to_original_map: Dict[int, int] = {}
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = None
        self.tokenizer_initial_state: Optional[int] = None

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        for uid, node in self.arena.items():
            self.max_depth[int(uid)] = int(node.get("max_depth", 0) or 0)
            children = node.get("children") or []
            if not children:
                node["children"] = []
                continue
            new_children = []
            for (pop, llm_bv_json), dest_map in children:
                llm_bv = bs_from_json(dumps(llm_bv_json))
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    new_dest_map.append((int(dest_idx), bs_from_json(dumps(state_bv_json))))
                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena = {int(k): v for k, v in arena_json.get("values", [])}
        model = Model(roots_map, arena)

        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.tokenizer = constraint.tokenizer()
        model.glr_parser = constraint.glr_parser()
        model.ignore_terminal_id = model.glr_parser.ignore_terminal_id
        model.tokenizer_initial_state = model.tokenizer.initial_state_id()

        parser_data = data['parser']
        table_data = parser_data['stage_7_table']
        start_state_id = parser_data['start_state_id']

        def _parse_action(ad):
            v = ad['variant']
            if v == 'Shift':
                return ad['state_id']
            if v == 'Reduce':
                return Reduce(ad['nonterminal_id'], ad['len'], tuple(sorted(ad['production_ids'])))
            if v == 'Split':
                reduces: Dict[int, Dict[int, Tuple[int, ...]]] = {
                    int(length): {int(nt): tuple(sorted(pids)) for nt, pids in nts}
                    for length, nts in ad['reduces']
                }
                return Split(ad['shift'], reduces)
            return None

        py_table: Dict[int, Row] = {}
        for state_id_str, row in table_data:
            state_id = int(state_id_str)
            actions = {int(tid): act for tid, ad in row['shifts_and_reduces_full'] if (act := _parse_action(ad)) is not None}
            gotos = {int(nt): gd['state_id'] for nt, gd in row['gotos'] if gd['state_id'] is not None}
            py_table[state_id] = Row(actions, gotos)
        model.parser_table = ParserTable(start_state_id, py_table)

        initial_acc = PyAcc(terminals_union=ffi.HybridL2Bitset.all())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        raw_pm = constraint.possible_matches()
        model.possible_matches_cache = {int(tsid): {int(tid): bv for tid, bv in inner.items()} for tsid, inner in raw_pm.items()}
        model.internal_to_original_map = constraint.internal_to_original_map()
        model.all_internal_llm_tokens_bitset = constraint.all_internal_llm_tokens_bitset()
        return model

    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, ffi.Bitset]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            allowed_l2 = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                if not matched_bv.is_subset(allowed_l2.get_l2_bitset(state_id)):
                    return False
            return True
        return gss.prune(predicate)

    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_l2 = acc.terminals_union
            new_bvs: Dict[int, ffi.Bitset] = collections.defaultdict(ffi.Bitset.zeros)
            for old_sid, new_sid in state_map.items():
                new_bvs[new_sid] = new_bvs[new_sid].union(old_l2.get_l2_bitset(old_sid))
            new_l2 = ffi.HybridL2Bitset.all()
            for new_sid, bv in new_bvs.items():
                new_l2.insert_l2_bitset(new_sid, bv)
            return PyAcc(new_l2)
        return gss.apply(apply_map)

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_l2 = acc.terminals_union
            new_l2 = current_l2.union(current_l2)  # clone
            curr_bv = current_l2.get_l2_bitset(state_id)
            if curr_bv.contains(terminal_id):
                new_l2.insert_l2_bitset(state_id, curr_bv.difference(ffi.Bitset.from_indices([terminal_id])))
            return PyAcc(new_l2)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end + 1):
                                yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        t0 = time.perf_counter()
        token_bytes = self.id_to_token[token_id]

        # Build tokenizer maps
        terminals_map: Dict[int, ffi.Bitset] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            b = ffi.Bitset.zeros()
            for terminal_id, _ in matches:
                b.insert(terminal_id)
            terminals_map[tokenizer_sid] = b

        # Prune and map per-state GSS
        current_state_for_processing: Dict[int, GSS] = {}
        for tokenizer_sid, gss in self.state.items():
            pruned = self._prune_disallowed_terminals(gss, terminals_map)
            if not pruned.is_empty():
                current_state_for_processing[tokenizer_sid] = self._map_allowed_terminals_tokenizer_states(pruned, state_map)

        new_states: Dict[int, List[GSS]] = collections.defaultdict(list)
        q = collections.deque((0, sid, gss) for sid, gss in current_state_for_processing.items())
        visited_q_items: set = set()

        while q:
            offset, tokenizer_sid, gss = q.popleft()
            q_key = (offset, tokenizer_sid, id(gss))
            if q_key in visited_q_items:
                continue
            visited_q_items.add(q_key)

            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tokenizer_sid)

            for terminal_id, width in matches:
                processed_gss = gss if terminal_id == self.ignore_terminal_id else self._process_token(gss, terminal_id)

                if end_state is not None:
                    accessible_terms = set(self.tokenizer.tokens_accessible_from_state(end_state))
                    if terminal_id in accessible_terms:
                        processed_gss = self._disallow_terminal_in_state(processed_gss, end_state, terminal_id)

                if not processed_gss.is_empty():
                    new_offset = offset + width
                    next_sid = self.tokenizer_initial_state
                    if new_offset == len(token_bytes):
                        new_states[next_sid].append(processed_gss)
                    else:
                        q.append((new_offset, next_sid, processed_gss))

            if end_state is not None:
                new_states[end_state].append(gss)

        merged_states = {sid: GSS.merge(glist) for sid, glist in new_states.items() if glist}
        self.state = {sid: st for sid, st in merged_states.items() if not st.is_empty()}

        t1 = time.perf_counter()
        if os.environ.get("REPORT_COMMIT_TIME") == "1":
            print(f"commit (ms): {round((t1 - t0) * 1000, 2)}")

    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        heads_by_state: Dict[int, List[GSS]] = collections.defaultdict(list)
        for state_id in gss.peek():
            heads_by_state[state_id].append(gss.isolate(state_id))

        shifted_gsses: List[GSS] = []

        while heads_by_state:
            state_id, state_gsss = heads_by_state.popitem()
            state_gss = GSS.merge(state_gsss)
            row = self.parser_table.table.get(state_id)
            if not row:
                continue
            action = row.actions.get(terminal_id)
            if not action:
                continue

            if isinstance(action, int):  # Shift
                shifted_gsses.append(state_gss.push(action))
                continue

            if isinstance(action, Reduce):
                popped = state_gss.popn(action.len)
                for from_state_id in popped.peek():
                    goto_state_id = self.parser_table.table[from_state_id].gotos[action.nonterminal_id]
                    heads_by_state[goto_state_id].append(popped.isolate(from_state_id).push(goto_state_id))
                continue

            # Split
            if action.shift is not None:
                shifted_gsses.append(state_gss.push(action.shift))
            for length, nts in action.reduces.items():
                popped = state_gss.popn(length)
                for from_state_id in popped.peek():
                    table_row = self.parser_table.table[from_state_id]
                    for nt_id in nts.keys():
                        goto_state_id = table_row.gotos[nt_id]
                        heads_by_state[goto_state_id].append(popped.isolate(from_state_id).push(goto_state_id))

        return GSS(node_factory=gss._factory) if not shifted_gsses else GSS.merge(shifted_gsses)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.
        """
        active_states = self.state
        all_ones_mask = self.all_internal_llm_tokens_bitset
        final_mask = ffi.Bitset.zeros()

        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        stopped: set[int] = set()
        todo: Dict[int, set[int]] = {}
        depth_heap: List[int] = []

        heappush = heapq.heappush
        heappop = heapq.heappop
        roots_map = self.roots_map
        max_depth = self.max_depth
        arena = self.arena
        is_end = self.is_end

        # Seed
        for sid, gss in active_states.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)
            existing = values.get(root_idx)
            if existing is not None:
                egss, emask = existing
                values[root_idx] = (GSS.merge([egss, gss]), emask.union(all_ones_mask))
            else:
                values[root_idx] = (gss, all_ones_mask)
            d = max_depth[root_idx]
            bucket = todo.get(d)
            if bucket is None:
                todo[d] = {root_idx}
                heappush(depth_heap, d)
            else:
                bucket.add(root_idx)

        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        # Main loop
        while True:
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

                # End-node handling
                if is_end(node_idx):
                    forbidden_llm_tokens = ffi.Bitset.zeros()
                    disallowed_l2 = get_disallowed_terminals_py(gss_node)
                    possible_matches = self.possible_matches_cache

                    for (start, end), disallowed_bv in disallowed_l2.range_values():
                        if disallowed_bv.is_empty():
                            continue
                        end = min(end, self.tokenizer.max_state())
                        for tsid in range(start, end + 1):
                            pm = possible_matches.get(tsid)
                            if not pm:
                                continue
                            for terminal_id, llm_tokens_for_terminal in pm.items():
                                if disallowed_bv.contains(terminal_id):
                                    forbidden_llm_tokens = forbidden_llm_tokens.union(llm_tokens_for_terminal)

                    allowed_tokens = llm_mask.difference(forbidden_llm_tokens)
                    if not allowed_tokens.is_empty():
                        final_mask = final_mask.union(allowed_tokens)

                if llm_mask.is_empty():
                    stopped.add(node_idx)
                    continue

                # Transitions
                for (pop, llm_bv), dests in arena.get(node_idx, {}).get("children") or []:
                    popped = gss_node.popn(pop)
                    llm_empty = llm_bv.is_empty()
                    for dest_idx, state_bv in dests:
                        matched: List[GSS] = []
                        if not state_bv.is_empty():
                            for sid_val in popped.peek():
                                if state_bv.contains(sid_val):
                                    matched.append(popped.isolate(sid_val))
                        if not matched:
                            continue
                        child_gss = GSS.merge(matched)
                        child_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)
                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            egss, emask = existing
                            values[d] = (GSS.merge([egss, child_gss]), emask.union(child_mask))
                        else:
                            values[d] = (child_gss, child_mask)
                        enqueue(max_depth[d], d)

        # Convert internal mask back to original IDs
        original_mask = ffi.Bitset.zeros()
        for internal_id in final_mask.to_indices():
            orig = self.internal_to_original_map.get(internal_id)
            if orig is not None:
                original_mask.insert(orig)
        return RangeSet.from_ranges(original_mask.to_ranges())
