# python/aug25/models/precompute3_model_pure_python_alt_commit.py
from __future__ import annotations

import json
import collections
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, replace

import _sep1 as ffi
from .precompute3_model_pure_python import (
    Model as _Model,
    PyAcc,
    GSS,
    _acc_memoize,
    Stats,
    RangeSet,
    profile,
    NodeID,
    LLMTokenSet,
    TerminalIdSet,
)

# --- Dataclasses for the Precompute0 Trie Structure ---

@dataclass
class Precompute0NodeContents:
    final_tokenizer_state: Optional[int]
    live_tokens: LLMTokenSet

@dataclass
class Precompute0Node:
    children: List[Tuple[Optional[Tuple[int, Optional[int]]], List[Tuple[NodeID, LLMTokenSet]]]] = field(default_factory=list)
    value: Precompute0NodeContents = field(default_factory=lambda: Precompute0NodeContents(None, RangeSet.empty()))

# --- The New Model ---

@dataclass
class Model(_Model):
    """
    A model that subclasses the base Model to provide an alternative `commit` implementation.
    This version uses the precomputed0 trie, which is specialized for single-token commits
    and avoids using the tokenizer at commit time, similar to the Rust `commit(token_id)` logic.
    """
    # New fields specific to this model
    arena0: Dict[NodeID, Precompute0Node] = field(default_factory=dict)
    roots_map0: Dict[int, NodeID] = field(default_factory=dict)
    terminal_map_by_llm: Dict[int, Dict[int, TerminalIdSet]] = field(default_factory=dict)
    state_map_by_llm: Dict[int, Dict[int, int]] = field(default_factory=dict)
    original_to_internal_map: Dict[int, int] = field(default_factory=dict)

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        """
        Constructs the model by first loading the base model and then parsing
        the additional data structures required for this alternative commit strategy.
        """
        base_model = _Model.from_json_string(s)
        data = json.loads(s)
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # --- Load precompute0 trie ---
        roots_map0_raw = data["precomputed0"]
        arena0_json = data["trie0_god"]
        arena0_values = arena0_json.get("values", [])
        arena0_dict = {int(k): v for k, v in arena0_values}
        roots_map0 = {int(s): int(r) for s, r in roots_map0_raw}

        for node in arena0_dict.values():
            children = node.get("children") or []
            new_children = []
            for edge_key_json, dest_map in children:
                if edge_key_json is None:
                    edge_key = None
                else:
                    gtid, disallow_opt = edge_key_json
                    edge_key = (gtid, disallow_opt)

                new_dest_map = []
                for dest_idx, llm_bv_json in dest_map:
                    llm_bv_bitset = bs_from_json(dumps(llm_bv_json))
                    llm_bv: LLMTokenSet = RangeSet.from_ranges(llm_bv_bitset.to_ranges())
                    new_dest_map.append((int(dest_idx), llm_bv))
                new_children.append((edge_key, new_dest_map))
            node["children"] = new_children
            value_json = node.get("value", {})
            live_tokens_json = value_json.get("live_tokens", [])
            live_tokens_bitset = bs_from_json(dumps(live_tokens_json))
            node["value"] = Precompute0NodeContents(
                final_tokenizer_state=value_json.get("final_tokenizer_state"),
                live_tokens=RangeSet.from_ranges(live_tokens_bitset.to_ranges())
            )
        arena0 = {uid: Precompute0Node(children=nd.get("children", []), value=nd["value"]) for uid, nd in arena0_dict.items()}

        # --- Load precomputed per-token maps ---
        def parse_dedup_map(raw_map, value_parser):
            id_to_value = {item[0]: value_parser(item[1]) for item in raw_map['values']}
            key_to_id = {item[0]: item[1] for item in raw_map['keys']}
            return {int(key): id_to_value[id_] for key, id_ in key_to_id.items()}

        def parse_terminal_map_value(raw_value):
            out = {}
            for sid_str, bv_json in raw_value:
                bs = bs_from_json(dumps(bv_json))
                out[int(sid_str)] = RangeSet.from_ranges(bs.to_ranges())
            return out

        def parse_state_map_value(raw_value):
            return {int(k): v for k, v in raw_value}

        terminal_map_by_llm = parse_dedup_map(data['terminal_map_by_llm'], parse_terminal_map_value)
        state_map_by_llm = parse_dedup_map(data['state_map_by_llm'], parse_state_map_value)

        original_to_internal_map = {}
        for internal, original_bv in base_model.internal_to_original_map.items():
            for original in original_bv.iter_indices():
                original_to_internal_map[original] = internal

        # Construct the final model using all fields from the base model plus the new ones
        return Model(
            **base_model.__dict__,
            arena0=arena0,
            roots_map0=roots_map0,
            terminal_map_by_llm=terminal_map_by_llm,
            state_map_by_llm=state_map_by_llm,
            original_to_internal_map=original_to_internal_map
        )

    def copy(self):
        return replace(self)

    def commit(self, token_id: int):
        """
        Overrides the base `commit` method. This version uses the precomputed0 trie
        to update the GLR state without invoking the tokenizer.
        """
        print("\n--- commit_precompute0 START ---")
        print(f"Committing token ID: {token_id}")
        self_copy = self.copy()
        _Model.commit(self_copy, token_id)

        stats = Stats.get()
        stats.start('commit_precompute0')

        internal_id = self.original_to_internal_map.get(token_id)
        if internal_id is None:
            raise ValueError(f"LLM token ID {token_id} not found in internal mapping.")

        terminals_map = self.terminal_map_by_llm.get(internal_id, {})
        state_map = self.state_map_by_llm.get(internal_id, {})

        # 1. Prepare GSS: Prune based on terminals matched by the token and map tokenizer states.
        @_acc_memoize()
        def mutator(acc: PyAcc) -> Optional[PyAcc]:
            disallowed_terminals_map = acc.terminals_union
            for tsid, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(tsid)
                if disallowed_for_state and not matched_bv.isdisjoint(disallowed_for_state):
                    return None
            new_bvs: Dict[int, TerminalIdSet] = {}
            for old_sid, new_sid in state_map.items():
                bv_source = acc.terminals_union.get(old_sid)
                if bv_source and not bv_source.is_empty():
                    if new_sid in new_bvs:
                        new_bvs[new_sid] |= bv_source
                    else:
                        new_bvs[new_sid] = bv_source
            return PyAcc(terminals_union=new_bvs, llm_mask=acc.llm_mask)

        cache = {}
        current_state = {tsid: gss.apply_and_prune(mutator, cache) for tsid, gss in self.state.items()}
        current_state = {tsid: gss for tsid, gss in current_state.items() if not gss.is_empty()}

        # 2. Traverse the precompute0 trie with the prepared GSS.
        q = collections.deque()
        for tokenizer_sid, gss in current_state.items():
            root_node_id = self.roots_map0[tokenizer_sid]
            q.append((root_node_id, gss))

        new_overall_state_parts = collections.defaultdict(list)
        visited: Dict[NodeID, GSS] = {}

        while q:
            node_id, gss = q.popleft()
            if node_id in visited:
                merged_gss = gss.merge(visited[node_id])
                if len(merged_gss.peek()) == len(visited[node_id].peek()):
                    continue
                gss = merged_gss
            visited[node_id] = gss

            node = self.arena0[node_id]
            if node.value.final_tokenizer_state is not None:
                new_overall_state_parts[node.value.final_tokenizer_state].append(gss)
                continue

            for edge_key, dest_map in node.children:
                for dest_node_id, edge_bv in dest_map:
                    if not edge_bv.contains(internal_id):
                        continue

                    processed_gss = gss
                    if edge_key is not None:
                        gtid, disallow_opt = edge_key
                        print(f"Processing edge with gtid {gtid}, GSS {gss}")
                        processed_gss = self._process_token(gss, gtid)
                        print(f"Processing edge with gtid {gtid}, disallow {disallow_opt}, resulting GSS: {processed_gss}")

                        if disallow_opt is not None and not processed_gss.is_empty():
                            end_state = disallow_opt
                            term_id = gtid
                            processed_gss = self._disallow_terminal_in_state(processed_gss, end_state, term_id)

                    if not processed_gss.is_empty():
                        q.append((dest_node_id, processed_gss))

        # 3. Finalize the new state.
        merged_states = {sid: GSS.merge_many(gss_list) for sid, gss_list in new_overall_state_parts.items() if gss_list}
        self.state = {sid: gss for sid, gss in merged_states.items() if not gss.is_empty()}

        stats.stop('commit_precompute0')

        if not self.state == self_copy.state:
            print("State mismatch after commit:")
            print("With _Model.commit:")
            print(GSS.merge_many(self_copy.state.values()))
            print("With Model.commit (ie precompute0):")
            print(GSS.merge_many(self.state.values()))
            raise AssertionError("The state of the model after committing the token does not match the state before committing.")


__all__ = ['Precompute0Model']