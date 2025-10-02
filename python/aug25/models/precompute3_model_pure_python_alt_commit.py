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
    original_to_internal_map0: Dict[int, int] = field(default_factory=dict)

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

        # This model's commit logic is based on precompute0, so we must use its vocab.
        vocab0 = data['precompute0_vocab']
        internal_to_original_map0_raw = dict(vocab0['internal_to_original'])
        original_to_internal_map0 = {}
        for internal, original_list in internal_to_original_map0_raw.items():
            for original in original_list:
                original_to_internal_map0[original] = int(internal)

        # Construct the final model using all fields from the base model plus the new ones
        return Model(
            **base_model.__dict__,
            arena0=arena0,
            roots_map0=roots_map0,
            terminal_map_by_llm=terminal_map_by_llm,
            state_map_by_llm=state_map_by_llm,
            original_to_internal_map0=original_to_internal_map0
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
        print(f"Current state before commit: {self.state}")
        self_copy = self.copy()
        _Model.commit(self_copy, token_id)

        stats = Stats.get()
        stats.start('commit_precompute0')

        internal_id = self.original_to_internal_map0.get(token_id)
        if internal_id is None:
            raise ValueError(f"LLM token ID {token_id} not found in internal mapping.")

        # terminals_map = self.terminal_map_by_llm.get(internal_id, {})
        # state_map = self.state_map_by_llm.get(internal_id, {})

        token_bytes = self.id_to_token[token_id]
        print(f"Token bytes: {token_bytes}")
        terminals_map: Dict[int, TerminalIdSet] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            matched_terminals = [terminal_id for terminal_id, _ in matches]
            terminals_map[tokenizer_sid] = RangeSet.from_indices(matched_terminals)

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
        node_gss_map: Dict[NodeID, GSS] = {}

        for tokenizer_sid, gss in current_state.items():
            root_node_id = self.roots_map0[tokenizer_sid]
            if root_node_id in node_gss_map:
                node_gss_map[root_node_id] = node_gss_map[root_node_id].merge(gss)
            else:
                node_gss_map[root_node_id] = gss
        q.extend(node_gss_map.keys())

        new_overall_state_parts = collections.defaultdict(list)

        while q:
            node_id = q.popleft()
            gss = node_gss_map.pop(node_id)

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
                            print(f"Disallowing terminal {term_id} in state {end_state}")
                            processed_gss = self._disallow_terminal_in_state(processed_gss, end_state, term_id)

                    if not processed_gss.is_empty():
                        existing_gss = node_gss_map.get(dest_node_id)
                        if existing_gss is None:
                            node_gss_map[dest_node_id] = processed_gss
                            q.append(dest_node_id)
                        else:
                            merged_gss = existing_gss.merge(processed_gss)
                            node_gss_map[dest_node_id] = merged_gss

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