from __future__ import annotations

import json
from typing import Dict, List, Tuple, Optional, Any

import _sep1 as ffi
from ..common_interface import GraphProvider
from .icl_rangeset import RangeSet
from . import precompute3_engine as eng


class Model(GraphProvider):
    """
    Thin Python wrapper around the C++ Engine. The core model logic (GSS, commit, get_mask)
    is implemented in C++ with a faithful LeveledGSS translation. Python is only used for:
      - Loading/normalizing the JSON model and constraint via _sep1
      - Providing GraphProvider helpers for the benchmark/visualizer
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # For GraphProvider compatibility and iteration helpers
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena

        # Interface-visible fields
        self.id_to_token: Dict[int, bytes] = {}
        self.internal_to_original_map: Dict[int, List[int]] = {}

        # Tokenizer/parser
        self.tokenizer = None
        self.glr_parser = None
        self.tokenizer_initial_state: Optional[int] = None
        self.tokenizer_max_state: Optional[int] = None

        # Engine instance
        self._engine: Optional[Any] = None

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)

        # Roots map
        roots_map = data["precomputed3"]

        # Arena
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena: Dict[int, dict] = {int(k): v for k, v in arena_values}

        model = Model(roots_map, arena)

        # Load tokenizer and parser from the full constraint JSON via _sep1
        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.tokenizer = constraint.tokenizer()
        model.tokenizer_max_state = model.tokenizer.max_state()
        model.glr_parser = constraint.glr_parser()
        model.tokenizer_initial_state = model.tokenizer.initial_state_id()

        # Parse parser JSON directly from data (no Python-side dataclasses)
        parser_data = data['parser']  # contains 'start_state_id' and 'stage_7_table'

        # Build id_to_token mapping (original LLM token map)
        model.id_to_token = {int(v): bytes(k) for k, v in data['llm_token_map']}

        vocab = data.get('precompute3_vocab') or data.get('precompute2_vocab') or data.get('precompute_vocab')
        if vocab:
            model.internal_to_original_map = {int(k): v for k, v in vocab['internal_to_original']}
            internal_max = vocab['internal_max_llm_token']
            all_internal_tokens_bitset = ffi.Bitset.from_ranges([(0, internal_max)])
        else:
            # Fallback for old format: one-to-one mapping
            i2o_map_one_to_one = constraint.internal_to_original_map()
            model.internal_to_original_map = {k: [v] for k, v in i2o_map_one_to_one.items()}
            all_internal_tokens_bitset = constraint.all_internal_llm_tokens_bitset()

        # Create the C++ Engine (self-contained algorithm in C++)
        # Pass raw structures; the engine will parse/normalize as needed.
        model._engine = eng.Engine(
            model.tokenizer,
            model.tokenizer.initial_state_id(),
            model.tokenizer.max_state(),
            model.glr_parser.ignore_terminal_id if model.glr_parser.ignore_terminal_id is not None else None,
            parser_data,
            dict(model.roots_map),
            model.arena,
            constraint.possible_matches(),                 # tsid -> term_id -> _sep1.Bitset
            all_internal_tokens_bitset,                    # _sep1.Bitset universe
            model.internal_to_original_map                 # int -> list[int]
        )
        model._engine.reset_stats()

        return model

    def commit(self, token_id: int):
        token_bytes = self.id_to_token[token_id]
        self._engine.commit(token_bytes)

    def finalize(self):
        """Called at the end of a benchmark run to perform any final actions, like printing stats."""
        print("\n--- Final Stats Report from Model ---")
        self._engine.report_stats()

    def get_mask(self) -> RangeSet:
        return self._engine.get_mask()

    # GraphProvider impl for benchmark_runner compatibility
    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        This is used only for visualization/analysis. The engine does not use this.
        """
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv_json), dests in children:
            # Convert llm_bv_json to a RangeSet once for membership check
            bv = ffi.Bitset.from_json_string(json.dumps(llm_bv_json))
            llm_bv_rs = RangeSet.from_ranges(bv.to_ranges())

            if llm_bv_rs.contains(int(token)):
                for dest_idx, state_bv_json in dests:
                    state_bv = ffi.Bitset.from_json_string(json.dumps(state_bv_json))
                    if state_bv.is_empty():
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(int(start), int(end) + 1):
                                yield (int(pop), sid, int(dest_idx))
