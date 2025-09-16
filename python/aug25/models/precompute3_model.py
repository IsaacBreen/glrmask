import json
import time
import heapq
from typing import Dict, List, Tuple, Optional, Type

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi  # the compiled module
from tqdm.auto import tqdm
from gss_tester.interface import GSS
from gss_tester.fast_impl import FastGSS


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation).
    Normalizes input arena by converting JSON bitsets into ffi.Bitset instances
    and provides graph traversal and mask computation interfaces.
    """
    def __init__(self, tokenizer, parser, roots_map, arena: Dict[int, dict]):
        self.tokenizer = tokenizer
        self.parser = parser
        self.roots_map = roots_map
        self.arena = arena

        self.gss_class: Type[GSS] = FastGSS
        self.acc_factory = lambda: 0
        self.merge_func = lambda a, b: a + b
        
        # The model's state is a Python dictionary of tokenizer_state -> GSS object
        initial_gss = self.gss_class.initial(self.acc_factory)
        self.state: Dict[int, GSS] = {
            self.tokenizer.initial_state_id(): initial_gss
        }
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
        constraint = ffi.GrammarConstraint.from_json_string(s)
        tokenizer = constraint.tokenizer()
        parser = constraint.get_parser()
        roots_map = {int(k): int(v) for k, v in constraint.precompute3_json_string().rsplit('],', 1)[0].split('[', 2)[-1].replace('],[', '],[').split('],[')}

        data = json.loads(s)
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        
        return Model(tokenizer, parser, roots_map, arena)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        node_data = self.arena.get(node)
        if not node_data: return False
        return bool((node_data.get("value") or {}).get("end", False))

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
                            for sid in range(start, end):
                                yield (int(pop), sid, int(dest_idx))

    def step(self, gss_state: GSS, terminal_id: int) -> GSS:
        """
        A pure-Python implementation of a GLR parser step.
        This function takes a GSS node and a terminal ID, and applies the
        shifts and reduces defined in the parse table to produce a new GSS.
        """
        raise NotImplementedError("Python-side GLR step is not implemented yet.")

    def commit(self, token_id: int):
        # NOTE: The logic of this method was tightly coupled with the RustGSS
        # implementation, which uses PyGSSNode objects with FFI methods to
        # manipulate parser state (e.g., .reset_llm_tokens, .prune_disallowed_terminals).
        # The pure-Python FastGSS is immutable and does not have an FFI backend,
        # making the original logic untranslatable without a significant redesign.
        # The original method also contained a NotImplementedError.
        pass

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask given a mapping from tokenizer state to
        GSS nodes. This is the performance-critical routine.
        """
        # NOTE: The logic of this method was tightly coupled with the RustGSS
        # implementation, relying on PyGSSNode features like .allowed_llm_tokens()
        # and .popn_fast(). FastGSS has a different API and does not store
        # parser-specific state in its nodes directly. Adapting this logic
        # requires a new traversal implementation and a strategy for managing
        # allowed token sets (which would be done in the now-disabled `commit`
        # method).
        return RangeSet.from_ranges([])
