from typing import List, Tuple, Callable, Set, Iterable, Dict, Any, Type

from .interface import GSS, T, Acc
from _sep1 import GSSNode as PyGSSNode

class RustGSS(GSS[T, Acc]):
    """
    A GSS implementation that wraps the high-performance Rust GSSNode.
    - The structure of all stacks is managed by the Rust GSSNode objects.
    - This wrapper class holds a dictionary mapping head nodes (PyGSSNode) to
      their corresponding Python accumulator values.
    - Structural equality of PyGSSNodes allows for implicit merging of common
      stack structures.
    """

    def __init__(self, stacks: Dict[PyGSSNode, Acc], acc_default_factory: Callable[[], Acc]):
        self.stacks = stacks
        self._acc_default_factory = acc_default_factory
        if not self.stacks:
            # Ensure there's always at least one stack, the empty one.
            self.stacks[PyGSSNode()] = self._acc_default_factory()

    @classmethod
    def initial(cls: Type['RustGSS'], acc_default_factory: Callable[[], Acc]) -> 'RustGSS[T, Acc]':
        return cls({}, acc_default_factory)

    def push(self, value: T) -> 'RustGSS[T, Acc]':
        new_stacks: Dict[PyGSSNode, Acc] = {}
        for node, acc in self.stacks.items():
            new_node = node.push(value)
            new_stacks[new_node] = acc
        return RustGSS(new_stacks, self._acc_default_factory)

    def pop(self, value: T) -> 'RustGSS[T, Acc]':
        new_stacks: Dict[PyGSSNode, Acc] = {}
        had_non_empty_stack = False
        for node, acc in self.stacks.items():
            if not node.is_root():
                had_non_empty_stack = True
            
            predecessors = node.popn_fast(1)
            for edge_val, parent_node in predecessors:
                if edge_val == value:
                    new_stacks[parent_node] = acc

        if not new_stacks and had_non_empty_stack:
            return RustGSS.initial(self._acc_default_factory)

        return RustGSS(new_stacks, self._acc_default_factory)

    def apply(self, func: Callable[[Acc], Acc]) -> 'RustGSS[T, Acc]':
        new_stacks = {node: func(acc) for node, acc in self.stacks.items()}
        return RustGSS(new_stacks, self._acc_default_factory)

    def peek(self) -> Set[T]:
        peek_values: Set[T] = set()
        for node in self.stacks.keys():
            predecessors = node.popn_fast(1)
            for edge_val, _ in predecessors:
                peek_values.add(edge_val)
        return peek_values

    @staticmethod
    def merge(gss_list: Iterable['RustGSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'RustGSS[T, Acc]':
        gss_list = list(gss_list)
        if not gss_list:
            raise ValueError("Cannot merge an empty list of GSS instances.")

        factory = gss_list[0]._acc_default_factory
        merged_stacks: Dict[PyGSSNode, Acc] = {}

        for gss in gss_list:
            # Don't merge the default empty stack if there are other, non-empty stacks.
            is_default_empty = len(gss.stacks) == 1 and next(iter(gss.stacks.keys())).is_root()
            
            has_other_stacks = any(len(g.stacks) > 1 or not next(iter(g.stacks.keys())).is_root() for g in gss_list)

            if is_default_empty and has_other_stacks:
                continue

            for node, acc in gss.stacks.items():
                if node in merged_stacks:
                    merged_stacks[node] = merge_func(merged_stacks[node], acc)
                else:
                    merged_stacks[node] = acc
        
        return RustGSS(merged_stacks, factory)

    def to_json_serializable(self) -> Any:
        all_stacks = []
        path_cache: Dict[PyGSSNode, List[List[T]]] = {}
        
        # Sort items by node pointer for deterministic traversal
        sorted_items = sorted(self.stacks.items(), key=lambda item: item[0].ptr())

        for node, acc in sorted_items:
            paths = self._reconstruct_paths(node, path_cache)
            for path in paths:
                all_stacks.append({"stack": path, "acc": acc})
        
        return sorted(all_stacks, key=lambda x: (tuple(x["stack"]), x["acc"]))

    def _reconstruct_paths(self, node: PyGSSNode, cache: Dict[PyGSSNode, List[List[T]]]) -> List[List[T]]:
        if node in cache:
            return cache[node]

        if node.is_root():
            return [[]]

        paths: List[List[T]] = []
        predecessors = node.popn_fast(1)
        
        # Sort predecessors by pointer for deterministic path reconstruction
        sorted_predecessors = sorted(predecessors, key=lambda p: p[1].ptr())

        for edge_val, parent_node in sorted_predecessors:
            parent_paths = self._reconstruct_paths(parent_node, cache)
            for p_path in parent_paths:
                paths.append(p_path + [edge_val])
        
        cache[node] = paths
        return paths
