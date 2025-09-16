from typing import List, Tuple, Callable, Set, Iterable, Dict, Any, Type

from .interface import GSS, T, Acc
from _sep1 import GSSNode

# A global cache is acceptable here since GSS nodes are immutable and the graph is append-only.
# A path from a given node will never change.
_path_cache: Dict[int, List[List[Any]]] = {}

class RustGSS(GSS[T, Acc]):
    """
    A GSS implementation that wraps the high-performance Rust GSSNode.
    - The Python-side state is a set of (GSSNode, accumulator) tuples representing the heads.
    - The integer accumulator from the test spec is stored on the Python side.
    - Operations are translated to calls on the wrapped GSSNode objects.
    - Path reconstruction for `to_json_serializable` is memoized for performance.
    """

    def __init__(self, heads: Set[Tuple[GSSNode, Acc]], acc_default_factory: Callable[[], Acc]):
        self._acc_default_factory = acc_default_factory
        if not heads:
            # Normalize an empty GSS to have one root node with a default accumulator.
            self._heads = frozenset([(GSSNode(), self._acc_default_factory())])
        else:
            self._heads = frozenset(heads)

    @classmethod
    def initial(cls: Type['RustGSS'], acc_default_factory: Callable[[], Acc]) -> 'RustGSS[T, Acc]':
        return cls(set(), acc_default_factory)

    def push(self, value: T) -> 'RustGSS[T, Acc]':
        new_heads = set()
        memo: Dict[int, GSSNode] = {}
        for node, acc in self._heads:
            node_ptr = node.ptr()
            if node_ptr in memo:
                new_node = memo[node_ptr]
            else:
                new_node = node.push(value)
                memo[node_ptr] = new_node
            new_heads.add((new_node, acc))
        return RustGSS(new_heads, self._acc_default_factory)

    def pop(self, value: T) -> 'RustGSS[T, Acc]':
        new_heads = set()
        for node, acc in self._heads:
            predecessors = node.popn_fast(1)
            for state_id, pred_node in predecessors:
                if state_id == value:
                    new_heads.add((pred_node, acc))
        
        had_non_root_stacks = any(node.max_depth() > 0 for node, _ in self._heads)
        if not new_heads and had_non_root_stacks:
            return RustGSS(set(), self._acc_default_factory)

        return RustGSS(new_heads, self._acc_default_factory)

    def apply(self, func: Callable[[Acc], Acc]) -> 'RustGSS[T, Acc]':
        new_heads = {(node, func(acc)) for node, acc in self._heads}
        return RustGSS(new_heads, self._acc_default_factory)

    def peek(self) -> Set[T]:
        peek_values = set()
        for node, _ in self._heads:
            if node.max_depth() > 0:
                predecessors = node.popn_fast(1)
                for state_id, _ in predecessors:
                    peek_values.add(state_id)
        return peek_values

    @staticmethod
    def merge(gss_list: Iterable['RustGSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'RustGSS[T, Acc]':
        gss_list = list(gss_list)
        if not gss_list:
            raise ValueError("Cannot merge an empty list of GSS instances.")

        factory = gss_list[0]._acc_default_factory

        # Filter out GSSs that only contain the initial empty stack, unless that's all there is.
        # This mirrors the reference implementation's behavior.
        active_gss = [g for g in gss_list if not (len(g._heads) == 1 and next(iter(g._heads))[0].max_depth() == 0)]
        if not active_gss:
            return gss_list[0] # All were initial/empty

        merged_heads_map: Dict[GSSNode, Acc] = {}
        for gss in active_gss:
            for node, acc in gss._heads:
                if node in merged_heads_map:
                    merged_heads_map[node] = merge_func(merged_heads_map[node], acc)
                else:
                    merged_heads_map[node] = acc
        
        new_heads = set(merged_heads_map.items())
        return RustGSS(new_heads, factory)

    def to_json_serializable(self) -> Any:
        all_stacks = []
        for node, acc in self._heads:
            paths = self._get_paths(node)
            for path in paths:
                all_stacks.append({"stack": path, "acc": acc})
        
        return sorted(all_stacks, key=lambda x: (tuple(x["stack"]), x["acc"]))

    def _get_paths(self, node: GSSNode) -> List[List[T]]:
        node_ptr = node.ptr()
        if node_ptr in _path_cache:
            return _path_cache[node_ptr]

        if node.max_depth() == 0:
            return [[]]

        paths = []
        predecessors = node.popn_fast(1)
        for state_id, pred_node in predecessors:
            parent_paths = self._get_paths(pred_node)
            for p_path in parent_paths:
                paths.append(p_path + [state_id])
        
        _path_cache[node_ptr] = paths
        return paths
