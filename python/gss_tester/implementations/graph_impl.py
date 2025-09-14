from typing import List, Tuple, Callable, Set, Iterable, Dict, Any, Type
from ..interface import GSS, T, Acc
import itertools

# Type alias for clarity
NodeId = int

class GraphGSS(GSS[T, Acc]):
    """
    A GSS implementation using a directed graph to merge common stack suffixes.
    - Nodes are defined by (value, parent_tuple) and are canonicalized.
    - The GSS state is a map from head_node_id -> accumulator.
    - This implementation has different pop() semantics than the reference one.
    """
    # Class-level storage for the graph nodes to ensure all instances share one graph
    _node_counter = itertools.count()
    _node_to_id: Dict[Tuple[T, Tuple[NodeId, ...]], NodeId] = {}
    _id_to_node: Dict[NodeId, Tuple[T, Tuple[NodeId, ...]]] = {}
    _root_id: NodeId = -1 # Special ID for the root (representing an empty stack)

    def __init__(self, heads: Dict[NodeId, Acc], acc_default_factory: Callable[[], Acc]):
        self.heads = heads
        self._acc_default_factory = acc_default_factory
        if not self.heads:
             self.heads[self._root_id] = self._acc_default_factory()

    @classmethod
    def _get_node_id(cls, value: T, parents: Tuple[NodeId, ...]) -> NodeId:
        # Sort parents to make the key canonical, ensuring (v, (p1,p2)) == (v, (p2,p1))
        parents = tuple(sorted(parents))
        key = (value, parents)
        if key in cls._node_to_id:
            return cls._node_to_id[key]
        
        new_id = next(cls._node_counter)
        cls._node_to_id[key] = new_id
        cls._id_to_node[new_id] = key
        return new_id

    @classmethod
    def initial(cls: Type['GraphGSS'], acc_default_factory: Callable[[], Acc]) -> 'GraphGSS[T, Acc]':
        return cls({}, acc_default_factory)

    def push(self, value: T) -> 'GraphGSS[T, Acc]':
        new_heads: Dict[NodeId, Acc] = {}
        acc_to_nodes: Dict[Acc, List[NodeId]] = {}
        for node_id, acc in self.heads.items():
            if acc not in acc_to_nodes:
                acc_to_nodes[acc] = []
            acc_to_nodes[acc].append(node_id)

        for acc, node_ids in acc_to_nodes.items():
            new_node_id = self._get_node_id(value, tuple(node_ids))
            new_heads[new_node_id] = acc
        
        return GraphGSS(new_heads, self._acc_default_factory)

    def pop(self, value: T) -> 'GraphGSS[T, Acc]':
        # This implementation merges accumulators on pop, which differs from the reference.
        # This is a deliberate choice to showcase how the analyzer would catch semantic differences.
        # NOTE: This assumes a merge operation on the accumulator. For the purpose of this test,
        # we assume `+` is the merge operation, as defined in the test spec. A more robust
        # implementation would require the merge function to be available here.
        new_heads: Dict[NodeId, Acc] = {}
        
        for node_id, acc in self.heads.items():
            if node_id == self._root_id:
                continue
            
            node_val, parents = self._id_to_node.get(node_id, (None, None))
            if node_val == value:
                for parent_id in parents:
                    if parent_id in new_heads:
                        new_heads[parent_id] = new_heads[parent_id] + acc
                    else:
                        new_heads[parent_id] = acc
        
        return GraphGSS(new_heads, self._acc_default_factory)

    def apply(self, func: Callable[[Acc], Acc]) -> 'GraphGSS[T, Acc]':
        new_heads = {node_id: func(acc) for node_id, acc in self.heads.items()}
        return GraphGSS(new_heads, self._acc_default_factory)

    def peek(self) -> Set[T]:
        return {self._id_to_node[node_id][0] for node_id in self.heads if node_id != self._root_id}

    @staticmethod
    def merge(gss_list: Iterable['GraphGSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'GraphGSS[T, Acc]':
        factory = None
        merged_heads: Dict[NodeId, Acc] = {}

        for gss in gss_list:
            if isinstance(gss, GraphGSS):
                if factory is None:
                    factory = gss._acc_default_factory
                for node_id, acc in gss.heads.items():
                    if node_id in merged_heads:
                        merged_heads[node_id] = merge_func(merged_heads[node_id], acc)
                    else:
                        merged_heads[node_id] = acc
        
        if factory is None:
            raise ValueError("Cannot merge empty list of GSS or list with non-graph implementations")

        return GraphGSS(merged_heads, factory)

    def to_json_serializable(self) -> Any:
        # To be comparable with the reference impl, we must reconstruct the full stacks.
        memo: Dict[NodeId, List[List[T]]] = {}

        def get_paths(node_id: NodeId) -> List[List[T]]:
            if node_id in memo:
                return memo[node_id]
            if node_id == self._root_id:
                return [[]]
            
            val, parents = self._id_to_node[node_id]
            all_paths = []
            for p_id in parents:
                parent_paths = get_paths(p_id)
                for p_path in parent_paths:
                    all_paths.append(p_path + [val])
            memo[node_id] = all_paths
            return all_paths

        output_stacks = []
        for node_id, acc in self.heads.items():
            paths = get_paths(node_id)
            for path in paths:
                output_stacks.append({"stack": path, "acc": acc})
        
        # Sort for canonical representation
        return sorted(output_stacks, key=lambda x: (tuple(x['stack']), x['acc']))
