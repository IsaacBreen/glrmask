import itertools
from functools import reduce
from typing import List, Tuple, Callable, Set, Iterable, Dict, Any, Type, Generic, FrozenSet

from .interface import GSS, T, Acc

class _Node(Generic[T, Acc]):
    """Represents a node in the GSS graph. Each node is a unique state."""
    _id_counter = itertools.count()

    def __init__(self, acc: Acc, depth: int):
        self.id = next(self._id_counter)
        self.acc = acc
        self.depth = depth
    
    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if not isinstance(other, _Node):
            return NotImplemented
        return self.id == other.id
    
    def __repr__(self):
        return f"_Node(id={self.id}, depth={self.depth}, acc={self.acc})"

    def __str__(self):
        return self.__repr__()

class FastGSS(GSS[T, Acc]):
    """
    A performant GSS implementation using a graph of shared nodes.
    
    - The GSS is represented as a DAG where nodes are states and edges are stack operations.
    - Nodes store accumulators and their depth from the root.
    - The structure is represented by a child-to-parents mapping, allowing efficient `pop` operations.
    - GSS instances are immutable. Operations return new instances with updated heads or structure maps.
    - Path reconstruction for `merge` and `to_json_serializable` is memoized for performance.
    """

    def __init__(self, 
                 heads: FrozenSet[_Node[T, Acc]], 
                 acc_default_factory: Callable[[], Acc], 
                 root: _Node[T, Acc],
                 child_to_parents: Dict[_Node[T, Acc], Set[Tuple[T, _Node[T, Acc]]]],
                 path_cache: Dict[int, FrozenSet[Tuple[T, ...]]]):
        self._heads = heads
        self._acc_default_factory = acc_default_factory
        self._root = root
        self._child_to_parents = child_to_parents
        self._path_cache = path_cache

    @classmethod
    def initial(cls: Type['FastGSS'], acc_default_factory: Callable[[], Acc]) -> 'FastGSS[T, Acc]':
        root = _Node(acc=acc_default_factory(), depth=0)
        return cls(
            heads=frozenset([root]),
            acc_default_factory=acc_default_factory,
            root=root,
            child_to_parents={},
            path_cache={root.id: frozenset([tuple()])}
        )

    @classmethod
    def from_stacks(cls: Type['FastGSS'], stacks: List[Tuple[List[T], Acc]], acc_default_factory: Callable[[], Acc]) -> 'FastGSS[T, Acc]':
        """Creates a GSS from a list of explicit stacks by building the node graph."""
        root = _Node(acc=acc_default_factory(), depth=0)
        child_to_parents: Dict[_Node[T, Acc], Set[Tuple[T, _Node[T, Acc]]]] = {}
        path_cache: Dict[int, FrozenSet[Tuple[T, ...]]] = {root.id: frozenset([tuple()])}
        heads: Set[_Node[T, Acc]] = set()

        # Memoize nodes created for stack prefixes to ensure structure is shared.
        # key: tuple(stack_prefix), value: _Node
        memoized_nodes: Dict[Tuple[T, ...], _Node[T, Acc]] = {tuple(): root}

        if not stacks:
            heads.add(root)

        for stack_list, acc in stacks:
            stack_list.reverse()
            current_node = root
            stack_tuple = tuple(stack_list)

            # Traverse/create nodes for the path
            for i, value in enumerate(stack_tuple):
                prefix = stack_tuple[:i + 1]
                if prefix in memoized_nodes:
                    current_node = memoized_nodes[prefix]
                else:
                    # Create a new node. Its accumulator is temporary; only head accumulators matter.
                    new_node = _Node(acc=current_node.acc, depth=current_node.depth + 1)
                    child_to_parents[new_node] = {(value, current_node)}
                    memoized_nodes[prefix] = new_node
                    current_node = new_node
            
            # Now `current_node` is the shared node for this stack path.
            # We need a head with the specific accumulator.
            if current_node.acc == acc:
                heads.add(current_node)
            else:
                # Create a new head node with the correct accumulator but same structure.
                new_head = _Node(acc=acc, depth=current_node.depth)
                if current_node in child_to_parents:
                    child_to_parents[new_head] = child_to_parents[current_node]
                heads.add(new_head)

        return cls(frozenset(heads), acc_default_factory, root, child_to_parents, path_cache)

    def push(self, value: T) -> 'FastGSS[T, Acc]':
        new_heads: Set[_Node[T, Acc]] = set()
        new_child_to_parents = self._child_to_parents.copy()
        
        # Memoize node creation within a single push operation
        memo: Dict[_Node[T, Acc], _Node[T, Acc]] = {}

        for head in self._heads:
            if head in memo:
                new_heads.add(memo[head])
                continue

            new_node = _Node(acc=head.acc, depth=head.depth + 1)
            
            if new_node not in new_child_to_parents:
                new_child_to_parents[new_node] = set()
            new_child_to_parents[new_node].add((value, head))
            
            new_heads.add(new_node)
            memo[head] = new_node

        return FastGSS(frozenset(new_heads), self._acc_default_factory, self._root, new_child_to_parents, self._path_cache.copy())

    def pop(self) -> 'FastGSS[T, Acc]':
        new_heads: Set[_Node[T, Acc]] = set()
        for head in self._heads:
            if head in self._child_to_parents:
                for _, parent in self._child_to_parents[head]:
                    new_heads.add(parent)

        if not new_heads:
            # A pop resulting in no stacks should yield a GSS with a single empty stack (the root).
            return FastGSS(frozenset([self._root]), self._acc_default_factory, self._root, self._child_to_parents, self._path_cache)

        return FastGSS(frozenset(new_heads), self._acc_default_factory, self._root, self._child_to_parents, self._path_cache)

    def popn(self, n: int) -> 'FastGSS[T, Acc]':
        gss = self
        for _ in range(n):
            gss = gss.pop()
        return gss

    def isolate(self, value: T) -> 'FastGSS[T, Acc]':
        new_heads: Set[_Node[T, Acc]] = set()
        for head in self._heads:
            if head in self._child_to_parents:
                if any(v == value for v, _ in self._child_to_parents[head]):
                    new_heads.add(head)

        if not new_heads:
            # If no stacks match, the result is an empty GSS (represented by the root).
            return FastGSS(frozenset([self._root]), self._acc_default_factory, self._root, self._child_to_parents, self._path_cache)
        return FastGSS(frozenset(new_heads), self._acc_default_factory, self._root, self._child_to_parents, self._path_cache)

    def apply(self, func: Callable[[Acc], Acc]) -> 'FastGSS[T, Acc]':
        new_heads: Set[_Node[T, Acc]] = set()
        new_child_to_parents = self._child_to_parents.copy()
        memo: Dict[_Node[T, Acc], _Node[T, Acc]] = {}

        for head in self._heads:
            if head in memo:
                new_heads.add(memo[head])
                continue

            new_acc = func(head.acc)
            new_node = _Node(acc=new_acc, depth=head.depth)
            
            if head in self._child_to_parents:
                new_child_to_parents[new_node] = self._child_to_parents[head]
            
            new_heads.add(new_node)
            memo[head] = new_node
            
        return FastGSS(frozenset(new_heads), self._acc_default_factory, self._root, new_child_to_parents, self._path_cache.copy())

    def prune(self, predicate: Callable[[Acc], bool]) -> 'FastGSS[T, Acc]':
        new_heads = {head for head in self._heads if predicate(head.acc)}

        if not new_heads:
            # If all stacks are pruned, the result is an empty GSS (represented by the root).
            return FastGSS(frozenset([self._root]), self._acc_default_factory, self._root, self._child_to_parents, self._path_cache)

        return FastGSS(frozenset(new_heads), self._acc_default_factory, self._root, self._child_to_parents, self._path_cache)

    def split_heads(self) -> Iterable['FastGSS[T, Acc]']:
        for head in self._heads:
            yield FastGSS(frozenset([head]), self._acc_default_factory, self._root, self._child_to_parents, self._path_cache)

    def peek(self) -> Set[T]:
        peek_values: Set[T] = set()
        for head in self._heads:
            if head in self._child_to_parents:
                for value, _ in self._child_to_parents[head]:
                    peek_values.add(value)
        return peek_values

    def get_acc(self, merge_func: Callable[[Acc, Acc], Acc]) -> Acc:
        """Merges the accumulators of all active stacks into a single value."""
        accumulators = [head.acc for head in self._heads]
        return reduce(merge_func, accumulators)

    @staticmethod
    def merge(gss_list: Iterable['FastGSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'FastGSS[T, Acc]':
        gss_list = list(gss_list)
        if not gss_list:
            raise ValueError("Cannot merge an empty list of GSS instances.")
        
        first_gss = gss_list[0]
        
        if len(gss_list) == 1:
            return first_gss

        # Filter out GSSs that only contain an empty stack if there are others with content.
        gss_with_content = [gss for gss in gss_list if any(h is not gss._root for h in gss._heads)]

        if gss_with_content:
            gss_list_to_merge = gss_with_content
        else:
            # All GSSs only contain empty stacks, so merge them all.
            gss_list_to_merge = gss_list

        # Combine all structural information and heads
        all_child_to_parents = {}
        all_path_caches = {}
        all_heads = set()
        for gss in gss_list_to_merge:
            all_child_to_parents.update(gss._child_to_parents)
            all_path_caches.update(gss._path_cache)
            all_heads.update(gss._heads)

        # Group heads by their structural path
        heads_by_path: Dict[FrozenSet[Tuple[T, ...]], List[_Node[T, Acc]]] = {}
        for head in all_heads:
            paths = first_gss._reconstruct_paths(head, all_child_to_parents, all_path_caches)
            if paths not in heads_by_path:
                heads_by_path[paths] = []
            heads_by_path[paths].append(head)

        final_heads: Set[_Node[T, Acc]] = set()
        for paths, nodes in heads_by_path.items():
            if len(nodes) == 1:
                final_heads.add(nodes[0])
                continue
            
            # Merge accumulators for nodes on the same path
            merged_acc = reduce(merge_func, (n.acc for n in nodes))
            
            # Try to find an existing node with the merged accumulator to reuse
            reused_node = next((n for n in nodes if n.acc == merged_acc), None)
            
            if reused_node:
                final_heads.add(reused_node)
            else:
                # Create a new node representing the merged state
                canonical_node = nodes[0]
                new_node = _Node(acc=merged_acc, depth=canonical_node.depth)
                if canonical_node in all_child_to_parents:
                    all_child_to_parents[new_node] = all_child_to_parents[canonical_node]
                final_heads.add(new_node)

        return FastGSS(frozenset(final_heads), first_gss._acc_default_factory, first_gss._root, all_child_to_parents, all_path_caches)

    def to_json_serializable(self) -> Any:
        all_stacks = []
        for head in self._heads:
            paths = self._reconstruct_paths(head, self._child_to_parents, self._path_cache)
            for path in paths:
                all_stacks.append({"stack": list(path), "acc": head.acc})
        
        # Sort for a canonical representation
        return sorted(all_stacks, key=lambda x: (x["stack"], repr(x["acc"])))

    def _reconstruct_paths(self, node: _Node, child_to_parents: Dict, cache: Dict) -> FrozenSet[Tuple[T, ...]]:
        if node.id in cache:
            return cache[node.id]

        if node == self._root:
            return frozenset([tuple()])

        if node not in child_to_parents:
            # This can happen if a GSS is empty (has no heads pointing to this node's parents)
            return frozenset()

        paths: Set[Tuple[T, ...]] = set()
        for value, parent in child_to_parents[node]:
            parent_paths = self._reconstruct_paths(parent, child_to_parents, cache)
            for p_path in parent_paths:
                paths.add(p_path + (value,))
        
        result = frozenset(paths)
        cache[node.id] = result
        return result

    def __hash__(self):
        serial = self.to_json_serializable()
        hashable_serial = tuple(
            (tuple(item['stack']), item['acc']) for item in serial
        )
        return hash(hashable_serial)

    def __eq__(self, other):
        if not isinstance(other, FastGSS):
            return NotImplemented
        return self.to_json_serializable() == other.to_json_serializable()

    def __repr__(self):
        return f"FastGSS({self.to_json_serializable()})"
