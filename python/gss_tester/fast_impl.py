import itertools
from functools import reduce
import collections
from typing import List, Tuple, Callable, Set, Iterable, Dict, Any, Type, Generic, FrozenSet

from .interface import GSS, T, Acc

class _Node(Generic[T, Acc]):
    """Represents a node in the GSS graph. Each node is a unique state."""
    _id_counter = itertools.count()

    def __init__(self, acc: Acc, depth: int):
        self.id: int = next(self._id_counter)
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

    def pop(self, value: T) -> 'FastGSS[T, Acc]':
        new_heads: Set[_Node[T, Acc]] = set()
        for head in self._heads:
            if head in self._child_to_parents:
                for v, parent in self._child_to_parents[head]:
                    if v == value:
                        new_heads.add(parent)

        if not new_heads:
            # A pop resulting in no stacks should yield a GSS with a single empty stack (the root).
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

    def peek(self) -> Set[T]:
        peek_values: Set[T] = set()
        for head in self._heads:
            if head in self._child_to_parents:
                for value, _ in self._child_to_parents[head]:
                peek_values.add(value)
        return peek_values

    def _with_heads(self, new_heads: FrozenSet[_Node]) -> 'FastGSS':
        return FastGSS(
            heads=new_heads,
            acc_default_factory=self._acc_default_factory,
            root=self._root,
            child_to_parents=self._child_to_parents,
            path_cache=self._path_cache
        )

    def popn_fast(self, n: int) -> List[Tuple[int, 'FastGSS']]:
        def popn_collect_nodes(gss: 'FastGSS', num_pops: int) -> Set[_Node]:
            level = gss._heads
            for _ in range(num_pops):
                next_level = set()
                for node in level:
                    if node in gss._child_to_parents:
                        for _, parent in gss._child_to_parents[node]:
                            next_level.add(parent)
                level = next_level
                if not level:
                    break
            return level

        nodes_at_n = popn_collect_nodes(self, n)

        result = []
        for node in nodes_at_n:
            if node in self._child_to_parents:
                for state_id, parent in self._child_to_parents[node]:
                    result.append((state_id, self._with_heads(frozenset([parent]))))
        return result

    def allowed_llm_tokens(self) -> Any:
        final_mask = self._acc_default_factory()['llms'].__class__.zeros()
        for head in self._heads:
            q_roots = collections.deque([head])
            visited_roots = {head}
            reachable_roots = set()
            while q_roots:
                node = q_roots.popleft()
                is_a_root = (node == self._root) or (node not in self._child_to_parents) or (not self._child_to_parents[node])

                if is_a_root:
                    reachable_roots.add(node)
                else:
                    for _, parent in self._child_to_parents[node]:
                        if parent not in visited_roots:
                            visited_roots.add(parent)
                            q_roots.append(parent)

            aggregated_llms = reduce(lambda a, b: a.union(b), (r.acc['llms'] for r in reachable_roots), self._acc_default_factory()['llms'].__class__.zeros())
            head_allowed = head.acc['llms'].intersection(aggregated_llms)
            final_mask = final_mask.union(head_allowed)
        return final_mask

    def is_alive(self) -> bool:
        for head in self._heads:
            local_llms = head.acc['llms']
            if local_llms.is_empty():
                continue

            q_roots = collections.deque([head])
            visited_roots = {head}
            while q_roots:
                node = q_roots.popleft()
                is_a_root = (node == self._root) or (node not in self._child_to_parents) or (not self._child_to_parents[node])
                if is_a_root:
                    if not local_llms.intersection(node.acc['llms']).is_empty():
                        return True
                elif node in self._child_to_parents:
                    for _, parent in self._child_to_parents[node]:
                        if parent not in visited_roots:
                            visited_roots.add(parent)
                            q_roots.append(parent)
        return False

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
        return sorted(all_stacks, key=lambda x: (x["stack"], x["acc"]))

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
