from __future__ import annotations
from typing import TypeVar, Generic, Callable, Iterable, Final, Any

# --- Type Variables ---
T = TypeVar("T")
Acc = TypeVar("Acc")

# --- Module-level cache for node canonicalization ---
_NODE_CACHE: dict[tuple[Any, tuple[tuple[Any, Any], ...]], FastGSS] = {}


class FastGSS(Generic[T, Acc]):
    """
    An optimized, immutable implementation of a Graph-Structured Stack (GSS).

    Each instance of `FastGSS` is a canonicalized, immutable node representing a
    set of stack tails. A node is defined by its accumulator and its edges, where
    each edge consists of a value of type `T` and a pointer to a parent `FastGSS` node.

    Operations are functional, returning new `FastGSS` instances rather than
    modifying existing ones.
    """
    __slots__ = ('acc', 'edges', 'max_depth', '_hash', '_edges_dict')

    # Private constructor to force use of the factory function `_create_node`
    def __init__(self, acc: Acc, edges: tuple[tuple[T, FastGSS[T, Acc]], ...]):
        self.acc: Acc = acc
        self.edges: tuple[tuple[T, FastGSS[T, Acc]], ...] = edges

        if not self.edges:
            self.max_depth: int = 0
        else:
            # The depth of this node is 1 + the max depth of its parents.
            self.max_depth: int = 1 + max(p.max_depth for _, p in self.edges)

        self._hash: int = hash((self.acc, self.edges))
        # Lazy-loaded dict view of edges for faster lookups
        self._edges_dict: dict[T, FastGSS[T, Acc]] | None = None

    @property
    def edges_as_dict(self) -> dict[T, FastGSS[T, Acc]]:
        """A dictionary view of the edges for convenient access."""
        if self._edges_dict is None:
            self._edges_dict = dict(self.edges)
        return self._edges_dict

    def __repr__(self) -> str:
        return f"FastGSS(acc={self.acc}, edges={len(self.edges)}, depth={self.max_depth})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        # Canonicalization ensures that two equal nodes are the same object.
        if not isinstance(other, FastGSS):
            return NotImplemented
        return self is other

    @staticmethod
    def get_root() -> FastGSS[Any, Any]:
        """Returns the canonical root node, representing the set of empty stacks."""
        return _ROOT

    @classmethod
    def push(
        cls,
        parents: Iterable[tuple[FastGSS[T, Acc], T]],
        acc: Acc
    ) -> FastGSS[T, Acc]:
        """
        Creates a new GSS node by pushing values onto a collection of parent nodes.
        This operation effectively performs a merge of all parent stacks, followed
        by a push of the corresponding value for each.

        Args:
            parents: An iterable of (parent_node, value) tuples.
            acc: The accumulator for the new node.

        Returns:
            A new, canonicalized `FastGSS` node.
        """
        edges_by_val: dict[T, list[FastGSS[T, Acc]]] = {}
        for node, value in parents:
            if value not in edges_by_val:
                edges_by_val[value] = []
            edges_by_val[value].append(node)

        final_edges: set[tuple[T, FastGSS[T, Acc]]] = set()
        for value, nodes_to_merge in edges_by_val.items():
            if len(nodes_to_merge) == 1:
                final_edges.add((value, nodes_to_merge[0]))
            else:
                # Structurally merge the parents for a given value.
                # The accumulator of the synthetic merged parent is taken from the
                # first node in the list, as no merge function is provided here.
                merged_parent_edges = set()
                for node in nodes_to_merge:
                    merged_parent_edges.update(node.edges)

                merged_parent_acc = nodes_to_merge[0].acc
                merged_parent = _create_node(merged_parent_acc, frozenset(merged_parent_edges))
                final_edges.add((value, merged_parent))

        return _create_node(acc, frozenset(final_edges))

    def pop(self, count: int, value: T) -> FastGSS[T, Acc]:
        """
        Pops `count` levels from the stacks represented by this node, and then
        pushes a new value `T` on top of all resulting stack tails.

        Args:
            count: The number of levels to pop.
            value: The new value to push on top of the resulting stacks.

        Returns:
            A new `FastGSS` node representing the result of the pop-and-push.
        """
        memo = {}
        def _find_tails(node: FastGSS[T, Acc], num_to_pop: int) -> frozenset[FastGSS[T, Acc]]:
            if num_to_pop <= 0:
                return frozenset([node])
            if node is _ROOT:
                return frozenset([_ROOT])
            if (node, num_to_pop) in memo:
                return memo[(node, num_to_pop)]

            tails = set()
            for _, parent in node.edges:
                tails.update(_find_tails(parent, num_to_pop - 1))

            result = frozenset(tails)
            memo[(node, num_to_pop)] = result
            return result

        tails = _find_tails(self, count)

        # If popping results in multiple tails, they are structurally merged
        # into a single parent before the final value is pushed.
        if not tails:
            parent_node = _ROOT
        elif len(tails) == 1:
            parent_node = next(iter(tails))
        else:
            merged_edges = set()
            for tail_node in tails:
                merged_edges.update(tail_node.edges)
            # Use self's acc for the synthetic merged parent.
            parent_node = _create_node(self.acc, frozenset(merged_edges))

        return _create_node(self.acc, frozenset([(value, parent_node)]))

    def mutate(self, mutator: Callable[[Acc], Acc]) -> FastGSS[T, Acc]:
        """
        Creates a new node identical to this one, but with a new accumulator
        value produced by the mutator function.

        Args:
            mutator: A function that takes the old accumulator and returns a new one.

        Returns:
            A new `FastGSS` node with the updated accumulator.
        """
        new_acc = mutator(self.acc)
        return _create_node(new_acc, frozenset(self.edges))

    def merge(
        self,
        others: Iterable[FastGSS[T, Acc]],
        merger: Callable[[Acc, Acc], Acc]
    ) -> FastGSS[T, Acc]:
        """
        Merges this node with other GSS nodes.

        The new node's accumulator is the result of applying the merger function
        to all node accumulators. Its edges are the union of all edges from all
        merged nodes.

        Args:
            others: An iterable of `FastGSS` nodes to merge with this one.
            merger: A function to combine two accumulators.

        Returns:
            A new, merged `FastGSS` node.
        """
        all_nodes = [self] + list(others)
        if not all_nodes:
            return _ROOT

        # Merge accumulators
        new_acc = all_nodes[0].acc
        for i in range(1, len(all_nodes)):
            new_acc = merger(new_acc, all_nodes[i].acc)

        # Merge edges
        all_edges: set[tuple[T, FastGSS[T, Acc]]] = set(self.edges)
        for other_node in others:
            all_edges.update(other_node.edges)

        return _create_node(new_acc, frozenset(all_edges))

    def peek(self, values: set[T]) -> list[tuple[T, Acc]]:
        """
        Returns the top values and accumulator for stacks whose top value is in
        the provided set.

        Args:
            values: A set of `T` values to look for at the top of the stacks.

        Returns:
            A list of (`T`, `Acc`) tuples for each matching edge.
        """
        results = []
        for value, _ in self.edges:
            if value in values:
                results.append((value, self.acc))
        return results

    def get_all_stacks(self) -> dict[tuple[T, ...], Acc]:
        """
        Unrolls all unique stack paths ending at this node.

        Returns:
            A dictionary mapping each unique stack (a tuple of values) to the
            accumulator of this node.
        """
        memo = {}
        def _unroll(node: FastGSS[T, Acc]) -> set[tuple[T, ...]]:
            if node is _ROOT:
                return {tuple()}
            if node in memo:
                return memo[node]

            paths = set()
            if not node.edges:
                 paths.add(tuple())
                 memo[node] = paths
                 return paths

            for value, parent in node.edges:
                parent_paths = _unroll(parent)
                for path in parent_paths:
                    paths.add(path + (value,))

            memo[node] = paths
            return paths

        paths = _unroll(self)
        # The accumulator is associated with the head node, so all unrolled
        # stacks from this node will have the same accumulator.
        return {path: self.acc for path in paths}


def _create_node(acc: Acc, edges: frozenset[tuple[T, FastGSS[T, Acc]]]) -> FastGSS[T, Acc]:
    """Factory function to create and canonicalize FastGSS nodes."""
    # Sort edges by value hash to ensure canonical representation in the key.
    # This is important because frozenset order is not guaranteed.
    sorted_edges = tuple(sorted(edges, key=lambda item: hash(item[0])))
    key = (acc, sorted_edges)

    if key in _NODE_CACHE:
        return _NODE_CACHE[key]

    new_node = FastGSS(acc, sorted_edges)
    _NODE_CACHE[key] = new_node
    return new_node

# --- Global Root Node ---
# The root represents the empty GSS state.
_ROOT: Final[FastGSS] = FastGSS(acc=None, edges=tuple())
_NODE_CACHE[(None, tuple())] = _ROOT
