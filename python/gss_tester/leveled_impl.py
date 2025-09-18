from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import reduce
from typing import (
    TypeVar,
    Generic,
    List,
    Tuple,
    Callable,
    Set,
    Iterable,
    Dict,
    Any,
    Optional,
    Type,
)

from .interface import GSS, T, Acc


class _ANode(Generic[T, Acc]):
    """
    Structural node for the GSS DAG.
    - No accumulator is stored here.
    - Each node has exactly one parent (except root), and is identified by a unique id.
    - Edges are maintained externally via the owning GSS instance.
    """
    _id_counter = itertools.count()

    __slots__ = ("id", "depth")

    def __init__(self, depth: int):
        self.id = next(self._id_counter)
        self.depth = depth

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ANode):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        return f"_ANode(id={self.id}, depth={self.depth})"


@dataclass(frozen=True, slots=True)
class _BHead(Generic[T, Acc]):
    """
    "Head" wrapper that carries an accumulator at a specific structural node.
    This models the invariant:
      - Accumulators live only at heads (one 'level'): B wraps an A node + Acc.
      - If a parent node has a head (acc), its children will not (as operations replace heads).
      - If all children's accs would be equal on pop, they collapse into the parent (dedup of heads).
    """
    node: _ANode[T, Acc]
    acc: Acc

    def __hash__(self) -> int:
        # Hash by node id and accumulator
        return hash((self.node.id, self.acc))

    def __repr__(self) -> str:
        return f"_BHead(node={self.node!r}, acc={self.acc!r})"


class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A fast, immutable Graph-Structured Stack (GSS) focusing on "accumulators-at-one-level" invariants.

    Design:
      - Structural nodes (_ANode) form a DAG from root via labeled edges (values of type T).
      - Active stacks ("heads") are represented by _BHead objects (node+acc), i.e., type B wrapping type A + Acc.
      - Only heads carry accumulators; internal nodes do not.
      - Push replaces each head with a new child head; Pop replaces head(s) by parent head(s).
      - If multiple children pop to the same parent with equal accumulators, they deduplicate into a single parent head.
      - Multiple different accumulators at the same parent path are allowed (they represent distinct heads on that stack).

    Data structures:
      - _root: the root _ANode (represents the empty stack).
      - _root_acc: accumulator of the empty stack (Reference semantics).
      - _heads: frozenset of _BHead (the active stacks with accumulators).
      - _parent_of: dict mapping child _ANode -> (parent _ANode, value T). Root has no parent.
      - _child_of: dict mapping (parent _ANode, value T) -> child _ANode (unique).
      - _path_cache: dict mapping node.id -> tuple of T representing the path from root to this node (cached).
    """

    __slots__ = (
        "_root",
        "_root_acc",
        "_heads",
        "_parent_of",
        "_child_of",
        "_path_cache",
    )

    def __init__(
        self,
        root: _ANode[T, Acc],
        root_acc: Acc,
        heads: frozenset[_BHead[T, Acc]],
        parent_of: Dict[_ANode[T, Acc], Tuple[_ANode[T, Acc], T]],
        child_of: Dict[Tuple[_ANode[T, Acc], T], _ANode[T, Acc]],
        path_cache: Dict[int, Tuple[T, ...]],
    ):
        self._root = root
        self._root_acc = root_acc
        self._heads = heads
        self._parent_of = parent_of
        self._child_of = child_of
        self._path_cache = path_cache

    # ---- Construction ----

    @classmethod
    def from_stacks(cls: Type["LeveledGSS[T, Acc]"], stacks: List[Tuple[List[T], Acc]]) -> "LeveledGSS[T, Acc]":
        """
        Build a GSS from explicit stacks. Matches ReferenceGSS semantics:
          - There must be an empty stack [] providing the root accumulator.
          - Multiple stacks can be present; structure is shared where possible.
        """
        # Find root acc from empty stack, as in reference implementation
        root_acc: Optional[Acc] = None
        has_empty = False
        for s, acc in stacks:
            if not s:
                root_acc = acc
                has_empty = True
                break
        if not has_empty:
            raise ValueError("LeveledGSS.from_stacks requires an empty stack to determine the root accumulator.")

        root = _ANode[T, Acc](depth=0)

        # We'll build using a memo from path tuples to nodes for structural sharing
        path_to_node: Dict[Tuple[T, ...], _ANode[T, Acc]] = {tuple(): root}
        parent_of: Dict[_ANode[T, Acc], Tuple[_ANode[T, Acc], T]] = {}
        child_of: Dict[Tuple[_ANode[T, Acc], T], _ANode[T, Acc]] = {}
        path_cache: Dict[int, Tuple[T, ...]] = {root.id: tuple()}

        def get_or_create_node(path: Tuple[T, ...]) -> _ANode[T, Acc]:
            if path in path_to_node:
                return path_to_node[path]
            # Create by extending parent path
            parent_path, val = path[:-1], path[-1]
            parent_node = get_or_create_node(parent_path)
            key = (parent_node, val)
            if key in child_of:
                node = child_of[key]
                path_to_node[path] = node
                return node
            node = _ANode[T, Acc](depth=parent_node.depth + 1)
            parent_of[node] = (parent_node, val)
            child_of[key] = node
            path_to_node[path] = node
            path_cache[node.id] = path
            return node

        heads: Set[_BHead[T, Acc]] = set()

        for stack, acc in stacks:
            path = tuple(stack)
            node = get_or_create_node(path)
            heads.add(_BHead(node=node, acc=acc))

        # Reference semantics: If no stacks specified (shouldn't happen due to empty root), ensure there's root:
        if not heads:
            heads.add(_BHead(node=root, acc=root_acc))

        return cls(
            root=root,
            root_acc=root_acc,  # store the root accumulator
            heads=frozenset(heads),
            parent_of=parent_of,
            child_of=child_of,
            path_cache=path_cache,
        )

    # ---- Core operations ----

    def push(self, value: T) -> "LeveledGSS[T, Acc]":
        """
        Push 'value' onto all active stacks.
        Implementation:
          - For each head (node, acc), generate the child node for (node, value), reusing structure if it exists.
          - Result heads are the set of (child, acc).
          - This respects invariants: parent heads are replaced by children heads; parents no longer carry accs.
        """
        new_child_of = dict(self._child_of)
        new_parent_of = dict(self._parent_of)
        new_path_cache = dict(self._path_cache)

        new_heads: Set[_BHead[T, Acc]] = set()

        for head in self._heads:
            parent = head.node
            key = (parent, value)
            child = new_child_of.get(key)
            if child is None:
                child = _ANode[T, Acc](depth=parent.depth + 1)
                new_child_of[key] = child
                new_parent_of[child] = (parent, value)

                # Update path cache
                parent_path = self._reconstruct_path_cached(parent, new_parent_of, new_path_cache)
                new_path_cache[child.id] = parent_path + (value,)

            new_heads.add(_BHead(node=child, acc=head.acc))

        return LeveledGSS(
            root=self._root,
            root_acc=self._root_acc,
            heads=frozenset(new_heads),
            parent_of=new_parent_of,
            child_of=new_child_of,
            path_cache=new_path_cache,
        )

    def pop(self) -> "LeveledGSS[T, Acc]":
        """
        Pop the top value from all active stacks.
        Result:
          - For each head (node, acc), replace it with head(s) at its parent node(s).
          - Due to our structure, each node has at most one parent. If none (root), it disappears.
          - If multiple children map to the same parent and have equal accs, dedup occurs naturally.
          - If all heads disappear, return a GSS representing the single empty stack (root with _root_acc).
        """
        new_heads: Set[_BHead[T, Acc]] = set()

        for head in self._heads:
            parent_edge = self._parent_of.get(head.node)
            if parent_edge is not None:
                parent_node, _ = parent_edge
                new_heads.add(_BHead(node=parent_node, acc=head.acc))
            # else: head at root, pop has no effect for that head (it's removed)

        if not new_heads:
            # Return GSS with a single empty stack (root with root_acc), as per ReferenceGSS semantics.
            return LeveledGSS(
                root=self._root,
                root_acc=self._root_acc,
                heads=frozenset({_BHead(node=self._root, acc=self._root_acc)}),
                parent_of=self._parent_of,
                child_of=self._child_of,
                path_cache=self._path_cache,
            )

        return LeveledGSS(
            root=self._root,
            root_acc=self._root_acc,
            heads=frozenset(new_heads),
            parent_of=self._parent_of,
            child_of=self._child_of,
            path_cache=self._path_cache,
        )

    def is_empty(self) -> bool:
        """
        Checks if the GSS contains only the initial empty stack (root).
        """
        return len(self._heads) == 1 and next(iter(self._heads)).node == self._root

    def isolate(self, value: T) -> "LeveledGSS[T, Acc]":
        """
        Keep only stacks with 'value' at the top.
        Implementation detail:
          - Top is the labeled edge into the head's node. We check the incoming edge label.
        """
        new_heads: Set[_BHead[T, Acc]] = set()
        for head in self._heads:
            parent_edge = self._parent_of.get(head.node)
            if parent_edge is None:
                # root has no top; it doesn't match any value
                continue
            _, v = parent_edge
            if v == value:
                new_heads.add(head)

        if not new_heads:
            # Result is the empty GSS (root only)
            return LeveledGSS(
                root=self._root,
                root_acc=self._root_acc,
                heads=frozenset({_BHead(node=self._root, acc=self._root_acc)}),
                parent_of=self._parent_of,
                child_of=self._child_of,
                path_cache=self._path_cache,
            )

        return LeveledGSS(
            root=self._root,
            root_acc=self._root_acc,
            heads=frozenset(new_heads),
            parent_of=self._parent_of,
            child_of=self._child_of,
            path_cache=self._path_cache,
        )

    def apply(self, func: Callable[[Acc], Acc]) -> "LeveledGSS[T, Acc]":
        """
        Apply 'func' to each head accumulator.
        This preserves structure and ensures dedup if multiple heads map to same (node, acc') after transform.
        """
        new_heads = {_BHead(node=h.node, acc=func(h.acc)) for h in self._heads}

        return LeveledGSS(
            root=self._root,
            root_acc=self._root_acc,
            heads=frozenset(new_heads),
            parent_of=self._parent_of,
            child_of=self._child_of,
            path_cache=self._path_cache,
        )

    def prune(self, predicate: Callable[[Acc], bool]) -> "LeveledGSS[T, Acc]":
        """
        Remove stacks whose accumulator does not satisfy the predicate.
        If all are removed, return the empty GSS (root with root_acc).
        """
        new_heads = {h for h in self._heads if predicate(h.acc)}
        if not new_heads:
            return LeveledGSS(
                root=self._root,
                root_acc=self._root_acc,
                heads=frozenset({_BHead(node=self._root, acc=self._root_acc)}),
                parent_of=self._parent_of,
                child_of=self._child_of,
                path_cache=self._path_cache,
            )
        return LeveledGSS(
            root=self._root,
            root_acc=self._root_acc,
            heads=frozenset(new_heads),
            parent_of=self._parent_of,
            child_of=self._child_of,
            path_cache=self._path_cache,
        )

    def peek(self) -> Set[T]:
        """
        Return the set of top-of-stack values across all heads.
        """
        vals: Set[T] = set()
        for head in self._heads:
            parent_edge = self._parent_of.get(head.node)
            if parent_edge is not None:
                _, v = parent_edge
                vals.add(v)
        return vals

    def get_acc(self, merge_func: Callable[[Acc, Acc], Acc]) -> Acc:
        """
        Merge the accumulators of all active stacks using merge_func (left fold).
        """
        it = iter(self._heads)
        try:
            first = next(it)
        except StopIteration:
            # Shouldn't happen; empty GSS has root with an acc
            return self._root_acc
        acc_val = first.acc
        for h in it:
            acc_val = merge_func(acc_val, h.acc)
        return acc_val

    @staticmethod
    def merge(gss_list: Iterable["LeveledGSS[T, Acc]"], merge_func: Callable[[Acc, Acc], Acc]) -> "LeveledGSS[T, Acc]":
        """
        Merge multiple GSS instances into one, combining accumulators for identical stacks.
        Semantics:
          - If there is at least one GSS with non-empty stacks (i.e., some head not at root),
            ignore GSS instances that contain only the empty stack.
          - For identical stack paths across different GSS inputs, combine accumulators via merge_func.
          - Within a single GSS, distinct accumulators on the same path remain distinct (this method merges across inputs).
        """
        gsss = list(gss_list)
        if not gsss:
            raise ValueError("Cannot merge an empty list of GSS instances.")

        # If any GSS has content beyond the empty stack, we ignore purely-empty ones.
        def has_content(g: LeveledGSS[T, Acc]) -> bool:
            return any(h.node is not g._root for h in g._heads)

        with_content = [g for g in gsss if has_content(g)]
        first = gsss[0]

        if not with_content:
            # All are empty -> return an empty GSS bearing the first's root_acc
            return LeveledGSS.from_stacks([([], first._root_acc)])

        # Aggregate across all non-empty heads, merging accumulator values for identical paths across inputs.
        path_to_acc: Dict[Tuple[T, ...], Acc] = {}

        for g in with_content:
            for h in g._heads:
                if h.node is g._root:
                    continue  # ignore root-only when others exist
                path = g._reconstruct_path_cached(h.node, g._parent_of, g._path_cache)
                if path in path_to_acc:
                    path_to_acc[path] = merge_func(path_to_acc[path], h.acc)
                else:
                    path_to_acc[path] = h.acc

        # Rebuild a new structure from aggregated paths
        root = _ANode[T, Acc](depth=0)
        parent_of: Dict[_ANode[T, Acc], Tuple[_ANode[T, Acc], T]] = {}
        child_of: Dict[Tuple[_ANode[T, Acc], T], _ANode[T, Acc]] = {}
        path_cache: Dict[int, Tuple[T, ...]] = {root.id: tuple()}
        heads: Set[_BHead[T, Acc]] = set()

        def get_or_create_node(path: Tuple[T, ...]) -> _ANode[T, Acc]:
            # Build deterministically using child_of
            node = root
            if not path:
                return node
            # We don't keep a full memo by tuple to stay lightweight; follow child_of and create as needed.
            # Still O(len(path)) per path.
            curr = root
            curr_path = ()
            for v in path:
                key = (curr, v)
                nxt = child_of.get(key)
                if nxt is None:
                    nxt = _ANode[T, Acc](depth=curr.depth + 1)
                    child_of[key] = nxt
                    parent_of[nxt] = (curr, v)
                    curr_path = curr_path + (v,)
                    path_cache[nxt.id] = curr_path
                    curr = nxt
                else:
                    curr = nxt
                    # When traversing an existing node, we must update curr_path
                    # to ensure the next segment is built on the correct prefix.
                    if curr.id not in path_cache:
                        # This is a best-effort cache fill.
                        path_cache[curr.id] = _reconstruct_path(curr, parent_of, path_cache)
                    curr_path = path_cache[curr.id]
                node = curr
            return node

        # Create heads for each aggregated path
        for path, acc in path_to_acc.items():
            node = get_or_create_node(path)
            heads.add(_BHead(node=node, acc=acc))

        return LeveledGSS(
            root=root,
            root_acc=first._root_acc,
            heads=frozenset(heads),
            parent_of=parent_of,
            child_of=child_of,
            path_cache=path_cache,
        )

    # ---- JSON and equality/hash ----

    def to_json_serializable(self) -> Any:
        """
        Canonical JSON-serializable snapshot of current heads as a list of {"stack": [...], "acc": ...}.
        Sorted for deterministic comparison.
        """
        items = []
        for h in self._heads:
            path = self._reconstruct_path_cached(h.node, self._parent_of, self._path_cache)
            items.append({"stack": list(path), "acc": h.acc})
        # Sort canonically: by path then repr(acc) for stability across arbitrary objects
        items.sort(key=lambda x: (tuple(x["stack"]), repr(x["acc"])))
        return items

    def __hash__(self) -> int:
        serial = self.to_json_serializable()
        return hash(tuple((tuple(item["stack"]), repr(item["acc"])) for item in serial))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LeveledGSS):
            return NotImplemented
        return self.to_json_serializable() == other.to_json_serializable()

    def __repr__(self) -> str:
        return f"LeveledGSS({self.to_json_serializable()})"

    # ---- Internals ----

    @staticmethod
    def _reconstruct_path_cached(
        node: _ANode[T, Acc],
        parent_of: Dict[_ANode[T, Acc], Tuple[_ANode[T, Acc], T]],
        cache: Dict[int, Tuple[T, ...]],
    ) -> Tuple[T, ...]:
        if node.id in cache:
            return cache[node.id]
        return _reconstruct_path(node, parent_of, cache)


def _reconstruct_path(
    node: _ANode[T, Acc],
    parent_of: Dict[_ANode[T, Acc], Tuple[_ANode[T, Acc], T]],
    cache: Dict[int, Tuple[T, ...]],
) -> Tuple[T, ...]:
    """
    Reconstruct the path from root to 'node' using parent_of map.
    Assumes a unique parent chain (DAG without multiple parents for the same node in practice).
    """
    # Iterative to avoid recursion overhead
    stack_nodes: List[_ANode[T, Acc]] = []
    curr = node
    while True:
        if curr.id in cache:
            prefix = cache[curr.id]
            break
        stack_nodes.append(curr)
        p = parent_of.get(curr)
        if p is None:
            # curr is root
            prefix = ()
            cache[curr.id] = prefix
            stack_nodes.pop()  # remove root from list to avoid duplicate write below
            break
        curr = p[0]

    # Now unwind
    path = list(prefix)
    # We need to also collect the edge labels along the way
    # We replay from the cached node to the original 'node'
    last = None
    # Recompute the chain with values
    # We'll walk the same nodes from where we had cache to original 'node'.
    # First, figure out the chain including values:
    # Build a reverse chain from the original node to the nearest cached ancestor
    chain: List[Tuple[_ANode[T, Acc], T]] = []
    curr = node
    while True:
        if curr.id in cache:
            break
        parent_tuple = parent_of.get(curr)
        if parent_tuple is None:
            # reached root with no cache
            break
        parent_node, val = parent_tuple
        chain.append((parent_node, val))
        curr = parent_node
    # chain is from child->... to cached ancestor; reverse to go forward
    for parent_node, val in reversed(chain):
        path.append(val)
        # fill cache for child
        # The child is the next in chain; we can retrieve it indirectly but we don't need to cache each intermediate child explicitly here
    # Cache final node path
    cache[node.id] = tuple(path)
    return tuple(path)
