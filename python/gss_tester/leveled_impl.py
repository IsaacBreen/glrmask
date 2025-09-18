from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    frozenset,
)

from .interface import GSS, T, Acc
from .reference_impl import ReferenceGSS


# ------------------------------
# Structural "A" DAG definition
# ------------------------------


class _A:
    """
    Structural node for a persistent linked representation (DAG) of stack cells.
    This is a union of:
      - _ARoot: the root (represents the empty prefix)
      - _ACell: a cell holding one T and a pointer to previous _A node

    Nodes are interned via the owning _Arena.
    """

    __slots__ = ()

    def is_root(self) -> bool:
        return isinstance(self, _ARoot)

    def depth(self) -> int:
        """Depth (length) of stack at this node."""
        raise NotImplementedError

    def prev(self) -> Optional[_A]:
        """Previous structural node if this is a cell, else None for root."""
        raise NotImplementedError

    def value(self) -> T:
        """Top value T if this is a cell. Undefined for root."""
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class _ARoot(_A):
    __slots__ = ()

    def depth(self) -> int:
        return 0

    def prev(self) -> Optional[_A]:
        return None

    def value(self) -> T:
        raise AttributeError("Root has no value")

    def __repr__(self) -> str:
        return "_ARoot()"


@dataclass(frozen=True, eq=True)
class _ACell(_A, Generic[T]):
    """A single stack cell: holds the pushed value and a pointer to the previous node."""

    _prev: _A
    _value: T
    _depth: int  # cached depth == _prev.depth + 1

    __slots__ = ("_prev", "_value", "_depth")

    def depth(self) -> int:
        return self._depth

    def prev(self) -> Optional[_A]:
        return self._prev

    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return f"_ACell(value={self._value!r}, depth={self._depth})"


class _Arena(Generic[T]):
    """
    Interns (_A, T) -> _ACell and provides helpers for reconstructing stacks.
    This allows for sharing of stack structure between GSS instances.
    """

    def __init__(self) -> None:
        self._root: _ARoot = _ARoot()
        # Intern table: key = (prev_node, value) -> _ACell
        self._intern: Dict[Tuple[_A, T], _ACell[T]] = {}

    @property
    def root(self) -> _ARoot:
        return self._root

    def get_child(self, prev: _A, value: T) -> _ACell[T]:
        key = (prev, value)
        node = self._intern.get(key)
        if node is None:
            node = _ACell(prev, value, prev.depth() + 1)
            self._intern[key] = node
        return node

    def reconstruct_list(self, node: _A) -> List[T]:
        """Return the explicit stack list for this structural head."""
        vals: List[T] = []
        cur: Optional[_A] = node
        while isinstance(cur, _ACell):
            vals.append(cur.value())
            cur = cur.prev()
        vals.reverse()
        return vals


# ------------------------------
# "B" variant representation
# ------------------------------


class _BKind:
    EMPTY = "empty"  # No stacks present
    LEAF = "leaf"  # A single (_A, Acc) pair
    GROUP = "group"  # A {frozenset[_A], Acc} pair (sucked-up)
    BRANCH = "branch"  # A Dict[_A, Acc] for multiple accs


@dataclass(eq=False)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A Graph-Structured Stack implementation using a leveled representation.
    - The stack structure is a persistent DAG of `_A` nodes, managed by an `_Arena`.
    - The `LeveledGSS` object represents the set of active stack heads and their
      accumulators, using one of four variants (_BKind) for efficiency:
        1) EMPTY: Represents no stacks.
        2) LEAF: A single active stack head and its accumulator.
        3) GROUP: Multiple active stack heads that share a common accumulator.
           This is the "sucked-up" representation.
        4) BRANCH: Multiple active stack heads with different accumulators.
    """

    _kind: str
    _arena: _Arena[T]

    # Variant payloads
    _head: Optional[_A] = None
    _acc: Optional[Acc] = None
    _heads: Optional[frozenset[_A]] = None
    _children: Optional[Dict[_A, Acc]] = None

    # -------------------------
    # Construction and helpers
    # -------------------------

    @classmethod
    def _empty(cls, arena: _Arena[T]) -> LeveledGSS[T, Acc]:
        return cls(_BKind.EMPTY, arena)

    @classmethod
    def _leaf(cls, arena: _Arena[T], head: _A, acc: Acc) -> LeveledGSS[T, Acc]:
        return cls(_BKind.LEAF, arena, _head=head, _acc=acc)

    @classmethod
    def _group(cls, arena: _Arena[T], heads: frozenset[_A], acc: Acc) -> LeveledGSS[T, Acc]:
        return cls(_BKind.GROUP, arena, _acc=acc, _heads=heads)

    @classmethod
    def _branch(cls, arena: _Arena[T], children: Dict[_A, Acc]) -> LeveledGSS[T, Acc]:
        return cls(_BKind.BRANCH, arena, _children=children)

    @classmethod
    def _from_heads_map(cls, arena: _Arena[T], heads_to_acc: Dict[_A, Acc]) -> LeveledGSS[T, Acc]:
        """
        Canonical factory for LeveledGSS. Analyzes the heads and creates the
        most compact representation (LEAF, GROUP, or BRANCH).
        This implements the "suck-up" logic.
        """
        if not heads_to_acc:
            return cls._empty(arena)

        if len(heads_to_acc) == 1:
            head, acc = next(iter(heads_to_acc.items()))
            return cls._leaf(arena, head, acc)

        # Group heads by accumulator to check for suck-up opportunities
        acc_groups: Dict[Acc, List[_A]] = defaultdict(list)
        for head, acc in heads_to_acc.items():
            acc_groups[acc].append(head)

        if len(acc_groups) == 1:
            # All heads share one accumulator: suck up into a GROUP
            acc = next(iter(acc_groups.keys()))
            heads = next(iter(acc_groups.values()))
            return cls._group(arena, frozenset(heads), acc)

        # Multiple accumulators: create a BRANCH
        return cls._branch(arena, heads_to_acc)

    def _iter_heads(self) -> Iterable[Tuple[_A, Acc]]:
        """Helper to iterate through all (head, acc) pairs, abstracting over variants."""
        if self._kind == _BKind.EMPTY:
            return
        elif self._kind == _BKind.LEAF:
            assert self._head is not None and self._acc is not None
            yield (self._head, self._acc)
        elif self._kind == _BKind.GROUP:
            assert self._heads is not None and self._acc is not None
            for head in self._heads:
                yield (head, self._acc)
        elif self._kind == _BKind.BRANCH:
            assert self._children is not None
            yield from self._children.items()

    # -------------------------
    # GSS interface
    # -------------------------

    @classmethod
    def from_stacks(cls: Type[LeveledGSS], stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        arena: _Arena[T] = _Arena()
        heads_to_acc: Dict[_A, Acc] = {}
        for vals, acc in stacks:
            cur: _A = arena.root
            for v in vals:
                cur = arena.get_child(cur, v)

            if cur in heads_to_acc:
                heads_to_acc[cur] = heads_to_acc[cur].merge(acc)
            else:
                heads_to_acc[cur] = acc
        return cls._from_heads_map(arena, heads_to_acc)

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        new_map: Dict[_A, Acc] = {}
        for head, acc in self._iter_heads():
            child = self._arena.get_child(head, value)
            if child in new_map:
                new_map[child] = new_map[child].merge(acc)
            else:
                new_map[child] = acc
        return self._from_heads_map(self._arena, new_map)

    def pop(self) -> LeveledGSS[T, Acc]:
        new_map: Dict[_A, Acc] = {}
        for head, acc in self._iter_heads():
            if not head.is_root():
                prev = head.prev()
                assert prev is not None
                if prev in new_map:
                    new_map[prev] = new_map[prev].merge(acc)
                else:
                    new_map[prev] = acc
        return self._from_heads_map(self._arena, new_map)

    def is_empty(self) -> bool:
        if self._kind == _BKind.LEAF:
            return self._head is not None and self._head.is_root()
        return False

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        new_map: Dict[_A, Acc] = {}
        for head, acc in self._iter_heads():
            if value is None:
                if head.is_root():
                    new_map[head] = acc if head not in new_map else new_map[head].merge(acc)
            else:
                if isinstance(head, _ACell) and head.value() == value:
                    new_map[head] = acc if head not in new_map else new_map[head].merge(acc)
        return self._from_heads_map(self._arena, new_map)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        if self._kind == _BKind.EMPTY:
            return self
        if self._kind == _BKind.LEAF:
            assert self._head is not None and self._acc is not None
            return self._leaf(self._arena, self._head, func(self._acc))
        if self._kind == _BKind.GROUP:
            assert self._heads is not None and self._acc is not None
            return self._group(self._arena, self._heads, func(self._acc))

        assert self._children is not None
        new_children = {h: func(a) for h, a in self._children.items()}
        # Re-canonicalize in case func makes accumulators equal
        return self._from_heads_map(self._arena, new_children)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        new_map: Dict[_A, Acc] = {h: a for h, a in self._iter_heads() if predicate(a)}
        return self._from_heads_map(self._arena, new_map)

    def peek(self) -> Set[T]:
        tops: Set[T] = set()
        for head, _ in self._iter_heads():
            if isinstance(head, _ACell):
                tops.add(head.value())
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        accs = [acc for _, acc in self._iter_heads()]
        if not accs:
            return None
        return reduce(lambda a, b: a.merge(b), accs)

    def to_reference_impl(self) -> ReferenceGSS[T, Acc]:
        stacks: List[Tuple[List[T], Acc]] = []
        for head, acc in self._iter_heads():
            vals = self._arena.reconstruct_list(head)
            stacks.append((vals, acc))
        return ReferenceGSS.from_stacks(stacks)

    @staticmethod
    def merge(gss_list: Iterable[GSS[T, Acc]]) -> LeveledGSS[T, Acc]:
        if not gss_list:
            # This case needs an arena. We can't create one, so we can't return an empty GSS.
            # The caller should handle this. For now, we assume gss_list is not empty.
            # A more robust solution would be to have a default/global arena.
            # Or require an arena to be passed in.
            # For this problem, we can assume the list is non-empty and use the first arena.
            gss_list = list(gss_list)
            if not gss_list:
                raise ValueError("Cannot merge an empty list of GSS instances without a shared arena context.")

        arena: Optional[_Arena[T]] = None
        all_heads: Dict[_A, Acc] = {}

        for gss in gss_list:
            # All merged GSSs must share the same arena for nodes to be compatible.
            if isinstance(gss, LeveledGSS):
                if arena is None:
                    arena = gss._arena
                elif arena is not gss._arena:
                    raise ValueError("Cannot merge LeveledGSS instances from different arenas.")

                for head, acc in gss._iter_heads():
                    if head in all_heads:
                        all_heads[head] = all_heads[head].merge(acc)
                    else:
                        all_heads[head] = acc
            else:
                # Fallback for merging with other GSS types
                ref_impl = gss.to_reference_impl()
                temp_gss = LeveledGSS.from_stacks(ref_impl._stacks)
                if arena is None:
                    arena = temp_gss._arena
                for head, acc in temp_gss._iter_heads():
                    if head in all_heads:
                        all_heads[head] = all_heads[head].merge(acc)
                    else:
                        all_heads[head] = acc

        if arena is None:
            # This happens if gss_list contained only empty non-LeveledGSS instances
            return LeveledGSS(_BKind.EMPTY, _Arena())

        return LeveledGSS._from_heads_map(arena, all_heads)

    # -------------------------
    # Dunder methods
    # -------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GSS):
            return NotImplemented
        # Compare via ReferenceGSS for robust, canonical semantics
        return self.to_reference_impl() == other.to_reference_impl()

    def __repr__(self) -> str:
        if self._kind == _BKind.EMPTY:
            return "LeveledGSS(EMPTY)"
        if self._kind == _BKind.LEAF:
            assert self._head is not None
            stack = self._arena.reconstruct_list(self._head)
            return f"LeveledGSS(LEAF stack={stack!r} acc={self._acc!r})"
        if self._kind == _BKind.GROUP:
            stacks = sorted([self._arena.reconstruct_list(h) for h in self._heads])
            return f"LeveledGSS(GROUP stacks={stacks!r} acc={self._acc!r})"

        # BRANCH
        assert self._children is not None
        parts = []
        sorted_children = sorted(
            self._children.items(),
            key=lambda p: (self._arena.reconstruct_list(p[0]), repr(p[1])),
        )
        for h, a in sorted_children:
            parts.append(f"{self._arena.reconstruct_list(h)!r}:{a!r}")
        return f"LeveledGSS(BRANCH {{{', '.join(parts)}}})"
