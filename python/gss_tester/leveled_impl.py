from __future__ import annotations

from abc import ABC, abstractmethod
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
)

from .interface import GSS, T, Acc
from .reference_impl import ReferenceGSS


# ------------------------------
# Structural DAG definition (_A nodes)
#
# These nodes represent the stack structure. They are immutable and are not
# interned in an arena; they behave as value types.
# ------------------------------


@dataclass(frozen=True, eq=True)
class _A(ABC, Generic[T]):
    """Abstract base class for a structural stack node."""

    __slots__ = ()

    @abstractmethod
    def depth(self) -> int:
        ...

    @abstractmethod
    def prev(self) -> Optional[_A[T]]:
        ...

    def is_root(self) -> bool:
        return self.depth() == 0


@dataclass(frozen=True, eq=True)
class _ARoot(_A[T]):
    """Represents the root of a stack (the empty stack)."""

    __slots__ = ()

    def depth(self) -> int:
        return 0

    def prev(self) -> Optional[_A[T]]:
        return None

    def __repr__(self) -> str:
        return "_ARoot()"


_A_ROOT: _ARoot = _ARoot()


@dataclass(frozen=True, eq=True)
class _ACell(_A[T], Generic[T]):
    """A single stack cell: holds a value and a pointer to the previous node."""

    _prev: _A[T]
    _value: T
    _depth: int

    __slots__ = ("_prev", "_value", "_depth")

    def __init__(self, prev: _A[T], value: T):
        # We need a custom __init__ to compute the depth, as frozen dataclasses
        # don't easily support fields calculated in __post_init__.
        # object.__setattr__ is used to bypass the frozen attribute restriction
        # during initialization.
        object.__setattr__(self, "_prev", prev)
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_depth", prev.depth() + 1)

    def depth(self) -> int:
        return self._depth

    def prev(self) -> Optional[_A[T]]:
        return self._prev

    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return f"_ACell(value={self._value!r}, depth={self._depth})"


def _reconstruct_list(node: _A[T]) -> List[T]:
    """Return the explicit stack list for a given structural head node."""
    vals: List[T] = []
    cur: Optional[_A[T]] = node
    while isinstance(cur, _ACell):
        vals.append(cur.value())
        cur = cur.prev()
    vals.reverse()
    return vals


# ------------------------------
# LeveledGSS implementation
# ------------------------------


class _BKind:
    EMPTY = "empty"  # Represents a GSS with no active stacks
    GROUP = "group"  # Multiple heads, one common accumulator ("sucked-up")
    BRANCH = "branch"  # Multiple heads, multiple distinct accumulators


@dataclass(eq=False)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A GSS implementation using a leveled, "sucked-up" representation.

    - Stack structure is a persistent DAG of immutable `_A` nodes (value types).
    - The `LeveledGSS` object represents the set of active stack heads and their
      accumulators, using one of three variants for efficiency:
        1) EMPTY: Represents no stacks.
        2) GROUP: Multiple stack heads that share a common accumulator.
        3) BRANCH: Multiple stack heads with different accumulators.

    The factory method `_from_heads_map` ensures the most compact representation
    is always used, which is how the "suck-up" logic is implemented.
    """

    _kind: str

    # Variant payloads
    _heads: Optional[frozenset[_A[T]]] = None
    _acc: Optional[Acc] = None
    _children: Optional[Dict[_A[T], Acc]] = None

    # -------------------------
    # Construction and helpers
    # -------------------------

    @classmethod
    def _empty(cls) -> LeveledGSS[T, Acc]:
        return cls(_BKind.EMPTY)

    @classmethod
    def _group(cls, heads: frozenset[_A[T]], acc: Acc) -> LeveledGSS[T, Acc]:
        return cls(_BKind.GROUP, _heads=heads, _acc=acc)

    @classmethod
    def _branch(cls, children: Dict[_A[T], Acc]) -> LeveledGSS[T, Acc]:
        return cls(_BKind.BRANCH, _children=children)

    @classmethod
    def _from_heads_map(cls, heads_to_acc: Dict[_A[T], Acc]) -> LeveledGSS[T, Acc]:
        """
        Canonical factory for LeveledGSS. Analyzes the heads and creates the
        most compact representation (GROUP or BRANCH). This implements the
        "suck-up" logic.
        """
        if not heads_to_acc:
            return cls._empty()

        # Check if all accumulators are the same to enable "suck-up"
        first_acc = next(iter(heads_to_acc.values()))
        all_same = all(acc == first_acc for acc in heads_to_acc.values())

        if all_same:
            # All heads share one accumulator: suck up into a GROUP
            return cls._group(frozenset(heads_to_acc.keys()), first_acc)
        else:
            # Multiple accumulators: create a BRANCH
            return cls._branch(heads_to_acc)

    def _iter_heads(self) -> Iterable[Tuple[_A[T], Acc]]:
        """Helper to iterate through all (head, acc) pairs, abstracting over variants."""
        if self._kind == _BKind.EMPTY:
            return
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
    def from_stacks(
        cls: Type[LeveledGSS], stacks: List[Tuple[List[T], Acc]]
    ) -> LeveledGSS[T, Acc]:
        heads_to_acc: Dict[_A[T], Acc] = {}
        root: _A[T] = _A_ROOT
        for vals, acc in stacks:
            cur = root
            for v in vals:
                cur = _ACell(cur, v)

            if cur in heads_to_acc:
                heads_to_acc[cur] = heads_to_acc[cur].merge(acc)
            else:
                heads_to_acc[cur] = acc
        return cls._from_heads_map(heads_to_acc)

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        new_map: Dict[_A[T], Acc] = {}
        for head, acc in self._iter_heads():
            child = _ACell(head, value)
            if child in new_map:
                new_map[child] = new_map[child].merge(acc)
            else:
                new_map[child] = acc
        return self._from_heads_map(new_map)

    def pop(self) -> LeveledGSS[T, Acc]:
        new_map: Dict[_A[T], Acc] = {}
        for head, acc in self._iter_heads():
            if not head.is_root():
                prev = head.prev()
                assert prev is not None
                if prev in new_map:
                    new_map[prev] = new_map[prev].merge(acc)
                else:
                    new_map[prev] = acc
        return self._from_heads_map(new_map)

    def is_empty(self) -> bool:
        if self._kind == _BKind.GROUP:
            # A GSS is "empty" if it contains exactly one stack, which is empty.
            assert self._heads is not None
            return len(self._heads) == 1 and next(iter(self._heads)).is_root()
        return False

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        new_map: Dict[_A[T], Acc] = {}
        for head, acc in self._iter_heads():
            if value is None:
                if head.is_root():
                    new_map[head] = (
                        acc if head not in new_map else new_map[head].merge(acc)
                    )
            else:
                if isinstance(head, _ACell) and head.value() == value:
                    new_map[head] = (
                        acc if head not in new_map else new_map[head].merge(acc)
                    )
        return self._from_heads_map(new_map)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        if self._kind == _BKind.EMPTY:
            return self
        if self._kind == _BKind.GROUP:
            assert self._heads is not None and self._acc is not None
            return self._group(self._heads, func(self._acc))

        assert self._children is not None
        new_children = {h: func(a) for h, a in self._children.items()}
        # Re-canonicalize in case func makes some accumulators equal
        return self._from_heads_map(new_children)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        new_map: Dict[_A[T], Acc] = {
            h: a for h, a in self._iter_heads() if predicate(a)
        }
        return self._from_heads_map(new_map)

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
            vals = _reconstruct_list(head)
            stacks.append((vals, acc))
        return ReferenceGSS.from_stacks(stacks)

    @staticmethod
    def merge(gss_list: Iterable[GSS[T, Acc]]) -> LeveledGSS[T, Acc]:
        all_heads: Dict[_A[T], Acc] = {}
        root: _A[T] = _A_ROOT

        for gss in gss_list:
            # To merge different GSS types, we convert them to a common
            # representation of (stack_values, acc) pairs via ReferenceGSS.
            ref_gss = gss.to_reference_impl()
            for vals, acc in ref_gss._stacks:
                # Reconstruct the _A node structure for this stack
                cur = root
                for v in vals:
                    cur = _ACell(cur, v)

                if cur in all_heads:
                    all_heads[cur] = all_heads[cur].merge(acc)
                else:
                    all_heads[cur] = acc

        return LeveledGSS._from_heads_map(all_heads)

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
        if self._kind == _BKind.GROUP:
            assert self._heads is not None and self._acc is not None
            stacks = sorted([_reconstruct_list(h) for h in self._heads])
            return f"LeveledGSS(GROUP stacks={stacks!r} acc={self._acc!r})"

        # BRANCH
        assert self._children is not None
        parts = []
        # Sort for deterministic output
        sorted_children = sorted(
            self._children.items(),
            key=lambda p: (_reconstruct_list(p[0]), repr(p[1])),
        )
        for h, a in sorted_children:
            parts.append(f"{_reconstruct_list(h)!r}:{a!r}")
        return f"LeveledGSS(BRANCH {{{', '.join(parts)}}})"
