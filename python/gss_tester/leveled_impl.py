from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
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

    We avoid a factory class; nodes are interned via the owning _Arena.
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


@dataclass(frozen=True, eq=False)
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


@dataclass(frozen=True, eq=False)
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
    Interns (_A, T) -> _ACell and provides helpers for reconstructing stacks and
    edge metadata. This replaces the previous _NodeFactory/_StackNode types.
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
    EMPTY = "empty"      # No stacks present
    WITH_ACC = "acc"     # Contains (A, Acc): a leaf that carries accumulator at this structural head
    BRANCH = "branch"    # Maps to other Bs (an aggregate of leaf nodes). Anchored at the arena root.


@dataclass(eq=False)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A Graph-Structured Stack implementation using:
      - A persistent structural DAG of stack cells (_A nodes) managed by an _Arena.
      - A leveled representation "B" with three variants:
          1) EMPTY: represents no stacks
          2) WITH_ACC: a leaf that contains one structural head (_A) and its accumulator (Acc)
          3) BRANCH: maps to other Bs (children), each of which must be a WITH_ACC leaf.
             This ensures accumulators occur only at one 'level' (i.e., at the leaves).

    Notes:
      - We do not store edges with max-depth directly in the structure; instead we can compute
        the edge->max-depth metadata on demand from the current set of heads.
      - Equality and hashing are based on the canonical ReferenceGSS view.
      - For identical stacks, we merge accumulators via Acc.merge() in canonicalization steps.
    """

    # Variant tag
    _kind: str

    # Common arena (shared across persistent states)
    _arena: _Arena[T]

    # Variant payloads:
    # - For EMPTY: none used
    # - For WITH_ACC: a single head and its accumulator
    _head: Optional[_A] = None
    _acc: Optional[Acc] = None

    # - For BRANCH: map from structural head to merged Acc
    #   The children are WITH_ACC leaves; we store them compactly as a dict head->Acc.
    _children: Optional[Dict[_A, Acc]] = None

    # -------------------------
    # Construction and helpers
    # -------------------------

    @classmethod
    def _empty(cls, arena: Optional[_Arena[T]] = None) -> "LeveledGSS[T, Acc]":
        return cls(_BKind.EMPTY, arena if arena is not None else _Arena(), None, None, None)

    @classmethod
    def _leaf(cls, arena: _Arena[T], head: _A, acc: Acc) -> "LeveledGSS[T, Acc]":
        return cls(_BKind.WITH_ACC, arena, head, acc, None)

    @classmethod
    def _branch(cls, arena: _Arena[T], children: Dict[_A, Acc]) -> "LeveledGSS[T, Acc]":
        # Canonicalize: merge identical heads (already merged via dict)
        if not children:
            return cls._empty(arena)

        # If exactly one child, present as a leaf
        if len(children) == 1:
            (head, acc) = next(iter(children.items()))
            return cls(_BKind.WITH_ACC, arena, head, acc, None)

        return cls(_BKind.BRANCH, arena, None, None, dict(children))

    def _ensure_branch(self) -> Dict[_A, Acc]:
        """
        Return a copy of children map for mutation logic. Converting leaves to branches if needed.
        This returns a new dict so that the current node remains persistent/immutable.
        """
        if self._kind == _BKind.EMPTY:
            return {}
        if self._kind == _BKind.BRANCH:
            return dict(self._children or {})
        if self._kind == _BKind.WITH_ACC:
            # Single leaf -> branch with one child
            assert self._head is not None and self._acc is not None
            return {self._head: self._acc}
        raise AssertionError("Unknown variant")

    def _heads_items(self) -> List[Tuple[_A, Acc]]:
        """
        Return the list of (head, acc) pairs in this state, merged per-head.
        """
        if self._kind == _BKind.EMPTY:
            return []
        if self._kind == _BKind.WITH_ACC:
            assert self._head is not None and self._acc is not None
            return [(self._head, self._acc)]
        # BRANCH
        assert self._children is not None
        return list(self._children.items())

    @staticmethod
    def _merge_acc(a: Acc, b: Acc) -> Acc:
        return a.merge(b)  # type: ignore[return-value]

    @classmethod
    def _from_heads_map(cls, arena: _Arena[T], heads_to_acc: Dict[_A, Acc]) -> "LeveledGSS[T, Acc]":
        """
        Build a LeveledGSS from a merged head->acc map, applying 'suck up' if possible.
        """
        return cls._branch(arena, heads_to_acc)

    # -------------------------
    # GSS interface
    # -------------------------

    @classmethod
    def from_stacks(cls: Type["LeveledGSS[T, Acc]"], stacks: List[Tuple[List[T], Acc]]) -> "LeveledGSS[T, Acc]":
        arena: _Arena[T] = _Arena()
        # Merge identical stacks by merging their accs
        per_head: Dict[_A, Acc] = {}
        for vals, acc in stacks:
            cur: _A = arena.root
            for v in vals:
                cur = arena.get_child(cur, v)
            # Merge acc for identical heads
            if cur in per_head:
                per_head[cur] = cls._merge_acc(per_head[cur], acc)
            else:
                per_head[cur] = acc
        return cls._from_heads_map(arena, per_head)

    def push(self, value: T) -> "LeveledGSS[T, Acc]":
        arena = self._arena
        new_map: Dict[_A, Acc] = {}
        for head, acc in self._heads_items():
            child = arena.get_child(head, value)
            if child in new_map:
                new_map[child] = self._merge_acc(new_map[child], acc)
            else:
                new_map[child] = acc
        # Note: pushing empty state remains empty
        return self._from_heads_map(arena, new_map)

    def pop(self) -> "LeveledGSS[T, Acc]":
        arena = self._arena
        new_map: Dict[_A, Acc] = {}
        # Popping empties yields nothing; popping one-element stacks yields root (empty stack)
        for head, acc in self._heads_items():
            if isinstance(head, _ARoot):
                # no-op: popping empty stack yields nothing (drop), same as ReferenceGSS
                continue
            prev = head.prev()
            assert prev is not None
            if prev in new_map:
                new_map[prev] = self._merge_acc(new_map[prev], acc)
            else:
                new_map[prev] = acc
        return self._from_heads_map(arena, new_map)

    def is_empty(self) -> bool:
        """
        True iff there is exactly one active stack and that stack is the empty stack.
        """
        items = self._heads_items()
        if len(items) != 1:
            return False
        (head, _) = items[0]
        return isinstance(head, _ARoot)

    def isolate(self, value: Optional[T]) -> "LeveledGSS[T, Acc]":
        arena = self._arena
        new_map: Dict[_A, Acc] = {}
        if value is None:
            # Keep only empty stacks (heads that are root)
            for head, acc in self._heads_items():
                if isinstance(head, _ARoot):
                    # Merge: there could be only one, but keep consistent
                    if head in new_map:
                        new_map[head] = self._merge_acc(new_map[head], acc)
                    else:
                        new_map[head] = acc
            return self._from_heads_map(arena, new_map)

        # Keep only stacks whose top equals `value`
        for head, acc in self._heads_items():
            if isinstance(head, _ACell) and head.value() == value:
                new_map[head] = acc if head not in new_map else self._merge_acc(new_map[head], acc)
        return self._from_heads_map(arena, new_map)

    def apply(self, func: Callable[[Acc], Acc]) -> "LeveledGSS[T, Acc]":
        arena = self._arena
        if self._kind == _BKind.EMPTY:
            return self

        if self._kind == _BKind.WITH_ACC:
            assert self._head is not None and self._acc is not None
            return LeveledGSS(_BKind.WITH_ACC, arena, self._head, func(self._acc), None)

        # BRANCH
        assert self._children is not None
        transformed: Dict[_A, Acc] = {h: func(a) for h, a in self._children.items()}
        return self._from_heads_map(arena, transformed)

    def prune(self, predicate: Callable[[Acc], bool]) -> "LeveledGSS[T, Acc]":
        arena = self._arena
        if self._kind == _BKind.EMPTY:
            return self

        if self._kind == _BKind.WITH_ACC:
            assert self._head is not None and self._acc is not None
            return self if predicate(self._acc) else self._empty(arena)

        # BRANCH
        assert self._children is not None
        kept: Dict[_A, Acc] = {h: a for h, a in self._children.items() if predicate(a)}
        return self._from_heads_map(arena, kept)

    def peek(self) -> Set[T]:
        # Return all top values across non-empty stacks
        tops: Set[T] = set()
        for head, _ in self._heads_items():
            if isinstance(head, _ACell):
                tops.add(head.value())
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        items = [acc for _, acc in self._heads_items()]
        if not items:
            return None
        return reduce(lambda a, b: a.merge(b), items)

    def to_reference_impl(self) -> "ReferenceGSS[T, Acc]":
        """
        Convert to ReferenceGSS by enumerating all heads with their merged accumulators,
        then letting the ReferenceGSS canonicalize.
        """
        stacks: List[Tuple[List[T], Acc]] = []
        for head, acc in self._heads_items():
            vals = self._arena.reconstruct_list(head)
            stacks.append((vals, acc))
        return ReferenceGSS.from_stacks(stacks)

    @staticmethod
    def merge(gss_list: Iterable["GSS[T, Acc]"]) -> "LeveledGSS[T, Acc]":
        """
        Merge multiple GSS states by:
          - converting each to ReferenceGSS,
          - merging via ReferenceGSS.merge,
          - building a new LeveledGSS from the merged stacks.
        """
        ref_inputs: List[ReferenceGSS[T, Acc]] = []
        for g in gss_list:
            if isinstance(g, ReferenceGSS):
                ref_inputs.append(g)
            else:
                ref_inputs.append(g.to_reference_impl())  # type: ignore[arg-type]
        merged_ref: ReferenceGSS[T, Acc] = ReferenceGSS.merge(ref_inputs)
        return LeveledGSS.from_stacks(merged_ref._stacks)  # type: ignore[attr-defined]

    # -------------------------
    # Internal utilities
    # -------------------------

    def _edge_max_depth_map(self) -> Dict[Tuple[Optional[_A], T], int]:
        """
        Compute internal edge->maxDepth metadata for debugging/introspection:
          key = (parent_node, symbol T). parent_node is None for edges from the root.
          value = max absolute depth (length) of any active stack that traverses that edge.
        """
        edge_depths: Dict[Tuple[Optional[_A], T], int] = {}

        # For each head, walk upward to root, updating edge max depth
        for head, _ in self._heads_items():
            # Skip empty head; it contributes no edges
            if isinstance(head, _ARoot):
                continue
            d = head.depth()
            cur: Optional[_A] = head
            while isinstance(cur, _ACell):
                parent = cur.prev()
                key = (parent, cur.value())
                prev_max = edge_depths.get(key)
                if prev_max is None or d > prev_max:
                    edge_depths[key] = d
                d -= 1
                cur = parent

        return edge_depths

    def __hash__(self) -> int:
        # Hash based on canonical representation: head->acc (merged)
        # Beware: Acc must be hashable or we fallback to string repr.
        items = []
        for head, acc in self._heads_items():
            vals = tuple(self._arena.reconstruct_list(head))
            items.append((vals, acc))
        try:
            return hash(frozenset(items))  # type: ignore[arg-type]
        except Exception:
            # Fallback to deterministic string-based hash
            as_str = tuple(sorted((repr(vals), repr(acc)) for vals, acc in items))
            return hash(as_str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LeveledGSS):
            return NotImplemented
        # Compare via ReferenceGSS equality for robust, canonical semantics
        return self.to_reference_impl() == other.to_reference_impl()

    def __repr__(self) -> str:
        if self._kind == _BKind.EMPTY:
            return "LeveledGSS(EMPTY)"
        if self._kind == _BKind.WITH_ACC:
            assert self._head is not None and self._acc is not None
            stack = self._arena.reconstruct_list(self._head)
            return f"LeveledGSS(ACC stack={stack!r} acc={self._acc!r})"
        # BRANCH
        assert self._children is not None
        parts = []
        for h, a in sorted(((h, a) for h, a in self._children.items()), key=lambda p: (self._arena.reconstruct_list(p[0]), repr(p[1]))):
            parts.append(f"{self._arena.reconstruct_list(h)!r}:{a!r}")
        return f"LeveledGSS(BRANCH {{{', '.join(parts)}}})"
