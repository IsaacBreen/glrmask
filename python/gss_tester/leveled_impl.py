from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Dict, Generic, List, Optional, Set, Tuple, Type, Callable

from .interface import GSS, T, Acc
from .reference_impl import ReferenceGSS


class InvariantViolation(Exception):
    pass


@dataclass(frozen=True, eq=True)
class _Node(Generic[T]):
    """
    A single stack node. value is the top value for this head; parent points to the rest of the stack.
    The canonical empty stack is represented by the singleton _EMPTY (value=None, parent=None).
    """
    value: Optional[T]
    parent: Optional['_Node[T]']

    def is_empty(self) -> bool:
        return self.value is None and self.parent is None


# Canonical empty stack node (shared across all instances)
_EMPTY = _Node(None, None)


def _node_from_list(vals: List[T]) -> _Node[T]:
    """Build a node chain from a bottom-to-top list representation (top is last)."""
    n: _Node[T] = _EMPTY  # type: ignore[assignment]
    for v in vals:
        n = _Node(v, n)
    return n


def _list_from_node(node: _Node[T]) -> List[T]:
    """Convert a node chain back to a bottom-to-top list representation."""
    out: List[T] = []
    cur = node
    while cur is not _EMPTY and cur.value is not None:
        out.append(cur.value)  # type: ignore[arg-type]
        assert cur.parent is not None
        cur = cur.parent
    out.reverse()
    return out


class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    High-performance, sharing-preserving GSS based on a DAG of stack nodes.
    - Each active stack is represented by a head node (_Node).
    - Operations are implemented via direct traversal and map transforms (no pair materialization).
    - Node sharing is preserved automatically via parent pointers; deduplication uses structural equality.
    """

    def __init__(self, heads: Dict[_Node[T], Acc]):
        # Canonicalize: merge accumulators for identical heads (structural equality)
        dedup: Dict[_Node[T], Acc] = {}
        for node, acc in heads.items():
            if node in dedup:
                dedup[node] = dedup[node].merge(acc)
            else:
                dedup[node] = acc
        self._heads: Dict[_Node[T], Acc] = dedup

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        heads: Dict[_Node[T], Acc] = {}
        for vals, acc in stacks:
            node = _node_from_list(vals)
            if node in heads:
                heads[node] = heads[node].merge(acc)
            else:
                heads[node] = acc
        return cls(heads)

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        if not self._heads:
            return LeveledGSS({})
        new_heads: Dict[_Node[T], Acc] = {}
        for node, acc in self._heads.items():
            child = _Node(value, node)
            if child in new_heads:
                new_heads[child] = new_heads[child].merge(acc)
            else:
                new_heads[child] = acc
        return LeveledGSS(new_heads)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        if not self._heads:
            return LeveledGSS({})
        new_heads: Dict[_Node[T], Acc] = {}
        for node, acc in self._heads.items():
            # Empty stacks are discarded by pop()
            if node is _EMPTY or node.value is None or node.parent is None:
                continue
            parent = node.parent
            if parent in new_heads:
                new_heads[parent] = new_heads[parent].merge(acc)
            else:
                new_heads[parent] = acc
        return LeveledGSS(new_heads)

    def is_empty(self) -> bool:
        return not self._heads

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        if value is None:
            acc = self._heads.get(_EMPTY)  # type: ignore[arg-type]
            return LeveledGSS({} if acc is None else {_EMPTY: acc})  # type: ignore[dict-item]
        filtered: Dict[_Node[T], Acc] = {}
        for node, acc in self._heads.items():
            if node is _EMPTY or node.value is None:
                continue
            if node.value == value:
                filtered[node] = acc
        return LeveledGSS(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        if not self._heads:
            return LeveledGSS({})
        # Memoize on acc (by value if hashable, else by id)
        cache: Dict[object, Acc] = {}

        def key_for(a: Acc) -> object:
            try:
                hash(a)
                return ("h", a)
            except Exception:
                return ("i", id(a))

        changed = False
        new_heads: Dict[_Node[T], Acc] = {}
        for node, acc in self._heads.items():
            k = key_for(acc)
            new_acc = cache.get(k)
            if new_acc is None:
                new_acc = func(acc)
                cache[k] = new_acc
            if new_acc is not acc:
                changed = True
            new_heads[node] = new_acc
        return self if not changed else LeveledGSS(new_heads)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        if not self._heads:
            return LeveledGSS({})
        # Memoize predicate results (by value if hashable, else by id)
        cache: Dict[object, bool] = {}

        def key_for(a: Acc) -> object:
            try:
                hash(a)
                return ("h", a)
            except Exception:
                return ("i", id(a))

        changed = False
        kept: Dict[_Node[T], Acc] = {}
        for node, acc in self._heads.items():
            k = key_for(acc)
            ok = cache.get(k)
            if ok is None:
                ok = predicate(acc)
                cache[k] = ok
            if ok:
                kept[node] = acc
            else:
                changed = True
        return self if not changed else LeveledGSS(kept)

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        # Identity fast-path
        if other is self:
            return self
        if isinstance(other, LeveledGSS):
            if not other._heads:
                return self
            if not self._heads:
                return other
            merged: Dict[_Node[T], Acc] = dict(self._heads)
            for node, acc in other._heads.items():
                if node in merged:
                    merged[node] = merged[node].merge(acc)
                else:
                    merged[node] = acc
            return LeveledGSS(merged)
        # Fallback: build nodes from the other's reference representation
        ref = other.to_reference_impl()
        assert isinstance(ref, ReferenceGSS)
        merged: Dict[_Node[T], Acc] = dict(self._heads)
        for vals, acc in ref._stacks:  # type: ignore[attr-defined]
            node = _node_from_list(vals)
            if node in merged:
                merged[node] = merged[node].merge(acc)
            else:
                merged[node] = acc
        return LeveledGSS(merged)

    def peek(self) -> Set[T]:
        tops: Set[T] = set()
        for node in self._heads.keys():
            if node is _EMPTY or node.value is None:
                continue
            tops.add(node.value)
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        if not self._heads:
            return None
        it = iter(self._heads.values())
        try:
            first = next(it)
        except StopIteration:
            return None
        total = first
        for a in it:
            total = total.merge(a)
        return total

    def to_reference_impl(self) -> 'ReferenceGSS[T, Acc]':
        # Materialize pairs only on demand (for comparison/serialization)
        stacks: List[Tuple[List[T], Acc]] = []
        for node, acc in self._heads.items():
            vals = _list_from_node(node)
            stacks.append((vals, acc))
        return ReferenceGSS.from_stacks(stacks)

    # Human-friendly validator (acyclic, canonical empty)
    def validate_invariants(self) -> None:
        for node in self._heads.keys():
            visited = set()
            cur = node
            while cur is not _EMPTY and cur.value is not None:
                ident = id(cur)
                if ident in visited:
                    raise InvariantViolation("Cycle detected in node parents.")
                visited.add(ident)
                if cur.parent is None:
                    raise InvariantViolation("Non-empty node has no parent.")
                cur = cur.parent
            if cur is not _EMPTY:
                if cur.value is not None or cur.parent is not None:
                    raise InvariantViolation("Empty node must be the canonical _EMPTY.")

    def __repr__(self) -> str:
        return f"LeveledGSS(ref={self.to_reference_impl()!r})"

    def __str__(self) -> str:
        return f"LeveledGSS({self.to_reference_impl().to_json_serializable()})"
