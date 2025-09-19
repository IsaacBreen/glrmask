from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any

from .interface import GSS, T, Acc


# ------------------------------
# Public node classes (unchanged API)
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    inner: 'UpperBranch[T, Acc]' | 'Interface[T, Acc]'


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, 'Upper[T, Acc]']]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: 'Lower[T]'
    acc: Acc | None  # Placeholder for top-level interface acc.


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    inner: 'LowerBranch[T]' | 'Leaf'


@dataclass(frozen=True, eq=True)
class LowerBranch(Generic[T]):
    children: Dict[Any, Dict[int, 'Lower[T]']]


@dataclass(frozen=True, eq=True)
class Leaf:
    pass


# ------------------------------
# Minimal internal representation: compact root map sentinel
# ------------------------------

class _PathMap(Generic[T, Acc]):
    """
    Compact sentinel storing the entire path->acc map at the root Lower node.

    We embed this object as the only key in LowerBranch.children.
    This keeps encode/decode O(1) and avoids traversals.
    """
    __slots__ = ("paths",)

    def __init__(self, paths: Dict[Tuple[T, ...], Acc]):
        # Treat as immutable after construction.
        self.paths = paths

    def __repr__(self) -> str:
        return f"<PathMap:{len(self.paths)} paths>"

    # Identity equality keeps structural sharing semantics simple and fast.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


def _empty_lower() -> Lower[T]:
    return Lower(Leaf())


def _lower_to_map(node: Lower[T]) -> Dict[Tuple[T, ...], Acc]:
    """
    Decode a Lower node into its path->acc map.

    This compact implementation supports either:
    - an empty Lower represented by Leaf, or
    - a LowerBranch whose children dict has exactly one key: a _PathMap sentinel.
    Any deviation returns an empty map (should not occur in this implementation).
    """
    inner = node.inner
    if isinstance(inner, Leaf):
        return {}
    children = inner.children
    if not children:
        return {}
    if len(children) == 1:
        only_key = next(iter(children))
        if isinstance(only_key, _PathMap):
            # Return the underlying dict by reference. Treat as read-only.
            return only_key.paths
    # Unexpected structure: treat as empty.
    return {}


def _map_to_lower(paths: Dict[Tuple[T, ...], Acc]) -> Lower[T]:
    """
    Encode a path->acc map into the compact Lower form using a single _PathMap.
    """
    if not paths:
        return _empty_lower()
    # Value map (Dict[int, Lower[T]]) is irrelevant to decoding; keep empty for minimal footprint.
    children: Dict[Any, Dict[int, Lower[T]]] = {_PathMap(paths): {}}
    return Lower(LowerBranch(children=children))


def _merge_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


def _map_merge_paths(a: Dict[Tuple[T, ...], Acc], b: Dict[Tuple[T, ...], Acc]) -> Dict[Tuple[T, ...], Acc]:
    """Merge two path->acc maps, combining accumulators for identical paths."""
    if not a:
        return dict(b)
    if not b:
        return dict(a)
    out = dict(a)
    for path, acc in b.items():
        prev = out.get(path)
        out[path] = acc if prev is None else prev.merge(acc)  # type: ignore[union-attr]
    return out


def _encode_for_sort(obj: Any) -> str:
    """
    Deterministic encoding for sorting mixed JSON-serializable values.
    Falls back to repr() if json.dumps fails.
    """
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


# ------------------------------
# Public LeveledGSS implementation (API unchanged)
# ------------------------------

@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        """
        Build a LeveledGSS from explicit stacks.
        - Merge accumulators for identical stacks.
        - Store a compact map encoding internally for O(1) encode/decode.
        """
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            prev = merged.get(key)
            merged[key] = acc if prev is None else prev.merge(acc)
        lower_root = _map_to_lower(merged)
        return LeveledGSS(inner=Upper(Interface(node=lower_root, acc=None)), empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Decode to a canonical, deterministic list of (stack, acc) pairs:
        - One entry per distinct stack (accumulators merged).
        - Sorted for stability across runs.
        """
        iface = self.inner.inner  # type: ignore[union-attr]
        assert isinstance(iface, Interface), "LeveledGSS must hold an Interface at the top."
        paths = _lower_to_map(iface.node)
        items = [(list(k), v) for k, v in paths.items()]
        items.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return items

    def _get_lower(self) -> Lower[T]:
        iface = self.inner.inner  # type: ignore[union-attr]
        assert isinstance(iface, Interface), "LeveledGSS must hold an Interface at the top."
        return iface.node

    def _with_lower(self, lower: Lower[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(inner=Upper(Interface(node=lower, acc=None)), empty=None)

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self  # No stacks to push onto; remains empty.
        pushed = {p + (value,): acc for p, acc in paths.items()}
        return self._with_lower(_map_to_lower(pushed))

    def pop(self) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self  # Already empty.
        if not any(p for p in paths.keys()):
            return self  # Only empty stacks -> pop discards them -> empty; return self (still empty).
        popped: Dict[Tuple[T, ...], Acc] = {}
        for p, acc in paths.items():
            if p:
                key = p[:-1]
                prev = popped.get(key)
                popped[key] = acc if prev is None else prev.merge(acc)
        return self._with_lower(_map_to_lower(popped))

    def is_empty(self) -> bool:
        return not bool(_lower_to_map(self._get_lower()))

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self
        if value is None:
            filtered = {(): acc for p, acc in paths.items() if p == ()}
        else:
            filtered = {p: acc for p, acc in paths.items() if p and p[-1] == value}
        if len(filtered) == len(paths):
            return self
        return self._with_lower(_map_to_lower(filtered))

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self
        transformed: Dict[Tuple[T, ...], Acc] = {}
        unchanged = True
        for p, acc in paths.items():
            new_acc = func(acc)
            transformed[p] = new_acc
            if unchanged and new_acc != acc:
                unchanged = False
        if unchanged:
            return self
        return self._with_lower(_map_to_lower(transformed))

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self
        kept = {p: acc for p, acc in paths.items() if predicate(acc)}
        if len(kept) == len(paths):
            return self
        return self._with_lower(_map_to_lower(kept))

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        map_a = _lower_to_map(self._get_lower())
        map_b = _lower_to_map(other._get_lower())
        if not map_a:
            return other
        if not map_b:
            return self
        merged = _map_merge_paths(map_a, map_b)
        return self._with_lower(_map_to_lower(merged))

    def peek(self) -> Set[T]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return set()
        return {p[-1] for p in paths.keys() if p}

    def reduce_acc(self) -> Optional[Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return None
        total: Optional[Acc] = None
        for acc in paths.values():
            total = _merge_acc(total, acc)
        return total


# ------------------------------
# Invariant validation (public API unchanged)
# ------------------------------

def _validate_upper(node: Upper[T, Acc]):
    """
    Recursively walk Upper nodes (no-op for Interfaces).
    This implementation always uses a single Interface at the top.
    """
    if isinstance(node.inner, UpperBranch):
        for children_by_val in node.inner.children.values():
            for child in children_by_val.values():
                _validate_upper(child)
    # If Interface: nothing to check.


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Checks internal invariants of the LeveledGSS structure.
    Raises AssertionError if an invariant is violated.
    """
    _validate_upper(gss.inner)

    # Invariant from original: If inner is an Interface and empty exists, their accs must differ.
    if isinstance(gss.inner.inner, Interface) and gss.empty is not None:
        if gss.inner.inner.acc == gss.empty:
            raise AssertionError(
                "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
            )
