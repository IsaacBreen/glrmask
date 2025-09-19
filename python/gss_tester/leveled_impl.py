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
# Minimal internal representation
# ------------------------------
# We store the entire mapping {tuple[T, ...] -> Acc} in the root Lower node.
# To keep the public types unchanged and the structure compact, we encode the
# map inside a single-entry LowerBranch whose only key is a private _MapBox
# object holding the mapping. Values are unused placeholders.

class _MapBox(Generic[T, Acc]):
    __slots__ = ("paths",)

    def __init__(self, paths: Dict[Tuple[T, ...], Acc]):
        # Treat as immutable externally; operations always create fresh dicts.
        self.paths = paths

    def __repr__(self) -> str:
        return f"<MapBox:{len(self.paths)} path(s)>"


def _empty_lower() -> Lower[T]:
    return Lower(Leaf())


def _lower_to_map(node: Lower[T]) -> Dict[Tuple[T, ...], Acc]:
    """
    Decode a Lower node into its path->acc map.

    Representation:
    - Empty: Lower(Leaf()) -> {}
    - Non-empty: Lower(LowerBranch(children={_MapBox(paths): {}}))
    Any deviation returns {} (shouldn't occur).
    """
    inner = node.inner
    if isinstance(inner, Leaf):
        return {}
    if not inner.children or len(inner.children) != 1:
        return {}
    (only_key, _) = next(iter(inner.children.items()))
    if isinstance(only_key, _MapBox):
        # Return reference; callers should not mutate.
        return only_key.paths
    return {}


def _map_to_lower(paths: Dict[Tuple[T, ...], Acc]) -> Lower[T]:
    if not paths:
        return _empty_lower()
    return Lower(LowerBranch(children={_MapBox(paths): {}}))


def _merge_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


def _merge_path_maps(a: Dict[Tuple[T, ...], Acc], b: Dict[Tuple[T, ...], Acc]) -> Dict[Tuple[T, ...], Acc]:
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

    # --- Construction / Serialization ---

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        """
        Build a LeveledGSS from explicit stacks.
        - Merge accumulators for identical stacks.
        - Store a compact map encoding at the root Lower node.
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
        iface = self._iface()
        paths = _lower_to_map(iface.node)
        items = [(list(k), v) for k, v in paths.items()]
        items.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return items

    # --- Internal helpers ---

    def _iface(self) -> Interface[T, Acc]:
        inner = self.inner.inner
        assert isinstance(inner, Interface), "LeveledGSS must hold an Interface at the top."
        return inner

    def _paths(self) -> Dict[Tuple[T, ...], Acc]:
        return _lower_to_map(self._iface().node)

    def _with_paths(self, paths: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(inner=Upper(Interface(node=_map_to_lower(paths), acc=None)), empty=None)

    # --- Core operations ---

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if not paths:
            return self  # No stacks to push onto; remains empty.
        pushed = {p + (value,): acc for p, acc in paths.items()}
        return self._with_paths(pushed)

    def pop(self) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if not paths:
            return self  # Already empty.
        popped: Dict[Tuple[T, ...], Acc] = {}
        for p, acc in paths.items():
            if p:
                key = p[:-1]
                prev = popped.get(key)
                popped[key] = acc if prev is None else prev.merge(acc)
        # If all stacks were empty, this becomes an empty GSS.
        return self._with_paths(popped)

    def is_empty(self) -> bool:
        return not bool(self._paths())

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if not paths:
            return self
        if value is None:
            filtered = {(): acc for p, acc in paths.items() if p == ()}
        else:
            filtered = {p: acc for p, acc in paths.items() if p and p[-1] == value}
        if len(filtered) == len(paths):
            return self
        return self._with_paths(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
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
        return self._with_paths(transformed)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if not paths:
            return self
        kept = {p: acc for p, acc in paths.items() if predicate(acc)}
        if len(kept) == len(paths):
            return self
        return self._with_paths(kept)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        map_a = self._paths()
        map_b = other._paths()
        if not map_a:
            return other
        if not map_b:
            return self
        merged = _merge_path_maps(map_a, map_b)
        return self._with_paths(merged)

    def peek(self) -> Set[T]:
        paths = self._paths()
        if not paths:
            return set()
        return {p[-1] for p in paths.keys() if p}

    def reduce_acc(self) -> Optional[Acc]:
        paths = self._paths()
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
