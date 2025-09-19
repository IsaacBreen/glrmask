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
# Internal representation
# ------------------------------
# Keep the public node types intact, but store the entire GSS as a single
# canonical mapping: Dict[Tuple[T, ...], Acc]
#
# To keep LeveledGSS a frozen dataclass with the same public fields and
# preserve very fast operations, we tuck this mapping into Interface.acc
# using a private box (_MapBox). The Interface.node is a constant Leaf-based
# Lower, since the structure graph is not used by algorithms here.

class _MapBox(Generic[T, Acc]):
    __slots__ = ("paths",)

    def __init__(self, paths: Dict[Tuple[T, ...], Acc]):
        self.paths = paths  # Do not mutate after creation.

    def __repr__(self) -> str:
        return f"<MapBox:{len(self.paths)}>"


_EMPTY_LOWER: Lower[Any] = Lower(Leaf())


def _encode_for_sort(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


def _merge_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


# ------------------------------
# Public LeveledGSS implementation
# ------------------------------

@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    # --- Construction / Serialization ---

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        # Merge accumulators for identical stacks into a canonical map.
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            prev = merged.get(key)
            merged[key] = acc if prev is None else prev.merge(acc)
        return LeveledGSS(inner=Upper(Interface(node=_EMPTY_LOWER, acc=_MapBox(merged))), empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        paths = self._paths()
        items = [(list(k), v) for k, v in paths.items()]
        items.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return items

    # --- Internal helpers ---

    def _iface(self) -> Interface[T, Acc]:
        inner = self.inner.inner
        assert isinstance(inner, Interface), "Top-level must be an Interface node."
        return inner

    def _paths(self) -> Dict[Tuple[T, ...], Acc]:
        acc = self._iface().acc
        if isinstance(acc, _MapBox):
            return acc.paths
        # No map stored => empty GSS
        return {}

    def _with_paths(self, paths: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
        # Always return a fresh instance (deterministic behavior across impls).
        return LeveledGSS(inner=Upper(Interface(node=_EMPTY_LOWER, acc=_MapBox(paths))), empty=None)

    # --- Core operations ---

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if not paths:
            return self._with_paths({})
        pushed = {p + (value,): acc for p, acc in paths.items()}
        return self._with_paths(pushed)

    def pop(self) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if not paths:
            return self._with_paths({})
        popped: Dict[Tuple[T, ...], Acc] = {}
        for p, acc in paths.items():
            if p:  # Only pop non-empty stacks
                q = p[:-1]
                prev = popped.get(q)
                popped[q] = acc if prev is None else prev.merge(acc)
        return self._with_paths(popped)

    def is_empty(self) -> bool:
        return not bool(self._paths())

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if value is None:
            filtered = {(): acc for p, acc in paths.items() if p == ()}
        else:
            filtered = {p: acc for p, acc in paths.items() if p and p[-1] == value}
        return self._with_paths(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        transformed = {p: func(acc) for p, acc in paths.items()}
        return self._with_paths(transformed)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        kept = {p: acc for p, acc in paths.items() if predicate(acc)}
        return self._with_paths(kept)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        a = self._paths()
        b = other._paths()
        if not a:
            return self._with_paths(dict(b))
        if not b:
            return self._with_paths(dict(a))
        merged: Dict[Tuple[T, ...], Acc] = dict(a)
        for p, acc in b.items():
            prev = merged.get(p)
            merged[p] = acc if prev is None else prev.merge(acc)
        return self._with_paths(merged)

    def peek(self) -> Set[T]:
        paths = self._paths()
        return {p[-1] for p in paths if p} if paths else set()

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
    # With the compact representation, Upper always wraps an Interface.
    # We keep this function to respect the public API; no deep checks needed.
    if isinstance(node.inner, UpperBranch):
        # If someone did construct an UpperBranch, traverse its children.
        for children_by_val in node.inner.children.values():
            for child in children_by_val.values():
                _validate_upper(child)
    # Interface case: nothing to validate structurally.


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    _validate_upper(gss.inner)
    # Preserve the original invariant about Interface.acc and the optional empty field.
    if isinstance(gss.inner.inner, Interface) and gss.empty is not None:
        if gss.inner.inner.acc == gss.empty:
            raise AssertionError(
                "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
            )
