from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any

from .interface import GSS, T, Acc


# Public node classes (unchanged API). Do not modify their fields or add new ones.

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    inner: 'UpperBranch[T, Acc]' | 'Interface[T, Acc]'


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, 'Upper[T, Acc]']]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: 'Lower[T]'
    acc: Acc | None  # Used internally to carry a private state box (see _MapBox below).


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    inner: 'LowerBranch[T]' | 'Leaf'


@dataclass(frozen=True, eq=True)
class LowerBranch(Generic[T]):
    children: Dict[Any, Dict[int, 'Lower[T]']]


@dataclass(frozen=True, eq=True)
class Leaf:
    pass


# Internal representation:
# We store the entire GSS state as a canonical mapping: Dict[Tuple[T, ...], Acc]
# Each key is the full stack (bottom -> top), and the value is its accumulator.
# To keep public types unchanged, we tuck that map into Interface.acc using a tiny private box.

class _MapBox(Generic[T, Acc]):
    __slots__ = ("paths",)

    def __init__(self, paths: Dict[Tuple[T, ...], Acc]):
        self.paths = paths  # Treat as immutable: never mutate in place.

    def __repr__(self) -> str:
        return f"<MapBox:{len(self.paths)}>"


# A shared inert Lower node to satisfy the public shape.
_EMPTY_LOWER: Lower[Any] = Lower(Leaf())


def _encode_for_sort(obj: Any) -> str:
    """
    Deterministic string key for sorting heterogeneous, possibly non-JSON values.
    Falls back to repr for objects json can't serialize.
    """
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A minimal and fast GSS implementation based on a canonical map:
      paths: Dict[Tuple[T, ...], Acc]
    All operations are pure and return fresh LeveledGSS instances.
    Duplicate stacks are always merged using Acc.merge.
    """
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    # --- Construction / Serialization ---

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
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

    # --- Core operations ---

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        pushed = {p + (value,): acc for p, acc in paths.items()}
        return self._with_paths(pushed)

    def pop(self) -> LeveledGSS[T, Acc]:
        popped: Dict[Tuple[T, ...], Acc] = {}
        for p, acc in self._paths().items():
            if p:  # Ignore empty stacks
                q = p[:-1]
                prev = popped.get(q)
                popped[q] = acc if prev is None else prev.merge(acc)
        return self._with_paths(popped)

    def is_empty(self) -> bool:
        return not bool(self._paths())

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if value is None:
            # Keep only empty stacks
            filtered = {(): acc for p, acc in paths.items() if p == ()}
        else:
            # Keep stacks whose top equals `value`
            filtered = {p: acc for p, acc in paths.items() if p and p[-1] == value}
        return self._with_paths(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        transformed = {p: func(acc) for p, acc in self._paths().items()}
        return self._with_paths(transformed)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        kept = {p: acc for p, acc in self._paths().items() if predicate(acc)}
        return self._with_paths(kept)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        merged: Dict[Tuple[T, ...], Acc] = dict(self._paths())
        for p, acc in other._paths().items():
            prev = merged.get(p)
            merged[p] = acc if prev is None else prev.merge(acc)
        return self._with_paths(merged)

    def peek(self) -> Set[T]:
        return {p[-1] for p in self._paths() if p}

    def reduce_acc(self) -> Optional[Acc]:
        it = iter(self._paths().values())
        try:
            total = next(it)
        except StopIteration:
            return None
        for acc in it:
            total = total.merge(acc)
        return total

    # --- Internal utilities ---

    def _paths(self) -> Dict[Tuple[T, ...], Acc]:
        inner = self.inner.inner
        # The top-level must be Upper(Interface(...)), and the map is carried in Interface.acc via _MapBox.
        if isinstance(inner, Interface) and isinstance(inner.acc, _MapBox):
            return inner.acc.paths
        return {}

    def _with_paths(self, paths: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(inner=Upper(Interface(node=_EMPTY_LOWER, acc=_MapBox(paths))), empty=None)


# Optional invariant validation helper (kept minimal).
def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Validates minimal structural invariants:
    - Top-level Upper must wrap an Interface.
    - The Interface.acc stores a _MapBox.
    """
    inner = gss.inner.inner
    if not isinstance(inner, Interface):
        raise AssertionError("Invariant violated: Top-level Upper must wrap an Interface.")
    if not isinstance(inner.acc, _MapBox):
        raise AssertionError("Invariant violated: Interface.acc must store the internal _MapBox.")
