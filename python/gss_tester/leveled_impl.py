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
    acc: Acc | None  # Internally used to hold _MapBox; see below.


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
# Keep a canonical mapping from full stack (bottom -> top) to its accumulator:
#   Dict[Tuple[T, ...], Acc]
# We stash this map inside Interface.acc via a tiny private box to avoid changing public types.

class _MapBox(Generic[T, Acc]):
    __slots__ = ("paths",)

    def __init__(self, paths: Dict[Tuple[T, ...], Acc]):
        self.paths = paths  # Treat as immutable by convention.

    def __repr__(self) -> str:
        return f"<MapBox:{len(self.paths)}>"


_EMPTY_LOWER: Lower[Any] = Lower(Leaf())


def _encode_for_sort(obj: Any) -> str:
    """
    Deterministic string key for sorting heterogeneous (possibly non-JSON) values.
    Falls back to repr for unknown types.
    """
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A minimal, fast GSS built on a single canonical map:
      paths: Dict[Tuple[T, ...], Acc]

    All operations return new LeveledGSS instances; duplicates are merged with Acc.merge.
    The public node types above are preserved, but only used as a thin wrapper.
    """
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    # -- Construction / Serialization --

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        merged: Dict[Tuple[T, ...], Acc] = {}
        for values, acc in stacks:
            key = tuple(values)
            prev = merged.get(key)
            merged[key] = acc if prev is None else prev.merge(acc)
        return LeveledGSS(Upper(Interface(_EMPTY_LOWER, _MapBox(merged))), None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        items = [(list(k), v) for k, v in self._paths().items()]
        items.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return items

    # -- Core operations --

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        pushed = {p + (value,): acc for p, acc in self._paths().items()}
        return self._with_paths(pushed)

    def pop(self) -> LeveledGSS[T, Acc]:
        popped: Dict[Tuple[T, ...], Acc] = {}
        for p, acc in self._paths().items():
            if p:  # discard empty stacks
                q = p[:-1]
                prev = popped.get(q)
                popped[q] = acc if prev is None else prev.merge(acc)
        return self._with_paths(popped)

    def is_empty(self) -> bool:
        return not self._paths()

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if value is None:
            filtered = {(): acc for p, acc in paths.items() if not p}
        else:
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

    # -- Internal helpers --

    def _paths(self) -> Dict[Tuple[T, ...], Acc]:
        """
        Extract the canonical map from the Interface.acc carrier.
        Returns {} if structure is missing (shouldn't happen in normal use).
        """
        inner = self.inner.inner
        if isinstance(inner, Interface) and isinstance(inner.acc, _MapBox):
            return inner.acc.paths
        return {}

    def _with_paths(self, paths: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(Upper(Interface(_EMPTY_LOWER, _MapBox(paths))), None)


# Optional invariant checker for debugging.
def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    inner = gss.inner.inner
    if not isinstance(inner, Interface):
        raise AssertionError("Top-level Upper must wrap an Interface.")
    if not isinstance(inner.acc, _MapBox):
        raise AssertionError("Interface.acc must carry the internal _MapBox.")
