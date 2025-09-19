from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any

from .interface import GSS, T, Acc


# Public node classes (unchanged API). These are kept exactly as-is to preserve
# the external typing/shape expected by other components.

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


# Internal representation:
# We model the entire GSS as a single immutable mapping:
#   Dict[Tuple[T, ...], Acc]
# where each key is the full stack (bottom->top), and the value is the
# accumulator for that stack. This is the canonical and minimal form for our
# operations. To keep the public dataclass fields untouched, we store this map
# inside Interface.acc using a tiny private box.

class _MapBox(Generic[T, Acc]):
    __slots__ = ("paths",)

    def __init__(self, paths: Dict[Tuple[T, ...], Acc]):
        # Treat as immutable (never mutate in-place).
        self.paths = paths

    def __repr__(self) -> str:
        return f"<MapBox:{len(self.paths)}>"


# A shared, inert Lower node. We never traverse it; it exists to satisfy the
# public type shape.
_EMPTY_LOWER: Lower[Any] = Lower(Leaf())


def _encode_for_sort(obj: Any) -> str:
    """
    Create a deterministic string key suitable for sorting heterogeneous values.
    Falls back to repr for objects json can't serialize.
    """
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    Minimal, fast GSS implementation:
    - The entire state is a canonical map: stack -> accumulator.
    - All operations are pure and return fresh LeveledGSS instances.
    - Duplicate stacks are always merged by combining accumulators via Acc.merge.
    """
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    # Construction / Serialization

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

    # Core operations

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
            if p:
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
        if not a and not b:
            return self._with_paths({})
        if not a:
            # Return fresh instance to keep construction semantics uniform.
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
        it = iter(paths.values())
        try:
            total = next(it)
        except StopIteration:
            return None
        for acc in it:
            total = total.merge(acc)
        return total

    # Internal utilities

    def _iface(self) -> Interface[T, Acc]:
        inner = self.inner.inner
        # We always store the map in the top-level Interface.acc.
        assert isinstance(inner, Interface), "Top-level node must be an Interface."
        return inner

    def _paths(self) -> Dict[Tuple[T, ...], Acc]:
        acc = self._iface().acc
        return acc.paths if isinstance(acc, _MapBox) else {}

    def _with_paths(self, paths: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
        # Always construct a fresh instance for clarity and determinism.
        return LeveledGSS(inner=Upper(Interface(node=_EMPTY_LOWER, acc=_MapBox(paths))), empty=None)


# Invariant validation (kept minimal and fast)
def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Minimal structural check to respect the original public API:
    - Top-level must be Upper(Interface(...)).
    - Keep the historical check that empty (if set) does not equal the Interface.acc.
    """
    inner = gss.inner.inner
    if not isinstance(inner, Interface):
        raise AssertionError("Invariant violated: Top-level Upper must wrap an Interface.")
    if gss.empty is not None and inner.acc == gss.empty:
        raise AssertionError(
            "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
        )
