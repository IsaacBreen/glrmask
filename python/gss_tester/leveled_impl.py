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
# Compact internal representation
# ------------------------------
# We represent the entire GSS as a single mapping:
#   paths: Dict[Tuple[T, ...], Acc]
# where each key is a full stack (bottom->top) and the value is the merged
# accumulator for that exact stack.
#
# To keep the public node types intact yet store this compact mapping, the
# root Lower node holds a single-entry LowerBranch whose only key is a private
# _MapBox(paths) object. Values are unused placeholders.
#
# All operations are simple transformations of this mapping with appropriate
# accumulator merging, yielding a new LeveledGSS instance.

class _MapBox(Generic[T, Acc]):
    __slots__ = ("paths",)

    def __init__(self, paths: Dict[Tuple[T, ...], Acc]):
        self.paths = paths  # Consumers must not mutate.

    def __repr__(self) -> str:
        return f"<MapBox:{len(self.paths)}>"


def _map_to_lower(paths: Dict[Tuple[T, ...], Acc]) -> Lower[T]:
    if not paths:
        return Lower(Leaf())
    return Lower(LowerBranch(children={_MapBox(paths): {}}))


def _lower_to_map(node: Lower[T]) -> Dict[Tuple[T, ...], Acc]:
    inner = node.inner
    if isinstance(inner, Leaf):
        return {}
    if len(inner.children) != 1:
        return {}
    (only_key, _placeholder) = next(iter(inner.children.items()))
    return only_key.paths if isinstance(only_key, _MapBox) else {}


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
        # Merge accumulators for identical stacks into a canonical path->acc map.
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            prev = merged.get(key)
            merged[key] = acc if prev is None else prev.merge(acc)
        return LeveledGSS(inner=Upper(Interface(node=_map_to_lower(merged), acc=None)), empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        # Deterministic, canonical list of (stack, acc) pairs.
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
        return _lower_to_map(self._iface().node)

    def _with_paths(self, paths: Dict[Tuple[T, ...], Acc]) -> LeveledGSS[T, Acc]:
        # Keep the public shape intact; Interface.acc is unused placeholder.
        return LeveledGSS(inner=Upper(Interface(node=_map_to_lower(paths), acc=None)), empty=None)

    # --- Core operations ---
    # Important behavioral note:
    # To keep fuzzing sequences deterministic across different implementations,
    # we return a fresh LeveledGSS instance even when the logical result is
    # unchanged. This mirrors ReferenceGSS which constructs a new instance for
    # every operation.

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if not paths:
            # Return a fresh empty instance to mirror ReferenceGSS behavior.
            return self._with_paths({})
        pushed = {p + (value,): acc for p, acc in paths.items()}
        return self._with_paths(pushed)

    def pop(self) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        if not paths:
            # Return a fresh empty instance to mirror ReferenceGSS behavior.
            return self._with_paths({})
        popped: Dict[Tuple[T, ...], Acc] = {}
        for p, acc in paths.items():
            if p:  # Ignore empty stacks (cannot pop).
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
        transformed: Dict[Tuple[T, ...], Acc] = {p: func(acc) for p, acc in paths.items()}
        return self._with_paths(transformed)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        paths = self._paths()
        kept = {p: acc for p, acc in paths.items() if predicate(acc)}
        return self._with_paths(kept)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        a = self._paths()
        b = other._paths()
        merged: Dict[Tuple[T, ...], Acc] = dict(a)
        for p, acc in b.items():
            prev = merged.get(p)
            merged[p] = acc if prev is None else prev.merge(acc)
        return self._with_paths(merged)

    def peek(self) -> Set[T]:
        paths = self._paths()
        return {p[-1] for p in paths.keys() if p} if paths else set()

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
    # In this compact representation, Upper always contains a single Interface.
    # Keep traversal minimal to respect the original public API.
    if isinstance(node.inner, UpperBranch):
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
