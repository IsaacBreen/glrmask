from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes (unchanged signatures)
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
    acc: Acc | None  # We store None for the top-level interface acc as placeholder.


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
# Sentinel used inside Lower to store accumulator at a node (as a special child key)
# ------------------------------

class _AccKey(Generic[Acc]):
    __slots__ = ("acc",)

    def __init__(self, acc: Acc):
        self.acc = acc

    def __repr__(self) -> str:
        return f"<AccKey:{self.acc!r}>"

    # Each instance is unique; equality by identity.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


def _is_acc_key(obj: Any) -> bool:
    return isinstance(obj, _AccKey)


def _merge_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


# ------------------------------
# Canonical helpers: Lower <-> path map
# ------------------------------

def _make_empty_lower() -> Lower[T]:
    return Lower(LowerBranch(children={}))


def _lower_to_map(node: Lower[T]) -> Dict[Tuple[T, ...], Acc]:
    """
    Decode a canonical Lower-trie into a map from stack paths (bottom->top) to accumulators.
    Canonical form:
      - Each label edge is at index 0.
      - Accumulator at a node is encoded as a child whose label is an _AccKey(acc) and points to Leaf at index 0.
    """
    result: Dict[Tuple[T, ...], Acc] = {}

    def visit(cur: Lower[T], prefix: List[T]) -> None:
        if isinstance(cur.inner, Leaf):
            return
        children = cur.inner.children

        # Accumulator at this node (at most one in canonical form)
        for label, _ in children.items():
            if _is_acc_key(label):
                result[tuple(prefix)] = label.acc  # type: ignore[attr-defined]
                break

        # Traverse label-children
        for label, idx_map in children.items():
            if _is_acc_key(label):
                continue
            child = idx_map.get(0)
            if child is not None:
                prefix.append(label)  # type: ignore[arg-type]
                visit(child, prefix)
                prefix.pop()

    visit(node, [])
    return result


def _map_merge_paths(a: Dict[Tuple[T, ...], Acc], b: Dict[Tuple[T, ...], Acc]) -> Dict[Tuple[T, ...], Acc]:
    """Merge two path->acc maps, combining accumulators for identical paths."""
    if not a:
        return dict(b)
    if not b:
        return dict(a)
    out = dict(a)
    for path, acc in b.items():
        if path in out:
            out[path] = out[path].merge(acc)  # type: ignore[union-attr]
        else:
            out[path] = acc
    return out


def _map_to_lower(paths: Dict[Tuple[T, ...], Acc]) -> Lower[T]:
    """
    Build a canonical Lower-trie from a path->acc map.
    Canonicalization rules:
      - For each node with an accumulator, add a child labeled with _AccKey(acc) -> Leaf at index 0.
      - For each label edge, use index 0 to the child.
      - Omit empty subtrees.
    """
    if not paths:
        return _make_empty_lower()

    ACC = object()  # sentinel for per-node accumulator in the temporary dict-trie

    # Build a nested dict trie
    root: Dict[Any, Dict] = {}
    for path, acc in paths.items():
        node = root
        for label in path:
            node = node.setdefault(label, {})
        prev = node.get(ACC)
        node[ACC] = acc if prev is None else prev.merge(acc)  # type: ignore[union-attr]

    # Convert nested dict trie into immutable Lower nodes
    def build(node_dict: Dict[Any, Dict]) -> Lower[T]:
        labels: Dict[T, Dict] = {}
        root_acc: Optional[Acc] = None

        for key, sub in node_dict.items():
            if key is ACC:
                root_acc = sub  # type: ignore[assignment]
            else:
                labels[key] = sub  # type: ignore[index]

        lb_children: Dict[Any, Dict[int, Lower[T]]] = {}
        for label, sub in labels.items():
            child = build(sub)
            # Skip truly empty branches
            if isinstance(child.inner, LowerBranch) and not child.inner.children:
                continue
            lb_children[label] = {0: child}

        if root_acc is not None:
            lb_children[_AccKey(root_acc)] = {0: Lower(Leaf())}  # type: ignore[arg-type]

        return Lower(LowerBranch(children=lb_children))

    return build(root)


def _encode_for_sort(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
    except Exception:
        return repr(obj)


# ------------------------------
# Public LeveledGSS implementation
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
        - Encode as a canonical trie (Lower) with _AccKey at nodes containing an accumulator.
        """
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc
        lower_root = _map_to_lower(merged)
        return LeveledGSS(inner=Upper(Interface(node=lower_root, acc=None)), empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """Decode the trie into a canonical, sorted list of (stack, acc) pairs."""
        iface = self.inner.inner  # type: ignore[union-attr]
        assert isinstance(iface, Interface), "LeveledGSS is expected to hold an Interface at the top."
        paths = _lower_to_map(iface.node)
        items = [(list(k), v) for k, v in paths.items()]
        items.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return items

    def _get_lower(self) -> Lower[T]:
        iface = self.inner.inner  # type: ignore[union-attr]
        assert isinstance(iface, Interface), "LeveledGSS is expected to hold an Interface at the top."
        return iface.node

    def _with_lower(self, lower: Lower[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(inner=Upper(Interface(node=lower, acc=None)), empty=None)

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self._with_lower(_make_empty_lower())
        pushed: Dict[Tuple[T, ...], Acc] = {}
        for p, acc in paths.items():
            new_path = p + (value,)
            if new_path in pushed:
                pushed[new_path] = pushed[new_path].merge(acc)
            else:
                pushed[new_path] = acc
        return self._with_lower(_map_to_lower(pushed))

    def pop(self) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self._with_lower(_make_empty_lower())
        popped: Dict[Tuple[T, ...], Acc] = {}
        for p, acc in paths.items():
            if p:  # only non-empty stacks can be popped
                new_path = p[:-1]
                if new_path in popped:
                    popped[new_path] = popped[new_path].merge(acc)
                else:
                    popped[new_path] = acc
        return self._with_lower(_map_to_lower(popped))

    def is_empty(self) -> bool:
        node = self._get_lower()
        if isinstance(node.inner, LowerBranch):
            return not node.inner.children
        return True

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self._with_lower(_make_empty_lower())
        if value is None:
            filtered = {(): acc for p, acc in paths.items() if p == ()}
        else:
            filtered = {p: acc for p, acc in paths.items() if p and p[-1] == value}
        return self._with_lower(_map_to_lower(filtered))

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self._with_lower(_make_empty_lower())
        transformed = {p: func(acc) for p, acc in paths.items()}
        return self._with_lower(_map_to_lower(transformed))

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        paths = _lower_to_map(self._get_lower())
        if not paths:
            return self._with_lower(_make_empty_lower())
        kept = {p: acc for p, acc in paths.items() if predicate(acc)}
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
# Invariant validation (minimal, unchanged public API)
# ------------------------------

def _validate_upper(node: Upper[T, Acc]):
    """
    Recursively walk Upper nodes. This implementation always uses a single Interface at the top.
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
