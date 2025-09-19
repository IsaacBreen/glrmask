from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Iterable

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
# Canonical helpers
# ------------------------------

def _is_empty_lower(node: Lower[T]) -> bool:
    """True iff this Lower represents no stacks at all."""
    return isinstance(node.inner, LowerBranch) and not node.inner.children


def _split(node: Lower[T]) -> Tuple[Dict[T, Lower[T]], Optional[Acc]]:
    """
    Split a canonical Lower node into (children_by_label, root_acc).
    Assumes canonical form:
      - For each label, there is at most index 0 in the inner map.
      - At most one _AccKey (with index 0) exists to encode the node's accumulator.
    """
    if isinstance(node.inner, Leaf):
        return {}, None

    labels: Dict[T, Lower[T]] = {}
    root_acc: Optional[Acc] = None

    for key, idx_map in node.inner.children.items():
        # Canonical: only index 0 used; fallback to any present for robustness.
        child = idx_map.get(0) or (next(iter(idx_map.values())) if idx_map else None)

        if _is_acc_key(key):
            if child is not None:
                # Merge defensively in case multiple sentinels appear (won't happen in canonical form).
                root_acc = _merge_acc(root_acc, key.acc)  # type: ignore[attr-defined]
        else:
            if child is not None and not _is_empty_lower(child):
                labels[key] = child  # type: ignore[assignment]

    return labels, root_acc


def _make_node(labels: Dict[T, Lower[T]], root_acc: Optional[Acc]) -> Lower[T]:
    """Create a canonical Lower node from child labels and an optional root accumulator."""
    if not labels and root_acc is None:
        return Lower(LowerBranch(children={}))

    children: Dict[Any, Dict[int, Lower[T]]] = {}
    for label, child in labels.items():
        if not _is_empty_lower(child):
            children[label] = {0: child}
    if root_acc is not None:
        children[_AccKey(root_acc)] = {0: Lower(Leaf())}
    return Lower(LowerBranch(children=children))


def _add_root_acc(node: Lower[T], acc: Optional[Acc]) -> Lower[T]:
    if acc is None:
        return node
    if isinstance(node.inner, Leaf):
        return _make_node({}, acc)
    labels, existing = _split(node)
    return _make_node(labels, _merge_acc(existing, acc))


def _remove_root_acc(node: Lower[T]) -> Tuple[Lower[T], Optional[Acc]]:
    if isinstance(node.inner, Leaf):
        return node, None
    labels, root_acc = _split(node)
    return _make_node(labels, None), root_acc


# ------------------------------
# Core trie operations (simple and canonical)
# ------------------------------

def _merge_lower(a: Lower[T], b: Lower[T]) -> Lower[T]:
    """Union of tries; accumulators on identical stacks are merged."""
    if _is_empty_lower(a):
        return b
    if _is_empty_lower(b):
        return a

    a_labels, a_acc = _split(a)
    b_labels, b_acc = _split(b)

    merged_labels: Dict[T, Lower[T]] = {}
    for key in set(a_labels) | set(b_labels):
        ca = a_labels.get(key)
        cb = b_labels.get(key)
        if ca is None:
            merged_labels[key] = cb  # type: ignore[assignment]
        elif cb is None:
            merged_labels[key] = ca
        else:
            merged_labels[key] = _merge_lower(ca, cb)

    return _make_node(merged_labels, _merge_acc(a_acc, b_acc))


def _push_lower(node: Lower[T], value: T) -> Lower[T]:
    """Push `value` on all stacks encoded by this trie."""
    if isinstance(node.inner, Leaf) or _is_empty_lower(node):
        return node

    labels, root_acc = _split(node)
    new_labels: Dict[T, Lower[T]] = {}

    # Recurse into children
    for label, child in labels.items():
        pushed_child = _push_lower(child, value)
        if not _is_empty_lower(pushed_child):
            new_labels[label] = pushed_child

    # Move this node's accumulator under the `value` child
    if root_acc is not None:
        if value in new_labels:
            new_labels[value] = _add_root_acc(new_labels[value], root_acc)
        else:
            new_labels[value] = _make_node({}, root_acc)

    return _make_node(new_labels, None)


def _pop_lower(node: Lower[T]) -> Lower[T]:
    """Pop one element from all non-empty stacks in this trie."""
    if isinstance(node.inner, Leaf) or _is_empty_lower(node):
        return node

    labels, _ = _split(node)  # root_acc ignored: empty stacks can't be popped

    carry_up: Optional[Acc] = None
    new_labels: Dict[T, Lower[T]] = {}

    for label, child in labels.items():
        child_wo_acc, acc_at_child = _remove_root_acc(child)
        carry_up = _merge_acc(carry_up, acc_at_child)

        popped_child = _pop_lower(child_wo_acc)
        if not _is_empty_lower(popped_child):
            new_labels[label] = popped_child

    return _make_node(new_labels, carry_up)


def _isolate_lower_by_top(
    node: Lower[T],
    target: Optional[T],
    incoming_label: Optional[T] = None
) -> Optional[Lower[T]]:
    """
    Keep only stacks whose top element equals `target` (or empty stacks if target is None).
    Returns None if the resulting subtree contains no stacks.
    """
    if isinstance(node.inner, Leaf) or _is_empty_lower(node):
        return None

    labels, root_acc = _split(node)

    keep_root = (incoming_label is None and target is None) or (incoming_label is not None and incoming_label == target)
    kept_root = root_acc if keep_root else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        filtered = _isolate_lower_by_top(child, target, incoming_label=label)
        if filtered is not None and not _is_empty_lower(filtered):
            new_labels[label] = filtered

    if not new_labels and kept_root is None:
        return None
    return _make_node(new_labels, kept_root)


def _apply_lower(node: Lower[T], func: Callable[[Acc], Acc]) -> Lower[T]:
    """Apply a transformation to every accumulator in the trie."""
    if isinstance(node.inner, Leaf) or _is_empty_lower(node):
        return node

    labels, root_acc = _split(node)
    transformed_root = func(root_acc) if root_acc is not None else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        t_child = _apply_lower(child, func)
        if not _is_empty_lower(t_child):
            new_labels[label] = t_child

    return _make_node(new_labels, transformed_root)


def _prune_lower(node: Lower[T], predicate: Callable[[Acc], bool]) -> Optional[Lower[T]]:
    """Remove stacks whose accumulator does not satisfy `predicate`."""
    if isinstance(node.inner, Leaf) or _is_empty_lower(node):
        return None

    labels, root_acc = _split(node)
    kept_root = root_acc if (root_acc is not None and predicate(root_acc)) else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        p_child = _prune_lower(child, predicate)
        if p_child is not None and not _is_empty_lower(p_child):
            new_labels[label] = p_child

    if not new_labels and kept_root is None:
        return None
    return _make_node(new_labels, kept_root)


def _has_any_stack(node: Lower[T]) -> bool:
    """True iff there is any accumulator anywhere in this trie."""
    if isinstance(node.inner, Leaf) or _is_empty_lower(node):
        return False
    labels, root_acc = _split(node)
    if root_acc is not None:
        return True
    return any(_has_any_stack(child) for child in labels.values())


def _peek_values(node: Lower[T], incoming_label: Optional[T], out: Set[T]) -> Set[T]:
    """Collect all top-of-stack values for non-empty stacks."""
    if isinstance(node.inner, Leaf) or _is_empty_lower(node):
        return out

    labels, root_acc = _split(node)

    # Root accumulator at this node means stacks with top == incoming_label (if any)
    if root_acc is not None and incoming_label is not None:
        out.add(incoming_label)

    for label, child in labels.items():
        _peek_values(child, label, out)
    return out


def _reduce_all_acc(node: Lower[T]) -> Optional[Acc]:
    """Merge all accumulators in the trie (or None if empty)."""
    if isinstance(node.inner, Leaf) or _is_empty_lower(node):
        return None

    labels, root_acc = _split(node)
    total: Optional[Acc] = root_acc
    for child in labels.values():
        total = _merge_acc(total, _reduce_all_acc(child))
    return total


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
        - Build a bottom->top trie; at each node, a special _AccKey(acc) child encodes the accumulator.
        """
        # Canonicalize and merge duplicate stacks
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        # Build a nested Python dict trie first for simplicity
        acc_key = object()
        trie: Dict[Any, Dict] = {}

        def insert(path: List[T], acc: Acc) -> None:
            node = trie
            for item in path:
                node = node.setdefault(item, {})
            prev = node.get(acc_key)
            node[acc_key] = acc if prev is None else prev.merge(acc)

        for key, acc in merged.items():
            insert(list(key), acc)

        # Convert the nested dict trie into immutable Lower nodes
        def build_lower(node_dict: Dict[Any, Dict]) -> Lower[T]:
            labels: Dict[T, Lower[T]] = {}
            root_acc: Optional[Acc] = None
            for key, sub in node_dict.items():
                if key is acc_key:
                    root_acc = sub  # type: ignore[assignment]
                else:
                    labels[key] = build_lower(sub)
            return _make_node(labels, root_acc)

        lower_root = build_lower(trie)
        return LeveledGSS(inner=Upper(Interface(node=lower_root, acc=None)), empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Decode the trie into a canonical, sorted list of (stack, acc) pairs.
        """
        results: List[Tuple[List[T], Acc]] = []

        def collect(node: Lower[T], prefix: List[T]) -> None:
            if isinstance(node.inner, Leaf) or _is_empty_lower(node):
                return
            labels, root_acc = _split(node)
            if root_acc is not None:
                results.append((list(prefix), root_acc))
            for label, child in labels.items():
                prefix.append(label)
                collect(child, prefix)
                prefix.pop()

        # We always use a single Interface at the top
        iface = self.inner.inner  # type: ignore[union-attr]
        if isinstance(iface, Interface):
            collect(iface.node, [])
        else:
            # Should never happen with this implementation, but keep a safe fallback traversal.
            def collect_from_upper(u: Upper[T, Acc], top_prefix: List[T]) -> None:
                if isinstance(u.inner, Interface):
                    collect(u.inner.node, top_prefix)
                    return
                br: UpperBranch[T, Acc] = u.inner
                for val, idx_map in br.children.items():
                    for child in idx_map.values():
                        top_prefix.append(val)
                        collect_from_upper(child, top_prefix)
                        top_prefix.pop()
            collect_from_upper(self.inner, [])

        # Merge again defensively (though structure should already be canonical), then sort
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in results:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        items = [(list(k), v) for k, v in merged.items()]

        def _encode_for_sort(obj: Any) -> str:
            try:
                return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
            except Exception:
                return repr(obj)

        items.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        return items

    def _get_lower(self) -> Lower[T]:
        iface = self.inner.inner  # type: ignore[union-attr]
        if isinstance(iface, Interface):
            return iface.node
        # Fallback conversion (should not occur)
        def upper_to_lower(u: Upper[T, Acc]) -> Lower[T]:
            if isinstance(u.inner, Interface):
                return u.inner.node
            br: UpperBranch[T, Acc] = u.inner
            lb_children: Dict[Any, Dict[int, Lower[T]]] = {}
            for label, idx_map in br.children.items():
                for child in idx_map.values():
                    lb_children.setdefault(label, {})[0] = upper_to_lower(child)
            return Lower(LowerBranch(children=lb_children))
        return upper_to_lower(self.inner)

    def _with_lower(self, lower: Lower[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(inner=Upper(Interface(node=lower, acc=None)), empty=None)

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        return self._with_lower(_push_lower(lower, value))

    def pop(self) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        return self._with_lower(_pop_lower(lower))

    def is_empty(self) -> bool:
        return not _has_any_stack(self._get_lower())

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        filtered = _isolate_lower_by_top(lower, value, incoming_label=None)
        if filtered is None:
            return LeveledGSS.from_stacks([])
        return self._with_lower(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        return self._with_lower(_apply_lower(lower, func))

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        pruned = _prune_lower(lower, predicate)
        if pruned is None:
            return LeveledGSS.from_stacks([])
        return self._with_lower(pruned)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        a = self._get_lower()
        b = other._get_lower()
        if not _has_any_stack(a):
            return other
        if not _has_any_stack(b):
            return self
        return self._with_lower(_merge_lower(a, b))

    def peek(self) -> Set[T]:
        return _peek_values(self._get_lower(), incoming_label=None, out=set())

    def reduce_acc(self) -> Optional[Acc]:
        return _reduce_all_acc(self._get_lower())


# ------------------------------
# Invariant validation (kept minimal, unchanged public API)
# ------------------------------

def _validate_upper(node: Upper[T, Acc]):
    """
    Recursively walk Upper nodes. This implementation always uses a single Interface at the top,
    so there's nothing to enforce beyond structural traversal.
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
