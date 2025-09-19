from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Iterable

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes (unchanged)
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    inner: UpperBranch[T, Acc] | Interface[T, Acc]


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, Upper[T, Acc]]]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: Lower[T]
    acc: Acc | None  # We store None for the top-level interface acc as placeholder.


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    inner: LowerBranch[T] | Leaf


@dataclass(frozen=True, eq=True)
class LowerBranch(Generic[T]):
    children: Dict[Any, Dict[int, Lower[T]]]


@dataclass(frozen=True, eq=True)
class Leaf:
    pass


# Internal helper: sentinel key used inside Lower to store accumulator at a leaf.
class _AccKey(Generic[Acc]):
    __slots__ = ("acc",)

    def __init__(self, acc: Acc):
        self.acc = acc

    def __repr__(self) -> str:
        return f"<AccKey:{self.acc!r}>"

    # Make each instance unique, even if the underlying acc compares equal.
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


def _acc_reduce(accs: Iterable[Acc]) -> Optional[Acc]:
    total: Optional[Acc] = None
    for a in accs:
        total = _merge_acc(total, a)
    return total


# ------------------------------
# Canonical helpers (simplified)
# ------------------------------

def _is_empty_branch(node: Lower[T]) -> bool:
    return isinstance(node.inner, LowerBranch) and not node.inner.children


def _empty_lower() -> Lower[T]:
    return Lower(LowerBranch(children={}))


def _split(node: Lower[T]) -> Tuple[Dict[T, Lower[T]], Optional[Acc]]:
    """
    Splits a canonical Lower node into (label_children, root_acc).
    Assumes children are canonical:
      - For each label key, only index 0 exists.
      - At most one _AccKey with index 0 contains the root accumulator.
    """
    if isinstance(node.inner, Leaf):
        return {}, None

    labels: Dict[T, Lower[T]] = {}
    root_acc: Optional[Acc] = None

    for key, idx_map in node.inner.children.items():
        if _is_acc_key(key):
            # Merge defensively if multiple sentinels appear (shouldn't happen in canonical form).
            root_acc = _merge_acc(root_acc, key.acc)  # type: ignore[attr-defined]
        else:
            # Canonical form uses only index 0; fall back to any if not present.
            child = idx_map.get(0)
            if child is None and idx_map:
                child = next(iter(idx_map.values()))
            if child is not None and not _is_empty_branch(child):
                labels[key] = child  # type: ignore[assignment]

    return labels, root_acc


def _make_node(labels: Dict[T, Lower[T]], root_acc: Optional[Acc]) -> Lower[T]:
    if not labels and root_acc is None:
        return _empty_lower()

    children: Dict[Any, Dict[int, Lower[T]]] = {}
    for label, child in labels.items():
        children[label] = {0: child}
    if root_acc is not None:
        children[_AccKey(root_acc)] = {0: Lower(Leaf())}
    return Lower(LowerBranch(children=children))


# ------------------------------
# Core trie operations (simplified yet equivalent)
# ------------------------------

def _merge_lower(a: Lower[T], b: Lower[T]) -> Lower[T]:
    """
    Merges two Lower tries by union of paths and merging accumulators on identical stacks.
    """
    if isinstance(a.inner, Leaf):
        return b
    if isinstance(b.inner, Leaf):
        return a

    a_labels, a_acc = _split(a)
    b_labels, b_acc = _split(b)

    merged_labels: Dict[T, Lower[T]] = {}

    # Merge children by label
    all_keys = set(a_labels.keys()) | set(b_labels.keys())
    for key in all_keys:
        if key in a_labels and key in b_labels:
            merged_labels[key] = _merge_lower(a_labels[key], b_labels[key])
        elif key in a_labels:
            merged_labels[key] = a_labels[key]
        else:
            merged_labels[key] = b_labels[key]

    merged_acc = _merge_acc(a_acc, b_acc)
    return _make_node(merged_labels, merged_acc)


def _add_root_acc(lower: Lower[T], acc: Optional[Acc]) -> Lower[T]:
    if acc is None:
        return lower
    if isinstance(lower.inner, Leaf):
        return _make_node({}, acc)
    labels, root_acc = _split(lower)
    return _make_node(labels, _merge_acc(root_acc, acc))


def _remove_root_acc(lower: Lower[T]) -> Tuple[Lower[T], Optional[Acc]]:
    if isinstance(lower.inner, Leaf):
        return lower, None
    labels, root_acc = _split(lower)
    return _make_node(labels, None), root_acc


def _push_lower(node: Lower[T], value: T) -> Lower[T]:
    """
    Pushes `value` onto all existing stacks represented by this Lower trie.
    Implementation: move the root accumulator under a `value` child and recurse into children.
    """
    if isinstance(node.inner, Leaf):
        return node

    labels, root_acc = _split(node)

    # Recurse into existing children
    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        pushed_child = _push_lower(child, value)
        if not _is_empty_branch(pushed_child):
            new_labels[label] = pushed_child

    # Move root accumulator under the `value` label
    if root_acc is not None:
        if value in new_labels:
            new_labels[value] = _add_root_acc(new_labels[value], root_acc)
        else:
            new_labels[value] = _make_node({}, root_acc)

    return _make_node(new_labels, None)


def _pop_lower(node: Lower[T]) -> Lower[T]:
    """
    Pops one element from all non-empty stacks represented by this Lower trie.
    Discards empty stacks at this level; contributes stacks of length 1 up to root.
    """
    if isinstance(node.inner, Leaf):
        return node

    labels, root_acc = _split(node)
    # Discard root_acc (empty stacks can't be popped)
    del root_acc

    new_labels: Dict[T, Lower[T]] = {}
    contributions: Optional[Acc] = None

    for label, child in labels.items():
        child_wo_acc, acc_at_child = _remove_root_acc(child)
        contributions = _merge_acc(contributions, acc_at_child)
        popped_child = _pop_lower(child_wo_acc)
        if not _is_empty_branch(popped_child):
            new_labels[label] = popped_child

    return _make_node(new_labels, contributions)


def _isolate_lower_by_top(node: Lower[T], target: Optional[T], incoming_label: Optional[T] = None) -> Optional[Lower[T]]:
    """
    Keeps only stacks whose top element equals `target` (or empty stacks if target is None).
    Returns None if the resulting subtree contains no stacks.
    """
    if isinstance(node.inner, Leaf):
        return None

    labels, root_acc = _split(node)

    keep_root = (incoming_label is None and target is None) or (incoming_label is not None and target == incoming_label)
    root_keep_acc = root_acc if keep_root else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        filtered = _isolate_lower_by_top(child, target, incoming_label=label)
        if filtered is not None and not _is_empty_branch(filtered):
            new_labels[label] = filtered

    if not new_labels and root_keep_acc is None:
        return None

    return _make_node(new_labels, root_keep_acc)


def _apply_lower(node: Lower[T], func: Callable[[Acc], Acc]) -> Lower[T]:
    """
    Applies `func` to each accumulator throughout the trie.
    """
    if isinstance(node.inner, Leaf):
        return node

    labels, root_acc = _split(node)
    transformed_root = func(root_acc) if root_acc is not None else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        transformed_child = _apply_lower(child, func)
        if not _is_empty_branch(transformed_child):
            new_labels[label] = transformed_child

    return _make_node(new_labels, transformed_root)


def _prune_lower(node: Lower[T], predicate: Callable[[Acc], bool]) -> Optional[Lower[T]]:
    """
    Removes stacks whose accumulator does not satisfy `predicate`.
    Prunes empty branches. Returns None if no stacks remain.
    """
    if isinstance(node.inner, Leaf):
        return None

    labels, root_acc = _split(node)
    kept_root: Optional[Acc] = root_acc if (root_acc is not None and predicate(root_acc)) else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        pruned_child = _prune_lower(child, predicate)
        if pruned_child is not None and not _is_empty_branch(pruned_child):
            new_labels[label] = pruned_child

    if not new_labels and kept_root is None:
        return None

    return _make_node(new_labels, kept_root)


def _has_any_stack(node: Lower[T]) -> bool:
    """
    Returns True if there exists at least one stack (i.e., any accumulator sentinel anywhere).
    """
    if isinstance(node.inner, Leaf):
        return False
    for key, idx_map in node.inner.children.items():
        if _is_acc_key(key):
            return True
        for child in idx_map.values():
            if _has_any_stack(child):
                return True
    return False


def _peek_values(node: Lower[T], incoming_label: Optional[T] = None, out: Optional[Set[T]] = None) -> Set[T]:
    """
    Collects the set of top-of-stack values across all non-empty stacks.
    """
    if out is None:
        out = set()
    if isinstance(node.inner, Leaf):
        return out

    labels, root_acc = _split(node)

    # Root accumulators imply top-of-stack == incoming_label (if not None)
    if root_acc is not None and incoming_label is not None:
        out.add(incoming_label)

    for label, child in labels.items():
        _peek_values(child, incoming_label=label, out=out)
    return out


def _reduce_all_acc(node: Lower[T]) -> Optional[Acc]:
    """
    Merges all accumulators across the entire trie (or returns None if no stacks).
    """
    if isinstance(node.inner, Leaf):
        return None

    labels, root_acc = _split(node)
    total: Optional[Acc] = root_acc
    for child in labels.values():
        total = _merge_acc(total, _reduce_all_acc(child))
    return total


# ------------------------------
# Public LeveledGSS implementation (simplified but functionally equivalent)
# ------------------------------

@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        """
        Build a LeveledGSS from explicit stacks.
        Implementation strategy:
        - Canonicalize input stacks by merging accumulators for identical lists.
        - Build a Lower trie that encodes each stack path from bottom to top.
        - At the end of each path, attach a special _AccKey(acc) edge to a Leaf.
        - Store the trie inside a top-level Upper Interface node with a placeholder acc (None).
        """
        # Canonicalize: merge accumulators for identical stacks
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        # Build a nested Python dict trie first: Dict[node_key, child_dict]
        acc_key = object()
        trie: Dict[Any, Dict] = {}

        def insert_path(path: List[T], acc: Acc) -> None:
            node = trie
            for item in path:
                node = node.setdefault(item, {})
            # Merge accumulator if a stack already ends here
            prev = node.get(acc_key)
            node[acc_key] = acc if prev is None else prev.merge(acc)

        for key, acc in merged.items():
            insert_path(list(key), acc)

        # Convert the trie into Lower nodes (immutable dataclasses)
        def build_lower(node_dict: Dict[Any, Dict]) -> Lower[T]:
            labels: Dict[T, Lower[T]] = {}
            root_acc: Optional[Acc] = None

            for key, sub in node_dict.items():
                if key is acc_key:
                    # Accumulator stored here
                    root_acc = sub  # type: ignore[assignment]
                else:
                    labels[key] = build_lower(sub)

            return _make_node(labels, root_acc)

        lower_root = build_lower(trie)

        # Top-level Upper is a single Interface to our Lower trie.
        upper = Upper(Interface(node=lower_root, acc=None))
        return LeveledGSS(inner=upper, empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Decode the Lower trie into a list of (stack, acc) pairs.
        The trie is encoded bottom->top; we traverse accordingly.
        """
        results: List[Tuple[List[T], Acc]] = []

        def collect(node: Lower[T], prefix: List[T]) -> None:
            if isinstance(node.inner, Leaf):
                return
            labels, root_acc = _split(node)
            if root_acc is not None:
                results.append((list(prefix), root_acc))
            for label, child in labels.items():
                prefix.append(label)
                collect(child, prefix)
                prefix.pop()

        if isinstance(self.inner.inner, Interface):
            collect(self.inner.inner.node, [])
        else:
            # Defensive fallback: traverse generic Upper (should not occur in normal operation)
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

        # Canonicalize and sort deterministically
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
        # Helper to extract the root Lower node; our instances keep Interface at the top.
        if isinstance(self.inner.inner, Interface):
            return self.inner.inner.node
        # Defensive conversion (should not happen): fold UpperBranch into a Lower by translating directly.
        def upper_to_lower(u: Upper[T, Acc]) -> Lower[T]:
            if isinstance(u.inner, Interface):
                return u.inner.node
            lb_children: Dict[Any, Dict[int, Lower[T]]] = {}
            br: UpperBranch[T, Acc] = u.inner
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
        new_lower = _push_lower(lower, value)
        return LeveledGSS(inner=Upper(Interface(node=new_lower, acc=None)), empty=None)

    def pop(self) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        new_lower = _pop_lower(lower)
        return LeveledGSS(inner=Upper(Interface(node=new_lower, acc=None)), empty=None)

    def is_empty(self) -> bool:
        lower = self._get_lower()
        return not _has_any_stack(lower)

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        filtered = _isolate_lower_by_top(lower, value, incoming_label=None)
        if filtered is None:
            return LeveledGSS.from_stacks([])
        return LeveledGSS(inner=Upper(Interface(node=filtered, acc=None)), empty=None)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        transformed = _apply_lower(lower, func)
        return LeveledGSS(inner=Upper(Interface(node=transformed, acc=None)), empty=None)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        pruned = _prune_lower(lower, predicate)
        if pruned is None:
            return LeveledGSS.from_stacks([])
        return LeveledGSS(inner=Upper(Interface(node=pruned, acc=None)), empty=None)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        a = self._get_lower()
        b = other._get_lower()
        if not _has_any_stack(a):
            return other
        if not _has_any_stack(b):
            return self
        merged = _merge_lower(a, b)
        return LeveledGSS(inner=Upper(Interface(node=merged, acc=None)), empty=None)

    def peek(self) -> Set[T]:
        lower = self._get_lower()
        return _peek_values(lower, incoming_label=None, out=set())

    def reduce_acc(self) -> Optional[Acc]:
        lower = self._get_lower()
        return _reduce_all_acc(lower)


# ------------------------------
# Invariant validation (unchanged public API)
# ------------------------------

def _validate_upper(node: Upper[T, Acc]):
    """
    Recursively validates simple invariants on Upper nodes.
    This implementation keeps the structure simple: a top-level Interface to a Lower trie.
    If UpperBranch appears, ensure that if all children are Interfaces, their accs are unique.
    """
    if isinstance(node.inner, UpperBranch):
        branch = node.inner
        all_children = [
            child
            for children_by_val in branch.children.values()
            for child in children_by_val.values()
        ]

        if all_children and all(isinstance(child.inner, Interface) for child in all_children):
            accs = [child.inner.acc for child in all_children]
            if len(set(accs)) != len(accs):
                raise AssertionError(
                    "Invariant violated: UpperBranch has children that are all Interfaces with duplicate accs."
                )

        for child in all_children:
            _validate_upper(child)
    # Base case: node.inner is an Interface; nothing to do.


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Checks internal invariants of the LeveledGSS structure.
    Raises AssertionError if an invariant is violated.
    """
    _validate_upper(gss.inner)

    # Invariant: If inner is an interface and empty exists, their accs must differ.
    if isinstance(gss.inner.inner, Interface) and gss.empty is not None:
        if gss.inner.inner.acc == gss.empty:
            raise AssertionError(
                "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
            )
