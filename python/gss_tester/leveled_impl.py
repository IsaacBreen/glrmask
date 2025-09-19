from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Iterable

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes
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
# Canonical children helpers
# ------------------------------

def _is_empty_branch(node: Lower[T]) -> bool:
    return isinstance(node.inner, LowerBranch) and not node.inner.children


def _normalize_children(children: Dict[Any, Dict[int, Lower[T]]]) -> Tuple[Dict[T, Lower[T]], Optional[Acc]]:
    """
    Converts the possibly multi-index 'children' mapping into:
    - a canonical label->Lower map with at most one Lower per label (merged if necessary),
    - a single optional accumulator at this node (merged if multiple sentinel entries exist).
    """
    labels: Dict[T, Lower[T]] = {}
    root_acc: Optional[Acc] = None

    # First collect per-label lists and all root accs
    temp_label_children: Dict[T, List[Lower[T]]] = {}
    accs: List[Acc] = []

    for key, idx_map in children.items():
        if _is_acc_key(key):
            # Sentinel(s) storing accumulator(s) for stacks ending at this node.
            # Merge later for canonical single acc at root.
            # There should be only one, but be defensive.
            # Extract acc from sentinel key.
            accs.append(key.acc)  # type: ignore[attr-defined]
        else:
            lst = temp_label_children.setdefault(key, [])
            lst.extend(idx_map.values())

    # Merge per-label lists down to a single Lower per label
    for label, lowers in temp_label_children.items():
        if not lowers:
            continue
        merged: Optional[Lower[T]] = None
        for child in lowers:
            merged = child if merged is None else _merge_lower(merged, child)
        if merged is not None and (not isinstance(merged.inner, LowerBranch) or merged.inner.children):
            labels[label] = merged

    # Merge all root accs into a single one
    root_acc = _acc_reduce(accs)
    return labels, root_acc


def _build_children(labels: Dict[T, Lower[T]], root_acc: Optional[Acc]) -> Dict[Any, Dict[int, Lower[T]]]:
    """
    Builds a children dict from canonical label map and an optional root accumulator.
    Uses only index 0 for deterministic placement and to keep structure simple.
    """
    out: Dict[Any, Dict[int, Lower[T]]] = {}
    for label, lower in labels.items():
        out[label] = {0: lower}
    if root_acc is not None:
        out[_AccKey(root_acc)] = {0: Lower(Leaf())}
    return out


def _make_lower(labels: Dict[T, Lower[T]], root_acc: Optional[Acc]) -> Lower[T]:
    return Lower(LowerBranch(children=_build_children(labels, root_acc)))


# ------------------------------
# Core trie operations
# ------------------------------

def _merge_lower(a: Lower[T], b: Lower[T]) -> Lower[T]:
    """
    Merges two Lower tries by union of paths and merging accumulators on identical stacks.
    """
    if isinstance(a.inner, Leaf):
        return b
    if isinstance(b.inner, Leaf):
        return a

    a_labels, a_acc = _normalize_children(a.inner.children)
    b_labels, b_acc = _normalize_children(b.inner.children)

    all_keys: Set[T] = set(a_labels.keys()) | set(b_labels.keys())
    merged_labels: Dict[T, Lower[T]] = {}
    for key in all_keys:
        if key in a_labels and key in b_labels:
            merged_labels[key] = _merge_lower(a_labels[key], b_labels[key])
        elif key in a_labels:
            merged_labels[key] = a_labels[key]
        else:
            merged_labels[key] = b_labels[key]

    merged_acc = _merge_acc(a_acc, b_acc)
    return _make_lower(merged_labels, merged_acc)


def _add_root_acc(lower: Lower[T], acc: Optional[Acc]) -> Lower[T]:
    """
    Returns a new Lower equal to `lower` but with its root having an added accumulator (merged if present).
    """
    if acc is None:
        return lower
    if isinstance(lower.inner, Leaf):
        return _make_lower({}, acc)

    labels, root_acc = _normalize_children(lower.inner.children)
    return _make_lower(labels, _merge_acc(root_acc, acc))


def _remove_root_acc(lower: Lower[T]) -> Tuple[Lower[T], Optional[Acc]]:
    """
    Removes all accumulator entries at the root of `lower` and returns (new_lower_without_root_acc, merged_acc_removed).
    """
    if isinstance(lower.inner, Leaf):
        return lower, None

    labels, root_acc = _normalize_children(lower.inner.children)
    return _make_lower(labels, None), root_acc


def _push_lower(node: Lower[T], value: T) -> Lower[T]:
    """
    Pushes `value` onto all existing stacks represented by this Lower trie.
    Implementation: move the root accumulator under a `value` child and recurse into children.
    """
    if isinstance(node.inner, Leaf):
        return node

    labels, root_acc = _normalize_children(node.inner.children)

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
            new_labels[value] = _make_lower({}, root_acc)

    return _make_lower(new_labels, None)


def _pop_lower(node: Lower[T]) -> Lower[T]:
    """
    Pops one element from all non-empty stacks represented by this Lower trie.
    Discards empty stacks at this level; contributes stacks of length 1 up to root.
    """
    if isinstance(node.inner, Leaf):
        return node

    labels, root_acc = _normalize_children(node.inner.children)
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

    return _make_lower(new_labels, contributions)


def _isolate_lower_by_top(node: Lower[T], target: Optional[T], incoming_label: Optional[T] = None) -> Optional[Lower[T]]:
    """
    Keeps only stacks whose top element equals `target` (or empty stacks if target is None).
    Returns None if the resulting subtree contains no stacks.
    """
    if isinstance(node.inner, Leaf):
        return None

    labels, root_acc = _normalize_children(node.inner.children)

    keep_root = (incoming_label is None and target is None) or (incoming_label is not None and target == incoming_label)
    root_keep_acc = root_acc if keep_root else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        filtered = _isolate_lower_by_top(child, target, incoming_label=label)
        if filtered is not None and not _is_empty_branch(filtered):
            new_labels[label] = filtered

    if not new_labels and root_keep_acc is None:
        return None

    return _make_lower(new_labels, root_keep_acc)


def _apply_lower(node: Lower[T], func: Callable[[Acc], Acc]) -> Lower[T]:
    """
    Applies `func` to each accumulator throughout the trie.
    """
    if isinstance(node.inner, Leaf):
        return node

    labels, root_acc = _normalize_children(node.inner.children)
    transformed_root = func(root_acc) if root_acc is not None else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        transformed_child = _apply_lower(child, func)
        if not _is_empty_branch(transformed_child):
            new_labels[label] = transformed_child

    return _make_lower(new_labels, transformed_root)


def _prune_lower(node: Lower[T], predicate: Callable[[Acc], bool]) -> Optional[Lower[T]]:
    """
    Removes stacks whose accumulator does not satisfy `predicate`.
    Prunes empty branches. Returns None if no stacks remain.
    """
    if isinstance(node.inner, Leaf):
        return None

    labels, root_acc = _normalize_children(node.inner.children)
    kept_root: Optional[Acc] = root_acc if (root_acc is not None and predicate(root_acc)) else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, child in labels.items():
        pruned_child = _prune_lower(child, predicate)
        if pruned_child is not None and not _is_empty_branch(pruned_child):
            new_labels[label] = pruned_child

    if not new_labels and kept_root is None:
        return None

    return _make_lower(new_labels, kept_root)


def _has_any_stack(node: Lower[T]) -> bool:
    """
    Returns True if there exists at least one stack (i.e., any accumulator sentinel anywhere).
    """
    if isinstance(node.inner, Leaf):
        return False
    # Check root
    for key in node.inner.children:
        if _is_acc_key(key):
            return True
    # Recurse
    for key, idx_map in node.inner.children.items():
        if not _is_acc_key(key):
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

    labels, root_acc = _normalize_children(node.inner.children)

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

    labels, root_acc = _normalize_children(node.inner.children)
    total: Optional[Acc] = root_acc
    for child in labels.values():
        total = _merge_acc(total, _reduce_all_acc(child))
    return total


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
        trie: Dict[Any, Dict] = {}

        def insert_path(path: List[T], acc: Acc) -> None:
            node = trie
            for item in path:
                node = node.setdefault(item, {})
            node[_AccKey(acc)] = {}

        for key, acc in merged.items():
            insert_path(list(key), acc)

        # Convert the trie into Lower nodes (immutable dataclasses)
        def build_lower(node_dict: Dict[Any, Dict]) -> Lower[T]:
            children_map: Dict[Any, Dict[int, Lower[T]]] = {}
            for key, sub in node_dict.items():
                if isinstance(key, _AccKey):
                    child_lower = Lower(Leaf())
                else:
                    child_lower = build_lower(sub)
                children_map.setdefault(key, {})[0] = child_lower  # type: ignore[arg-type]
            return Lower(LowerBranch(children=children_map))

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
            labels, root_acc = _normalize_children(node.inner.children)
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
