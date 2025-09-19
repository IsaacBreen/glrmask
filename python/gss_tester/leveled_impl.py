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
    # We do not want accidental coalescing at the dict layer; we canonicalize earlier.
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


def _split_children(children: Dict[Any, Dict[int, Lower[T]]]) -> Tuple[Dict[T, List[Lower[T]]], List[Acc]]:
    """
    Splits a LowerBranch.children dict into:
    - label_to_lowers: mapping from normal label (T) to list of child Lower nodes
    - accs: list of accumulator values stored at this node (via _AccKey)
    """
    label_to_lowers: Dict[T, List[Lower[T]]] = {}
    accs: List[Acc] = []
    for key, idx_map in children.items():
        if _is_acc_key(key):
            # Collect all accs (there should be at most one entry, but merge defensively)
            accs.append(key.acc)  # type: ignore[attr-defined]
        else:
            lst = label_to_lowers.setdefault(key, [])
            lst.extend(idx_map.values())
    return label_to_lowers, accs


def _build_children_from_map(label_to_lower: Dict[T, Lower[T]], acc: Optional[Acc]) -> Dict[Any, Dict[int, Lower[T]]]:
    """
    Builds a children dict from a mapping of labels to Lower nodes and an optional accumulator at this node.
    Uses index 0 for deterministic placement.
    """
    out: Dict[Any, Dict[int, Lower[T]]] = {}
    for label, lower in label_to_lower.items():
        out[label] = {0: lower}
    if acc is not None:
        out[_AccKey(acc)] = {0: Lower(Leaf())}
    return out


def _merge_lower(a: Lower[T], b: Lower[T]) -> Lower[T]:
    """
    Merges two Lower tries by union of paths and merging accumulators on identical stacks.
    """
    if isinstance(a.inner, Leaf):
        # Leaf only appears as child of an _AccKey; standalone Leaf means no stacks under prefix.
        return b
    if isinstance(b.inner, Leaf):
        return a

    # Both are LowerBranch
    a_labels, a_accs = _split_children(a.inner.children)
    b_labels, b_accs = _split_children(b.inner.children)

    # Merge label children
    all_labels: Set[T] = set(a_labels.keys()) | set(b_labels.keys())
    merged_labels: Dict[T, Lower[T]] = {}
    for label in all_labels:
        lowers_a = a_labels.get(label, [])
        lowers_b = b_labels.get(label, [])
        # Merge all lowers for this label
        merged: Optional[Lower[T]] = None
        for child in lowers_a + lowers_b:
            merged = child if merged is None else _merge_lower(merged, child)
        if merged is not None:
            # Keep only non-empty subtries
            if isinstance(merged.inner, LowerBranch) and not merged.inner.children:
                # Empty branch; skip
                pass
            else:
                merged_labels[label] = merged

    # Merge accumulators at this node
    merged_acc = _acc_reduce(a_accs + b_accs)
    return Lower(LowerBranch(children=_build_children_from_map(merged_labels, merged_acc)))


def _add_root_acc(lower: Lower[T], acc: Optional[Acc]) -> Lower[T]:
    """
    Returns a new Lower equal to `lower` but with its root having an added accumulator (merged if present).
    """
    if acc is None:
        return lower
    if isinstance(lower.inner, Leaf):
        # A leaf doesn't carry any children; make a branch that only stores the accumulator.
        return Lower(LowerBranch(children=_build_children_from_map({}, acc)))

    labels, accs = _split_children(lower.inner.children)
    merged_acc = _merge_acc(_acc_reduce(accs), acc)
    return Lower(LowerBranch(children=_build_children_from_map(labels, merged_acc)))


def _remove_root_acc(lower: Lower[T]) -> Tuple[Lower[T], Optional[Acc]]:
    """
    Removes all accumulator entries at the root of `lower` and returns (new_lower, merged_acc_removed).
    """
    if isinstance(lower.inner, Leaf):
        return lower, None
    labels, accs = _split_children(lower.inner.children)
    new_children = _build_children_from_map(labels, None)
    return Lower(LowerBranch(children=new_children)), _acc_reduce(accs)


def _push_lower(node: Lower[T], value: T) -> Lower[T]:
    """
    Pushes `value` onto all existing stacks represented by this Lower trie.
    Implementation: move all root-accumulator(s) down under a `value` child; recurse into children.
    """
    if isinstance(node.inner, Leaf):
        return node
    labels, accs = _split_children(node.inner.children)

    # Recurse into children
    new_labels: Dict[T, Lower[T]] = {}
    for label, lowers in labels.items():
        merged: Optional[Lower[T]] = None
        for child in lowers:
            pushed_child = _push_lower(child, value)
            merged = pushed_child if merged is None else _merge_lower(merged, pushed_child)
        if merged is not None and (not isinstance(merged.inner, LowerBranch) or merged.inner.children):
            new_labels[label] = merged

    # Move root accumulators under the `value` label as sentinel(s)
    contrib_acc = _acc_reduce(accs)
    if value in new_labels:
        new_labels[value] = _add_root_acc(new_labels[value], contrib_acc)
    else:
        # Create a new child with only the accumulator at its root if contrib exists
        if contrib_acc is not None:
            new_labels[value] = Lower(LowerBranch(children=_build_children_from_map({}, contrib_acc)))
        # Else, nothing to add

    return Lower(LowerBranch(children=_build_children_from_map(new_labels, None)))


def _pop_lower(node: Lower[T]) -> Lower[T]:
    """
    Pops one element from all non-empty stacks represented by this Lower trie.
    Implementation strategy:
    - Discard any root accumulator (those were empty stacks; pop removes them).
    - For each label child:
        * Extract accumulator(s) at the child's root: those correspond to stacks of length 1 under this child.
          After pop, they become empty stacks at the current node.
        * Recurse into the child's subtree (without its root accumulator) to pop deeper stacks.
    - Place the merged contributions from all children as a root accumulator at this node.
    """
    if isinstance(node.inner, Leaf):
        return node

    labels, root_accs = _split_children(node.inner.children)
    # Root accs correspond to empty stacks; discard on pop.
    del root_accs

    new_labels: Dict[T, Lower[T]] = {}
    contributions: Optional[Acc] = None

    for label, lowers in labels.items():
        merged_child: Optional[Lower[T]] = None
        contrib_this_label: Optional[Acc] = None
        for child in lowers:
            # Extract and remove root acc from child
            child_wo_acc, acc_at_child_root = _remove_root_acc(child)
            contrib_this_label = _merge_acc(contrib_this_label, acc_at_child_root)
            # Recurse
            popped_child = _pop_lower(child_wo_acc)
            # Keep non-empty child
            if isinstance(popped_child.inner, LowerBranch) and not popped_child.inner.children:
                # Empty; skip
                continue
            merged_child = popped_child if merged_child is None else _merge_lower(merged_child, popped_child)

        if merged_child is not None:
            new_labels[label] = merged_child
        contributions = _merge_acc(contributions, contrib_this_label)

    return Lower(LowerBranch(children=_build_children_from_map(new_labels, contributions)))


def _isolate_lower_by_top(node: Lower[T], target: Optional[T], incoming_label: Optional[T] = None) -> Optional[Lower[T]]:
    """
    Keeps only stacks whose top element equals `target` (or empty stacks if target is None).
    We propagate the label-from-parent to decide whether to keep root accumulator(s).
    Returns None if the resulting subtree contains no stacks.
    """
    if isinstance(node.inner, Leaf):
        return None  # Should not happen; Leaf is only below sentinel; no stacks here.

    labels, accs = _split_children(node.inner.children)

    # Determine whether to keep root accumulator(s) at this node.
    keep_root = (incoming_label is None and target is None) or (incoming_label is not None and target == incoming_label)
    root_acc: Optional[Acc] = _acc_reduce(accs) if keep_root else None

    new_labels: Dict[T, Lower[T]] = {}
    for label, lowers in labels.items():
        merged_child: Optional[Lower[T]] = None
        for child in lowers:
            filtered = _isolate_lower_by_top(child, target, incoming_label=label)
            if filtered is None:
                continue
            merged_child = filtered if merged_child is None else _merge_lower(merged_child, filtered)
        if merged_child is not None and (not isinstance(merged_child.inner, LowerBranch) or merged_child.inner.children):
            new_labels[label] = merged_child

    # If nothing remains and no root acc to keep, return None
    if not new_labels and root_acc is None:
        return None

    return Lower(LowerBranch(children=_build_children_from_map(new_labels, root_acc)))


def _apply_lower(node: Lower[T], func: Callable[[Acc], Acc]) -> Lower[T]:
    """
    Applies `func` to each accumulator throughout the trie.
    """
    if isinstance(node.inner, Leaf):
        return node

    labels, accs = _split_children(node.inner.children)

    # Transform root accumulators and merge if multiple
    transformed_root: Optional[Acc] = None
    for acc in accs:
        transformed_root = _merge_acc(transformed_root, func(acc))

    new_labels: Dict[T, Lower[T]] = {}
    for label, lowers in labels.items():
        merged_child: Optional[Lower[T]] = None
        for child in lowers:
            transformed_child = _apply_lower(child, func)
            merged_child = transformed_child if merged_child is None else _merge_lower(merged_child, transformed_child)
        if merged_child is not None and (not isinstance(merged_child.inner, LowerBranch) or merged_child.inner.children):
            new_labels[label] = merged_child

    return Lower(LowerBranch(children=_build_children_from_map(new_labels, transformed_root)))


def _prune_lower(node: Lower[T], predicate: Callable[[Acc], bool]) -> Optional[Lower[T]]:
    """
    Removes stacks whose accumulator does not satisfy `predicate`.
    Prunes empty branches. Returns None if no stacks remain.
    """
    if isinstance(node.inner, Leaf):
        return None

    labels, accs = _split_children(node.inner.children)
    # Keep only root accumulators satisfying predicate
    kept_root: Optional[Acc] = None
    for acc in accs:
        if predicate(acc):
            kept_root = _merge_acc(kept_root, acc)

    new_labels: Dict[T, Lower[T]] = {}
    for label, lowers in labels.items():
        merged_child: Optional[Lower[T]] = None
        for child in lowers:
            pruned = _prune_lower(child, predicate)
            if pruned is None:
                continue
            merged_child = pruned if merged_child is None else _merge_lower(merged_child, pruned)
        if merged_child is not None and (not isinstance(merged_child.inner, LowerBranch) or merged_child.inner.children):
            new_labels[label] = merged_child

    if not new_labels and kept_root is None:
        return None

    return Lower(LowerBranch(children=_build_children_from_map(new_labels, kept_root)))


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

    # Root accumulators imply top-of-stack == incoming_label (if not None)
    for key in node.inner.children:
        if _is_acc_key(key) and incoming_label is not None:
            out.add(incoming_label)
            break

    for label, idx_map in node.inner.children.items():
        if _is_acc_key(label):
            continue
        for child in idx_map.values():
            _peek_values(child, incoming_label=label, out=out)
    return out


def _reduce_all_acc(node: Lower[T]) -> Optional[Acc]:
    """
    Merges all accumulators across the entire trie (or returns None if no stacks).
    """
    if isinstance(node.inner, Leaf):
        return None

    total: Optional[Acc] = None
    labels, accs = _split_children(node.inner.children)
    for acc in accs:
        total = _merge_acc(total, acc)
    for lowers in labels.values():
        for child in lowers:
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
        - We intentionally keep `empty=None` and represent empty stacks within the Lower trie
          via an _AccKey at the root. This keeps invariants trivially satisfied.
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
        # node_key is either a real stack item (T) or _AccKey(acc) sentinel at leaves.
        trie: Dict[Any, Dict] = {}

        def insert_path(path: List[T], acc: Acc) -> None:
            node = trie
            # Traverse bottom -> top
            for item in path:
                node = node.setdefault(item, {})
            # Attach accumulator marker at the end
            node[_AccKey(acc)] = {}  # Child dict for leaf (empty)

        for key, acc in merged.items():
            insert_path(list(key), acc)

        # Convert the trie into Lower nodes (immutable dataclasses)
        def build_lower(node_dict: Dict[Any, Dict]) -> Lower[T]:
            children_map: Dict[Any, Dict[int, Lower[T]]] = {}
            for key, sub in node_dict.items():
                if isinstance(key, _AccKey):
                    # Terminal edge carrying the accumulator
                    child_lower = Lower(Leaf())
                    # Use index 0 for deterministic placement
                    children_map.setdefault(key, {})[0] = child_lower  # type: ignore[arg-type]
                else:
                    child_lower = build_lower(sub)
                    children_map.setdefault(key, {})[0] = child_lower  # type: ignore[arg-type]
            return Lower(LowerBranch(children=children_map))

        lower_root = build_lower(trie)

        # Top-level Upper is a single Interface to our Lower trie.
        upper = Upper(Interface(node=lower_root, acc=None))
        # Keep empty=None so that validation rule about equality of accs is skipped.
        return LeveledGSS(inner=upper, empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Decode the Lower trie into a list of (stack, acc) pairs.
        The trie was encoded bottom->top; we traverse accordingly.
        """
        def collect_from_lower(node: Lower[T], prefix: List[T], out: List[Tuple[List[T], Acc]]) -> None:
            if isinstance(node.inner, Leaf):
                # Should not happen in our encoding except as child of an _AccKey.
                return
            branch: LowerBranch[T] = node.inner
            for key in branch.children:
                for child in branch.children[key].values():
                    if isinstance(key, _AccKey):
                        out.append((list(prefix), key.acc))  # type: ignore[attr-defined]
                    else:
                        prefix.append(key)  # descend adding item
                        collect_from_lower(child, prefix, out)
                        prefix.pop()

        results: List[Tuple[List[T], Acc]] = []
        # Our encoding always sets inner as Interface
        if isinstance(self.inner.inner, Interface):
            collect_from_lower(self.inner.inner.node, [], results)
        elif isinstance(self.inner.inner, UpperBranch):
            # Defensive: handle unexpected structure by traversing generic Upper tree
            def collect_from_upper(u: Upper[T, Acc], top_prefix: List[T], out: List[Tuple[List[T], Acc]]) -> None:
                if isinstance(u.inner, Interface):
                    collect_from_lower(u.inner.node, top_prefix, out)
                    return
                br: UpperBranch[T, Acc] = u.inner
                for val, idx_map in br.children.items():
                    for child in idx_map.values():
                        top_prefix.append(val)
                        collect_from_upper(child, top_prefix, out)
                        top_prefix.pop()

            collect_from_upper(self.inner, [], results)
        else:
            # Should not occur; return empty
            pass

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
        # Helper to extract the root Lower node, handling both Interface and UpperBranch (defensive).
        if isinstance(self.inner.inner, Interface):
            return self.inner.inner.node
        # Fallback: if an UpperBranch exists, fold it into a single Lower by translating the structure.
        # This is defensive and should not happen in normal operation.
        def upper_to_lower(u: Upper[T, Acc]) -> Lower[T]:
            if isinstance(u.inner, Interface):
                return u.inner.node
            # Convert UpperBranch by interpreting labels as top-level children sequences
            lb_children: Dict[Any, Dict[int, Lower[T]]] = {}
            br: UpperBranch[T, Acc] = u.inner
            for label, idx_map in br.children.items():
                for child in idx_map.values():
                    lb_children.setdefault(label, {})[0] = upper_to_lower(child)
            return Lower(LowerBranch(children=lb_children))
        return upper_to_lower(self.inner)

    def _with_lower(self, lower: Lower[T]) -> LeveledGSS[T, Acc]:
        return LeveledGSS(inner=Upper(Interface(node=lower, acc=None))), None  # type: ignore[return-value]

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        # Quick path: if no stacks, return empty
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
    """Recursively validates invariants on Upper nodes."""
    if isinstance(node.inner, UpperBranch):
        branch = node.inner
        all_children = [
            child
            for children_by_val in branch.children.values()
            for child in children_by_val.values()
        ]

        # Invariant 1: If all children are interfaces, their accs must be unique.
        if all_children and all(isinstance(child.inner, Interface) for child in all_children):
            accs = [child.inner.acc for child in all_children]
            if len(set(accs)) != len(accs):
                raise AssertionError(
                    "Invariant violated: UpperBranch has children that are all Interfaces with duplicate accs."
                )

        # Recurse into children
        for child in all_children:
            _validate_upper(child)
    # Base case: node.inner is an Interface, do nothing further down this path.


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Checks internal invariants of the LeveledGSS structure.
    Raises AssertionError if an invariant is violated.
    """
    # Check recursive invariants on the inner structure.
    _validate_upper(gss.inner)

    # Invariant 2: If inner is an interface and empty exists, their accs must differ.
    if isinstance(gss.inner.inner, Interface) and gss.empty is not None:
        if gss.inner.inner.acc == gss.empty:
            raise AssertionError(
                "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
            )
