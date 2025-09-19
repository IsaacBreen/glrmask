from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Iterable

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes (public API shape preserved)
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


# ------------------------------
# Private helpers
# ------------------------------

class _AccKey(Generic[Acc]):
    """
    Sentinel used inside LowerBranch.children to encode the presence of an accumulator
    at a node. Each _AccKey instance is unique even if the underlying acc compares equal,
    so we explicitly merge during canonicalization rather than relying on dict keys.
    """
    __slots__ = ("acc",)

    def __init__(self, acc: Acc):
        self.acc = acc

    def __repr__(self) -> str:
        return f"<AccKey:{self.acc!r}>"

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


# A simple, canonical trie node used privately to implement all logic in a clear way.
# It encodes stacks as paths from the root (empty prefix) to a node. Each node holds:
# - acc: Optional accumulator for the stack that terminates at this node (the exact path).
# - kids: mapping from next label (stack item) to child node.
@dataclass
class _Node(Generic[T, Acc]):
    acc: Optional[Acc]
    kids: Dict[T, "_Node[T, Acc]"]


def _node_is_empty(n: _Node[T, Acc]) -> bool:
    if n.acc is not None:
        return False
    for ch in n.kids.values():
        if not _node_is_empty(ch):
            return False
    return True


def _merge_nodes(a: _Node[T, Acc], b: _Node[T, Acc]) -> _Node[T, Acc]:
    merged_acc = _merge_acc(a.acc, b.acc)
    # Merge children label-wise
    all_labels = set(a.kids.keys()) | set(b.kids.keys())
    merged_kids: Dict[T, _Node[T, Acc]] = {}
    for label in all_labels:
        ca = a.kids.get(label)
        cb = b.kids.get(label)
        if ca is None:
            child = cb  # type: ignore[assignment]
        elif cb is None:
            child = ca
        else:
            child = _merge_nodes(ca, cb)
        if child is not None and not _node_is_empty(child):
            merged_kids[label] = child
    return _Node(acc=merged_acc, kids=merged_kids)


def _lower_to_node(lower: Lower[T]) -> _Node[T, Acc]:
    """
    Converts our Lower encoding (which uses _AccKey sentinels to represent a node's acc)
    into the canonical _Node representation.
    """
    if isinstance(lower.inner, Leaf):
        # A standalone Leaf only appears under an _AccKey edge. Represent as empty node.
        return _Node(acc=None, kids={})

    # Collect acc at this node and merge duplicate children by label
    acc_at_node: Optional[Acc] = None
    kids: Dict[T, _Node[T, Acc]] = {}

    for key, idx_map in lower.inner.children.items():
        if _is_acc_key(key):
            # Merge all accs attached at this node (defensive; normal form has at most one).
            for _child in idx_map.values():
                # child is always Leaf in our encoding; ignore it and merge the acc
                acc_at_node = _merge_acc(acc_at_node, key.acc)  # type: ignore[attr-defined]
        else:
            # Merge multiple entries for same label by recursively merging nodes
            child_node: Optional[_Node[T, Acc]] = None
            for child_lower in idx_map.values():
                decoded = _lower_to_node(child_lower)
                child_node = decoded if child_node is None else _merge_nodes(child_node, decoded)
            if child_node is not None and not _node_is_empty(child_node):
                if key in kids:
                    kids[key] = _merge_nodes(kids[key], child_node)
                else:
                    kids[key] = child_node

    return _Node(acc=acc_at_node, kids=kids)


def _node_to_lower(node: _Node[T, Acc]) -> Lower[T]:
    """
    Converts the canonical _Node representation back to the Lower encoding with _AccKey sentinels.
    """
    children_map: Dict[Any, Dict[int, Lower[T]]] = {}

    # Add child for acc sentinel if present
    if node.acc is not None:
        children_map[_AccKey(node.acc)] = {0: Lower(Leaf())}

    # Add labeled children deterministically at index 0
    for label, child in node.kids.items():
        if _node_is_empty(child):
            continue
        children_map[label] = {0: _node_to_lower(child)}

    return Lower(LowerBranch(children=children_map))


def _push_node(node: _Node[T, Acc], value: T) -> _Node[T, Acc]:
    """
    Pushes `value` onto all existing stacks represented by this trie.
    Implementation: move acc at every node one step down under `value`; recurse.
    """
    # Recurse into children first
    new_kids: Dict[T, _Node[T, Acc]] = {}
    for label, child in node.kids.items():
        pushed_child = _push_node(child, value)
        if not _node_is_empty(pushed_child):
            new_kids[label] = pushed_child

    # Move this node's acc down under `value`
    if node.acc is not None:
        if value in new_kids:
            # Merge the moved acc into the existing child
            target = new_kids[value]
            new_kids[value] = _Node(acc=_merge_acc(target.acc, node.acc), kids=target.kids)
        else:
            new_kids[value] = _Node(acc=node.acc, kids={})
        new_acc = None
    else:
        new_acc = None

    return _Node(acc=new_acc, kids=new_kids)


def _pop_node(node: _Node[T, Acc]) -> _Node[T, Acc]:
    """
    Pops one element from all non-empty stacks.
    - Discard this node's own acc (those were already empty stacks).
    - For each child:
        * Child.acc contributes to this node's new acc (those were length-1 stacks under that child).
        * Recursively pop the child without its acc; keep as a child if it remains non-empty.
    """
    new_acc: Optional[Acc] = None
    new_kids: Dict[T, _Node[T, Acc]] = {}

    for label, child in node.kids.items():
        # Contribution from stacks of length exactly one under `label`
        new_acc = _merge_acc(new_acc, child.acc)

        # Pop deeper stacks: drop child's own acc, then recurse
        child_wo_acc = _Node(acc=None, kids=child.kids)
        popped_child = _pop_node(child_wo_acc)
        if not _node_is_empty(popped_child):
            new_kids[label] = popped_child

    return _Node(acc=new_acc, kids=new_kids)


def _isolate_node_by_top(node: _Node[T, Acc], target: Optional[T], incoming_label: Optional[T] = None) -> _Node[T, Acc]:
    """
    Keeps only stacks whose top element equals `target` (or empty stacks if target is None).
    Does not modify stack contents; prunes accs that don't match the condition.
    """
    keep_acc = (incoming_label is None and target is None) or (incoming_label is not None and incoming_label == target)
    new_acc = node.acc if keep_acc else None

    new_kids: Dict[T, _Node[T, Acc]] = {}
    for label, child in node.kids.items():
        filtered = _isolate_node_by_top(child, target, incoming_label=label)
        if not _node_is_empty(filtered):
            new_kids[label] = filtered

    return _Node(acc=new_acc, kids=new_kids)


def _apply_node(node: _Node[T, Acc], func: Callable[[Acc], Acc]) -> _Node[T, Acc]:
    new_acc = func(node.acc) if node.acc is not None else None
    new_kids: Dict[T, _Node[T, Acc]] = {}
    for label, child in node.kids.items():
        transformed = _apply_node(child, func)
        if not _node_is_empty(transformed):
            new_kids[label] = transformed
    return _Node(acc=new_acc, kids=new_kids)


def _prune_node(node: _Node[T, Acc], predicate: Callable[[Acc], bool]) -> _Node[T, Acc]:
    new_acc = node.acc if (node.acc is not None and predicate(node.acc)) else None
    new_kids: Dict[T, _Node[T, Acc]] = {}
    for label, child in node.kids.items():
        kept_child = _prune_node(child, predicate)
        if not _node_is_empty(kept_child):
            new_kids[label] = kept_child
    return _Node(acc=new_acc, kids=new_kids)


def _peek_node(node: _Node[T, Acc], incoming_label: Optional[T], out: Set[T]) -> None:
    # Root acc represents empty stack (no top), so only record if there is an incoming label.
    if node.acc is not None and incoming_label is not None:
        out.add(incoming_label)
    for label, child in node.kids.items():
        _peek_node(child, label, out)


def _reduce_node_acc(node: _Node[T, Acc]) -> Optional[Acc]:
    total = node.acc
    for child in node.kids.values():
        total = _merge_acc(total, _reduce_node_acc(child))
    return total


def _has_any_stack(lower: Lower[T]) -> bool:
    """
    Fast check for emptiness directly on Lower (avoid building _Node).
    Returns True if any _AccKey sentinel exists anywhere in the trie.
    """
    if isinstance(lower.inner, Leaf):
        return False
    for key, idx_map in lower.inner.children.items():
        if _is_acc_key(key):
            return True
        for child in idx_map.values():
            if _has_any_stack(child):
                return True
    return False


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        """
        Build a LeveledGSS from explicit stacks by constructing a canonical trie:
        - Merge accumulators for identical stacks.
        - Store each stack path bottom->top.
        """
        # Canonicalize input: merge identical stacks
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        # Build _Node trie
        root = _Node[T, Acc](acc=None, kids={})

        def insert(path: List[T], acc: Acc) -> None:
            cur = root
            for item in path:
                nxt = cur.kids.get(item)
                if nxt is None:
                    nxt = _Node[T, Acc](acc=None, kids={})
                    cur.kids[item] = nxt
                cur = nxt
            cur.acc = _merge_acc(cur.acc, acc)

        for key, acc in merged.items():
            insert(list(key), acc)

        lower_root = _node_to_lower(root)
        upper = Upper(Interface(node=lower_root, acc=None))
        return LeveledGSS(inner=upper, empty=None)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        """
        Decode to a canonical, sorted list of stacks from the internal trie.
        """
        def collect(node: _Node[T, Acc], prefix: List[T], out: List[Tuple[List[T], Acc]]) -> None:
            if node.acc is not None:
                out.append((list(prefix), node.acc))
            for label, child in node.kids.items():
                prefix.append(label)
                collect(child, prefix, out)
                prefix.pop()

        lower = self._get_lower()
        node = _lower_to_node(lower)

        results: List[Tuple[List[T], Acc]] = []
        collect(node, [], results)

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
        # Extract the root Lower node (we always use an Interface at the top).
        if isinstance(self.inner.inner, Interface):
            return self.inner.inner.node

        # Defensive fallback: if an UpperBranch exists (shouldn't in normal use), fold it.
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

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        node = _lower_to_node(lower)
        pushed = _push_node(node, value)
        new_lower = _node_to_lower(pushed)
        return LeveledGSS(inner=Upper(Interface(node=new_lower, acc=None)), empty=None)

    def pop(self) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        node = _lower_to_node(lower)
        popped = _pop_node(node)
        if _node_is_empty(popped):
            return LeveledGSS.from_stacks([])
        new_lower = _node_to_lower(popped)
        return LeveledGSS(inner=Upper(Interface(node=new_lower, acc=None)), empty=None)

    def is_empty(self) -> bool:
        lower = self._get_lower()
        return not _has_any_stack(lower)

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        node = _lower_to_node(lower)
        filtered = _isolate_node_by_top(node, value, incoming_label=None)
        if _node_is_empty(filtered):
            return LeveledGSS.from_stacks([])
        new_lower = _node_to_lower(filtered)
        return LeveledGSS(inner=Upper(Interface(node=new_lower, acc=None)), empty=None)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        if not _has_any_stack(lower):
            return LeveledGSS.from_stacks([])
        node = _lower_to_node(lower)
        transformed = _apply_node(node, func)
        if _node_is_empty(transformed):
            return LeveledGSS.from_stacks([])
        new_lower = _node_to_lower(transformed)
        return LeveledGSS(inner=Upper(Interface(node=new_lower, acc=None)), empty=None)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        lower = self._get_lower()
        node = _lower_to_node(lower)
        pruned = _prune_node(node, predicate)
        if _node_is_empty(pruned):
            return LeveledGSS.from_stacks([])
        new_lower = _node_to_lower(pruned)
        return LeveledGSS(inner=Upper(Interface(node=new_lower, acc=None)), empty=None)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        a = self._get_lower()
        b = other._get_lower()
        if not _has_any_stack(a):
            return other
        if not _has_any_stack(b):
            return self
        na = _lower_to_node(a)
        nb = _lower_to_node(b)
        merged = _merge_nodes(na, nb)
        new_lower = _node_to_lower(merged)
        return LeveledGSS(inner=Upper(Interface(node=new_lower, acc=None)), empty=None)

    def peek(self) -> Set[T]:
        lower = self._get_lower()
        node = _lower_to_node(lower)
        out: Set[T] = set()
        _peek_node(node, incoming_label=None, out=out)
        return out

    def reduce_acc(self) -> Optional[Acc]:
        lower = self._get_lower()
        node = _lower_to_node(lower)
        return _reduce_node_acc(node)


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
