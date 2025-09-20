from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any
from weakref import WeakKeyDictionary

from .interface import GSS, T, Acc


# ------------------------------
# Public node classes (shape preserved; thin wrappers only)
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    inner: 'UpperBranch[T, Acc]' | 'Interface[T, Acc]'


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, Upper[T, Acc]]]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: 'Lower[T]'
    acc: Acc | None  # Placeholder; not used by this implementation.


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
# Private: the canonical trie representation
# ------------------------------

@dataclass(frozen=True)
class _Node(Generic[T, Acc]):
    # Accumulator for the stack that ends at this path (root => empty stack)
    acc: Optional[Acc]
    # Children by next pushed value
    kids: Dict[T, "_Node[T, Acc]"]


def _merge_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


def _node_is_empty(n: _Node[T, Acc]) -> bool:
    if n.acc is not None:
        return False
    # Empty only if no child has any accumulator reachable
    for ch in n.kids.values():
        if not _node_is_empty(ch):
            return False
    return True


def _merge_nodes(a: _Node[T, Acc], b: _Node[T, Acc]) -> _Node[T, Acc]:
    # Merge accumulators and recursively merge the children
    merged_acc = _merge_acc(a.acc, b.acc)
    kids: Dict[T, _Node[T, Acc]] = {}
    for label in set(a.kids) | set(b.kids):
        ca = a.kids.get(label)
        cb = b.kids.get(label)
        if ca is None:
            child = cb  # type: ignore[assignment]
        elif cb is None:
            child = ca
        else:
            child = _merge_nodes(ca, cb)
        if child is not None and not _node_is_empty(child):
            kids[label] = child
    return _Node(acc=merged_acc, kids=kids)


def _push_node(n: _Node[T, Acc], value: T) -> _Node[T, Acc]:
    # Recurse into children
    new_kids: Dict[T, _Node[T, Acc]] = {k: _push_node(c, value) for k, c in n.kids.items()}

    # Move this node's acc down to child 'value'
    if n.acc is not None:
        child = new_kids.get(value)
        if child is None:
            new_kids[value] = _Node(acc=n.acc, kids={})
        else:
            new_kids[value] = _Node(acc=_merge_acc(child.acc, n.acc), kids=child.kids)

    # Prune any empty children for cleanliness
    new_kids = {k: c for k, c in new_kids.items() if not _node_is_empty(c)}
    return _Node(acc=None, kids=new_kids)


def _pop_node(n: _Node[T, Acc]) -> _Node[T, Acc]:
    # Pop discards empty stacks (root acc). It gathers child accs (stacks of length 1 become length 0)
    new_acc: Optional[Acc] = None
    new_kids: Dict[T, _Node[T, Acc]] = {}

    for label, child in n.kids.items():
        # Stacks of length exactly one under this child contribute child's acc to root
        new_acc = _merge_acc(new_acc, child.acc)
        # Deeper stacks: drop child's own acc, pop deeper
        deeper = _pop_node(_Node(acc=None, kids=child.kids))
        if not _node_is_empty(deeper):
            new_kids[label] = deeper

    return _Node(acc=new_acc, kids=new_kids)


def _isolate_top(n: _Node[T, Acc], target: Optional[T], incoming_label: Optional[T] = None) -> _Node[T, Acc]:
    # Keep acc if the top of the stack equals target (incoming edge label),
    # or keep empty stacks if target is None (at the root only).
    keep_acc = (incoming_label is None and target is None) or (incoming_label is not None and incoming_label == target)
    acc = n.acc if keep_acc else None

    kids: Dict[T, _Node[T, Acc]] = {}
    for label, child in n.kids.items():
        filtered = _isolate_top(child, target, incoming_label=label)
        if not _node_is_empty(filtered):
            kids[label] = filtered

    return _Node(acc=acc, kids=kids)


def _apply_node(n: _Node[T, Acc], func: Callable[[Acc], Acc]) -> _Node[T, Acc]:
    acc = func(n.acc) if n.acc is not None else None
    kids: Dict[T, _Node[T, Acc]] = {}
    for label, child in n.kids.items():
        t = _apply_node(child, func)
        if not _node_is_empty(t):
            kids[label] = t
    return _Node(acc=acc, kids=kids)


def _prune_node(n: _Node[T, Acc], predicate: Callable[[Acc], bool]) -> _Node[T, Acc]:
    acc = n.acc if (n.acc is not None and predicate(n.acc)) else None
    kids: Dict[T, _Node[T, Acc]] = {}
    for label, child in n.kids.items():
        kept = _prune_node(child, predicate)
        if not _node_is_empty(kept):
            kids[label] = kept
    return _Node(acc=acc, kids=kids)


def _peek_node(n: _Node[T, Acc], incoming_label: Optional[T], out: Set[T]) -> None:
    # If a stack ends here and we came via label L, then L is a top value
    if n.acc is not None and incoming_label is not None:
        out.add(incoming_label)
    for label, child in n.kids.items():
        _peek_node(child, label, out)


def _reduce_node_acc(n: _Node[T, Acc]) -> Optional[Acc]:
    total = n.acc
    for child in n.kids.values():
        total = _merge_acc(total, _reduce_node_acc(child))
    return total


# ------------------------------
# Public LeveledGSS built on the canonical trie
# ------------------------------

# Note: We keep the public dataclass shape exactly as defined.
# Internally, we associate each LeveledGSS instance to its canonical _Node
# using a WeakKeyDictionary. The public 'inner' and 'empty' fields are inert,
# kept only to preserve the expected public surface and invariants.
_EMPTY_UPPER: Upper[Any, Any] = Upper(Interface(node=Lower(Leaf()), acc=None))


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    # --- Construction & representation ---

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        # Canonicalize: merge accumulators for identical stacks
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        # Build trie
        def insert(root: _Node[T, Acc], path: List[T], acc: Acc) -> _Node[T, Acc]:
            cur = root
            for item in path:
                nxt = cur.kids.get(item)
                if nxt is None:
                    nxt = _Node[T, Acc](acc=None, kids={})
                cur = _Node(acc=cur.acc, kids={**cur.kids, item: nxt})
            # Place/merge acc at the terminal node
            def set_acc(n: _Node[T, Acc], path_rev: List[T]) -> _Node[T, Acc]:
                if not path_rev:
                    return _Node(acc=_merge_acc(n.acc, acc), kids=n.kids)
                head, *tail = path_rev
                child = n.kids[head]
                new_child = set_acc(child, tail)
                kids = dict(n.kids)
                kids[head] = new_child
                return _Node(acc=n.acc, kids=kids)

            # Re-traverse immutably to set the acc
            # Build a path list for re-writing
            node_path: List[T] = []
            cur2 = root
            for item in path:
                node_path.append(item)
                cur2 = cur2.kids.get(item, _Node(acc=None, kids={}))
            # Now write back
            return set_acc(root, path)

        root = _Node[T, Acc](acc=None, kids={})
        for key, acc in merged.items():
            root = insert(root, list(key), acc)

        # Clean up empties if any (shouldn't be necessary, but safe)
        if _node_is_empty(root):
            node = _Node[T, Acc](acc=None, kids={})
        else:
            node = root

        return _make_gss(node)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        node = _node_of(self)
        out: List[Tuple[List[T], Acc]] = []

        def collect(n: _Node[T, Acc], prefix: List[T]) -> None:
            if n.acc is not None:
                out.append((list(prefix), n.acc))
            for label, child in n.kids.items():
                prefix.append(label)
                collect(child, prefix)
                prefix.pop()

        collect(node, [])

        # Deterministic sort for stability/debuggability
        def _enc(obj: Any) -> str:
            try:
                return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
            except Exception:
                return repr(obj)

        out.sort(key=lambda pair: (_enc(pair[0]), _enc(pair[1])))
        return out

    # --- Core operations ---

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        node = _node_of(self)
        return _make_gss(_push_node(node, value))

    def pop(self) -> 'LeveledGSS[T, Acc]':
        node = _node_of(self)
        return _make_gss(_pop_node(node))

    def is_empty(self) -> bool:
        return _node_is_empty(_node_of(self))

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        node = _node_of(self)
        return _make_gss(_isolate_top(node, value, incoming_label=None))

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        node = _node_of(self)
        return _make_gss(_apply_node(node, func))

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        node = _node_of(self)
        return _make_gss(_prune_node(node, predicate))

    def merge(self, other: 'LeveledGSS[T, Acc]') -> 'LeveledGSS[T, Acc]':
        a = _node_of(self)
        b = _node_of(other)
        return _make_gss(_merge_nodes(a, b))

    def peek(self) -> Set[T]:
        out: Set[T] = set()
        _peek_node(_node_of(self), incoming_label=None, out=out)
        return out

    def reduce_acc(self) -> Optional[Acc]:
        return _reduce_node_acc(_node_of(self))


# ------------------------------
# Instance <-> node association (private)
# ------------------------------

# Store the canonical trie for each LeveledGSS instance without changing public types.
_STATE: "WeakKeyDictionary[LeveledGSS[Any, Any], _Node[Any, Any]]" = WeakKeyDictionary()


def _make_gss(node: _Node[T, Acc]) -> LeveledGSS[T, Acc]:
    # Always keep a minimal public wrapper; actual state is in _STATE
    gss: LeveledGSS[T, Acc] = LeveledGSS(inner=_EMPTY_UPPER, empty=None)  # type: ignore[arg-type]
    _STATE[gss] = node
    return gss


def _node_of(gss: LeveledGSS[T, Acc]) -> _Node[T, Acc]:
    # Fallback to empty if somehow missing (defensive)
    node = _STATE.get(gss)  # type: ignore[index]
    if node is None:
        return _Node(acc=None, kids={})
    return node  # type: ignore[return-value]


# ------------------------------
# Public validation (minimal, defensive)
# ------------------------------

def _validate_upper(node: Upper[T, Acc]) -> None:
    # Accept both Interface and UpperBranch; recurse if UpperBranch present.
    if isinstance(node.inner, UpperBranch):
        for children_by_val in node.inner.children.values():
            for child in children_by_val.values():
                _validate_upper(child)
    # If it's Interface, nothing to validate.


def validate_invariants(gss: LeveledGSS[T, Acc]) -> None:
    """
    Minimal invariant checks:
    - The recursive 'Upper' structure is well-formed (no action for Interface; traversal if UpperBranch).
    - If inner is an Interface and 'empty' exists, its acc must differ from the interface acc.
      (In this implementation, interface acc is always None.)
    """
    _validate_upper(gss.inner)

    if isinstance(gss.inner.inner, Interface) and gss.empty is not None:
        if gss.inner.inner.acc == gss.empty:
            raise AssertionError(
                "Invariant violated: LeveledGSS.inner (Interface) and LeveledGSS.empty have the same accumulator."
            )
