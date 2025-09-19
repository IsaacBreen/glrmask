from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Any, Iterable

from .interface import GSS, T, Acc


# ------------------------------
# Public node classes (shape preserved; used only as a thin wrapper)
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
    acc: Acc | None  # Placeholder; not used by this implementation.


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
# Private: canonical trie at the heart of the implementation
# ------------------------------

@dataclass
class _Node(Generic[T, Acc]):
    # acc holds the accumulator for the stack that ends exactly at this path (root => empty stack)
    acc: Optional[Acc]
    # kids maps the next pushed value to the child node
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
    for ch in n.kids.values():
        if not _node_is_empty(ch):
            return False
    return True


def _merge_nodes(a: _Node[T, Acc], b: _Node[T, Acc]) -> _Node[T, Acc]:
    merged_acc = _merge_acc(a.acc, b.acc)
    all_labels = set(a.kids.keys()) | set(b.kids.keys())
    kids: Dict[T, _Node[T, Acc]] = {}
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
            kids[label] = child
    return _Node(acc=merged_acc, kids=kids)


def _push_node(n: _Node[T, Acc], value: T) -> _Node[T, Acc]:
    # Recurse into children
    new_kids: Dict[T, _Node[T, Acc]] = {}
    for label, child in n.kids.items():
        pushed_child = _push_node(child, value)
        if not _node_is_empty(pushed_child):
            new_kids[label] = pushed_child

    # Move this node's acc one step down on 'value'
    if n.acc is not None:
        if value in new_kids:
            target = new_kids[value]
            new_kids[value] = _Node(acc=_merge_acc(target.acc, n.acc), kids=target.kids)
        else:
            new_kids[value] = _Node(acc=n.acc, kids={})

    return _Node(acc=None, kids=new_kids)


def _pop_node(n: _Node[T, Acc]) -> _Node[T, Acc]:
    # Discard n.acc (empty stacks do not survive pop)
    new_acc: Optional[Acc] = None
    new_kids: Dict[T, _Node[T, Acc]] = {}

    for label, child in n.kids.items():
        # Stacks of length exactly one under this child contribute child's acc
        new_acc = _merge_acc(new_acc, child.acc)

        # Pop deeper stacks: drop child's own acc and recurse
        deeper = _pop_node(_Node(acc=None, kids=child.kids))
        if not _node_is_empty(deeper):
            new_kids[label] = deeper

    return _Node(acc=new_acc, kids=new_kids)


def _isolate_top(n: _Node[T, Acc], target: Optional[T], incoming_label: Optional[T] = None) -> _Node[T, Acc]:
    # Keep acc only if this node represents stacks whose top equals target,
    # or keep empty stacks if target is None (incoming_label is None for root).
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
# Private: Lower <-> _Node adapter (kept minimal, deterministic)
# ------------------------------

class _AccKey(Generic[Acc]):
    """
    Sentinel used to encode the presence of an accumulator at a node in LowerBranch.children.
    Kept simple: rely on object identity so distinct sentinels never collide.
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


def _is_acc_key(x: Any) -> bool:
    return isinstance(x, _AccKey)


def _lower_to_node(lower: Lower[T]) -> _Node[T, Acc]:
    if isinstance(lower.inner, Leaf):
        # Shouldn't appear at root, but treat as empty for robustness.
        return _Node(acc=None, kids={})

    acc: Optional[Acc] = None
    kids: Dict[T, _Node[T, Acc]] = {}

    for key, idx_map in lower.inner.children.items():
        if _is_acc_key(key):
            # Merge all accs attached at this node (normally one)
            for _child in idx_map.values():
                acc = _merge_acc(acc, key.acc)  # type: ignore[attr-defined]
        else:
            # Merge potential duplicates by index
            merged_child: Optional[_Node[T, Acc]] = None
            for child_lower in idx_map.values():
                decoded = _lower_to_node(child_lower)
                merged_child = decoded if merged_child is None else _merge_nodes(merged_child, decoded)
            if merged_child is not None and not _node_is_empty(merged_child):
                if key in kids:
                    kids[key] = _merge_nodes(kids[key], merged_child)
                else:
                    kids[key] = merged_child

    return _Node(acc=acc, kids=kids)


def _node_to_lower(node: _Node[T, Acc]) -> Lower[T]:
    children: Dict[Any, Dict[int, Lower[T]]] = {}

    if node.acc is not None:
        children[_AccKey(node.acc)] = {0: Lower(Leaf())}

    for label, child in node.kids.items():
        if _node_is_empty(child):
            continue
        children[label] = {0: _node_to_lower(child)}

    return Lower(LowerBranch(children=children))


# ------------------------------
# Public LeveledGSS built on top of the canonical trie
# ------------------------------

@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Upper[T, Acc]
    empty: Optional[Acc]

    # --- Construction & representation ---

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        # Canonicalize input: merge accumulators for identical stacks.
        merged: Dict[Tuple[Any, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merged[key].merge(acc)
            else:
                merged[key] = acc

        # Build trie
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

        # Encode to public wrapper
        lower_root = _node_to_lower(root)
        return LeveledGSS(inner=Upper(Interface(node=lower_root, acc=None)), empty=None)

    def _as_node(self) -> _Node[T, Acc]:
        # Extract underlying node (we always store Interface at top)
        if isinstance(self.inner.inner, Interface):
            return _lower_to_node(self.inner.inner.node)

        # Defensive: fold UpperBranch if ever present
        def upper_to_lower(u: Upper[T, Acc]) -> Lower[T]:
            if isinstance(u.inner, Interface):
                return u.inner.node
            lb_children: Dict[Any, Dict[int, Lower[T]]] = {}
            br: UpperBranch[T, Acc] = u.inner
            for label, idx_map in br.children.items():
                for child in idx_map.values():
                    lb_children.setdefault(label, {})[0] = upper_to_lower(child)
            return Lower(LowerBranch(children=lb_children))

        return _lower_to_node(upper_to_lower(self.inner))

    @staticmethod
    def _from_node(node: _Node[T, Acc]) -> LeveledGSS[T, Acc]:
        if _node_is_empty(node):
            return LeveledGSS.from_stacks([])
        lower = _node_to_lower(node)
        return LeveledGSS(inner=Upper(Interface(node=lower, acc=None))), None  # type: ignore[misc]

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        node = self._as_node()
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

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        node = self._as_node()
        pushed = _push_node(node, value)
        return LeveledGSS._from_node(pushed)

    def pop(self) -> LeveledGSS[T, Acc]:
        node = self._as_node()
        popped = _pop_node(node)
        return LeveledGSS._from_node(popped)

    def is_empty(self) -> bool:
        return _node_is_empty(self._as_node())

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        node = self._as_node()
        filtered = _isolate_top(node, value, incoming_label=None)
        return LeveledGSS._from_node(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        node = self._as_node()
        transformed = _apply_node(node, func)
        return LeveledGSS._from_node(transformed)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        node = self._as_node()
        pruned = _prune_node(node, predicate)
        return LeveledGSS._from_node(pruned)

    def merge(self, other: LeveledGSS[T, Acc]) -> LeveledGSS[T, Acc]:
        a = self._as_node()
        b = other._as_node()
        merged = _merge_nodes(a, b)
        return LeveledGSS._from_node(merged)

    def peek(self) -> Set[T]:
        out: Set[T] = set()
        _peek_node(self._as_node(), incoming_label=None, out=out)
        return out

    def reduce_acc(self) -> Optional[Acc]:
        return _reduce_node_acc(self._as_node())


# ------------------------------
# Public validation (kept minimal and defensive)
# ------------------------------

def _validate_upper(node: Upper[T, Acc]) -> None:
    # This implementation always keeps an Interface at the top, but we accept UpperBranch defensively.
    if isinstance(node.inner, UpperBranch):
        # Recurse into all children; do not enforce additional invariants here
        for children_by_val in node.inner.children.values():
            for child in children_by_val.values():
                _validate_upper(child)
    # If it's Interface, nothing to validate here.


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
