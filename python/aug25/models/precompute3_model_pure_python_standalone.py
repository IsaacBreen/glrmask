from __future__ import annotations

import json
import heapq
import collections
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Any,
    Generator,
    TypeVar,
    Iterator,
    Iterable,
    Union,
    Protocol,
)
from collections import defaultdict

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi


# ------------------------------
# Minimal Mergeable protocol and ReferenceGSS used by LeveledGSS
# ------------------------------

T = TypeVar("T")


class Mergeable(Protocol):
    def merge(self, other: Any) -> Any:
        ...


Acc = TypeVar("Acc", bound=Mergeable)
NewAcc = TypeVar("NewAcc", bound=Mergeable)


class ReferenceGSS(Generic[T, Acc]):
    """
    Minimal canonicalizer used by LeveledGSS.from_stacks() and to_stacks().
    It merges accumulators for identical stacks using acc.merge().
    """

    def __init__(self, stacks: List[Tuple[List[T], Acc]]):
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in merged:
                prev = merged[key]
                merged[key] = prev if prev is acc else prev.merge(acc)
            else:
                merged[key] = acc
        self._map: Dict[Tuple[T, ...], Acc] = merged
        self._stacks: List[Tuple[List[T], Acc]] = [(list(k), v) for k, v in merged.items()]

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        # Canonical order: by length then lexicographically by values
        items = sorted(self._map.items(), key=lambda kv: (len(kv[0]), kv[0]))
        return [(list(k), v) for k, v in items]


# ------------------------------
# LeveledGSS implementation (inlined)
# ------------------------------

# Internal node classes
type Upper[T, Acc] = "UpperBranch[T, Acc]" | "Interface[T, Acc]"


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T, Acc]):
    children: Dict[T, Dict[int, Upper[T, Acc]]]
    empty: Optional[Acc]
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Generator[Upper[T, Acc], None, None]:
        """Returns an iterator over all child nodes."""
        for children_at_depth in self.children.values():
            yield from children_at_depth.values()


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    children: Dict[T, Dict[int, "Lower[T]"]]
    acc: Acc
    empty: Optional[Acc]
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Iterator["Lower[T]"]:
        for v_children in self.children.values():
            yield from v_children.values()


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    children: Dict[T, Dict[int, "Lower[T]"]]
    empty: bool
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Iterator["Lower[T]"]:
        for v_children in self.children.values():
            yield from v_children.values()


@dataclass(frozen=True, eq=True)
class LeveledGSS(Generic[T, Acc]):
    inner: Upper[T, Acc]

    @classmethod
    def from_stacks(cls, stacks: List[Tuple[List[T], Acc]]) -> "LeveledGSS[T, Acc]":
        # Use ReferenceGSS to canonicalize stacks by merging accumulators.
        # We access _stacks directly to avoid the sorting done by to_stacks().
        canonical_stacks = ReferenceGSS(stacks)._stacks

        empty_acc: Optional[Acc] = None
        # A simple trie: { val: { "end": Optional[Acc], "sub": <subtrie> } }
        trie: Dict[T, Dict[str, Any]] = {}

        for vals, acc in canonical_stacks:
            if not vals:
                empty_acc = acc
                continue
            node = trie
            for i, v in enumerate(reversed(vals)):
                entry = node.setdefault(v, {"end": None, "sub": {}})
                if i == len(vals) - 1:
                    # Since input is canonical, there's no need to merge.
                    entry["end"] = acc
                else:
                    node = entry["sub"]

        def build(d: Dict[T, Dict[str, Any]], root_empty: Optional[Acc] = None) -> Upper[T, Acc]:
            # Build children recursively
            children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            all_child_nodes: List[Upper[T, Acc]] = []
            for v, e in d.items():
                nodes_for_v: List[Upper[T, Acc]] = []
                end_acc = e.get("end")
                sub = e.get("sub", {})
                if end_acc is not None:
                    # Represent a leaf/end-of-stack using an UpperBranch with the accumulator in 'empty', then promote
                    nodes_for_v.append(try_promote(UpperBranch(children={}, empty=end_acc)))
                if sub:
                    nodes_for_v.append(build(sub))
                if nodes_for_v:
                    children[v] = {n._max_depth: n for n in nodes_for_v}
                    all_child_nodes.extend(nodes_for_v)

            # Check for promotion
            if all(isinstance(child, Interface) for child in all_child_nodes):
                accs: Set[Acc] = set()
                for c in all_child_nodes:
                    # This must be an Interface, based on the check above.
                    accs.add(c.acc)
                    if c.empty is not None:
                        accs.add(c.empty)

                if root_empty is not None:
                    accs.add(root_empty)

                if len(accs) <= 1:
                    the_acc = accs.pop() if accs else None
                    if the_acc is None:
                        # This is a truly empty GSS.
                        return UpperBranch(children={}, empty=None)

                    def build_lower(sub_d: Dict[T, Dict[str, Any]]) -> Lower[T]:
                        l_children: Dict[T, Dict[int, Lower[T]]] = {}
                        for v_l, e_l in sub_d.items():
                            sub_l = e_l.get("sub", {})
                            has_end = e_l.get("end") is not None
                            sub_lower = build_lower(sub_l) if sub_l else Lower(children={}, empty=False)
                            node_for_v = Lower(children=sub_lower.children, empty=has_end)
                            l_children[v_l] = {node_for_v._max_depth: node_for_v}
                        return Lower(children=l_children, empty=False)

                    lower_tree = build_lower(d)
                    return Interface(
                        children=lower_tree.children,
                        acc=the_acc,
                        empty=root_empty
                    )

            return UpperBranch(children=children, empty=root_empty)

        return LeveledGSS(build(trie, empty_acc))

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []

        def dfs_lower(l: Lower[T], pref: List[T], acc: Acc) -> None:
            if l.empty:
                res.append((list(reversed(pref)), acc))
            for v, kids in l.children.items():
                for child in kids.values():
                    dfs_lower(child, pref + [v], acc)

        def dfs_upper(u: Upper[T, Acc], pref: List[T]) -> None:
            if isinstance(u, UpperBranch):
                if u.empty is not None:
                    res.append((list(reversed(pref)), u.empty))
                for v, kids in u.children.items():
                    for child in kids.values():
                        dfs_upper(child, pref + [v])
            elif isinstance(u, Interface):
                # The interface's `empty` slot is for a stack ending at `pref`.
                if u.empty is not None:
                    res.append((list(reversed(pref)), u.empty))

                if not u.children:
                    # If there are no lower children, this interface represents the end of a stack
                    # with accumulator u.acc.
                    res.append((list(reversed(pref)), u.acc))
                else:
                    # The interface's `children` are for stacks extending `pref`.
                    # All these stacks share accumulator `u.acc`.
                    for v, kids in u.children.items():
                        for child in kids.values():
                            dfs_lower(child, pref + [v], u.acc)

        dfs_upper(self.inner, [])

        # The internal representation is canonical. We use ReferenceGSS to sort
        # the stacks into a canonical list representation.
        return ReferenceGSS(res).to_stacks()

    def push(self, value: T) -> "LeveledGSS[T, Acc]":
        if self.is_empty():
            return self
        if isinstance(self.inner, Interface):
            lower_node = Lower(children=self.inner.children, empty=self.inner.empty is not None)
            new_children = {value: {lower_node._max_depth: lower_node}}
            return LeveledGSS(Interface(children=new_children, acc=self.inner.acc, empty=None))
        else:
            return LeveledGSS(UpperBranch(children={value: {self.inner._max_depth: self.inner}}, empty=None))

    def pop(self) -> "LeveledGSS[T, Acc]":
        if isinstance(self.inner, Interface):
            all_children = list(self.inner._all_children())
            merged = reduce(merge_lower, all_children[1:], all_children[0]) if all_children else Lower(children={}, empty=False)
            merged_empty = self.inner.acc if merged.empty else None
            if merged_empty is None and not merged.children:
                return LeveledGSS(UpperBranch(children={}, empty=merged_empty))
            else:
                return LeveledGSS(Interface(children=merged.children, acc=self.inner.acc, empty=merged_empty))
        else:
            all_children = list(self.inner._all_children())
            merged = reduce(merge_upper, all_children[1:], all_children[0]) if all_children else UpperBranch(children={}, empty=None)
            return LeveledGSS(try_promote(merged))

    def popn(self, n: int) -> "LeveledGSS[T, Acc]":
        if n <= 0:
            return self
        if self.is_empty():
            return self

        # Memoization caches to avoid recomputing on shared subtrees
        memo_upper: Dict[Tuple[int, int], Upper[T, Acc]] = {}
        memo_lower: Dict[Tuple[int, int], Lower[T]] = {}

        def _popn_lower(node: Lower[T], k: int) -> Lower[T]:
            """Recursively pop k levels from a Lower node."""
            if k == 0:
                return node

            key = (id(node), k)
            if key in memo_lower:
                return memo_lower[key]

            all_children = list(node._all_children())
            if not all_children:
                res = Lower(children={}, empty=False)
                memo_lower[key] = res
                return res

            # Recursively pop k-1 from all children and merge the results
            popped_children = [_popn_lower(child, k - 1) for child in all_children]
            res = reduce(merge_lower, popped_children[1:], popped_children[0])
            memo_lower[key] = res
            return res

        def _popn_upper(node: Upper[T, Acc], k: int) -> Upper[T, Acc]:
            """Recursively pop k levels from an Upper node."""
            if k == 0:
                return node

            key = (id(node), k)
            if key in memo_upper:
                return memo_upper[key]

            # Base case for recursion: no children to pop from
            all_children = list(node._all_children())
            if not all_children:
                res = UpperBranch(children={}, empty=None)
                memo_upper[key] = res
                return res

            if isinstance(node, Interface):
                # For an Interface, we pop k-1 levels from its Lower children
                popped_lower_children = [_popn_lower(child, k - 1) for child in all_children]
                merged = reduce(merge_lower, popped_lower_children[1:], popped_lower_children[0])

                # The result is a new Interface with the same accumulator
                new_empty = node.acc if merged.empty else None
                if not merged.children and new_empty is None:
                    res = UpperBranch(children={}, empty=None)
                else:
                    res = Interface(children=merged.children, acc=node.acc, empty=new_empty)
            else:  # UpperBranch
                # For an UpperBranch, we pop k-1 levels from its Upper children
                popped_upper_children = [_popn_upper(child, k - 1) for child in all_children]
                merged = reduce(merge_upper, popped_upper_children[1:], popped_upper_children[0])
                res = try_promote(merged)

            memo_upper[key] = res
            return res

        return LeveledGSS(_popn_upper(self.inner, n))

    def is_empty(self) -> bool:
        # An empty GSS is represented by an UpperBranch with no children and no empty accumulator.
        if isinstance(self.inner, UpperBranch):
            return not self.inner.children and self.inner.empty is None
        # An Interface always represents at least one stack, as it has an accumulator.
        return False

    def isolate(self, value: Optional[T]) -> "LeveledGSS[T, Acc]":
        if value is None:
            # Keep only empty stacks.
            if isinstance(self.inner, UpperBranch):
                empty_acc = self.inner.empty
            else:
                empty_acc = self.inner.empty
            new_root = UpperBranch(children={}, empty=empty_acc)
            # Promote to canonical form if applicable (avoids validator errors).
            return LeveledGSS(try_promote(new_root))

        # Keep stacks with `value` at the top.
        if isinstance(self.inner, UpperBranch):
            filtered_children = {value: self.inner.children[value]} if value in self.inner.children else {}
            return LeveledGSS(try_promote(UpperBranch(children=filtered_children, empty=None)))
        else:
            if value not in self.inner.children:
                return LeveledGSS(UpperBranch(children={}, empty=None))
            else:
                filtered_children = {value: self.inner.children[value]} if value in self.inner.children else {}
                return LeveledGSS(Interface(children=filtered_children, acc=self.inner.acc, empty=None))

    def isolate_many(self, values: Iterable[Optional[T]]) -> "LeveledGSS[T, Acc]":
        values_set = set(values)

        new_empty: Optional[Acc] = None
        if None in values_set and isinstance(self.inner, (UpperBranch, Interface)):
            new_empty = self.inner.empty

        if isinstance(self.inner, UpperBranch):
            filtered_children = {
                v: kids for v, kids in self.inner.children.items() if v in values_set
            }
            new_inner = try_promote(UpperBranch(children=filtered_children, empty=new_empty))
            return LeveledGSS(new_inner)
        else:  # The root is an Interface
            filtered_children = {
                v: kids for v, kids in self.inner.children.items() if v in values_set
            }

            if filtered_children:
                # Children remain, so we build an Interface.
                new_inner = Interface(children=filtered_children, acc=self.inner.acc, empty=new_empty)
                return LeveledGSS(new_inner)
            else:
                # No children remain. The result only contains the empty stack (if requested).
                # This is represented by an UpperBranch.
                new_inner = try_promote(UpperBranch(children={}, empty=new_empty))
                return LeveledGSS(new_inner)

    def apply(self, func: Callable[[Acc], NewAcc], memo: Optional[Dict[int, Any]] = None) -> "LeveledGSS[T, NewAcc]":
        if memo is None:
            memo = {}

        def transform(node: Upper[T, Acc]) -> Upper[T, NewAcc]:
            if id(node) in memo:
                return memo[id(node)]  # type: ignore[return-value]

            if isinstance(node, Interface):
                new_acc = func(node.acc)
                new_empty = func(node.empty) if node.empty is not None else None
                res = Interface(children=node.children, acc=new_acc, empty=new_empty)
                memo[id(node)] = res
                return res

            # It's an UpperBranch
            new_empty = func(node.empty) if node.empty is not None else None
            new_children: Dict[T, Dict[int, Upper[T, NewAcc]]] = {}

            for v, kids in node.children.items():
                new_kids_for_v: Dict[int, Upper[T, NewAcc]] = {}
                for d, child in kids.items():
                    new_child = transform(child)
                    new_kids_for_v[new_child._max_depth] = new_child
                new_children[v] = new_kids_for_v

            res = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res)  # type: ignore[arg-type]
            memo[id(node)] = promoted
            return promoted  # type: ignore[return-value]

        return LeveledGSS(transform(self.inner))  # type: ignore[type-var]

    def prune(self, predicate: Callable[[Acc], bool]) -> "LeveledGSS[T, Acc]":
        memo: Dict[int, Optional[Upper[T, Acc]]] = {}

        def transform(node: Upper[T, Acc]) -> Optional[Upper[T, Acc]]:
            if id(node) in memo:
                return memo[id(node)]

            if isinstance(node, Interface):
                keep_acc = predicate(node.acc)
                keep_empty = node.empty is not None and predicate(node.empty)
                new_empty = node.empty if keep_empty else None

                if keep_acc and new_empty == node.empty:
                    memo[id(node)] = node
                    return node

                if not keep_acc and not keep_empty:
                    memo[id(node)] = None
                    return None

                if not keep_acc and keep_empty:
                    res = UpperBranch(children={}, empty=new_empty)
                    promoted = try_promote(res)
                    memo[id(node)] = promoted
                    return promoted

                # keep_acc is True, but empty might have been pruned.
                res = Interface(children=node.children, acc=node.acc, empty=new_empty)
                memo[id(node)] = res
                return res

            # It's an UpperBranch
            new_empty = node.empty if node.empty is not None and predicate(node.empty) else None
            changed = new_empty != node.empty

            new_children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
            for v, kids in node.children.items():
                new_kids_for_v: Dict[int, Upper[T, Acc]] = {}
                child_map_changed = False
                for d, child in kids.items():
                    new_child = transform(child)
                    if new_child is not child:
                        child_map_changed = True
                    if new_child is not None:
                        new_kids_for_v[new_child._max_depth] = new_child

                if len(new_kids_for_v) != len(kids):
                    child_map_changed = True

                if child_map_changed:
                    changed = True
                    if new_kids_for_v:
                        new_children[v] = new_kids_for_v
                else:
                    new_children[v] = kids  # Reuse

            if not changed:
                memo[id(node)] = node
                return node

            if not new_children and new_empty is None:
                memo[id(node)] = None
                return None

            res = UpperBranch(children=new_children, empty=new_empty)
            promoted = try_promote(res)
            memo[id(node)] = promoted
            return promoted

        res_inner = transform(self.inner)
        if res_inner is None:
            return LeveledGSS(UpperBranch(children={}, empty=None))
        return LeveledGSS(res_inner)

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[NewAcc]]) -> "LeveledGSS[T, NewAcc]":
        """
        Fast single-pass implementation of apply_and_prune for LeveledGSS.
        - mutator(acc) -> Optional[NewAcc]
            * return None to prune stacks carrying `acc`
            * return NewAcc (possibly unchanged) to keep/update stacks
        This fuses the behavior of `apply` and `prune` and minimizes reconstruction.
        """
        acc_cache: Dict[int, Optional[NewAcc]] = {}

        def mutate_acc(a: Acc) -> Optional[NewAcc]:
            k = id(a)
            if k in acc_cache:
                return acc_cache[k]
            r = mutator(a)
            acc_cache[k] = r
            return r

        memo: Dict[int, Optional[Upper[T, NewAcc]]] = {}

        def transform(node: Upper[T, Acc]) -> Optional[Upper[T, NewAcc]]:
            nid = id(node)
            if nid in memo:
                return memo[nid]

            if isinstance(node, Interface):
                # Mutate/prune the primary accumulator
                new_acc_opt = mutate_acc(node.acc)
                # Mutate/prune the explicit empty, if present
                new_empty_opt = mutate_acc(node.empty) if node.empty is not None else None

                keep_acc = new_acc_opt is not None
                keep_empty = new_empty_opt is not None

                if not keep_acc and not keep_empty:
                    memo[nid] = None
                    return None

                if not keep_acc and keep_empty:
                    # Acc is pruned, but the interface's explicit empty survives as a terminal stack.
                    # Promote the leaf to maintain canonical form.
                    res = UpperBranch(children={}, empty=new_empty_opt)
                    promoted = try_promote(res)  # type: ignore[arg-type]
                    memo[nid] = promoted
                    return promoted

                # keep_acc is True
                new_acc = new_acc_opt  # type: ignore[assignment]
                # Detect if anything changed; children are reused verbatim.
                res = Interface(children=node.children, acc=new_acc, empty=new_empty_opt)
                memo[nid] = res
                return res

            # UpperBranch
            new_empty_opt = mutate_acc(node.empty) if node.empty is not None else None
            new_children: Dict[T, Dict[int, Upper[T, NewAcc]]] = {}

            for v, kids in node.children.items():
                new_kids_for_v: Dict[int, Upper[T, NewAcc]] = {}
                for d, child in kids.items():
                    new_child = transform(child)
                    if new_child is not None:
                        new_kids_for_v[new_child._max_depth] = new_child
                if new_kids_for_v:
                    new_children[v] = new_kids_for_v

            if not new_children and new_empty_opt is None:
                memo[nid] = None
                return None

            res = UpperBranch(children=new_children, empty=new_empty_opt)
            promoted = try_promote(res)  # type: ignore[arg-type]
            memo[nid] = promoted
            return promoted

        res_inner = transform(self.inner)
        if res_inner is None:
            return LeveledGSS(UpperBranch(children={}, empty=None))  # type: ignore[arg-type]
        return LeveledGSS(res_inner)

    def merge(self, other: "LeveledGSS[T, Acc]") -> "LeveledGSS[T, Acc]":
        return LeveledGSS(merge_upper(self.inner, other.inner))

    @staticmethod
    def merge_many(gss_list: List["LeveledGSS[T, Acc]"]) -> "LeveledGSS[T, Acc]":
        if not gss_list:
            return LeveledGSS(UpperBranch(children={}, empty=None))
        it = iter(gss_list)
        first = next(it)
        inner = first.inner
        for g in it:
            inner = merge_upper(inner, g.inner)
        return LeveledGSS(inner)

    def peek(self) -> Set[T]:
        if isinstance(self.inner, Interface):
            return set(self.inner.children.keys())
        else:
            return set(self.inner.children.keys())

    def reduce_acc(self) -> Optional[Acc]:
        # 1. Collect unique accumulator objects by their ID to avoid redundant merges
        # of the same object, and to be proportional to unique accs, not total stacks.
        unique_acc_objects: Dict[int, Acc] = {}
        visited_nodes: Set[int] = set()

        # Use a queue for iterative traversal to avoid recursion depth issues.
        queue: List[Upper[T, Acc]] = [self.inner]

        while queue:
            node = queue.pop()
            node_id = id(node)
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)

            if isinstance(node, Interface):
                unique_acc_objects[id(node.acc)] = node.acc
                if node.empty is not None:
                    unique_acc_objects[id(node.empty)] = node.empty
                # No need to traverse children, they are Lower nodes without accumulators.
            elif isinstance(node, UpperBranch):
                if node.empty is not None:
                    unique_acc_objects[id(node.empty)] = node.empty
                # Recurse into children.
                for children_at_depth in node.children.values():
                    for child in children_at_depth.values():
                        queue.append(child)

        accumulators = list(unique_acc_objects.values())

        # 2. Merge the collected unique accumulators.
        if not accumulators:
            return None

        if len(accumulators) == 1:
            return accumulators[0]

        return reduce(_merge_acc, accumulators)


Node = TypeVar("Node")
AccPromote = TypeVar("AccPromote", bound=Mergeable)


def _merge_optional_acc(a: Optional[Acc], b: Optional[Acc]) -> Optional[Acc]:
    if a is None:
        return b
    if b is None:
        return a
    return a.merge(b)


def _merge_acc(a: Acc, b: Acc) -> Acc:
    return a if a is b else a.merge(b)


def _merge_children_by_depth(
    c1: Dict[T, Dict[int, Node]],
    c2: Dict[T, Dict[int, Node]],
    merge_func: Callable[[Node, Node], Node],
) -> Dict[T, Dict[int, Node]]:
    if c1 is c2:
        return c1
    merged_children: Dict[T, Dict[int, Node]] = {}
    all_vals = c1.keys() | c2.keys()
    for v in all_vals:
        nodes_by_depth: Dict[int, list[Node]] = defaultdict(list)
        children_c1 = c1.get(v, {}).items()
        children_c2 = c2.get(v, {}).items()
        for depth, child in chain(children_c1, children_c2):
            nodes_by_depth[depth].append(child)
        if not nodes_by_depth:
            continue
        v_out = {
            (merged := reduce(merge_func, nodes))._max_depth: merged  # type: ignore[attr-defined]
            for nodes in nodes_by_depth.values()
        }
        merged_children[v] = v_out
    return merged_children


def try_promote(node: UpperBranch[T, AccPromote]) -> Upper[T, AccPromote]:
    all_children: List[Upper[T, AccPromote]] = list(node._all_children())
    if not all_children:
        # Leaf UpperBranch: if it represents an explicit empty stack (empty is not None),
        # it can be represented canonically as an Interface with no children.
        if node.empty is not None:
            return Interface(children={}, acc=node.empty, empty=node.empty)
        return node
    if not all(isinstance(c, Interface) for c in all_children):
        return node

    accs: Set[AccPromote] = set()
    if node.empty is not None:
        accs.add(node.empty)
    for c in all_children:
        ic: Interface[T, AccPromote] = c  # type: ignore[assignment]
        accs.add(ic.acc)
        if ic.empty is not None:
            accs.add(ic.empty)

    if len(accs) <= 1:
        the_acc: Optional[AccPromote] = next(iter(accs)) if accs else None
        if the_acc is None:
            return UpperBranch(children={}, empty=None)
        l_children: Dict[T, Dict[int, Lower[T]]] = {}
        for v, kids in node.children.items():
            v_map: Dict[int, Lower[T]] = {}
            for child in kids.values():
                ci: Interface[T, AccPromote] = child  # type: ignore[assignment]
                lower = Lower(children=ci.children, empty=(ci.empty is not None))
                v_map[lower._max_depth] = lower
            if v_map:
                l_children[v] = v_map
        return Interface(children=l_children, acc=the_acc, empty=node.empty)
    return node


def interface_to_upperbranch(it: Interface[T, Acc]) -> UpperBranch[T, Acc]:
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, kids in it.children.items():
        v_map: Dict[int, Upper[T, Acc]] = {}
        for lchild in kids.values():
            ci = Interface(
                children=lchild.children,
                acc=it.acc,
                empty=(it.acc if lchild.empty else None),
            )
            v_map[ci._max_depth] = ci
        if v_map:
            children[v] = v_map
    new_empty = it.empty
    if not it.children:
        # An interface with no children represents a stack ending here with its own accumulator.
        # This must be merged with any existing `empty` accumulator from a prefix stack.
        new_empty = _merge_optional_acc(it.empty, it.acc)
    return UpperBranch(children=children, empty=new_empty)


def merge_upper(u1: Upper[T, Acc], u2: Upper[T, Acc]) -> Upper[T, Acc]:
    if u1 is u2:
        return u1
    # If both are the same type, use the appropriate merge function
    if isinstance(u1, Interface) and isinstance(u2, Interface):
        return merge_interfaces(u1, u2)
    if isinstance(u1, UpperBranch) and isinstance(u2, UpperBranch):
        return merge_upperbranches(u1, u2)
    # Mixed types: convert Interface(s) to UpperBranch and merge
    ub1 = u1 if isinstance(u1, UpperBranch) else interface_to_upperbranch(u1)
    ub2 = u2 if isinstance(u2, UpperBranch) else interface_to_upperbranch(u2)
    return merge_upperbranches(ub1, ub2)  # type: ignore[arg-type]


def merge_upperbranches(a: UpperBranch[T, Acc], b: UpperBranch[T, Acc]) -> Upper[T, Acc]:
    if a is b:
        return a
    new_empty = _merge_optional_acc(a.empty, b.empty)
    merged_children = _merge_children_by_depth(a.children, b.children, merge_upper)
    return try_promote(UpperBranch(children=merged_children, empty=new_empty))


def merge_interfaces(a: Interface[T, Acc], b: Interface[T, Acc]) -> Upper[T, Acc]:
    if a.acc == b.acc or a.children is b.children:
        merged_children = _merge_children_by_depth(a.children, b.children, merge_lower)
        new_acc = _merge_acc(a.acc, b.acc)
        new_empty = _merge_optional_acc(a.empty, b.empty)
        return Interface(children=merged_children, acc=new_acc, empty=new_empty)
    return merge_upperbranches(interface_to_upperbranch(a), interface_to_upperbranch(b))


def merge_lower(l1: Lower[T], l2: Lower[T]) -> Lower[T]:
    # Fast paths
    if l1 is l2:
        return l1
    new_empty = l1.empty or l2.empty
    merged_children = _merge_children_by_depth(l1.children, l2.children, merge_lower)
    return Lower(children=merged_children, empty=new_empty)


def lower_to_upper(l: Lower[T], acc: Acc) -> Upper[T, Acc]:
    # Convert a Lower subtree to an Upper subtree; the accumulator for all stacks is 'acc'.
    children: Dict[T, Dict[int, Upper[T, Acc]]] = {}
    for v, kids in l.children.items():
        v_map: Dict[int, Upper[T, Acc]] = {}
        for lchild in kids.values():
            up_child = lower_to_upper(lchild, acc)
            v_map[up_child._max_depth] = up_child
        if v_map:
            children[v] = v_map
    ub = UpperBranch(children=children, empty=(acc if l.empty else None))
    return try_promote(ub)


# Alias for downstream code to use as GSS
GSS = LeveledGSS


# ------------------------------
# Standalone model using inlined LeveledGSS
# ------------------------------

@dataclass(frozen=True, eq=False)
class PyAcc:
    terminals_union: Dict[int, RangeSet]
    llm_mask: RangeSet

    def __eq__(self, other):
        if not isinstance(other, PyAcc):
            return NotImplemented
        return self.llm_mask == other.llm_mask and self.terminals_union == other.terminals_union

    def __hash__(self):
        # frozenset of items for hashable dict
        return hash((len(self.terminals_union), self.llm_mask))

    def merge(self, other: "PyAcc") -> "PyAcc":
        # The dataclass is frozen, so we can't modify in-place.
        # But terminals_union is a dict, which is mutable.
        # We must be careful to create copies.
        d1 = self.terminals_union
        d2 = other.terminals_union
        new_terminals_union = d1.copy()
        for k, v in d2.items():
            if k in new_terminals_union:
                new_terminals_union[k] = new_terminals_union[k].union(v)
            else:
                new_terminals_union[k] = v

        return PyAcc(
            terminals_union=new_terminals_union,
            llm_mask=self.llm_mask.union(other.llm_mask),
        )


@dataclass(frozen=True)
class Reduce:
    nonterminal_id: int
    len: int
    production_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Split:
    shift: Optional[int]
    reduces: Dict[int, Dict[int, Tuple[int, ...]]]  # len -> nt_id -> pids


# Action can be a Shift (int), Reduce, or Split
Action = Union[int, Reduce, Split]


@dataclass
class Row:
    actions: Dict[int, Action] = field(default_factory=dict)  # terminal_id -> Action
    gotos: Dict[int, int] = field(default_factory=dict)  # nonterminal_id -> state_id


@dataclass
class ParserTable:
    start_state_id: int
    table: Dict[int, Row]


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation), simplified and concise.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.id_to_token: Dict[int, bytes] = {}
        self.max_depth: Dict[int, int] = {}
        self.possible_matches_cache: Optional[Dict[int, Dict[int, RangeSet]]] = None
        self.tokenizer: Optional[ffi.Regex] = None
        self.glr_parser: Optional[ffi.GLRParser] = None
        self.ignore_terminal_id: Optional[int] = None
        self.parser_table: Optional[ParserTable] = None
        self.state: Dict[int, GSS] = {}
        self.internal_to_original_map: Dict[int, int] = {}
        self.all_internal_llm_tokens_bitset: Optional[RangeSet] = None
        self.tokenizer_initial_state: Optional[int] = None
        self.tokenizer_max_state: Optional[int] = None
        self.all_terminals_bitset: Optional[RangeSet] = None

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # Normalize arena children bitsets and cache max_depth
        for uid, node in self.arena.items():
            uid_int = int(uid)
            self.max_depth[uid_int] = int(node.get("max_depth", 0) or 0)

            children = node.get("children") or []
            if not children:
                node["children"] = []
                continue

            new_children = []
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv_bitset = bs_from_json(dumps(llm_bv_json))
                # Convert to RangeSet for ffi-free operations in commit/get_mask
                llm_bv = RangeSet.from_ranges(llm_bv_bitset.to_ranges())
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv_bitset = bs_from_json(dumps(state_bv_json))
                    state_bv = RangeSet.from_ranges(state_bv_bitset.to_ranges())
                    new_dest_map.append((int(dest_idx), state_bv))
                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)

        # Load tokenizer and parser table from the full constraint JSON
        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.tokenizer = constraint.tokenizer()
        model.tokenizer_max_state = model.tokenizer.max_state()
        model.glr_parser = constraint.glr_parser()
        model.ignore_terminal_id = model.glr_parser.ignore_terminal_id
        model.tokenizer_initial_state = model.tokenizer.initial_state_id()

        parser_data = data['parser']
        table_data = parser_data['stage_7_table']
        start_state_id = parser_data['start_state_id']
        py_table: Dict[int, Row] = {}
        for state_id_str, row_data in table_data:
            state_id = int(state_id_str)
            py_row = Row()
            for term_id_str, action_data in row_data['shifts_and_reduces_full']:
                term_id = int(term_id_str)
                variant = action_data['variant']
                if variant == 'Shift':
                    py_row.actions[term_id] = action_data['state_id']
                elif variant == 'Reduce':
                    pids = tuple(sorted(action_data['production_ids']))
                    py_row.actions[term_id] = Reduce(action_data['nonterminal_id'], action_data['len'], pids)
                elif variant == 'Split':
                    shift = action_data['shift']
                    reduces: Dict[int, Dict[int, Tuple[int, ...]]] = {}
                    for len_str, nts_data in action_data['reduces']:
                        len_int = int(len_str)
                        nts: Dict[int, Tuple[int, ...]] = {}
                        for nt_id_str, pids in nts_data:
                            nt_id_int = int(nt_id_str)
                            nts[nt_id_int] = tuple(sorted(pids))
                        reduces[len_int] = nts
                    py_row.actions[term_id] = Split(shift, reduces)
            for nt_id_str, goto_data in row_data['gotos']:
                nt_id = int(nt_id_str)
                if goto_data['state_id'] is not None:
                    py_row.gotos[nt_id] = goto_data['state_id']
            py_table[state_id] = py_row
        model.parser_table = ParserTable(start_state_id, py_table)

        all_terminals = set()
        for row in model.parser_table.table.values():
            all_terminals.update(row.actions.keys())
        if model.ignore_terminal_id is not None:
            all_terminals.add(model.ignore_terminal_id)
        model.all_terminals_bitset = RangeSet.from_indices(list(all_terminals))

        initial_acc = PyAcc(terminals_union={}, llm_mask=RangeSet.empty())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        # Convert possible_matches_cache to RangeSet
        pmc_ffi: Dict[int, Dict[int, ffi.Bitset]] = constraint.possible_matches()
        pmc_rs: Dict[int, Dict[int, RangeSet]] = {}
        for tsid, inner in pmc_ffi.items():
            mapped: Dict[int, RangeSet] = {}
            for term_id, bit in inner.items():
                mapped[int(term_id)] = RangeSet.from_ranges(bit.to_ranges())
            pmc_rs[int(tsid)] = mapped
        model.possible_matches_cache = pmc_rs
        model.internal_to_original_map = constraint.internal_to_original_map()
        # Convert universe LLM tokens bitset to RangeSet
        all_internal = constraint.all_internal_llm_tokens_bitset()
        model.all_internal_llm_tokens_bitset = RangeSet.from_ranges(all_internal.to_ranges())
        return model

    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, RangeSet]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            disallowed_terminals_map = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(state_id, RangeSet.empty())
                if not matched_bv.intersection(disallowed_for_state).is_empty():
                    return False
            return True
        return gss.prune(predicate)

    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_map = acc.terminals_union
            new_bvs: Dict[int, RangeSet] = collections.defaultdict(RangeSet.empty)
            for old_sid, new_sid in state_map.items():
                bv_source = old_map.get(old_sid, RangeSet.empty())
                new_bvs[new_sid] = new_bvs[new_sid].union(bv_source)

            return PyAcc(terminals_union=dict(new_bvs), llm_mask=acc.llm_mask)
        return gss.apply(apply_map)

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_map = acc.terminals_union.copy()
            curr_bv = current_map.get(state_id, RangeSet.empty())
            to_add = RangeSet.from_indices([terminal_id])
            new_bv = curr_bv.union(to_add)
            current_map[state_id] = new_bv
            return PyAcc(terminals_union=current_map, llm_mask=acc.llm_mask)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        """
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end + 1):
                                yield (int(pop), sid, int(dest_idx))

    def commit(self, token_id: int):
        token_bytes = self.id_to_token[token_id]

        # Build tokenizer maps
        terminals_map: Dict[int, RangeSet] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            matched_terminals = [terminal_id for terminal_id, _ in matches]
            terminals_map[tokenizer_sid] = RangeSet.from_indices(matched_terminals)

        # Prune and map per-state GSS
        temp_states: Dict[int, GSS] = {}
        for tokenizer_sid, gss in self.state.items():
            pruned_gss = self._prune_disallowed_terminals(gss, terminals_map)
            if not pruned_gss.is_empty():
                mapped_gss = self._map_allowed_terminals_tokenizer_states(pruned_gss, state_map)
                temp_states[tokenizer_sid] = mapped_gss

        current_state_for_processing = temp_states

        new_states: Dict[int, List[GSS]] = collections.defaultdict(list)
        q = collections.deque()
        for tokenizer_sid, gss in current_state_for_processing.items():
            q.append((0, tokenizer_sid, gss))  # offset, tokenizer_state, gss

        visited_q_items: set = set()

        while q:
            offset, tokenizer_sid, gss = q.popleft()
            q_item_key = (offset, tokenizer_sid, id(gss))
            if q_item_key in visited_q_items:
                continue
            visited_q_items.add(q_item_key)

            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tokenizer_sid)

            for terminal_id, width in matches:
                processed_gss = gss if terminal_id == self.ignore_terminal_id else self._process_token(gss, terminal_id)

                # Immediate re-match disallow
                if end_state is not None:
                    accessible_terms = set(self.tokenizer.tokens_accessible_from_state(end_state))
                    if terminal_id in accessible_terms:
                        processed_gss = self._disallow_terminal_in_state(processed_gss, end_state, terminal_id)

                if not processed_gss.is_empty():
                    new_offset = offset + width
                    next_tokenizer_sid = self.tokenizer_initial_state
                    if new_offset == len(token_bytes):
                        new_states[next_tokenizer_sid].append(processed_gss)
                    else:
                        q.append((new_offset, next_tokenizer_sid, processed_gss))

            if end_state is not None:
                new_states[end_state].append(gss)

        merged_states = {
            sid: GSS.merge_many(gss_list)
            for sid, gss_list in new_states.items()
            if gss_list
        }
        merged_states = {sid: state for sid, state in merged_states.items() if not state.is_empty()}

        self.state = merged_states

    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        heads_by_state: Dict[int, List[GSS]] = collections.defaultdict(list)
        for state_id in gss.peek():
            heads_by_state[state_id].append(gss.isolate(state_id))

        shifted_gsses: List[GSS] = []

        while heads_by_state:
            state_id, state_gsss = heads_by_state.popitem()
            state_gss = GSS.merge_many(state_gsss)
            row = self.parser_table.table.get(state_id)
            if not row:
                continue
            action = row.actions.get(terminal_id)
            if not action:
                continue

            def handle_shift(shift_to_state_id, gss_to_shift):
                shifted_gsses.append(gss_to_shift.push(shift_to_state_id))

            def handle_reduce(reduce_action: Reduce, gss_to_reduce: GSS):
                popped_gss = gss_to_reduce
                for _ in range(reduce_action.len):
                    popped_gss = popped_gss.pop()
                for from_state_id in popped_gss.peek():
                    goto_state_id = self.parser_table.table[from_state_id].gotos[reduce_action.nonterminal_id]
                    goto_gss = popped_gss.isolate(from_state_id).push(goto_state_id)
                    heads_by_state[goto_state_id].append(goto_gss)

            if isinstance(action, int):
                handle_shift(action, state_gss)
            elif isinstance(action, Reduce):
                handle_reduce(action, state_gss)
            elif isinstance(action, Split):
                if action.shift is not None:
                    handle_shift(action.shift, state_gss)
                for length, nts in action.reduces.items():
                    for nt_id, pids in nts.items():
                        handle_reduce(Reduce(nt_id, length, pids), state_gss)

        return GSS.merge_many(shifted_gsses)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Changes for get_mask_only:
        - Initialize a per-accumulator LLM mask (PyAcc.llm_mask) BEFORE traversal by computing
          the forbidden terminals -> forbidden LLM tokens and taking the complement.
        - Consume terminals_union (set to HybridL2Bitset.all()) after initialization.
        - As we traverse edges, intersect llm_mask with the edge's LLM bitset using apply.
        - At end nodes, simply reduce acc over the GSS and union the llm_mask into the final.
        """
        state_map: Dict[int, GSS] = self.state

        all_ones: Optional[RangeSet] = self.all_internal_llm_tokens_bitset
        final_mask: RangeSet = RangeSet.empty()

        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask
        values: Dict[int, GSS] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, RangeSet]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        # Seed: Initialize llm_mask in each GSS, consume terminals_union, and enqueue roots.
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask = RangeSet.empty()
            disallowed_map = acc.terminals_union

            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]
                for terminal_id in disallowed_terminals.to_indices():
                    if terminal_id in terminals_to_llm:
                        disallowed_llm_mask = disallowed_llm_mask.union(
                            terminals_to_llm[terminal_id]
                        )

            allowed_mask = (all_ones if all_ones is not None else RangeSet.empty()).difference(disallowed_llm_mask)
            return PyAcc(
                terminals_union={},  # consume
                llm_mask=allowed_mask,
            )

        apply_memo: Dict[PyAcc, PyAcc] = {}
        for sid, gss in state_map.items():
            r: int = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)
            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                bucket.add(r)

        def enqueue(d: int, n: int) -> None:
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                bucket.add(n)

        # Main loop
        while depth_heap:
            depth: int = hpop(depth_heap)
            while todo[depth]:
                node: int = todo[depth].pop()
                gss_node: GSS = values.pop(node)

                # End-node handling: just union the allowed LLM tokens
                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
                        final_mask = final_mask.union(reduced_acc.llm_mask)

                # Traverse edges and propagate masks
                edges = arena.get(node, {}).get("children") or []
                for (pop, llm_bv), dests in edges:
                    popped: GSS = gss_node.popn(pop)
                    if popped.is_empty():
                        continue

                    for dest_idx, state_bv in dests:
                        peeked = popped.peek()
                        values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]

                        if not values_to_keep:
                            continue

                        child_gss: GSS = popped.isolate_many(values_to_keep)
                        if child_gss.is_empty():
                            continue

                        # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv
                        acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}

                        def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                            if acc in acc_memo:
                                return acc_memo[acc]
                            new_mask = acc.llm_mask.intersection(llm_bv)
                            if new_mask.is_empty():
                                result = None
                            else:
                                result = PyAcc(
                                    terminals_union=acc.terminals_union,
                                    llm_mask=new_mask
                                )
                            acc_memo[acc] = result
                            return result

                        child_gss = child_gss.apply_and_prune(intersect_and_prune)
                        if child_gss.is_empty():
                            continue

                        d: int = int(dest_idx)
                        if d in values:
                            values[d] = values[d].merge(child_gss)
                        else:
                            values[d] = child_gss
                        enqueue(max_depth[d], d)

            todo.pop(depth)

        # Convert internal mask back to original IDs
        original_indices: List[int] = []
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_indices.append(self.internal_to_original_map[i])

        return RangeSet.from_indices(original_indices)
