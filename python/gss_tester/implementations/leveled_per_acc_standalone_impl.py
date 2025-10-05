from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from typing import (Any, Callable, Dict, Generic, Iterable, Iterator, List,
                    Optional, Set, Tuple, Type, TypeVar)

from ..interface import GSS, Acc, Mergeable, NewAcc, T
from .reference_impl import ReferenceGSS

# ------------------------------
# Internal, accumulator-less GSS implementation
# ------------------------------

@dataclass(frozen=True, eq=True)
class _Node(Generic[T]):
    """Internal node for representing a set of stack suffixes (a trie)."""
    children: Dict[T, Dict[int, "_Node[T]"]]
    empty: bool  # True if a stack can end at this node
    _max_depth: int = field(init=False)

    def __post_init__(self):
        if not self.children:
            depth = 0
        else:
            depth = max(child._max_depth for child in self._all_children()) + 1
        object.__setattr__(self, '_max_depth', depth)

    def _all_children(self) -> Iterator["_Node[T]"]:
        for children_at_depth in self.children.values():
            yield from children_at_depth.values()

def _merge_children_by_depth(
    c1: Dict[T, Dict[int, _Node[T]]],
    c2: Dict[T, Dict[int, _Node[T]]],
    merge_func: Callable[[_Node[T], _Node[T]], _Node[T]],
) -> Dict[T, Dict[int, _Node[T]]]:
    if c1 is c2:
        return c1
    merged_children: Dict[T, Dict[int, _Node[T]]] = {}
    all_vals = c1.keys() | c2.keys()
    for v in all_vals:
        nodes_by_depth: Dict[int, list[_Node[T]]] = defaultdict(list)
        children_c1 = c1.get(v, {}).items()
        children_c2 = c2.get(v, {}).items()
        for depth, child in chain(children_c1, children_c2):
            nodes_by_depth[depth].append(child)
        if not nodes_by_depth:
            continue

        v_out: Dict[int, _Node[T]] = {}
        for nodes in nodes_by_depth.values():
            merged = reduce(merge_func, nodes)
            v_out[merged._max_depth] = merged
        merged_children[v] = v_out
    return merged_children

def _merge_nodes(n1: _Node[T], n2: _Node[T]) -> _Node[T]:
    if n1 is n2:
        return n1
    new_empty = n1.empty or n2.empty
    merged_children = _merge_children_by_depth(n1.children, n2.children, _merge_nodes)
    return _Node(children=merged_children, empty=new_empty)

@dataclass(frozen=True, eq=True)
class _InternalGSS(Generic[T]):
    """A GSS-like structure for a set of stacks, without accumulators."""
    root: _Node[T]

    @classmethod
    def empty(cls) -> "_InternalGSS[T]":
        return cls(_Node(children={}, empty=False))

    @classmethod
    def from_stacks(cls, stacks: List[List[T]]) -> "_InternalGSS[T]":
        if not stacks:
            return cls.empty()

        trie: Dict[Any, Any] = {"#empty": False}
        for stack in stacks:
            if not stack:
                trie["#empty"] = True
                continue
            node = trie
            for item in reversed(stack):
                node = node.setdefault(item, {"#empty": False})
            node["#empty"] = True

        def build_node(sub_trie: Dict) -> _Node[T]:
            is_empty = sub_trie.pop("#empty", False)
            children: Dict[T, Dict[int, _Node[T]]] = {}
            for val, next_trie in sub_trie.items():
                child_node = build_node(next_trie)
                children[val] = {child_node._max_depth: child_node}
            return _Node(children=children, empty=is_empty)

        return cls(build_node(trie))

    def to_stacks(self) -> List[List[T]]:
        res: List[List[T]] = []
        def dfs(node: _Node[T], prefix: List[T]):
            if node.empty:
                res.append(list(reversed(prefix)))
            for val, children_at_depths in node.children.items():
                for child in children_at_depths.values():
                    dfs(child, prefix + [val])
        dfs(self.root, [])
        return res

    def push(self, value: T) -> "_InternalGSS[T]":
        if self.is_empty():
            return self
        new_root = _Node(children={value: {self.root._max_depth: self.root}}, empty=False)
        return _InternalGSS(new_root)

    def pop(self) -> "_InternalGSS[T]":
        all_children = list(self.root._all_children())
        if not all_children:
            return self.empty()
        merged_root = reduce(_merge_nodes, all_children)
        return _InternalGSS(merged_root)

    def popn(self, n: int) -> "_InternalGSS[T]":
        if n <= 0:
            return self
        if self.is_empty():
            return self
        memo: Dict[Tuple[int, int], _Node[T]] = {}
        def _popn_node(node: _Node[T], k: int) -> _Node[T]:
            if k == 0:
                return node
            key = (id(node), k)
            if key in memo:
                return memo[key]
            all_children = list(node._all_children())
            if not all_children:
                res = _Node(children={}, empty=False)
                memo[key] = res
                return res
            popped_children = [_popn_node(child, k - 1) for child in all_children]
            res = reduce(_merge_nodes, popped_children)
            memo[key] = res
            return res
        return _InternalGSS(_popn_node(self.root, n))

    def is_empty(self) -> bool:
        return not self.root.children and not self.root.empty

    def isolate(self, value: Optional[T]) -> "_InternalGSS[T]":
        if value is None:
            if self.root.empty:
                return _InternalGSS(_Node(children={}, empty=True))
            else:
                return self.empty()
        children_for_v = self.root.children.get(value)
        if not children_for_v:
            return self.empty()
        new_root = _Node(children={value: children_for_v}, empty=False)
        return _InternalGSS(new_root)

    def isolate_many(self, values: Iterable[Optional[T]]) -> "_InternalGSS[T]":
        values_set = set(values)
        new_empty = None in values_set and self.root.empty
        filtered_children = {
            v: kids for v, kids in self.root.children.items() if v in values_set
        }
        new_root = _Node(children=filtered_children, empty=new_empty)
        return _InternalGSS(new_root)

    def filter_by_length(self, min_len: Optional[int] = None, max_len: Optional[int] = None) -> "_InternalGSS[T]":
        _min = min_len if min_len is not None else 0
        _max = max_len if max_len is not None else float('inf')

        memo: Dict[Tuple[int, int], Optional[_Node[T]]] = {}

        def _filter(node: _Node[T], depth: int) -> Optional[_Node[T]]:
            key = (id(node), depth)
            if key in memo:
                return memo[key]

            if depth > _max:
                memo[key] = None
                return None

            keep_empty = node.empty and _min <= depth <= _max

            new_children: Dict[T, Dict[int, _Node[T]]] = {}
            if depth < _max:
                for v, kids in node.children.items():
                    new_kids_for_v: Dict[int, _Node[T]] = {}
                    for d, child in kids.items():
                        new_child = _filter(child, depth + 1)
                        if new_child:
                            new_kids_for_v[new_child._max_depth] = new_child
                    if new_kids_for_v:
                        new_children[v] = new_kids_for_v

            if not new_children and not keep_empty:
                memo[key] = None
                return None

            if new_children == node.children and keep_empty == node.empty:
                memo[key] = node
                return node

            res = _Node(children=new_children, empty=keep_empty)
            memo[key] = res
            return res

        new_root = _filter(self.root, 0)
        if new_root is None:
            return self.empty()
        return _InternalGSS(new_root)

    def merge(self, other: "_InternalGSS[T]") -> "_InternalGSS[T]":
        if self is other: return self
        if self.is_empty(): return other
        if other.is_empty(): return self
        return _InternalGSS(_merge_nodes(self.root, other.root))

    def peek(self) -> Set[T]:
        return set(self.root.children.keys())

# ------------------------------
# Public GSS implementation
# ------------------------------

@dataclass(eq=False)
class LeveledPerAccGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A GSS implementation that partitions the graph by accumulator.
    It uses an internal, simplified, accumulator-less GSS representation
    for each partition, which can be efficient when there are few distinct
    accumulators.
    """
    _parts: Dict[Acc, _InternalGSS[T]]

    def __post_init__(self):
        self._parts = {a: g for a, g in self._parts.items() if not g.is_empty()}

    @classmethod
    def from_stacks(cls: Type["LeveledPerAccGSS[T, Acc]"], stacks: List[Tuple[List[T], Acc]]) -> "LeveledPerAccGSS[T, Acc]":
        by_acc: Dict[Acc, List[List[T]]] = defaultdict(list)
        for vals, acc in stacks:
            by_acc[acc].append(list(vals))
        parts = {
            acc: _InternalGSS.from_stacks(acc_stacks)
            for acc, acc_stacks in by_acc.items()
        }
        return cls(_parts=parts)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        out: List[Tuple[List[T], Acc]] = []
        for acc, inner in self._parts.items():
            for vals in inner.to_stacks():
                out.append((vals, acc))
        return ReferenceGSS.from_stacks(out).to_stacks()

    def push(self, value: T) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.push(value) for acc, g in self._parts.items()})

    def pop(self) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.pop() for acc, g in self._parts.items()})

    def popn(self, n: int) -> "LeveledPerAccGSS[T, Acc]":
        if n <= 0 or self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.popn(n) for acc, g in self._parts.items()})

    def is_empty(self) -> bool:
        return not self._parts

    def isolate(self, value: Optional[T]) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.isolate(value) for acc, g in self._parts.items()})

    def isolate_many(self, values: Iterable[Optional[T]]) -> "LeveledPerAccGSS[T, Acc]":
        valset = set(values)
        if not valset or self.is_empty():
            return self.empty()
        return LeveledPerAccGSS({acc: g.isolate_many(valset) for acc, g in self._parts.items()})

    def filter_by_length(self, min_len: Optional[int] = None, max_len: Optional[int] = None) -> "LeveledPerAccGSS[T, Acc]":
        if self.is_empty():
            return self
        return LeveledPerAccGSS({acc: g.filter_by_length(min_len, max_len) for acc, g in self._parts.items()})

    def apply(self, func: Callable[[Acc], NewAcc], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        new_parts: Dict[NewAcc, _InternalGSS[T]] = {}
        for acc, g in self._parts.items():
            new_acc = func(acc)
            if new_acc in new_parts:
                new_parts[new_acc] = new_parts[new_acc].merge(g)
            else:
                new_parts[new_acc] = g
        return LeveledPerAccGSS(new_parts)  # type: ignore[arg-type]

    def prune(self, predicate: Callable[[Acc], bool], memo: Optional[Dict[int, Any]] = None) -> "LeveledPerAccGSS[T, Acc]":
        return LeveledPerAccGSS({acc: g for acc, g in self._parts.items() if predicate(acc)})

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[NewAcc]], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        cache: Dict[int, Optional[NewAcc]] = {} if memo is None else memo
        def decide(a: Acc) -> Optional[NewAcc]:
            k = id(a)
            if k in cache:
                return cache[k]
            r = mutator(a)
            cache[k] = r
            return r

        new_parts: Dict[NewAcc, _InternalGSS[T]] = {}
        for acc, g in self._parts.items():
            new_acc = decide(acc)
            if new_acc is None:
                continue
            if new_acc in new_parts:
                new_parts[new_acc] = new_parts[new_acc].merge(g)
            else:
                new_parts[new_acc] = g
        return LeveledPerAccGSS(new_parts)  # type: ignore[arg-type]

    def merge(self, other: "LeveledPerAccGSS[T, Acc]") -> "LeveledPerAccGSS[T, Acc]":
        if self is other: return self
        if self.is_empty(): return other
        if other.is_empty(): return self
        
        merged: Dict[Acc, _InternalGSS[T]] = dict(self._parts)
        for acc, g in other._parts.items():
            if acc in merged:
                merged[acc] = merged[acc].merge(g)
            else:
                merged[acc] = g
        return LeveledPerAccGSS(merged)

    @classmethod
    def merge_many(cls: Type["LeveledPerAccGSS[T, Acc]"], gss_list: Iterable["LeveledPerAccGSS[T, Acc]"]) -> "LeveledPerAccGSS[T, Acc]":
        result: Dict[Acc, _InternalGSS[T]] = {}
        for g in gss_list:
            for acc, inner in g._parts.items():
                if acc in result:
                    result[acc] = result[acc].merge(inner)
                else:
                    result[acc] = inner
        return cls(result)

    def peek(self) -> Set[T]:
        tops: Set[T] = set()
        for g in self._parts.values():
            tops.update(g.peek())
        return tops

    def reduce_acc(self) -> Optional[Acc]:
        if not self._parts:
            return None
        
        accs = iter(self._parts.keys())
        return reduce(lambda a, b: a.merge(b), accs)


Leveled_per_acc_standaloneGSS = LeveledPerAccGSS