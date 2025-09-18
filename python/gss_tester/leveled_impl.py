from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

from .interface import GSS, T, Acc
from .reference_impl import ReferenceGSS


@dataclass(frozen=True, eq=False)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A Graph-Structured Stack (GSS) implementation using a persistent, recursive
    Trie-like structure on reversed stacks.

    Design highlights:
    - Structure: An instance of LeveledGSS is the root node of a DAG. Each node
      represents a set of stack suffixes. Stacks are stored in reverse, so a
      path from the root represents a reversed stack. For a stack [a, b, c],
      the path is root -> 'c' -> 'b' -> 'a' -> node with accumulator.
    - Immutability: The structure is immutable. Operations return new LeveledGSS instances.
    - Sharing: Nodes are interned, so common stack suffixes are shared, forming a DAG.
    - Equivalence: Behaves equivalently to ReferenceGSS.
    """
    _accs: Tuple[Acc, ...] = field(default_factory=tuple, compare=False)
    _children: Dict[T, LeveledGSS[T, Acc]] = field(default_factory=dict, compare=False)
    _hash: Optional[int] = field(default=None, compare=False, repr=False)

    _cache: Dict[Any, LeveledGSS[T, Acc]] = {}

    @classmethod
    def create(
        cls: Type[LeveledGSS[T, Acc]],
        accs: Iterable[Acc],
        children: Dict[T, LeveledGSS[T, Acc]],
    ) -> LeveledGSS[T, Acc]:
        """Creates and interns a new GSS node."""
        accs_tuple = tuple(accs)
        children_tuple = tuple(sorted(children.items()))
        key = (accs_tuple, children_tuple)

        if key in cls._cache:
            return cls._cache[key]

        instance = cls(_accs=accs_tuple, _children=dict(children_tuple))
        cls._cache[key] = instance
        return instance

    @classmethod
    def from_stacks(
        cls: Type[LeveledGSS[T, Acc]], stacks: List[Tuple[List[T], Acc]]
    ) -> LeveledGSS[T, Acc]:
        memo: Dict[Any, LeveledGSS[T, Acc]] = {}

        reversed_stacks = tuple((tuple(reversed(s)), acc) for s, acc in stacks)

        def build(
            stacks_to_process: Tuple[Tuple[Tuple[T, ...], Acc], ...]
        ) -> LeveledGSS[T, Acc]:
            key = stacks_to_process
            if key in memo:
                return memo[key]

            accs_here = [acc for s, acc in stacks_to_process if not s]

            children_map: Dict[T, List[Tuple[Tuple[T, ...], Acc]]] = defaultdict(list)
            for s, acc in stacks_to_process:
                if s:
                    children_map[s[0]].append((s[1:], acc))

            children = {
                val: build(tuple(child_stacks))
                for val, child_stacks in children_map.items()
            }

            node = cls.create(accs_here, children)
            memo[key] = node
            return node

        return build(reversed_stacks)

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        return LeveledGSS.create([], {value: self})

    def pop(self) -> LeveledGSS[T, Acc]:
        if not self._children:
            return LeveledGSS.create([], {})
        return LeveledGSS.merge(self._children.values())

    def is_empty(self) -> bool:
        return len(self._accs) == 1 and not self._children

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        if value is None:
            return LeveledGSS.create(self._accs, {})

        if value in self._children:
            new_root = LeveledGSS.create([], {value: self._children[value]})
            return new_root
        else:
            return LeveledGSS.create([], {})

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        new_accs = [func(acc) for acc in self._accs]
        new_children = {
            val: child.apply(func) for val, child in self._children.items()
        }
        return LeveledGSS.create(new_accs, new_children)

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        new_accs = [acc for acc in self._accs if predicate(acc)]
        new_children = {}
        for val, child in self._children.items():
            new_child = child.prune(predicate)
            if new_child._accs or new_child._children:
                new_children[val] = new_child
        return LeveledGSS.create(new_accs, new_children)

    def peek(self) -> Set[T]:
        return set(self._children.keys())

    def reduce_acc(self) -> Optional[Acc]:
        all_accs: List[Acc] = []

        def collect(node: LeveledGSS[T, Acc]):
            all_accs.extend(node._accs)
            for child in node._children.values():
                collect(child)

        collect(self)
        if not all_accs:
            return None
        return reduce(lambda a, b: a.merge(b), all_accs)

    def to_reference_impl(self) -> ReferenceGSS[T, Acc]:
        stacks: List[Tuple[List[T], Acc]] = []

        def traverse(node: LeveledGSS[T, Acc], current_rev_stack: List[T]):
            if node._accs:
                s = list(current_rev_stack)
                s.reverse()
                for acc in node._accs:
                    stacks.append((s, acc))

            for val, child in node._children.items():
                current_rev_stack.append(val)
                traverse(child, current_rev_stack)
                current_rev_stack.pop()

        traverse(self, [])
        return ReferenceGSS.from_stacks(stacks)

    @staticmethod
    def merge(gss_list: Iterable[GSS[T, Acc]]) -> LeveledGSS[T, Acc]:
        leveled_gss_list: List[LeveledGSS[T, Acc]] = []
        other_gss_list: List[GSS[T, Acc]] = []
        for g in gss_list:
            if isinstance(g, LeveledGSS):
                leveled_gss_list.append(g)
            else:
                other_gss_list.append(g)

        if other_gss_list:
            all_gss = leveled_gss_list + other_gss_list
            ref_inputs = [g.to_reference_impl() for g in all_gss]
            merged_ref = ReferenceGSS.merge(ref_inputs)
            return LeveledGSS.from_stacks(merged_ref._stacks)

        if not leveled_gss_list:
            return LeveledGSS.create([], {})

        all_accs: List[Acc] = []
        all_children_map: Dict[T, List[LeveledGSS[T, Acc]]] = defaultdict(list)
        for gss in leveled_gss_list:
            all_accs.extend(gss._accs)
            for val, child in gss._children.items():
                all_children_map[val].append(child)

        merged_children = {
            val: LeveledGSS.merge(children)
            for val, children in all_children_map.items()
        }

        return LeveledGSS.create(all_accs, merged_children)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GSS):
            return NotImplemented
        return self.to_reference_impl() == other.to_reference_impl()

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash

        ref = self.to_reference_impl()
        canonical_set = frozenset(
            (tuple(vals), acc) for vals, acc in ref._stacks
        )
        h = hash(canonical_set)
        object.__setattr__(self, "_hash", h)
        return h
