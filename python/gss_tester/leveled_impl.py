from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Type, Union

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    # children: T -> depth -> LeveledGSS
    children: Dict[T, Dict[int, 'LeveledGSS[T, Acc]']]


@dataclass(frozen=True, eq=True)
class Interface(Generic[T, Acc]):
    node: Lower[T]
    acc: Acc


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    # children: T -> depth -> LeveledGSSInner
    children: Dict[T, Dict[int, 'Lower[T]']]
    is_leaf: bool

    @classmethod
    def from_stacks(cls: Type['Lower[T]'], stacks: List[List[T]]) -> 'Lower[T]':
        is_leaf = any(not s for s in stacks)
        children: Dict[T, Dict[int, 'Lower[T]']] = {}

        stacks_by_top: Dict[T, List[List[T]]] = defaultdict(list)
        for stack in stacks:
            if stack:
                stacks_by_top[stack[-1]].append(stack[:-1])

        for top, prefixes in stacks_by_top.items():
            children[top] = {}
            prefixes_by_len: Dict[int, List[List[T]]] = defaultdict(list)
            for prefix in prefixes:
                prefixes_by_len[len(prefix)].append(prefix)

            for length, group in prefixes_by_len.items():
                children[top][length] = Lower.from_stacks(group)

        return Lower(children=children, is_leaf=is_leaf)

    def to_stacks(self) -> List[List[T]]:
        stacks: List[List[T]] = []
        if self.is_leaf:
            stacks.append([])
        for top, children_at_depths in self.children.items():
            for _, child_node in children_at_depths.items():
                for prefix in child_node.to_stacks():
                    stacks.append(prefix + [top])
        return stacks

    def validate_invariants(self) -> None:
        # Invariant: lower must either have children or be a leaf (or both)
        if not self.children and not self.is_leaf:
            raise InvariantViolation("Lower node must have children or be a leaf.")

        # Recurse to children
        for children_at_depth in self.children.values():
            for child in children_at_depth.values():
                child.validate_invariants()


class InvariantViolation(Exception):
    pass


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    inner: Union[Upper[T, Acc], Interface[T, Acc]]

    def __post_init__(self):
        self.validate_invariants()

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        if not stacks:
            return LeveledGSS(Upper({}))

        accs = {s[1] for s in stacks}
        if len(accs) == 1:
            acc = accs.pop()
            list_of_stacks = [s[0] for s in stacks]
            return LeveledGSS(Interface(Lower.from_stacks(list_of_stacks), acc))

        children: Dict[T, Dict[int, 'LeveledGSS[T, Acc]']] = {}
        stacks_by_top_and_len: Dict[Tuple[T, int], List[Tuple[List[T], Acc]]] = defaultdict(list)

        for stack, acc in stacks:
            if stack:
                top = stack[-1]
                prefix = stack[:-1]
                stacks_by_top_and_len[(top, len(prefix))].append((prefix, acc))

        for (top, length), group in stacks_by_top_and_len.items():
            if top not in children:
                children[top] = {}
            children[top][length] = cls.from_stacks(group)

        return LeveledGSS(Upper(children))

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        if isinstance(self.inner, Interface):
            return [(s, self.inner.acc) for s in self.inner.node.to_stacks()]
        elif isinstance(self.inner, Upper):
            all_stacks: List[Tuple[List[T], Acc]] = []
            for top, children_at_depths in self.inner.children.items():
                for _, child_gss in children_at_depths.items():
                    all_stacks.extend([(prefix + [top], acc) for prefix, acc in child_gss.to_stacks()])
            return all_stacks
        return []

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def pop(self) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        raise NotImplementedError

    def peek(self) -> Set[T]:
        raise NotImplementedError

    def reduce_acc(self) -> Optional[Acc]:
        raise NotImplementedError

    def validate_invariants(self) -> None:
        if isinstance(self.inner, Upper):
            # Invariant: upper must have at least one child.
            all_children = [
                child
                for children_at_depth in self.inner.children.values()
                for child in children_at_depth.values()
            ]
            if not all_children:
                raise InvariantViolation("Upper node must have at least one child.")

            # Recurse validation to children.
            for child in all_children:
                child.validate_invariants()

            # Invariant: if all children are interfaces, they must not have equal accs.
            all_children_are_interfaces = all(isinstance(c.inner, Interface) for c in all_children)
            if all_children_are_interfaces:
                accs = [c.inner.acc for c in all_children]
                # Accumulators may not be hashable, so we can't use a set.
                # This is O(n^2) but likely fine for tests.
                if len(accs) > 1:
                    for i in range(len(accs)):
                        for j in range(i + 1, len(accs)):
                            if accs[i] == accs[j]:
                                raise InvariantViolation(
                                    "Upper with all-interface children has duplicate accumulators."
                                )

        elif isinstance(self.inner, Interface):
            # Delegate validation to the inner Lower node.
            self.inner.node.validate_invariants()
