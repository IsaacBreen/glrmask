from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Dict, Generic, List, Optional, Set, Tuple, Type, Union

from .interface import GSS, T, Acc


# ------------------------------
# Internal node classes
# ------------------------------

@dataclass(frozen=True, eq=True)
class Upper(Generic[T, Acc]):
    # children: T -> depth -> LeveledGSS
    children: Dict[T, Dict[int, 'LeveledGSS[T, Acc]']]
    empty_stack_acc: Optional[Acc] = None


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

    def pop(self) -> 'Lower[T]':
        if not self.children:
            return Lower(children={}, is_leaf=False)
        all_children = []
        for t, children_at_depths in self.children.items():
            all_children[t] = {}
            for depth, child_node in children_at_depths.items():
                all_children.append(child_node)
        return reduce(lambda c, n: c.merge(n), all_children[1:], all_children[0])

    def is_empty(self) -> bool:
        return not self.children and not self.is_leaf

    def isolate(self, value: Optional[T]) -> 'Lower[T]':
        if value is None:
            return Lower(children={}, is_leaf=self.is_leaf)
        if value not in self.children:
            return Lower(children={}, is_leaf=False)
        return Lower(children={value: self.children[value]}, is_leaf=False)

    def merge(self, other: Lower[T]) -> 'Lower[T]':
        raise NotImplementedError

    def peek(self) -> Set[T]:
        return set(self.children.keys())

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

    def _suck_up_accs(self) -> 'LeveledGSS[T, Acc]':
        if not isinstance(self.inner, Upper):
            return self

        # Cannot suck up if there's an empty stack at this level, as Interface can't represent it.
        if self.inner.empty_stack_acc is not None:
            return self

        all_children = [
            child
            for children_at_depth in self.inner.children.values()
            for child in children_at_depth.values()
        ]

        if not all_children or not all(isinstance(child.inner, Interface) for child in all_children):
            return self

        first_acc = all_children[0].inner.acc
        if not all(child.inner.acc == first_acc for child in all_children[1:]):
            return self

        # Invariant violated: suck up the accumulator.
        new_lower_children: Dict[T, Dict[int, 'Lower[T]']] = {}
        for top, children_at_depths in self.inner.children.items():
            new_lower_children[top] = {}
            for depth, child_gss in children_at_depths.items():
                # We know child_gss.inner is an Interface
                new_lower_children[top][depth] = child_gss.inner.node

        # The new Lower node represents stacks that had a top element, so it is not a leaf.
        new_lower = Lower(children=new_lower_children, is_leaf=False)
        return LeveledGSS(Interface(new_lower, first_acc))

    @classmethod
    def from_stacks(cls: Type['LeveledGSS[T, Acc]'], stacks: List[Tuple[List[T], Acc]]) -> 'LeveledGSS[T, Acc]':
        if not stacks:
            return LeveledGSS(Upper({}, None))

        children: Dict[T, Dict[int, 'LeveledGSS[T, Acc]']] = {}
        stacks_by_top_and_len: Dict[Tuple[T, int], List[Tuple[List[T], Acc]]] = defaultdict(list)
        empty_stack_accs: List[Acc] = []

        for stack, acc in stacks:
            if stack:
                top = stack[-1]
                prefix = stack[:-1]
                stacks_by_top_and_len[(top, len(prefix))].append((prefix, acc))
            else:
                empty_stack_accs.append(acc)

        for (top, length), group in stacks_by_top_and_len.items():
            if top not in children:
                children[top] = {}
            children[top][length] = cls.from_stacks(group)

        merged_empty_acc: Optional[Acc] = None
        if empty_stack_accs:
            merged_empty_acc = reduce(lambda a, b: a.merge(b), empty_stack_accs)

        # Optimization: if there are no non-empty stacks, we can use an Interface node.
        if not children and merged_empty_acc is not None:
            return LeveledGSS(Interface(Lower(children={}, is_leaf=True), merged_empty_acc))

        return LeveledGSS(Upper(children, merged_empty_acc))._suck_up_accs()

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        if isinstance(self.inner, Interface):
            return [(s, self.inner.acc) for s in self.inner.node.to_stacks()]
        elif isinstance(self.inner, Upper):
            all_stacks: List[Tuple[List[T], Acc]] = []
            if self.inner.empty_stack_acc is not None:
                all_stacks.append(([], self.inner.empty_stack_acc))
            for top, children_at_depths in self.inner.children.items():
                for _, child_gss in children_at_depths.items():
                    all_stacks.extend([(prefix + [top], acc) for prefix, acc in child_gss.to_stacks()])
            return all_stacks

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        ref_gss = self.to_reference_impl()
        new_ref_gss = ref_gss.push(value)
        return LeveledGSS.from_stacks(new_ref_gss.to_stacks())

    def pop(self) -> 'LeveledGSS[T, Acc]':
        ref_gss = self.to_reference_impl()
        new_ref_gss = ref_gss.pop()
        return LeveledGSS.from_stacks(new_ref_gss.to_stacks())

    def is_empty(self) -> bool:
        if isinstance(self.inner, Interface):
            return self.inner.node.is_empty()
        # It's an Upper node. It's empty if it has no children and no empty stack.
        return not self.inner.children and self.inner.empty_stack_acc is None

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        ref_gss = self.to_reference_impl()
        new_ref_gss = ref_gss.isolate(value)
        return LeveledGSS.from_stacks(new_ref_gss.to_stacks())

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        ref_gss = self.to_reference_impl()
        new_ref_gss = ref_gss.apply(func)
        return LeveledGSS.from_stacks(new_ref_gss.to_stacks())

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        ref_gss = self.to_reference_impl()
        new_ref_gss = ref_gss.prune(predicate)
        return LeveledGSS.from_stacks(new_ref_gss.to_stacks())

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        ref_gss = self.to_reference_impl()
        new_ref_gss = ref_gss.merge(other)
        return LeveledGSS.from_stacks(new_ref_gss.to_stacks())

    def peek(self) -> Set[T]:
        if isinstance(self.inner, Interface):
            return self.inner.node.peek()
        # It's an Upper node. The keys of its children are the top elements.
        return set(self.inner.children.keys())

    def reduce_acc(self) -> Optional[Acc]:
        if isinstance(self.inner, Interface):
            return self.inner.acc

        # It's an Upper node
        if not self.inner.children and self.inner.empty_stack_acc is None:
            return None

        child_accs = (
            child.reduce_acc()
            for children_at_depth in self.inner.children.values()
            for child in children_at_depth.values()
        )
        valid_accs = [acc for acc in child_accs if acc is not None]

        if self.inner.empty_stack_acc is not None:
            valid_accs.append(self.inner.empty_stack_acc)

        if not valid_accs:
            return None

        return reduce(lambda a, b: a.merge(b), valid_accs)

    def validate_invariants(self) -> None:
        if isinstance(self.inner, Upper):
            if not self.inner.children:
                return  # An empty Upper represents a valid empty GSS.

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

        elif isinstance(self.inner, Interface):
            # Delegate validation to the inner Lower node.
            self.inner.node.validate_invariants()
