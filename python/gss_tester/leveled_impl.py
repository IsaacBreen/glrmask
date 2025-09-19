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

    def _suck_up_accs(self) -> 'LeveledGSS[T, Acc]':
        if not isinstance(self.inner, Upper):
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
        # --------------------------------------------------------------------
        # New handling for the case where *all* stacks are empty.
        # --------------------------------------------------------------
        # In this situation the GSS must still carry the accumulator(s) of the
        # empty stack(s).  The original implementation returned an empty Upper
        # node, which the analyzer interpreted as an empty GSS.  We now build
        # an Interface node that contains a leaf Lower (representing the empty
        # stack) together with the merged accumulator.
        #
        # Example: after popping two merged stacks `[100, 101]` and `[100, 102]`
        # we obtain a single stack `[100]`.  The recursion in `from_stacks`
        # reaches a point where the remaining prefixes are `[]`.  The code
        # below ensures that those empty‑prefix cases are represented correctly.
        # --------------------------------------------------------------------
        if all(not s for s, _ in stacks):
            # Merge all accumulators of the empty stacks.
            # `reduce` needs an initial value; we can safely take the first
            # accumulator because `stacks` is non‑empty here.
            merged_acc: Acc = reduce(lambda a, b: a.merge(b),
                                    (acc for _, acc in stacks),
                                    stacks[0][1])
            # A leaf Lower node represents the empty stack.
            leaf_lower = Lower(children={}, is_leaf=True)
            return LeveledGSS(Interface(leaf_lower, merged_acc))

        if not stacks:
            return LeveledGSS(Upper({}))

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

        return LeveledGSS(Upper(children))._suck_up_accs()

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        if isinstance(self.inner, Interface):
            return [(s, self.inner.acc) for s in self.inner.node.to_stacks()]
        elif isinstance(self.inner, Upper):
            all_stacks: List[Tuple[List[T], Acc]] = []
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
            return False
        # It's an Upper node. It's empty if it has no children.
        return not self.inner.children

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
            return set(self.inner.node.children.keys())
        # It's an Upper node. The keys of its children are the top elements.
        return set(self.inner.children.keys())

    def reduce_acc(self) -> Optional[Acc]:
        if isinstance(self.inner, Interface):
            return self.inner.acc

        # It's an Upper node
        if not self.inner.children:
            return None

        child_accs = (
            child.reduce_acc()
            for children_at_depth in self.inner.children.values()
            for child in children_at_depth.values()
        )
        valid_accs = [acc for acc in child_accs if acc is not None]

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
