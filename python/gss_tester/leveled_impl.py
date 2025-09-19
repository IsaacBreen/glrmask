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

    def pop(self) -> 'Lower[T]':
        # Pop one element from all represented stacks:
        # This flattens one level by merging all child Lower nodes,
        # effectively dropping the current "top" layer.
        all_children: List[Lower[T]] = [
            child_node
            for children_at_depths in self.children.values()
            for child_node in children_at_depths.values()
        ]
        if not all_children:
            # No non-empty stacks to pop; result is empty (no stacks).
            return Lower(children={}, is_leaf=False)
        acc = all_children[0]
        for node in all_children[1:]:
            acc = acc.merge(node)
        return acc

    def is_empty(self) -> bool:
        return not self.children and not self.is_leaf

    def isolate(self, value: Optional[T]) -> 'Lower[T]':
        if value is None:
            return Lower(children={}, is_leaf=self.is_leaf)
        if value not in self.children:
            return Lower(children={}, is_leaf=False)
        return Lower(children={value: self.children[value]}, is_leaf=False)

    def merge(self, other: Lower[T]) -> 'Lower[T]':
        # Union of two Lower nodes (set union of represented stacks)
        if self is other:
            return self
        new_is_leaf = self.is_leaf or other.is_leaf
        new_children: Dict[T, Dict[int, 'Lower[T]']] = {}

        all_tops: Set[T] = set(self.children.keys()) | set(other.children.keys())
        for top in all_tops:
            depths_left = self.children.get(top, {})
            depths_right = other.children.get(top, {})
            all_depths = set(depths_left.keys()) | set(depths_right.keys())
            if not all_depths:
                continue
            merged_by_depth: Dict[int, 'Lower[T]'] = {}
            for d in all_depths:
                left_child = depths_left.get(d)
                right_child = depths_right.get(d)
                if left_child is not None and right_child is not None:
                    merged_by_depth[d] = left_child.merge(right_child)
                else:
                    merged_by_depth[d] = left_child if left_child is not None else right_child  # type: ignore
            new_children[top] = merged_by_depth

        return Lower(children=new_children, is_leaf=new_is_leaf)

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
        # Build a leveled structure from a list of stacks with accumulators.
        # Canonicalizes by merging accumulators for identical stacks.
        if not stacks:
            return LeveledGSS(Upper({}))

        # Base case: all stacks are empty (only [] in the input).
        if all(not s for s, _ in stacks):
            # Merge all accumulators for identical [] stacks.
            accs = [acc for _, acc in stacks]
            merged_acc = reduce(lambda a, b: a.merge(b), accs)
            lower = Lower(children={}, is_leaf=True)
            return LeveledGSS(Interface(lower, merged_acc))

        # Group by top and prefix length (depth).
        children: Dict[T, Dict[int, 'LeveledGSS[T, Acc]']] = {}
        stacks_by_top_and_len: Dict[Tuple[T, int], List[Tuple[List[T], Acc]]] = defaultdict(list)

        for stack, acc in stacks:
            if not stack:
                # Top-level empty stacks are ignored for canonical representation.
                # They will be represented only when nested under a parent via depth=0 leaves.
                continue
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
            # Drop empty stacks in canonical external representation (align with ReferenceGSS).
            return [(s, self.inner.acc) for s in self.inner.node.to_stacks() if s]
        elif isinstance(self.inner, Upper):
            all_stacks: List[Tuple[List[T], Acc]] = []
            for top, children_at_depths in self.inner.children.items():
                for _, child_gss in children_at_depths.items():
                    all_stacks.extend([(prefix + [top], acc) for prefix, acc in child_gss.to_stacks()])
            return all_stacks

    def push(self, value: T) -> 'LeveledGSS[T, Acc]':
        # Native implementation: operate on explicit stacks then rebuild.
        stacks = self.to_stacks()
        pushed = [(vals + [value], acc) for vals, acc in stacks]
        return LeveledGSS.from_stacks(pushed)

    def pop(self) -> 'LeveledGSS[T, Acc]':
        # Native implementation: pop from all non-empty stacks.
        stacks = self.to_stacks()
        popped = [(vals[:-1], acc) for vals, acc in stacks if vals]
        return LeveledGSS.from_stacks(popped)

    def is_empty(self) -> bool:
        if isinstance(self.inner, Interface):
            # Treat "only empty stacks" as empty (aligning with ReferenceGSS behavior).
            return not bool(self.inner.node.children)
        # It's an Upper node. It's empty if it has no children.
        return not self.inner.children

    def isolate(self, value: Optional[T]) -> 'LeveledGSS[T, Acc]':
        stacks = self.to_stacks()
        if value is None:
            # In canonical form, empty stacks are not retained.
            return LeveledGSS.from_stacks([])
        filtered = [(vals, acc) for vals, acc in stacks if vals and vals[-1] == value]
        return LeveledGSS.from_stacks(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> 'LeveledGSS[T, Acc]':
        stacks = self.to_stacks()
        transformed = [(vals, func(acc)) for vals, acc in stacks]
        return LeveledGSS.from_stacks(transformed)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'LeveledGSS[T, Acc]':
        stacks = self.to_stacks()
        kept = [(vals, acc) for vals, acc in stacks if predicate(acc)]
        return LeveledGSS.from_stacks(kept)

    def merge(self, other: GSS[T, Acc]) -> 'LeveledGSS[T, Acc]':
        # Native union + canonicalization: concatenation then rebuild merges dup stacks by accumulator.
        other_stacks = other.to_stacks()
        stacks = self.to_stacks() + other_stacks
        return LeveledGSS.from_stacks(stacks)

    def peek(self) -> Set[T]:
        if isinstance(self.inner, Interface):
            return self.inner.node.peek()
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

