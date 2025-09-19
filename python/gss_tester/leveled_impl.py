from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
)

from .interface import GSS, Acc, T
from .reference_impl import ReferenceGSS

# Increase recursion limit for deep stacks that can occur during construction
sys.setrecursionlimit(2000)


@dataclass(frozen=True, eq=True)
class LeveledGSS(GSS[T, Acc]):
    """
    An efficient, graph-based implementation of the GSS interface.

    This implementation represents the GSS as a tree (a directed acyclic graph)
    where nodes are shared. Each LeveledGSS instance represents a set of
    possible states (a collection of stacks).

    - `_branches`: A frozenset of (value, child_gss) pairs. This represents
      all stacks that are not empty. The `value` is the top element of the
      stack, and `child_gss` is the GSS for the rest of the stacks.
    - `_empty_acc`: An optional accumulator for the single empty stack, if it
      exists in the set of states.

    This structure naturally shares prefixes and maintains the "accumulator at
    one level" invariant: for any given stack, its accumulator is stored
    at the node where that stack becomes empty. The "suck-up" optimization
    happens implicitly: if a set of stacks with a common prefix also share
    an accumulator, they will all lead to a common descendant node of the form
    `LeveledGSS(_branches=frozenset(), _empty_acc=common_acc)`.
    """
    _branches: frozenset[tuple[T, LeveledGSS[T, Acc]]] = field(default_factory=frozenset)
    _empty_acc: Optional[Acc] = None

    # Caching the dictionary view of branches for performance
    _branches_dict: Dict[T, LeveledGSS[T, Acc]] = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, '_branches_dict', dict(self._branches))

    @classmethod
    def from_stacks(cls: Type[LeveledGSS], stacks: List[Tuple[List[T], Acc]]) -> LeveledGSS[T, Acc]:
        if not stacks:
            return EMPTY_GSS

        # 1. Find and merge accumulator for the empty stack
        empty_accs = [acc for s, acc in stacks if not s]
        merged_empty_acc = reduce(lambda a, b: a.merge(b), empty_accs) if empty_accs else None

        # 2. Group non-empty stacks by their top element
        non_empty_stacks = [(s, acc) for s, acc in stacks if s]
        grouped_by_head: Dict[T, List[Tuple[List[T], Acc]]] = {}
        for stack, acc in non_empty_stacks:
            head, tail = stack[-1], stack[:-1]
            if head not in grouped_by_head:
                grouped_by_head[head] = []
            grouped_by_head[head].append((tail, acc))

        # 3. Recursively build the GSS for each group of tails
        branches = frozenset(
            (head, cls.from_stacks(tails)) for head, tails in grouped_by_head.items()
        )

        return LeveledGSS(branches, merged_empty_acc)

    def is_empty(self) -> bool:
        return self is EMPTY_GSS

    def push(self, value: T) -> LeveledGSS[T, Acc]:
        # Pushing a value creates a new top level. The new GSS has only one
        # branch, and the GSS under that branch is the original GSS.
        # An empty GSS has no active stacks, so push results in an empty GSS.
        if self.is_empty():
            return EMPTY_GSS
        return LeveledGSS(frozenset([(value, self)]))

    def pop(self) -> LeveledGSS[T, Acc]:
        # Popping removes the top level of all non-empty stacks. This is
        # equivalent to merging all the child GSSs. The empty stack is preserved.
        if not self._branches:
            return self  # Only contains an empty stack or is empty

        # The new GSS is the merge of all children, plus the current empty stack.
        merged_children = self.merge_many(self._branches_dict.values())

        if self._empty_acc is None:
            return merged_children

        # Create a GSS representing only the current empty stack and merge it.
        empty_gss_part = LeveledGSS(_empty_acc=self._empty_acc)
        return merged_children.merge(empty_gss_part)

    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        if value is None:
            # Keep only the empty stack
            if self._empty_acc is not None:
                return LeveledGSS(_empty_acc=self._empty_acc)
            else:
                return EMPTY_GSS
        else:
            # Keep only the branch corresponding to `value`
            child_gss = self._branches_dict.get(value)
            if child_gss:
                return child_gss
            else:
                return EMPTY_GSS

    def apply(self, func: Callable[[Acc], Acc]) -> LeveledGSS[T, Acc]:
        # Use a memoization dictionary to preserve sharing of identical nodes.
        return self._apply_memo(func, {})

    def _apply_memo(self, func: Callable[[Acc], Acc], memo: Dict[int, LeveledGSS[T, Acc]]) -> LeveledGSS[T, Acc]:
        if self.is_empty():
            return self
        if id(self) in memo:
            return memo[id(self)]

        new_empty_acc = func(self._empty_acc) if self._empty_acc is not None else None

        new_branches = frozenset(
            (k, v._apply_memo(func, memo)) for k, v in self._branches_dict.items()
        )

        result = LeveledGSS(new_branches, new_empty_acc)
        memo[id(self)] = result
        return result

    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        if self.is_empty():
            return self

        new_empty_acc = self._empty_acc if (self._empty_acc is not None and predicate(self._empty_acc)) else None

        new_branches_dict = {}
        for k, v in self._branches_dict.items():
            pruned_v = v.prune(predicate)
            if not pruned_v.is_empty():
                new_branches_dict[k] = pruned_v

        if not new_branches_dict and new_empty_acc is None:
            return EMPTY_GSS

        return LeveledGSS(frozenset(new_branches_dict.items()), new_empty_acc)

    def merge(self, other: GSS[T, Acc]) -> LeveledGSS[T, Acc]:
        if not isinstance(other, LeveledGSS):
            # Fallback for merging with other GSS types by converting both to reference
            return LeveledGSS.from_stacks(
                self.to_reference_impl()._stacks + other.to_reference_impl()._stacks
            )

        if self.is_empty():
            return other
        if other.is_empty():
            return self

        # 1. Merge empty stack accumulators
        new_empty_acc: Optional[Acc]
        if self._empty_acc is not None and other._empty_acc is not None:
            new_empty_acc = self._empty_acc.merge(other._empty_acc)
        else:
            new_empty_acc = self._empty_acc or other._empty_acc

        # 2. Merge branches
        new_branches_dict = self._branches_dict.copy()
        for key, other_child in other._branches_dict.items():
            if key in new_branches_dict:
                new_branches_dict[key] = new_branches_dict[key].merge(other_child)
            else:
                new_branches_dict[key] = other_child

        return LeveledGSS(frozenset(new_branches_dict.items()), new_empty_acc)

    def peek(self) -> Set[T]:
        return set(self._branches_dict.keys())

    def reduce_acc(self) -> Optional[Acc]:
        accs: List[Acc] = []
        if self._empty_acc is not None:
            accs.append(self._empty_acc)

        for child in self._branches_dict.values():
            child_acc = child.reduce_acc()
            if child_acc is not None:
                accs.append(child_acc)

        if not accs:
            return None

        return reduce(lambda a, b: a.merge(b), accs)

    def to_reference_impl(self) -> ReferenceGSS[T, Acc]:
        return ReferenceGSS.from_stacks(self._to_stacks_recursive([]))

    def _to_stacks_recursive(self, prefix: List[T]) -> List[Tuple[List[T], Acc]]:
        """Helper to recursively walk the graph and flatten it into stacks."""
        stacks = []
        if self._empty_acc is not None:
            stacks.append((prefix, self._empty_acc))

        for key, child_gss in self._branches_dict.items():
            stacks.extend(child_gss._to_stacks_recursive([key] + prefix))

        return stacks

    def _with_empty_acc(self, acc: Optional[Acc]) -> LeveledGSS[T, Acc]:
        """Internal helper to create a new GSS with a different empty_acc."""
        if self._empty_acc == acc:
            return self
        return LeveledGSS(self._branches, acc)


# Singleton for the empty GSS to save memory and allow identity checks
EMPTY_GSS = LeveledGSS()

# --- Singleton Pattern ---
# We monkey-patch the class to return the singleton from __new__ when creating
# an empty instance. This is a common optimization for immutable objects.
_original_new = LeveledGSS.__new__

def _new_gss_new(cls, _branches=frozenset(), _empty_acc=None):
    if not _branches and _empty_acc is None:
        return EMPTY_GSS
    # Mypy has trouble with dynamically changing __new__
    return _original_new(cls)

if not TYPE_CHECKING:
    LeveledGSS.__new__ = _new_gss_new
