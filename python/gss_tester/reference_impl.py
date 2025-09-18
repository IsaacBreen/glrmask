from typing import List, Tuple, Callable, Set, Iterable, Any, Type, Optional, Dict
from functools import reduce
from dataclasses import dataclass
import json

from .interface import GSS, T, Acc


@dataclass(eq=False)
class ReferenceGSS(GSS[T, Acc]):
    """
    A simple, 'dumb' reference implementation of the GSS interface using a list of explicit stacks.
    Each stack is represented as a pair: (list_of_values, accumulator).

    Notes on semantics (aligned with GSS interface):
    - from_stacks: constructs a new GSS from explicit stacks.
    - push(value): pushes `value` onto all active stack heads; returns a new GSS.
    - pop(): for all active stacks with at least one element, pop the top; returns those popped stacks.
             Stacks that are already empty are dropped by pop().
    - isolate(value): keeps only stacks whose top value equals `value` (does not modify the stacks).
    - apply(func): transforms each accumulator independently; returns a new GSS.
    - prune(predicate): removes stacks whose accumulator does not satisfy predicate.
    - peek(): returns the set of top values across all stacks (ignores empty stacks).
    - reduce_acc(merge_func): reduces/merges all accumulators into one using `merge_func`, or None
                              if there are no active stacks.
    - merge(gss_list, merge_func): merges multiple GSS instances into one, combining accumulators for
                                   identical stacks using `merge_func`.
    - is_empty(): True iff there is exactly one active stack and it is the empty stack (i.e., []).
    """

    _stacks: List[Tuple[List[T], Acc]]

    def __post_init__(self):
        # Ensure we have our own copies of the stack lists to prevent external mutation.
        self._stacks = [(list(vals), acc) for vals, acc in self._stacks]

    @classmethod
    def from_stacks(cls: Type['ReferenceGSS'], stacks: List[Tuple[List[T], Acc]]) -> 'ReferenceGSS[T, Acc]':
        # The dataclass __init__ will be called, and __post_init__ will handle copying.
        return cls(stacks)

    def push(self, value: T) -> 'ReferenceGSS[T, Acc]':
        # Push `value` onto all stacks (copy each list to avoid mutating original)
        new_stacks: List[Tuple[List[T], Acc]] = []
        for vals, acc in self._stacks:
            new_vals = list(vals)
            new_vals.append(value)
            new_stacks.append((new_vals, acc))
        return ReferenceGSS(new_stacks)

    def pop(self) -> 'ReferenceGSS[T, Acc]':
        # Pop from all non-empty stacks, without merging.
        popped_stacks: List[Tuple[List[T], Acc]] = []
        for vals, acc in self._stacks:
            if vals:
                popped_stacks.append((vals[:-1], acc))
        return ReferenceGSS(popped_stacks)

    def isolate(self, value: Optional[T]) -> 'ReferenceGSS[T, Acc]':
        # Keep only stacks whose top equals `value`, or empty stacks if `value` is None.
        filtered: List[Tuple[List[T], Acc]] = []
        for vals, acc in self._stacks:
            if value is None:
                if not vals:
                    filtered.append((list(vals), acc))
            else:
                if vals and vals[-1] == value:
                    filtered.append((list(vals), acc))
        return ReferenceGSS(filtered)

    def apply(self, func: Callable[[Acc], Acc]) -> 'ReferenceGSS[T, Acc]':
        # Apply func to each accumulator independently
        transformed: List[Tuple[List[T], Acc]] = []
        for vals, acc in self._stacks:
            transformed.append((list(vals), func(acc)))
        return ReferenceGSS(transformed)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'ReferenceGSS[T, Acc]':
        # Keep only stacks where predicate(acc) is True
        kept: List[Tuple[List[T], Acc]] = []
        for vals, acc in self._stacks:
            if predicate(acc):
                kept.append((list(vals), acc))
        return ReferenceGSS(kept)

    def peek(self) -> Set[T]:
        # Return all top values across non-empty stacks
        tops: Set[T] = set()
        for vals, _ in self._stacks:
            if vals:
                tops.add(vals[-1])
        return tops

    def reduce_acc(self, merge_func: Callable[[Acc, Acc], Acc]) -> Optional[Acc]:
        # Reduce all accumulators into a single Acc (or None if no stacks)
        if not self._stacks:
            return None
        accs = [acc for _, acc in self._stacks]
        return reduce(merge_func, accs)

    @staticmethod
    def merge(
        gss_list: Iterable['ReferenceGSS[T, Acc]'],
        merge_func: Callable[[Acc, Acc], Acc]
    ) -> 'ReferenceGSS[T, Acc]':
        # Merge multiple GSS states, combining accumulators for identical stacks (by stack content)
        merged: Dict[Tuple[T, ...], Acc] = {}
        for gss in gss_list:
            for vals, acc in gss._stacks:
                key = tuple(vals)
                if key in merged:
                    merged[key] = merge_func(merged[key], acc)
                else:
                    merged[key] = acc
        result_stacks: List[Tuple[List[T], Acc]] = [(list(key), acc) for key, acc in merged.items()]
        return ReferenceGSS(result_stacks)

    def to_reference_impl(self, merge_func: Callable[[Acc, Acc], Acc]) -> 'ReferenceGSS[T, Acc]':
        """Converts to canonical ReferenceGSS by merging duplicate stacks."""
        return ReferenceGSS.merge([self], merge_func)

    def to_json_serializable(self, merge_func: Callable[[Acc, Acc], Acc]) -> Any:
        # Canonical, deterministic representation: a list of [values, acc] pairs, sorted
        # First, merge any stacks with identical values.
        merged: Dict[Tuple[T, ...], Acc] = {}
        for vals, acc in self._stacks:
            key = tuple(vals)
            if key in merged:
                merged[key] = merge_func(merged[key], acc)
            else:
                merged[key] = acc

        items: List[Tuple[List[T], Acc]] = [(list(key), acc) for key, acc in merged.items()]

        def _encode_for_sort(obj: Any) -> str:
            # Produce a stable string for sorting-comparison, even if obj isn't natively JSON-serializable
            try:
                return json.dumps(obj, sort_keys=True, default=repr, separators=(",", ":"))
            except Exception:
                # Fallback to repr if something goes wrong
                return repr(obj)

        items.sort(key=lambda pair: (_encode_for_sort(pair[0]), _encode_for_sort(pair[1])))
        # Return a plain JSON-serializable structure
        return [[vals, acc] for vals, acc in items]

    def is_empty(self) -> bool:
        # True iff there is exactly one active stack and that stack is empty.
        return len(self._stacks) == 1 and len(self._stacks[0][0]) == 0
