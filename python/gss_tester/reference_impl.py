from typing import List, Tuple, Callable, Set, Iterable, Dict, Any, Type, Optional
from functools import reduce
from .interface import GSS, T, Acc

class ReferenceGSS(GSS[T, Acc]):
    """
    A simple, 'dumb' reference implementation of the GSS interface using a list of explicit stacks.
    Its behavior is the gold standard for the consistency tests.
    """
    def __init__(self, stacks: List[Tuple[List[T], Acc]], root_acc: Acc):
        pass

    @classmethod
    def from_stacks(cls: Type['ReferenceGSS'], stacks: List[Tuple[List[T], Acc]]) -> 'ReferenceGSS[T, Acc]':
        """Creates a GSS from a list of explicit stacks."""
        pass

    def push(self, value: T) -> 'ReferenceGSS[T, Acc]':
        pass

    def pop(self) -> 'ReferenceGSS[T, Acc]':
        pass

    def isolate(self, value: T) -> 'ReferenceGSS[T, Acc]':
        pass

    def apply(self, func: Callable[[Acc], Acc]) -> 'ReferenceGSS[T, Acc]':
        pass

    def prune(self, predicate: Callable[[Acc], bool]) -> 'ReferenceGSS[T, Acc]':
        pass

    def peek(self) -> Set[T]:
        pass

    def reduce_acc(self, merge_func: Callable[[Acc, Acc], Acc]) -> Optional[Acc]:
        """
        Merges the accumulators of all active stacks into a single optional value.
        Returns None if there are no active stacks.
        """
        pass

    @staticmethod
    def merge(gss_list: Iterable['ReferenceGSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'ReferenceGSS[T, Acc]':
        pass

    def to_json_serializable(self) -> Any:
        pass

    def __hash__(self):
        pass

    def is_empty(self) -> bool:
        pass
