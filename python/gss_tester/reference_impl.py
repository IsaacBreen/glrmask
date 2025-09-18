from typing import List, Tuple, Callable, Set, Iterable, Dict, Any, Type
from functools import reduce
from .interface import GSS, T, Acc

class ReferenceGSS(GSS[T, Acc]):
    """
    A simple, 'dumb' reference implementation of the GSS interface using a list of explicit stacks.
    Its behavior is the gold standard for the consistency tests.
    """
    def __init__(self, stacks: List[Tuple[List[T], Acc]], acc_default_factory: Callable[[], Acc]):
        self.stacks = stacks
        self._acc_default_factory = acc_default_factory
        if not self.stacks:
            self.stacks.append(([], self._acc_default_factory()))

    @classmethod
    def initial(cls: Type['ReferenceGSS'], acc_default_factory: Callable[[], Acc]) -> 'ReferenceGSS[T, Acc]':
        """Creates a GSS with a single empty stack and a default accumulator."""
        return cls([], acc_default_factory)

    @classmethod
    def from_stacks(cls: Type['ReferenceGSS'], stacks: List[Tuple[List[T], Acc]], acc_default_factory: Callable[[], Acc]) -> 'ReferenceGSS[T, Acc]':
        """Creates a GSS from a list of explicit stacks."""
        return cls(stacks, acc_default_factory)

    def push(self, value: T) -> 'ReferenceGSS[T, Acc]':
        new_stacks = [(stack + [value], acc) for stack, acc in self.stacks]
        return ReferenceGSS(new_stacks, self._acc_default_factory)

    def pop(self) -> 'ReferenceGSS[T, Acc]':
        new_stacks = []
        for stack, acc in self.stacks:
            if stack:
                new_stacks.append((stack[:-1], acc))

        return ReferenceGSS(new_stacks, self._acc_default_factory)

    def isolate(self, value: T) -> 'ReferenceGSS[T, Acc]':
        new_stacks = []
        for stack, acc in self.stacks:
            if stack and stack[-1] == value:
                new_stacks.append((stack, acc))

        return ReferenceGSS(new_stacks, self._acc_default_factory)

    def apply(self, func: Callable[[Acc], Acc]) -> 'ReferenceGSS[T, Acc]':
        new_stacks = [(stack, func(acc)) for stack, acc in self.stacks]
        return ReferenceGSS(new_stacks, self._acc_default_factory)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'ReferenceGSS[T, Acc]':
        new_stacks = [(stack, acc) for stack, acc in self.stacks if predicate(acc)]
        return ReferenceGSS(new_stacks, self._acc_default_factory)

    def split_heads(self) -> Iterable['ReferenceGSS[T, Acc]']:
        for stack, acc in self.stacks:
            yield ReferenceGSS([(stack, acc)], self._acc_default_factory)

    def peek(self) -> Set[T]:
        return {stack[-1] for stack, acc in self.stacks if stack}

    def get_acc(self, merge_func: Callable[[Acc, Acc], Acc]) -> Acc:
        """Merges the accumulators of all active stacks into a single value."""
        accumulators = [acc for _, acc in self.stacks]
        return reduce(merge_func, accumulators)

    @staticmethod
    def merge(gss_list: Iterable['ReferenceGSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'ReferenceGSS[T, Acc]':
        all_stacks: List[Tuple[List[T], Acc]] = []
        factory = None
        
        gss_list = list(gss_list)
        if not gss_list:
            raise ValueError("Cannot merge empty list of GSS")

        for gss in gss_list:
            if isinstance(gss, ReferenceGSS):
                # Don't add the default empty stack if there are other stacks
                if len(gss.stacks) > 1 or gss.stacks[0][0]:
                    all_stacks.extend(gss.stacks)
                if factory is None:
                    factory = gss._acc_default_factory
        
        if factory is None:
             # This can happen if all GSSs were empty.
             factory = gss_list[0]._acc_default_factory

        merged_map: Dict[Tuple[T, ...], Acc] = {}
        for stack, acc in all_stacks:
            key = tuple(stack)
            if key in merged_map:
                merged_map[key] = merge_func(merged_map[key], acc)
            else:
                merged_map[key] = acc
        
        final_stacks = [(list(key), acc) for key, acc in merged_map.items()]
        return ReferenceGSS(final_stacks, factory)

    def to_json_serializable(self) -> Any:
        # Sort stacks for a canonical representation, making comparisons reliable.
        # Stacks (lists) are converted to tuples to be sortable keys.
        sorted_stacks = sorted(self.stacks, key=lambda x: (tuple(x[0]), x[1]))
        return [{"stack": stack, "acc": acc} for stack, acc in sorted_stacks]

    def __hash__(self):
        return hash(tuple((tuple(stack), acc) for stack, acc in self.stacks))