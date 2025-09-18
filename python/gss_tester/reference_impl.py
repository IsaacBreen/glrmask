from typing import List, Tuple, Callable, Set, Iterable, Dict, Any, Type
from functools import reduce
from .interface import GSS, T, Acc

class ReferenceGSS(GSS[T, Acc]):
    """
    A simple, 'dumb' reference implementation of the GSS interface using a list of explicit stacks.
    Its behavior is the gold standard for the consistency tests.
    """
    def __init__(self, stacks: List[Tuple[List[T], Acc]], root_acc: Acc):
        self.stacks = stacks
        self._root_acc = root_acc
        if not self.stacks:
            self.stacks.append(([], self._root_acc))

    @classmethod
    def from_stacks(cls: Type['ReferenceGSS'], stacks: List[Tuple[List[T], Acc]]) -> 'ReferenceGSS[T, Acc]':
        """Creates a GSS from a list of explicit stacks."""
        root_acc: Acc = None
        has_empty_stack = False
        for s, acc in stacks:
            if not s:
                has_empty_stack = True
                root_acc = acc
                break
        if not has_empty_stack:
            raise ValueError("ReferenceGSS.from_stacks requires an empty stack to determine the root accumulator.")
        return cls(stacks, root_acc)

    def push(self, value: T) -> 'ReferenceGSS[T, Acc]':
        new_stacks = [(stack + [value], acc) for stack, acc in self.stacks]
        return ReferenceGSS(new_stacks, self._root_acc)

    def pop(self) -> 'ReferenceGSS[T, Acc]':
        new_stacks = []
        for stack, acc in self.stacks:
            if stack:
                new_stacks.append((stack[:-1], acc))
        return ReferenceGSS(new_stacks, self._root_acc)

    def isolate(self, value: T) -> 'ReferenceGSS[T, Acc]':
        new_stacks = []
        for stack, acc in self.stacks:
            if stack and stack[-1] == value:
                new_stacks.append((stack, acc))
        return ReferenceGSS(new_stacks, self._root_acc)

    def apply(self, func: Callable[[Acc], Acc]) -> 'ReferenceGSS[T, Acc]':
        new_stacks = [(stack, func(acc)) for stack, acc in self.stacks]
        return ReferenceGSS(new_stacks, self._root_acc)

    def prune(self, predicate: Callable[[Acc], bool]) -> 'ReferenceGSS[T, Acc]':
        new_stacks = [(stack, acc) for stack, acc in self.stacks if predicate(acc)]
        return ReferenceGSS(new_stacks, self._root_acc)

    def peek(self) -> Set[T]:
        return {stack[-1] for stack, acc in self.stacks if stack}

    def get_acc(self, merge_func: Callable[[Acc, Acc], Acc]) -> Acc:
        """Merges the accumulators of all active stacks into a single value."""
        accumulators = [acc for _, acc in self.stacks]
        return reduce(merge_func, accumulators)

    @staticmethod
    def merge(gss_list: Iterable['ReferenceGSS[T, Acc]'], merge_func: Callable[[Acc, Acc], Acc]) -> 'ReferenceGSS[T, Acc]':
        all_stacks: List[Tuple[List[T], Acc]] = []
        root_acc = None
        
        gss_list = list(gss_list)
        if not gss_list:
            raise ValueError("Cannot merge empty list of GSS")

        root_acc = gss_list[0]._root_acc

        for gss in gss_list:
            if isinstance(gss, ReferenceGSS):
                # Don't add the default empty stack if there are other stacks
                if len(gss.stacks) > 1 or gss.stacks[0][0]:
                    all_stacks.extend(gss.stacks)
        
        if root_acc is None:
             # This can happen if all GSSs were empty.
             root_acc = gss_list[0]._root_acc

        merged_map: Dict[Tuple[T, ...], Acc] = {}
        for stack, acc in all_stacks:
            key = tuple(stack)
            if key in merged_map:
                merged_map[key] = merge_func(merged_map[key], acc)
            else:
                merged_map[key] = acc
        
        final_stacks = [(list(key), acc) for key, acc in merged_map.items()]
        return ReferenceGSS(final_stacks, root_acc)

    def to_json_serializable(self) -> Any:
        # Sort stacks for a canonical representation, making comparisons reliable.
        # Stacks (lists) are converted to tuples to be sortable keys.
        sorted_stacks = sorted(self.stacks, key=lambda x: (tuple(x[0]), x[1]))
        return [{"stack": stack, "acc": acc} for stack, acc in sorted_stacks]

    def __hash__(self):
        return hash(tuple((tuple(stack), acc) for stack, acc in self.stacks))

    def is_empty(self) -> bool:
        return len(self.stacks) == 1 and not self.stacks[0][0]

