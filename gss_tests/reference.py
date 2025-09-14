from typing import Callable, Dict, List, Set, Tuple


class GSS:
    def __init__(self, stacks: List[Tuple[List[int], int]] = None):
        self.stacks = stacks if stacks is not None else [([], 0)]

    def push(self, inputs: List[Tuple['GSS', int]]) -> 'GSS':
        new_stacks = []
        for gss, token in inputs:
            for stack, acc in gss.stacks:
                new_stack = (stack + [token], acc)
                new_stacks.append(new_stack)
        return GSS(new_stacks)

    def pop(self, count: int, token: int) -> 'GSS':
        filtered_stacks = [
            (stack, acc) for stack, acc in self.stacks
            if stack and stack[-1] == token
        ]
        return GSS(filtered_stacks)

    def mutate(self, func: Callable[[int], int]) -> None:
        self.stacks = [
            (stack, func(acc)) for stack, acc in self.stacks
        ]

    def get_stacks(self) -> Dict[Tuple[int, ...], int]:
        return {tuple(stack): acc for stack, acc in self.stacks}

    def peek(self) -> Set[int]:
        return {stack[-1] for stack, _ in self.stacks if stack}

    def merge(self, other: 'GSS', merge_func: Callable[[int, int], int]) -> 'GSS':
        merged_dict: Dict[Tuple[int, ...], int] = {}
        
        for stack, acc in self.stacks:
            key = tuple(stack)
            merged_dict[key] = acc
        
        for stack, acc in other.stacks:
            key = tuple(stack)
            if key in merged_dict:
                merged_dict[key] = merge_func(merged_dict[key], acc)
            else:
                merged_dict[key] = acc
                
        merged_stacks = [
            (list(stack), acc) for stack, acc in merged_dict.items()
        ]
        return GSS(merged_stacks)
