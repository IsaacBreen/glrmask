import inspect
from typing import Generator, Tuple, Any
from .reference import GSS


def run_tests() -> Generator[Tuple[dict, int], None, None]:
    # Initial state
    gss = GSS()
    yield gss.get_stacks(), inspect.currentframe().f_lineno

    # Basic push
    gss = gss.push([(gss, 5)])
    yield gss.get_stacks(), inspect.currentframe().f_lineno

    # Multiple pushes
    gss = gss.push([(gss, 10), (gss, 15)])
    yield gss.get_stacks(), inspect.currentframe().f_lineno

    # Pop operation
    gss = gss.pop(1, 15)
    yield gss.get_stacks(), inspect.currentframe().f_lineno

    # Mutate accumulators
    gss.mutate(lambda x: x * 2)
    yield gss.get_stacks(), inspect.currentframe().f_lineno

    # Merge two GSS instances
    other = GSS([([5, 10], 100)])
    merged = gss.merge(other, lambda a, b: a + b)
    yield merged.get_stacks(), inspect.currentframe().f_lineno

    # Peek operation
    _ = merged.peek()
    yield merged.get_stacks(), inspect.currentframe().f_lineno

    # Empty GSS tests
    empty_gss = GSS([])
    yield empty_gss.get_stacks(), inspect.currentframe().f_lineno

    # Pop from empty
    empty_popped = empty_gss.pop(1, 5)
    yield empty_popped.get_stacks(), inspect.currentframe().f_lineno

    # Push with empty inputs
    empty_pushed = empty_gss.push([])
    yield empty_pushed.get_stacks(), inspect.currentframe().f_lineno

    # Mutate empty
    empty_gss.mutate(lambda x: x + 1)
    yield empty_gss.get_stacks(), inspect.currentframe().f_lineno

    # Complex merge scenario
    gss1 = GSS([([1, 2], 10), ([3, 4], 20)])
    gss2 = GSS([([1, 2], 5), ([5, 6], 15)])
    complex_merged = gss1.merge(gss2, lambda a, b: max(a, b))
    yield complex_merged.get_stacks(), inspect.currentframe().f_lineno

    # Push with nested GSS
    nested_gss = GSS([([7], 50)])
    nested_push = gss1.push([(nested_gss, 99)])
    yield nested_push.get_stacks(), inspect.currentframe().f_lineno

    # Final state
    yield nested_push.get_stacks(), inspect.currentframe().f_lineno
