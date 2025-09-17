import inspect
from typing import Generator, Tuple, Any, Type
from .interface import GSS

def run_test_spec(gss_class: Type[GSS]) -> Generator[Tuple[Any, int], None, None]:
    """
    A generator that defines a sequence of GSS operations and yields states at
    various points for consistency checking.

    Yields:
        Tuple[Any, int]: A tuple of (json_serializable_gss_state, line_number).
    """
    
    # For this test, the accumulator is an integer and stack items are integers.
    acc_factory = lambda: 0
    merge_func = lambda a, b: a + b

    def _yield_state(gss_state: GSS):
        """Helper to yield the state and the caller's line number."""
        caller_frame = inspect.currentframe().f_back
        line_no = caller_frame.f_lineno
        yield (gss_state.to_json_serializable(), line_no)

    # --- Test 1: Basic push/pop ---
    gss1 = gss_class.initial(acc_factory)
    yield from _yield_state(gss1)

    gss1 = gss1.push(10)
    yield from _yield_state(gss1)

    gss1 = gss1.push(20)
    yield from _yield_state(gss1)

    gss_popped_20 = gss1.isolate(20).pop()
    yield from _yield_state(gss_popped_20)

    gss_popped_10 = gss_popped_20.isolate(10).pop()
    yield from _yield_state(gss_popped_10)

    # --- Test 2: Forking and Merging ---
    gss_a = gss_class.initial(acc_factory).push(1).push(2)
    gss_b = gss_class.initial(acc_factory).push(1).push(3)
    
    merged1 = gss_class.merge([gss_a, gss_b], merge_func)
    yield from _yield_state(merged1)
    
    # Merge a stack that already exists to test accumulator merging
    gss_c = gss_class.initial(acc_factory).push(1).push(2).apply(lambda acc: acc + 5)
    merged2 = gss_class.merge([merged1, gss_c], merge_func)
    yield from _yield_state(merged2)

    # --- Test 3: Apply accumulator function ---
    gss_to_apply = gss_class.initial(acc_factory).push(5)
    gss_to_apply = gss_to_apply.apply(lambda acc: acc + 100)
    yield from _yield_state(gss_to_apply)

    # --- Test 4: Pop with no match ---
    gss_no_match = gss_class.initial(acc_factory).push(88).push(99)
    gss_after_failed_pop = gss_no_match.isolate(1).pop() # This should result in an empty GSS
    yield from _yield_state(gss_after_failed_pop)

    # --- Test 5: Complex sequence (diamond pattern) ---
    s0 = gss_class.initial(acc_factory)
    s1 = s0.push(1)
    s2 = s1.push(2)
    s3 = s1.push(3)
    s4 = gss_class.merge([s2, s3], merge_func)
    yield from _yield_state(s4)
    
    s5 = s4.push(4)
    yield from _yield_state(s5)
    
    s6 = s5.isolate(4).pop()
    yield from _yield_state(s6) # should be same as s4
    
    s7 = s6.isolate(2).pop()
    s8 = s6.isolate(3).pop()
    yield from _yield_state(s7)
    yield from _yield_state(s8)
    
    s9 = gss_class.merge([s7, s8], merge_func)
    yield from _yield_state(s9) # should be same as s1
    
    # --- Test 6: Merging empty GSS ---
    gss_d = gss_class.initial(acc_factory).push(1)
    gss_empty = gss_d.isolate(2).pop() # creates an empty GSS
    yield from _yield_state(gss_empty)
    
    merged_with_empty = gss_class.merge([gss_d, gss_empty], merge_func)
    yield from _yield_state(merged_with_empty) # should be same as gss_d
