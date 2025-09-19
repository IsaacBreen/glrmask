import inspect
from typing import Generator, Tuple, Any, Type
from .interface import GSS, MergeableInt
from .fuzzer import run_fuzz_test

def run_test_spec(gss_class: Type[GSS]) -> Generator[Tuple[Any, int], None, None]:
    """
    A generator that defines a sequence of GSS operations and yields states at
    various points for consistency checking.

    Yields:
        Tuple[Any, int]: A tuple of (json_serializable_gss_state, line_number).
    """
    
    # For this test, the accumulator is an integer and stack items are integers.
    acc_factory = lambda: MergeableInt(0)

    def _yield_state(gss_state: GSS):
        """Helper to yield the state and the caller's line number."""
        caller_frame = inspect.currentframe().f_back
        line_no = caller_frame.f_lineno
        yield (gss_state.to_stacks(), line_no)

    # --- Test 1: Basic push/pop ---
    gss1 = gss_class.from_stacks([([], acc_factory())])
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
    gss_a = gss_class.from_stacks([([], acc_factory())]).push(1).push(2)
    gss_b = gss_class.from_stacks([([], acc_factory())]).push(1).push(3)
    
    merged1 = gss_a.merge(gss_b)
    yield from _yield_state(merged1)
    
    # Merge a stack that already exists to test accumulator merging
    gss_c = gss_class.from_stacks([([], acc_factory())]).push(1).push(2).apply(lambda acc: acc + 5)
    merged2 = merged1.merge(gss_c)
    yield from _yield_state(merged2)

    # --- Test 3: Apply accumulator function ---
    gss_to_apply = gss_class.from_stacks([([], acc_factory())]).push(5)
    gss_to_apply = gss_to_apply.apply(lambda acc: acc + 100)
    yield from _yield_state(gss_to_apply)

    # --- Test 4: Pop with no match ---
    gss_no_match = gss_class.from_stacks([([], acc_factory())]).push(88).push(99)
    gss_after_failed_pop = gss_no_match.isolate(1).pop() # This should result in an empty GSS
    yield from _yield_state(gss_after_failed_pop)

    # --- Test 5: Complex sequence (diamond pattern) ---
    s0 = gss_class.from_stacks([([], acc_factory())])
    s1 = s0.push(1)
    s2 = s1.push(2)
    s3 = s1.push(3)
    s4 = s2.merge(s3)
    yield from _yield_state(s4)
    
    s5 = s4.push(4)
    yield from _yield_state(s5)
    
    s6 = s5.isolate(4).pop()
    yield from _yield_state(s6) # should be same as s4
    
    s7 = s6.isolate(2).pop()
    s8 = s6.isolate(3).pop()
    yield from _yield_state(s7)
    yield from _yield_state(s8)
    
    s9 = s7.merge(s8)
    yield from _yield_state(s9) # should be same as s1
    
    # --- Test 6: Merging empty GSS ---
    gss_d = gss_class.from_stacks([([], acc_factory())]).push(1)
    gss_empty = gss_d.isolate(2).pop() # creates an empty GSS
    yield from _yield_state(gss_empty)
    
    merged_with_empty = gss_d.merge(gss_empty)
    yield from _yield_state(merged_with_empty) # should be same as gss_d

    # --- Test 7: Pruning ---
    gss_e = gss_class.from_stacks([([], acc_factory())]).push(1).apply(lambda acc: MergeableInt(5))
    gss_f = gss_class.from_stacks([([], acc_factory())]).push(2).apply(lambda acc: MergeableInt(10))
    gss_to_prune = gss_e.merge(gss_f)
    yield from _yield_state(gss_to_prune)

    pruned_gss = gss_to_prune.prune(lambda acc: acc > 7)
    yield from _yield_state(pruned_gss) # Should only contain stack [2] with acc 10

    pruned_all_gss = gss_to_prune.prune(lambda acc: acc > 100)
    yield from _yield_state(pruned_all_gss) # Should be an empty GSS

    # --- Test 8: Pop operations on merged stacks ---
    # Case 8a: Pop to a common parent, should result in one stack.
    # This exposes a bug in ReferenceGSS which creates duplicate stacks.
    s_base = gss_class.from_stacks([([], acc_factory())]).push(100)
    s_branch1 = s_base.push(101)
    s_branch2 = s_base.push(102)
    s_merged = s_branch1.merge(s_branch2)
    yield from _yield_state(s_merged) # has [100, 101] and [100, 102], both acc 0

    s_popped = s_merged.pop()
    yield from _yield_state(s_popped) # Should have one stack: [100] with acc 0

    # Case 8b: Push after a pop that should have merged.
    # This further highlights the divergence from the duplicate stacks.
    s_pushed_after_pop = s_popped.push(103)
    yield from _yield_state(s_pushed_after_pop) # Should have one stack: [100, 103] acc 0

    # Case 8c: Pop to a common parent with different accumulators.
    # This exposes a bug in FastGSS which loses accumulator information on pop.
    s_base2 = gss_class.from_stacks([([], acc_factory())]).push(200)
    s_branch3 = s_base2.push(201).apply(lambda _: MergeableInt(7))
    s_branch4 = s_base2.push(202).apply(lambda _: MergeableInt(8))
    s_merged2 = s_branch3.merge(s_branch4)
    yield from _yield_state(s_merged2) # has [200, 201] acc 7 and [200, 202] acc 8

    s_popped2 = s_merged2.pop()
    # Should have two stacks: [200] acc 7, and [200] acc 8
    yield from _yield_state(s_popped2)

    # --- Test 9: popn ---
    # Case 9a: popn less than stack depth
    gss_pn1 = gss_class.from_stacks([([], MergeableInt(0))]).push(1).push(2).push(3)
    gss_pn2 = gss_class.from_stacks([([], MergeableInt(0))]).push(4).push(5)
    gss_pn_merged = gss_pn1.merge(gss_pn2)
    yield from _yield_state(gss_pn_merged) # has [1,2,3] and [4,5]

    gss_pn_pop2 = gss_pn_merged.popn(2)
    yield from _yield_state(gss_pn_pop2) # has [1] and []

    # Case 9b: popn more than stack depth

    gss_pn_pop4 = gss_pn_merged.popn(4)
    yield from _yield_state(gss_pn_pop4) # has []

    # --- Test 10: Isolate empty stacks ---
    gss_i1 = gss_class.from_stacks([([], MergeableInt(1))]) # An empty stack
    gss_i2 = gss_class.from_stacks([([10], MergeableInt(2))]) # A non-empty stack
    gss_i_merged = gss_i1.merge(gss_i2)
    yield from _yield_state(gss_i_merged) # has [] acc 1 and [10] acc 2

    gss_isolated_empty = gss_i_merged.isolate(None)
    yield from _yield_state(gss_isolated_empty) # should have only [] acc 1

    gss_isolated_10 = gss_i_merged.isolate(10)
    yield from _yield_state(gss_isolated_10) # should have only [10] acc 2

    # --- Test 11: reduce_acc and peek ---
    gss_rp1 = gss_class.from_stacks([([1, 2], MergeableInt(3))])
    gss_rp2 = gss_class.from_stacks([([1, 3], MergeableInt(4))])
    gss_rp3 = gss_class.from_stacks([([], MergeableInt(5))])
    gss_rp_merged = gss_class.merge_many([gss_rp1, gss_rp2, gss_rp3])
    yield from _yield_state(gss_rp_merged)

    # peek() should return {2, 3}
    peek_result = gss_rp_merged.peek()
    # reduce_acc() should be 3 + 4 + 5 = 12
    reduce_result = gss_rp_merged.reduce_acc()
    
    # We can't yield non-GSS objects, so we'll create GSSs that represent the result
    # This is a bit of a hack, but it allows us to check for consistency.
    # A GSS with a single empty stack and the peek result as the accumulator.
    gss_peek_check = gss_class.from_stacks([([], MergeableInt(len(peek_result)))])
    yield from _yield_state(gss_peek_check)

    # A GSS with a single empty stack and the reduce_acc result as the accumulator.
    gss_reduce_check = gss_class.from_stacks([([], reduce_result if reduce_result is not None else MergeableInt(-1))])
    yield from _yield_state(gss_reduce_check)

    # Test on empty GSS
    gss_rp_empty = gss_class.from_stacks([])
    yield from _yield_state(gss_rp_empty)
    reduce_empty_result = gss_rp_empty.reduce_acc() # Should be None
    gss_reduce_empty_check = gss_class.from_stacks([([], MergeableInt(-1 if reduce_empty_result is None else 999))])
    yield from _yield_state(gss_reduce_empty_check)


    # --- Test 12: merge_many ---
    gss_m1 = gss_class.from_stacks([(['a'], MergeableInt(1))])
    gss_m2 = gss_class.from_stacks([(['a'], MergeableInt(2)), (['b'], MergeableInt(3))])
    gss_m3 = gss_class.from_stacks([(['c'], MergeableInt(4))])
    
    gss_merged_many = gss_class.merge_many([gss_m1, gss_m2, gss_m3])
    # Expected: [a] acc 3, [b] acc 3, [c] acc 4
    yield from _yield_state(gss_merged_many)

    # --- Test 13: Pop from GSS containing an empty stack ---
    gss_pe1 = gss_class.from_stacks([([], MergeableInt(1))])
    gss_pe2 = gss_class.from_stacks([([10], MergeableInt(2))])
    gss_pe_merged = gss_pe1.merge(gss_pe2)
    yield from _yield_state(gss_pe_merged) # has [] acc 1 and [10] acc 2

    gss_pe_popped = gss_pe_merged.pop()
    # pop should only affect non-empty stacks.
    # Expected: [] acc 1, [] acc 2 -> merged to [] acc 3
    yield from _yield_state(gss_pe_popped)

    # --- Test 14: String values and popn(0) ---
    gss_s1 = gss_class.from_stacks([(['x', 'y'], MergeableInt(1))])
    yield from _yield_state(gss_s1)

    gss_s_pop0 = gss_s1.popn(0) # Should be a no-op
    yield from _yield_state(gss_s_pop0)

    gss_s_pop1 = gss_s1.popn(1)
    yield from _yield_state(gss_s_pop1)

    # --- Test 15: Operations on Empty GSS ---
    gss_empty_start = gss_class.from_stacks([])
    yield from _yield_state(gss_empty_start)

    gss_empty_pushed = gss_empty_start.push(1) # Should still be empty
    yield from _yield_state(gss_empty_pushed)

    gss_empty_popped = gss_empty_start.pop() # Should still be empty
    yield from _yield_state(gss_empty_popped)

    gss_empty_isolated = gss_empty_start.isolate(1) # Should still be empty
    yield from _yield_state(gss_empty_isolated)

    gss_empty_applied = gss_empty_start.apply(lambda acc: acc + 1) # Should still be empty
    yield from _yield_state(gss_empty_applied)

    gss_empty_pruned = gss_empty_start.prune(lambda acc: True) # Should still be empty
    yield from _yield_state(gss_empty_pruned)

    # --- Test 16: Fuzz tests ---
    # Run a short, seeded fuzz test to catch more complex interactions.
    # The seed ensures the test is deterministic.
    fuzz_seed = 42
    fuzz_steps = 100
    for gss_state in run_fuzz_test(gss_class, seed=fuzz_seed, num_steps=fuzz_steps):
        yield from _yield_state(gss_state)
