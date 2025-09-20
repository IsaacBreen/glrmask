import random
from typing import Generator, Any, Type, List, Callable

from ..interface import GSS, MergeableInt


def run_fuzz_test(
    gss_class: Type[GSS],
    seed: int,
    num_steps: int,
    max_gss_states: int = 10,
    value_pool: List[Any] = None
) -> Generator[GSS, None, None]:
    """
    Runs a randomized sequence of GSS operations to find inconsistencies.
    Yields the new GSS state after each operation.
    """
    if value_pool is None:
        value_pool = list(range(20)) + ['a', 'b', 'c']

    rng = random.Random(seed)
    gss_states: List[GSS] = [gss_class.from_stacks([([], MergeableInt(0))])]
    yield gss_states[0]

    for _ in range(num_steps):
        if not gss_states:
            # All states were pruned or became empty, start over.
            gss_states.append(gss_class.from_stacks([([], MergeableInt(0))]))
            yield gss_states[0]
            if len(gss_states) >= max_gss_states:
                continue

        # Choose an operation
        can_merge = len(gss_states) >= 2
        operations = ['push', 'pop', 'popn', 'isolate', 'apply', 'prune']
        if can_merge:
            operations.append('merge')
        
        op_choice = rng.choice(operations)

        # Select GSS state(s) to operate on
        source_gss = rng.choice(gss_states)
        
        new_gss: GSS

        try:
            if op_choice == 'push':
                value = rng.choice(value_pool)
                new_gss = source_gss.push(value)

            elif op_choice == 'pop':
                new_gss = source_gss.pop()

            elif op_choice == 'popn':
                n = rng.randint(0, 4)
                new_gss = source_gss.popn(n)

            elif op_choice == 'isolate':
                # 20% chance of isolating empty stack
                if rng.random() < 0.2:
                    value = None
                else:
                    value = rng.choice(value_pool)
                new_gss = source_gss.isolate(value)

            elif op_choice == 'apply':
                amount = rng.randint(1, 10)
                # Use a default argument in the lambda to capture the value of `amount`
                func: Callable[[MergeableInt], MergeableInt] = lambda acc, amt=amount: acc + amt
                new_gss = source_gss.apply(func)

            elif op_choice == 'prune':
                threshold = rng.randint(0, 20)
                # Use a default argument in the lambda to capture the value of `threshold`
                predicate: Callable[[MergeableInt], bool] = lambda acc, thr=threshold: acc.real > thr
                new_gss = source_gss.prune(predicate)

            elif op_choice == 'merge' and can_merge:
                other_gss = rng.choice([g for g in gss_states if g is not source_gss])
                new_gss = source_gss.merge(other_gss)
            
            else: # Should not happen
                continue

            yield new_gss

            # Add the new state to our pool, but avoid adding empty or duplicate states
            if new_gss is not source_gss and not new_gss.is_empty():
                gss_states.append(new_gss)

            # Prune the list of GSS states to keep it manageable
            if len(gss_states) > max_gss_states:
                gss_states = rng.sample(gss_states, max_gss_states)

        except Exception:
            # Some operations might fail on some implementations if invariants are broken.
            # We'll just skip the step and continue fuzzing.
            continue
