import random
import json
from typing import Generator, Any, Type, List, Callable, Tuple, Dict, Optional, Set

from ..interface import GSS, MergeableInt


def run_fuzz_test(
    gss_class: Type[GSS],
    seed: int,
    num_steps: int,
    max_gss_states: int = 10,
    value_pool: List[Any] = None
) -> Generator[Tuple[GSS, Dict[str, Any]], None, None]:
    """
    Runs a randomized sequence of GSS operations to find inconsistencies.
    Yields (new_gSS_state, trace) after each operation.
    The trace includes operation name, arguments, source stacks, result stacks, and more.
    """
    if value_pool is None:
        value_pool = list(range(20)) + ['a', 'b', 'c']

    ALL_OPERATIONS = ['push', 'pop', 'popn', 'isolate', 'apply', 'prune', 'merge']

    def _state_sig(stacks: List[Any]) -> str:
        # Canonical fingerprint of a GSS state, independent of object identity
        return json.dumps(stacks, sort_keys=True, ensure_ascii=False)

    rng = random.Random(seed)
    gss_states: List[GSS] = [gss_class.from_stacks([([], MergeableInt(0))])]
    step_idx = 0
    yield gss_states[0], {
        "phase": "fuzz",
        "op": "init",
        "step": step_idx,
        "seed": seed,
        "args": {},
        "source_index": None,
        "other_index": None,
        "pool_size_before": 0,
        "pool_size_after": len(gss_states),
        "source_stacks": None,
        "other_stacks": None,
        "result_stacks": gss_states[0].to_stacks(),
        "added_to_pool": False,
    }

    # Track pool membership by structural signature so identity-only changes
    # don’t perturb RNG/control flow.
    pool_sigs: Set[str] = set()
    pool_sigs.add(_state_sig(gss_states[0].to_stacks()))

    for _ in range(num_steps):
        if not gss_states:
            # All states were pruned or became empty, start over.
            gss_states.append(gss_class.from_stacks([([], MergeableInt(0))]))
            step_idx += 1
            yield gss_states[0], {
                "phase": "fuzz",
                "op": "restart_empty_pool",
                "step": step_idx,
                "seed": seed,
                "args": {},
                "source_index": None,
                "other_index": None,
                "pool_size_before": 0,
                "pool_size_after": len(gss_states),
                "source_stacks": None,
                "other_stacks": None,
                "result_stacks": gss_states[0].to_stacks(),
                "added_to_pool": False,
            }
            # Reset pool signatures on restart
            pool_sigs = {_state_sig(gss_states[0].to_stacks())}
            if len(gss_states) >= max_gss_states:
                continue

        # Choose an operation
        op_choice = rng.choice(ALL_OPERATIONS)

        # Skip impossible operations
        if op_choice == 'merge' and len(gss_states) < 2:
            continue

        # Select GSS state(s) to operate on
        source_index = rng.randrange(len(gss_states))
        source_gss = gss_states[source_index]
        source_stacks = source_gss.to_stacks()
        
        new_gss: GSS
        args: Dict[str, Any] = {}
        other_stacks: Optional[List[Any]] = None
        other_index: Optional[int] = None

        try:
            if op_choice == 'push':
                value = rng.choice(value_pool)
                new_gss = source_gss.push(value)
                args = {"value": value}

            elif op_choice == 'pop':
                new_gss = source_gss.pop()

            elif op_choice == 'popn':
                n = rng.randint(0, 4)
                new_gss = source_gss.popn(n)
                args = {"n": n}

            elif op_choice == 'isolate':
                # 20% chance of isolating empty stack
                if rng.random() < 0.2:
                    value = None
                else:
                    value = rng.choice(value_pool)
                new_gss = source_gss.isolate(value)
                args = {"value": value}

            elif op_choice == 'apply':
                amount = rng.randint(1, 10)
                # Use a default argument in the lambda to capture the value of `amount`
                func: Callable[[MergeableInt], MergeableInt] = lambda acc, amt=amount: acc + amt
                new_gss = source_gss.apply(func)
                args = {"amount": amount}

            elif op_choice == 'prune':
                threshold = rng.randint(0, 20)
                # Use a default argument in the lambda to capture the value of `threshold`
                predicate: Callable[[MergeableInt], bool] = lambda acc, thr=threshold: acc.real > thr
                new_gss = source_gss.prune(predicate)
                args = {"threshold": threshold}

            elif op_choice == 'merge':
                candidates = [i for i in range(len(gss_states)) if i != source_index]
                other_index = rng.choice(candidates)
                other_gss = gss_states[other_index]
                new_gss = source_gss.merge(other_gss)
                other_stacks = other_gss.to_stacks()
                args = {"other_index": other_index}
            
            else: # Should not happen
                continue

            pool_size_before = len(gss_states)
            step_idx += 1
            result_stacks = new_gss.to_stacks()
            new_sig = _state_sig(result_stacks)
            # Add the new state to our pool, but avoid adding empty or duplicate states
            added_to_pool = False
            if not new_gss.is_empty():
                if new_sig not in pool_sigs:
                    gss_states.append(new_gss)
                    pool_sigs.add(new_sig)
                    added_to_pool = True

            yield new_gss, {
                "phase": "fuzz",
                "op": op_choice,
                "step": step_idx,
                "seed": seed,
                "args": args,
                "source_index": source_index,
                "other_index": other_index,
                "pool_size_before": pool_size_before,
                "pool_size_after": len(gss_states),
                "source_stacks": source_stacks,
                "other_stacks": other_stacks,
                "result_stacks": result_stacks,
                "added_to_pool": added_to_pool,
            }

            # Prune the list of GSS states to keep it manageable
            if len(gss_states) > max_gss_states:
                gss_states = rng.sample(gss_states, max_gss_states)
                # Rebuild the signature set to match the pruned pool
                pool_sigs = {_state_sig(s.to_stacks()) for s in gss_states}

        except Exception:
            # Some operations might fail on some implementations if invariants are broken.
            # We'll just skip the step and continue fuzzing.
            continue
