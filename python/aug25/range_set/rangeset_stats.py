import time
import collections
from typing import Callable, Any

# Wall-time per labeled operation
RANGESET_STATS = collections.defaultdict(float)   # label -> seconds
# Call counts per labeled operation
RANGESET_CALLS = collections.defaultdict(int)     # label -> calls
# Additional numeric metrics (sums/counters); keys are arbitrary metric names
RANGESET_METRICS = collections.defaultdict(float) # metric_name -> value

def reset_rangeset_stats() -> None:
    RANGESET_STATS.clear()
    RANGESET_CALLS.clear()
    RANGESET_METRICS.clear()

def record_metric(name: str, value: float = 1.0) -> None:
    RANGESET_METRICS[name] += float(value)

def time_method(func: Callable) -> Callable:
    """
    Decorator for instance methods; label includes the runtime class name.
    """
    def wrapper(*args, **kwargs):
        # Expecting instance method, so args[0] is self
        self = args[0] if args else None
        cls_name = type(self).__name__ if self is not None else "<unknown>"
        label = f"{cls_name}.{func.__name__}"
        t0 = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            RANGESET_STATS[label] += time.time() - t0
            RANGESET_CALLS[label] += 1
    return wrapper

def time_func(label: str) -> Callable[[Callable], Callable]:
    """
    Decorator for static functions or for forcing a specific label.
    """
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                RANGESET_STATS[label] += time.time() - t0
                RANGESET_CALLS[label] += 1
        return wrapper
    return deco

def print_rangeset_stats() -> None:
    print("\n--- RangeSet stats ---")
    total_time = sum(RANGESET_STATS.values())
    # Times and calls
    for label in sorted(RANGESET_STATS.keys()):
        v = RANGESET_STATS[label]
        calls = RANGESET_CALLS.get(label, 0)
        if total_time > 1e-6:
            print(f"t_{label:<30}: {v:8.4f}s ({v/total_time*100:5.1f}%) | calls: {calls}")
        else:
            print(f"t_{label:<30}: {v:8.4f}s | calls: {calls}")
    # Additional metrics
    if RANGESET_METRICS:
        print("\nRangeSet metrics (sums/counters):")
        for k in sorted(RANGESET_METRICS.keys()):
            v = RANGESET_METRICS[k]
            # print ints where applicable
            if abs(v - int(v)) < 1e-9:
                print(f"{k:<35}: {int(v)}")
            else:
                print(f"{k:<35}: {v:.4f}")
    print("----------------------\n")
