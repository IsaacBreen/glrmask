import time
from collections import defaultdict

class Stats:
    """A simple singleton class for collecting performance stats."""
    _instance = None

    def __init__(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.timers = {}
        self.enabled = True  # Can be turned off globally if needed

    @staticmethod
    def get():
        """Get the singleton instance."""
        if Stats._instance is None:
            Stats._instance = Stats()
        return Stats._instance

    def reset(self):
        """Clear all collected stats."""
        self.counts.clear()
        self.times.clear()
        self.timers.clear()

    def inc(self, key: str, value: int = 1):
        """Increment a counter."""
        if not self.enabled: return
        self.counts[key] += value

    def start(self, key: str):
        """Start a timer."""
        if not self.enabled: return
        self.timers[key] = time.perf_counter()

    def stop(self, key: str):
        """Stop a timer and record the duration."""
        if not self.enabled: return
        if key in self.timers:
            self.times[key] += time.perf_counter() - self.timers[key]
            del self.timers[key]

    def report(self):
        """Print a formatted report of all collected stats."""
        if not self.enabled: return
        print("\n--- Performance Stats ---")
        if self.counts:
            print("--- Counts ---")
            for key, value in sorted(self.counts.items()):
                print(f"  {key}: {value}")
        if self.times:
            print("\n--- Timings (ms) ---")
            for key, value in sorted(self.times.items()):
                print(f"  {key}: {value * 1000:.3f}")
        print("-------------------------\n")

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.report()