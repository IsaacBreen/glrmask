import time
import inspect
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


class Stats:
    """A simple singleton class for collecting performance stats with grouping.

    Features:
    - Counters via inc(key).
    - Timers via start(key)/stop(key) with duration totals AND hit counts.
    - Optional groups defined by key prefixes, e.g. "get_mask" or "get_mask.seeding".
      Each group reports:
        * All timed members whose keys are equal to the group prefix or start with "prefix.".
        * The group's "hit count": if there is a timer count for the exact group key, use it;
          otherwise, use the maximum timer count among descendant members.
        * For each member: total ms, hits, avg ms per hit, and ms per group hit.
      Notes:
        - Defining nested groups (e.g. "get_mask" and "get_mask.seeding") is supported.
          A key may appear in multiple groups independently. Each group computes its own
          per-group-hit normalization using its own group hit count.
        - Group totals are simple sums of member totals and may double-count if some timers
          are nested within others. These totals are provided to aid quick inspection rather
          than to guarantee exclusivity.
    """
    _instance = None

    def __init__(self):
        # General counters (manual via inc()).
        self.counts: Dict[str, int] = defaultdict(int)

        # Timings: total duration (seconds) per key.
        self.times: Dict[str, float] = defaultdict(float)

        # Timings: hit counts per key (number of successful stop() calls).
        self.time_counts: Dict[str, int] = defaultdict(int)

        # Location of first call for each key: key -> (file, line)
        self.key_positions: Dict[str, Tuple[str, int]] = {}

        # Active timers: key -> start_time
        self.timers: Dict[str, float] = {}

        # Enabled flag lets callers noop the collection if needed.
        self.enabled = True

        # Optional group prefixes (strings).
        self.groups = set()

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
        self.time_counts.clear()
        self.timers.clear()
        self.key_positions.clear()

    def inc(self, key: str, value: int = 1):
        """Increment a counter."""
        if not self.enabled:
            return
        self._record_key_position(key)
        self.counts[key] += value

    def start(self, key: str):
        """Start a timer."""
        if not self.enabled:
            return
        self._record_key_position(key)
        self.timers[key] = time.perf_counter()

    def stop(self, key: str):
        """Stop a timer and record the duration and hit count."""
        if not self.enabled:
            return
        if key in self.timers:
            self.times[key] += time.perf_counter() - self.timers[key]
            self.time_counts[key] += 1
            del self.timers[key]

    def _record_key_position(self, key: str):
        """If seeing a key for the first time, record its call site (file, line)."""
        if self.enabled and key not in self.key_positions:
            # Inspection is expensive, so we only do it once per key.
            try:
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    # We want the caller of inc() or start(), which is one level up.
                    info = inspect.getframeinfo(frame.f_back)
                    self.key_positions[key] = (info.filename, info.lineno)
                else:
                    self.key_positions[key] = ("<unknown>", 0)
            finally:
                # Avoid reference cycles
                del frame

    # -------- Group management --------

    def add_group(self, prefix: str):
        """Register a single group prefix."""
        self.groups.add(prefix)

    def set_groups(self, prefixes: Iterable[str]):
        """Replace current group prefixes with the provided list/iterable."""
        self.groups = set(prefixes)

    def clear_groups(self):
        """Clear configured groups."""
        self.groups.clear()

    # -------- Reporting --------

    def report(self):
        """Print a formatted report of all collected stats."""
        if not self.enabled:
            return

        print("\n--- Performance Stats ---")

        # General counts (manual inc()).
        if self.counts:
            print("--- Counts ---")
            sorted_keys = sorted(self.counts.keys(), key=lambda k: self.key_positions.get(k, ("", 0)))
            rows = [(key, self.counts[key]) for key in sorted_keys]
            self._print_table(
                headers=("key", "count"),
                rows=rows,
                formats=(str, self._fmt_int),
                indent="  "
            )

        # Timings with counts and per-hit averages.
        if self.times:
            print("\n--- Timings (ms) ---")
            rows = []
            sorted_keys = sorted(self.times.keys(), key=lambda k: self.key_positions.get(k, ("", 0)))
            for key in sorted_keys:
                total_ms = self.times[key] * 1000.0
                hits = self.time_counts.get(key, 0)
                avg_ms = (total_ms / hits) if hits else 0.0
                rows.append((key, total_ms, hits, avg_ms))

            self._print_table(
                headers=("key", "total_ms", "hits", "avg_ms"),
                rows=rows,
                formats=(str, self._fmt_ms, self._fmt_int, self._fmt_ms),
                indent="  "
            )

        # Grouped reporting (if any groups are defined)
        if self.groups:
            print("\n--- Groups (prefix-based) ---")
            group_sort_keys = {}
            for g in self.groups:
                members = self._group_members(g)
                if members:
                    # Sort key for a group is the minimum position of its members.
                    min_pos = min(self.key_positions.get(m, ("~", float("inf"))) for m in members)
                    group_sort_keys[g] = min_pos

            sorted_groups = sorted(self.groups, key=lambda g: group_sort_keys.get(g, ("~", float("inf"))))
            for g in sorted_groups:
                members = self._group_members(g)
                if not members:
                    # No timed members under this prefix; skip for brevity
                    continue

                group_hits = self._group_hits(g, members)
                group_total_ms = sum(self.times[k] for k in members) * 1000.0
                group_avg_ms = (group_total_ms / group_hits) if group_hits else 0.0

                print(f"\nGroup: {g}")
                print(f"  members: {len(members)} | group_hits: {self._fmt_int(group_hits)} | group_total_ms: {self._fmt_ms(group_total_ms)} | per_group_hit: {self._fmt_ms(group_avg_ms)}")

                # For each member, show both per-hit and per-group-hit metrics.
                rows = []
                sorted_members = sorted(members, key=lambda k: self.key_positions.get(k, ("", 0)))
                for k in sorted_members:
                    total_ms = self.times[k] * 1000.0
                    hits = self.time_counts.get(k, 0)
                    avg_ms = (total_ms / hits) if hits else 0.0
                    per_group_ms = (total_ms / group_hits) if group_hits else 0.0
                    rows.append((k, total_ms, hits, avg_ms, per_group_ms))

                self._print_table(
                    headers=("member", "total_ms", "hits", "avg_ms", "per_group_hit_ms"),
                    rows=rows,
                    formats=(str, self._fmt_ms, self._fmt_int, self._fmt_ms, self._fmt_ms),
                    indent="    "
                )

        print("-------------------------\n")

    # -------- Helpers --------

    @staticmethod
    def _fmt_int(value: int) -> str:
        """Format integer with thousands separator."""
        return f"{value:,}"

    @staticmethod
    def _fmt_ms(value: float) -> str:
        """Format milliseconds with 3 decimals and thousands separator."""
        return f"{value:,.3f}"

    def _print_table(
        self,
        headers: Tuple[str, ...],
        rows: List[Tuple],
        formats: Tuple = None,
        indent: str = "",
    ):
        """Print a simple aligned table.

        headers: tuple of column names.
        rows: list of tuples aligned with headers.
        formats: tuple of formatter callables (len == len(headers)), applied to each cell.
                 If None, str() is used.
        indent: left indentation for each printed row.
        """
        if not rows:
            return

        ncols = len(headers)
        fmts = formats if formats is not None else tuple([str] * ncols)

        # Determine which columns are numeric (should be right-aligned)
        # Check against the static formatters defined in the class.
        numeric_formatters = (self._fmt_int, self._fmt_ms)
        is_numeric_col = [f in numeric_formatters for f in fmts]

        # Convert cells to strings using provided formatters
        str_rows: List[List[str]] = []
        for r in rows:
            str_row = []
            for i in range(ncols):
                val = r[i]
                str_row.append(fmts[i](val) if callable(fmts[i]) else str(val))
            str_rows.append(str_row)

        # Compute column widths
        widths = [len(h) for h in headers]
        for r in str_rows:
            for i, cell in enumerate(r):
                widths[i] = max(widths[i], len(cell))

        # Print header
        header_line = indent + " | ".join(
            h.rjust(widths[i]) if is_numeric_col[i] else h.ljust(widths[i])
            for i, h in enumerate(headers)
        )
        sep_line = indent + "-+-".join("-" * widths[i] for i in range(ncols))
        print(header_line)
        print(sep_line)

        # Print rows
        for r in str_rows:
            print(
                indent + " | ".join(
                    r[i].rjust(widths[i]) if is_numeric_col[i] else r[i].ljust(widths[i])
                    for i in range(ncols)
                )
            )

    def _group_members(self, prefix: str) -> List[str]:
        """Return all timing keys that belong to the group 'prefix'.

        A key belongs if key == prefix or key.startswith(prefix + ".").
        """
        pfx_dot = prefix + "."
        members = [k for k in self.times.keys() if k == prefix or k.startswith(pfx_dot)]
        return members

    def _group_hits(self, prefix: str, members: List[str]) -> int:
        """Determine the 'group hit count' for a group.

        Rule:
        - If the exact group key has a timing hit count, use that.
        - Else use the maximum timing hit count among its members (descendants).
        This approximates "how many times the group ran" even if the root itself
        wasn't timed explicitly.
        """
        direct = self.time_counts.get(prefix, 0)
        if direct > 0:
            return direct
        max_desc = 0
        for k in members:
            max_desc = max(max_desc, self.time_counts.get(k, 0))
        return max_desc

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.report()
