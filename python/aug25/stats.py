import time
import inspect
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Optional
import statistics
import math
import os
import contextlib
import functools


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

        # Timings: list of durations (seconds) per key.
        self.durations: Dict[str, List[float]] = defaultdict(list)

        # Location of first call for each key: key -> (file, line)
        self.key_positions: Dict[str, Tuple[str, int]] = {}

        # Active timers: key -> start_time
        self.timers: Dict[str, float] = {}

        # Stack for tracking keys started within a generator scope.
        self.scope_stack: List[set] = []

        # Enabled flag lets callers noop the collection if needed.
        self.enabled = os.environ.get("DISABLE_STATS") is None

        # Optional group prefixes (strings).
        self.groups = set()

        # Flag to control printing of the 'All Stats' table (default: False)
        self.show_all_stats = False

    @staticmethod
    def get():
        """Get the singleton instance."""
        if Stats._instance is None:
            Stats._instance = Stats()
        return Stats._instance

    def reset(self):
        """Clear all collected stats."""
        self.counts.clear()
        self.durations.clear()
        self.timers.clear()
        self.key_positions.clear()
        self.scope_stack.clear()
        self.show_all_stats = False

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
        if self.scope_stack:
            self.scope_stack[-1].add(key)
        self.timers[key] = time.perf_counter()

    def stop(self, key: str):
        """Stop a timer and record the duration and hit count."""
        if not self.enabled:
            return
        if key in self.timers:
            duration = time.perf_counter() - self.timers[key]
            self.durations[key].append(duration)
            del self.timers[key]

    @contextlib.contextmanager
    def scope(self):
        """A context manager to ensure timers started within it are stopped.

        Useful for generators that might not run to completion.
        """
        if not self.enabled:
            yield
            return

        self.scope_stack.append(set())
        try:
            yield
        finally:
            if self.scope_stack:  # Check if not empty (e.g. after a reset)
                my_scope_keys = self.scope_stack.pop()
                active_keys = set(self.timers.keys())
                keys_to_stop = my_scope_keys & active_keys
                for key in keys_to_stop:
                    self.stop(key)

    def _record_key_position(self, key: str):
        """If seeing a key for the first time, record its call site (file, line)."""
        if self.enabled and key not in self.key_positions:
            # Inspection is expensive. Adjust active timers to account for the pause.
            pause_start = time.perf_counter()

            # --- Expensive operation ---
            caller_frame = None
            try:
                frame = inspect.currentframe()
                # The call stack is:
                #   - model code (the one we want)
                #   - inc() or start() in this class
                #   - _record_key_position() (current frame)
                # So we need to go up two levels.
                if frame and frame.f_back and frame.f_back.f_back:
                    caller_frame = frame.f_back.f_back
                    info = inspect.getframeinfo(caller_frame)
                    self.key_positions[key] = (info.filename, info.lineno)
                else:
                    self.key_positions[key] = ("<unknown>", 0)
            finally:
                # Avoid reference cycles
                del frame
                if caller_frame:
                    del caller_frame
            # --- End of expensive operation ---

            # Adjust start time of all active timers
            pause_duration = time.perf_counter() - pause_start
            for k in self.timers:
                self.timers[k] += pause_duration

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

    # -------- Reporting Configuration --------
    def set_show_all_stats(self, value: bool):
        """Set whether to include the 'All Stats' table in the report."""
        self.show_all_stats = value

    # -------- Reporting --------

    def report(self, sort_by: str = "position"):
        """Print a formatted report of all collected stats.

        Args:
            sort_by (str): Sorting order for keys.
                - "position": Sort by order of appearance in the code (default).
                - "alpha": Sort alphabetically.
        """
        if not self.enabled:
            return

        # --- Data Preparation Phase ---
        # This phase gathers all data and rows for all tables before printing.

        # 1. Prepare combined stats data  
        stats_headers = ("key", "hits", "total_ms", "mean_ms", "stdev_ms", "min_ms", "p50_ms", "p95_ms", "max_ms")
        stats_formats = (str, self._fmt_int_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank)
        stats_rows = []
        if self.show_all_stats:
            all_keys = self.counts.keys() | self.durations.keys()
            if all_keys:
                if sort_by == "alpha":
                    sorted_keys = sorted(all_keys)
                else:  # Default to "position"
                    sorted_keys = sorted(all_keys, key=lambda k: self.key_positions.get(k, ("", 0)))
                for key in sorted_keys:
                    # 'hits' is timer hits if available, otherwise counter value.
                    hits = None
                    if key in self.durations:
                        hits = len(self.durations[key])

                    if hits is None:
                        hits = self.counts.get(key)

                    if key in self.durations:
                        durs_ms = [d * 1000.0 for d in self.durations[key]]
                        if not durs_ms:
                            stats_rows.append((key, 0, 0.0, None, None, None, None, None, None))
                            continue

                        total_ms = sum(durs_ms)
                        mean_ms = statistics.mean(durs_ms)
                        min_ms = min(durs_ms)
                        max_ms = max(durs_ms)
                        if len(durs_ms) > 1:
                            stdev_ms = statistics.stdev(durs_ms)
                            # Use median for p50, more robust for even-sized lists
                            p50_ms = statistics.median(durs_ms)
                            # quantiles() is new in 3.8. It splits data into n+1 intervals.
                            # For percentiles (n=100), it gives 99 points. p95 is index 94.
                            qs = statistics.quantiles(durs_ms, n=100)
                            p95_ms = qs[94]
                        else:
                            stdev_ms = 0.0
                            p50_ms = mean_ms
                            p95_ms = mean_ms
                        stats_rows.append((key, hits, total_ms, mean_ms, stdev_ms, min_ms, p50_ms, p95_ms, max_ms))
                    else:
                        # It's a counter-only key
                        stats_rows.append((key, hits, None, None, None, None, None, None, None))

        # 2. Prepare Groups data
        groups_data = []
        group_members_headers = ("member", "hits", "hits/group_hit", "total_ms", "mean_ms", "stdev_ms", "min_ms", "max_ms", "ms/group_hit")
        group_members_formats = (str, self._fmt_int_or_blank, self._fmt_ratio_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank, self._fmt_ms_or_blank)
        if self.groups:
            group_sort_keys = {}
            for g in self.groups:
                members = self._group_members(g)
                if members:
                    min_pos = min(self.key_positions.get(m, ("~", float("inf"))) for m in members)
                    group_sort_keys[g] = min_pos

            if sort_by == "alpha":
                sorted_groups = sorted(self.groups)
            else:  # Default to "position"
                sorted_groups = sorted(self.groups, key=lambda g: group_sort_keys.get(g, ("~", float("inf"))))
            for g in sorted_groups:
                all_members = self._group_members(g)
                if not all_members:
                    continue

                timing_members = [m for m in all_members if m in self.durations]

                group_hits = self._group_hits(g, timing_members)
                group_total_ms = sum(sum(self.durations.get(k, [])) for k in timing_members) * 1000.0
                group_avg_ms = (group_total_ms / group_hits) if group_hits else 0.0

                group_info = {
                    "name": g,
                    "members_count": len(all_members),
                    "timing_members_count": len(timing_members),
                    "hits": group_hits,
                    "total_ms": group_total_ms,
                    "avg_ms": group_avg_ms,
                }

                member_rows = []
                if all_members:
                    if sort_by == "alpha":
                        sorted_members = sorted(all_members)
                    else:  # Default to "position"
                        sorted_members = sorted(all_members, key=lambda k: self.key_positions.get(k, ("", 0)))
                    for k in sorted_members:
                        # 'hits' is timer hits if available, otherwise counter value.
                        hits = None
                        if k in self.durations:
                            hits = len(self.durations[k])

                        if hits is None:
                            hits = self.counts.get(k)

                        hits_per_group_hit = None
                        if hits is not None and group_hits > 0:
                            hits_per_group_hit = hits / group_hits

                        if k in self.durations:
                            durs_ms = [d * 1000.0 for d in self.durations[k]]
                            if not durs_ms:
                                member_rows.append((k, 0, hits_per_group_hit, 0.0, None, None, None, None, 0.0))
                                continue

                            total_ms = sum(durs_ms)
                            mean_ms = statistics.mean(durs_ms)
                            min_ms = min(durs_ms)
                            max_ms = max(durs_ms)
                            if len(durs_ms) > 1:
                                stdev_ms = statistics.stdev(durs_ms)
                            else:
                                stdev_ms = 0.0
                            per_group_ms = (total_ms / group_hits) if group_hits else 0.0
                            member_rows.append((k, hits, hits_per_group_hit, total_ms, mean_ms, stdev_ms, min_ms, max_ms, per_group_ms))
                        else:
                            # Counter-only member
                            member_rows.append((k, hits, hits_per_group_hit, None, None, None, None, None, None))

                groups_data.append({"info": group_info, "member_rows": member_rows})

        # --- Width Calculation Phase ---
        # This phase determines the max width for each column across all tables.
        max_widths = defaultdict(int)

        def update_widths(headers, rows, formats):
            fmts = formats if formats is not None else tuple([str] * len(headers))
            for i, h in enumerate(headers):
                max_widths[h] = max(max_widths[h], len(h))
            for r in rows:
                for i, h in enumerate(headers):
                    val = r[i]
                    cell_str = fmts[i](val) if callable(fmts[i]) else str(val)
                    max_widths[h] = max(max_widths[h], len(cell_str))

        if stats_rows: update_widths(stats_headers, stats_rows, stats_formats)
        if groups_data:
            for group in groups_data:
                if group["member_rows"]:
                    update_widths(group_members_headers, group["member_rows"], group_members_formats)

        # Unify 'key' and 'member' widths for consistent alignment
        if 'key' in max_widths or 'member' in max_widths:
            unified_width = max(max_widths.get('key', 0), max_widths.get('member', 0))
            max_widths['key'] = unified_width
            max_widths['member'] = unified_width

        # --- Printing Phase ---
        # This phase prints all the tables using the pre-calculated widths.

        print("\n═══ Performance Stats ═══")

        if self.show_all_stats and stats_rows:
            print("\n▶ All Stats")
            self._print_table(headers=stats_headers, rows=stats_rows, formats=stats_formats, indent="  ", widths=max_widths)

        if self.groups and groups_data:
            print("\n▶ Groups")
            for group in groups_data:
                info = group["info"]
                print(f"\n  [{info['name']}]")
                summary_parts = []
                if info['timing_members_count'] > 0:
                    summary_parts.append(f"{info['members_count']} members")
                    summary_parts.append(f"{self._fmt_int(info['hits'])} hits")
                    summary_parts.append(f"{self._fmt_ms(info['total_ms'])}ms total")
                    summary_parts.append(f"{self._fmt_ms(info['avg_ms'])}ms/hit")
                print(f"    {' · '.join(summary_parts)}")

                if group["member_rows"]:
                    self._print_table(headers=group_members_headers, rows=group["member_rows"], formats=group_members_formats, indent="    ", widths=max_widths)

        print("\n═════════════════════════\n")

    # -------- Helpers --------

    @staticmethod
    def _fmt_int(value: int) -> str:
        """Format integer with thousands separator."""
        return f"{value:,}"

    @staticmethod
    def _fmt_ms(value: float) -> str:
        """Format milliseconds with 3 decimals and thousands separator."""
        return f"{value:,.3f}"

    @staticmethod
    def _fmt_int_or_blank(value: Optional[int]) -> str:
        """Format integer with thousands separator, or return blank string if None."""
        if value is None:
            return ""
        return f"{value:,}"

    @staticmethod
    def _fmt_ms_or_blank(value: Optional[float]) -> str:
        """Format ms with 3 decimals and thousands separator, or return blank string if None."""
        if value is None:
            return ""
        return f"{value:,.3f}"

    @staticmethod
    def _fmt_ratio_or_blank(value: Optional[float]) -> str:
        """Format a ratio with 2 decimals, or return blank string if None."""
        if value is None:
            return ""
        return f"{value:.2f}"

    def _print_table(
        self,
        headers: Tuple[str, ...],
        rows: List[Tuple],
        formats: Tuple = None,
        indent: str = "",
        widths: Dict[str, int] = None,
    ):
        """Print a simple aligned table.

        headers: tuple of column names.
        rows: list of tuples aligned with headers.
        formats: tuple of formatter callables (len == len(headers)), applied to each cell.
                 If None, str() is used.
        indent: left indentation for each printed row.
        widths: optional dict of {header: width} to enforce column widths.
        """
        if not rows:
            return

        ncols = len(headers)
        fmts = formats if formats is not None else tuple([str] * ncols)

        # Determine which columns are numeric (should be right-aligned)
        # Check against the static formatters defined in the class.
        numeric_formatters = (self._fmt_int, self._fmt_ms, self._fmt_int_or_blank, self._fmt_ms_or_blank, self._fmt_ratio_or_blank)
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
        if widths:
            col_widths = [widths[h] for h in headers]
        else:
            col_widths = [len(h) for h in headers]
            for r in str_rows:
                for i, cell in enumerate(r):
                    col_widths[i] = max(col_widths[i], len(cell))

        # Print header
        header_line = indent + " | ".join(
            h.rjust(col_widths[i]) if is_numeric_col[i] else h.ljust(col_widths[i])
            for i, h in enumerate(headers)
        )
        sep_line = indent + "-+-".join("-" * col_widths[i] for i in range(ncols))
        print(header_line)
        print(sep_line)

        # Print rows
        for r in str_rows:
            print(
                indent + " | ".join(
                    r[i].rjust(col_widths[i]) if is_numeric_col[i] else r[i].ljust(col_widths[i])
                    for i in range(ncols)
                )
            )

    def _group_members(self, prefix: str) -> List[str]:
        """Return all timing and count keys that belong to the group 'prefix'.

        A key belongs if key == prefix or key.startswith(prefix + ".").
        """
        pfx_dot = prefix + "."
        all_keys = self.durations.keys() | self.counts.keys()
        members = [k for k in all_keys if k == prefix or k.startswith(pfx_dot)]
        return members

    def _group_hits(self, prefix: str, members: List[str]) -> int:
        """Determine the 'group hit count' for a group.

        Rule:
        - If the exact group key has a timing hit count, use that.
        - Else use the maximum timing hit count among its members (descendants).
        This approximates "how many times the group ran" even if the root itself
        wasn't timed explicitly.
        """
        direct = len(self.durations.get(prefix, []))
        if direct > 0:
            return direct
        max_desc = 0
        for k in members:
            max_desc = max(max_desc, len(self.durations.get(k, [])))
        return max_desc

    @property
    def times(self) -> Dict[str, float]:
        """Backwards compatibility: returns total duration (seconds) per key."""
        return {k: sum(v) for k, v in self.durations.items()}

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.report()


def stats_generator(func):
    """Decorator for timing generators.

    Features:
    1. Ensures any timers started within the generator are eventually stopped,
       even if the generator is not fully exhausted (via `Stats.scope`).
    2. Pauses any active timers during `yield`. This means that time spent
       in the consumer, between generator iterations, is excluded from timer
       durations. This applies to timers started both inside and outside
       the generator.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        stats = Stats.get()
        if not stats.enabled:
            yield from func(*args, **kwargs)
            return

        # The scope ensures cleanup if the generator is abandoned mid-iteration.
        with stats.scope():
            gen = func(*args, **kwargs)
            yielded_value = None
            try:
                # Prime the generator to the first yield.
                yielded_value = next(gen)

                while True:
                    # `gen` is now paused at a `yield`. We have the value.
                    # Before we yield to the consumer, "pause" all active timers.
                    active_timers_before_yield = dict(stats.timers)
                    pause_start_time = time.perf_counter()

                    try:
                        # Yield to the consumer and get a value back from `send()`.
                        sent_value = yield yielded_value
                    except GeneratorExit:
                        # Consumer called close(). Pass it on and exit.
                        gen.close()
                        raise
                    except BaseException as e:
                        # Consumer called throw(). Pass it on. The generator might
                        # handle it and yield a new value, or it might raise.
                        try:
                            yielded_value = gen.throw(type(e), e, e.__traceback__)
                            continue  # Loop to pause and yield the new value
                        except StopIteration:
                            return  # Generator finished
                        except BaseException:
                            raise  # Generator re-raised, propagate up

                    # Resumed from consumer. Calculate pause duration.
                    pause_duration = time.perf_counter() - pause_start_time

                    # Adjust start times for all timers that were active before we yielded.
                    # This logic correctly handles nested adjustments (e.g. from _record_key_position)
                    # that may have occurred in the consumer while we were paused, preventing
                    # double-counting of pause time which can lead to negative durations.
                    for k, original_start_time in active_timers_before_yield.items():
                        if k in stats.timers:  # Check if it wasn't stopped by the consumer
                            current_start_time = stats.timers[k]
                            adjustment_during_pause = current_start_time - original_start_time
                            needed_adjustment = pause_duration - adjustment_during_pause
                            stats.timers[k] += needed_adjustment

                    # Resume the generator by sending the value from the consumer.
                    yielded_value = gen.send(sent_value)

            except StopIteration:
                # Generator finished, either during priming or in the loop.
                return

    return wrapper
