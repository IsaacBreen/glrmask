from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Generic, Iterable, List, Set

# We avoid importing the abstract ABC GSS directly here to keep this module generic.
# The runner will pass us the chosen implementation class and we will wrap instances.

T = TypeVar("T")
Acc = TypeVar("Acc")


@dataclass
class MethodAgg:
    count: int = 0
    total_ns: int = 0
    min_ns: int = field(default_factory=lambda: 1 << 62)
    max_ns: int = 0
    errors: int = 0

    def add(self, elapsed_ns: int):
        self.count += 1
        self.total_ns += elapsed_ns
        if elapsed_ns < self.min_ns:
            self.min_ns = elapsed_ns
        if elapsed_ns > self.max_ns:
            self.max_ns = elapsed_ns

    def add_error(self):
        self.errors += 1

    def to_json(self) -> Dict[str, Any]:
        avg_ns = self.total_ns // self.count if self.count > 0 else 0
        min_ns = self.min_ns if self.min_ns != (1 << 62) else 0
        return {
            "count": self.count,
            "total_ns": self.total_ns,
            "avg_ns": avg_ns,
            "min_ns": min_ns,
            "max_ns": self.max_ns,
            "errors": self.errors,
        }


@dataclass
class PhaseStats:
    name: str
    start_ns: int = 0
    end_ns: int = 0
    elapsed_ns: int = 0
    method_stats: Dict[str, MethodAgg] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    aborted: bool = False

    def start(self):
        self.start_ns = time.perf_counter_ns()

    def stop(self):
        self.end_ns = time.perf_counter_ns()
        self.elapsed_ns = self.end_ns - self.start_ns

    def record(self, method: str, elapsed_ns: int):
        agg = self.method_stats.get(method)
        if agg is None:
            agg = MethodAgg()
            self.method_stats[method] = agg
        agg.add(elapsed_ns)

    def record_error(self, method: str, message: str):
        agg = self.method_stats.get(method)
        if agg is None:
            agg = MethodAgg()
            self.method_stats[method] = agg
        agg.add_error()
        self.notes.append(f"{method} error: {message}")

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "elapsed_ns": self.elapsed_ns,
            "aborted": self.aborted,
            "error": self.error,
            "notes": self.notes[:],
            "methods": {k: v.to_json() for k, v in sorted(self.method_stats.items())},
        }


class TimingRecorder:
    """
    Records per-method and per-phase timing.
    """
    def __init__(self):
        self.global_method_stats: Dict[str, MethodAgg] = {}
        self.current_phase: Optional[PhaseStats] = None
        self.phases: List[PhaseStats] = []

    def start_phase(self, name: str):
        if self.current_phase is not None:
            self.end_phase()
        self.current_phase = PhaseStats(name=name)
        self.current_phase.start()

    def end_phase(self):
        if self.current_phase is not None:
            self.current_phase.stop()
            self.phases.append(self.current_phase)
            self.current_phase = None

    def abort_current_phase(self, reason: str = ""):
        if self.current_phase is not None:
            self.current_phase.aborted = True
            if reason:
                self.current_phase.notes.append(f"aborted: {reason}")
            self.end_phase()

    def record(self, method: str, elapsed_ns: int):
        # Global
        g = self.global_method_stats.get(method)
        if g is None:
            g = MethodAgg()
            self.global_method_stats[method] = g
        g.add(elapsed_ns)
        # Phase-local
        if self.current_phase is not None:
            self.current_phase.record(method, elapsed_ns)

    def record_error(self, method: str, message: str):
        g = self.global_method_stats.get(method)
        if g is None:
            g = MethodAgg()
            self.global_method_stats[method] = g
        g.add_error()
        if self.current_phase is not None:
            self.current_phase.record_error(method, message)

    def to_json(self) -> Dict[str, Any]:
        return {
            "phases": [p.to_json() for p in self.phases],
            "overall_methods": {k: v.to_json() for k, v in sorted(self.global_method_stats.items())},
        }


class TimedGSS(Generic[T, Acc]):
    """
    A lightweight proxy around a concrete GSS instance that times method calls.
    It wraps/unwraps across method boundaries so that chained operations remain instrumented.
    """
    __slots__ = ("inner", "recorder", "factory")

    def __init__(self, inner: Any, recorder: TimingRecorder, factory: "GSSFactory"):
        self.inner = inner
        self.recorder = recorder
        self.factory = factory

    # Wrapping helpers
    def _wrap(self, obj: Any) -> "TimedGSS":
        return TimedGSS(obj, self.recorder, self.factory)

    # Instrumented methods
    def to_stacks(self):
        t0 = time.perf_counter_ns()
        try:
            res = self.inner.to_stacks()
            return res
        finally:
            self.recorder.record("to_stacks", time.perf_counter_ns() - t0)

    def push(self, value: Any) -> "TimedGSS":
        t0 = time.perf_counter_ns()
        try:
            res = self.inner.push(value)
            return self._wrap(res)
        finally:
            self.recorder.record("push", time.perf_counter_ns() - t0)

    def pop(self) -> "TimedGSS":
        t0 = time.perf_counter_ns()
        try:
            res = self.inner.pop()
            return self._wrap(res)
        finally:
            self.recorder.record("pop", time.perf_counter_ns() - t0)

    def popn(self, n: int) -> "TimedGSS":
        # We expand into repeated pop() calls so individual pop() calls are timed as well.
        t0 = time.perf_counter_ns()
        try:
            g = self
            for _ in range(n):
                g = g.pop()
            return g
        finally:
            self.recorder.record("popn", time.perf_counter_ns() - t0)

    def is_empty(self) -> bool:
        t0 = time.perf_counter_ns()
        try:
            return self.inner.is_empty()
        finally:
            self.recorder.record("is_empty", time.perf_counter_ns() - t0)

    def isolate(self, value: Optional[Any]) -> "TimedGSS":
        t0 = time.perf_counter_ns()
        try:
            res = self.inner.isolate(value)
            return self._wrap(res)
        finally:
            self.recorder.record("isolate", time.perf_counter_ns() - t0)

    def apply(self, func: Callable[[Any], Any]) -> "TimedGSS":
        t0 = time.perf_counter_ns()
        try:
            res = self.inner.apply(func)
            return self._wrap(res)
        finally:
            self.recorder.record("apply", time.perf_counter_ns() - t0)

    def prune(self, predicate: Callable[[Any], bool]) -> "TimedGSS":
        t0 = time.perf_counter_ns()
        try:
            res = self.inner.prune(predicate)
            return self._wrap(res)
        finally:
            self.recorder.record("prune", time.perf_counter_ns() - t0)

    def merge(self, other: "TimedGSS") -> "TimedGSS":
        t0 = time.perf_counter_ns()
        try:
            # Unwrap other if necessary
            other_inner = other.inner if isinstance(other, TimedGSS) else other
            res = self.inner.merge(other_inner)
            return self._wrap(res)
        finally:
            self.recorder.record("merge", time.perf_counter_ns() - t0)

    @classmethod
    def merge_many(cls, gss_list: Iterable["TimedGSS"]) -> "TimedGSS":
        # This classmethod won't be used directly; the runner will sequence merges to ensure timing.
        raise NotImplementedError("Use factory.merge_many(...) to merge a list with instrumentation")

    def peek(self) -> Set[Any]:
        t0 = time.perf_counter_ns()
        try:
            return self.inner.peek()
        finally:
            self.recorder.record("peek", time.perf_counter_ns() - t0)

    def reduce_acc(self) -> Optional[Any]:
        t0 = time.perf_counter_ns()
        try:
            return self.inner.reduce_acc()
        finally:
            self.recorder.record("reduce_acc", time.perf_counter_ns() - t0)

    def to_reference_impl(self) -> "TimedGSS":
        t0 = time.perf_counter_ns()
        try:
            res = self.inner.to_reference_impl()
            return self._wrap(res)
        finally:
            self.recorder.record("to_reference_impl", time.perf_counter_ns() - t0)

    def __str__(self) -> str:
        return f"TimedGSS({self.inner})"

    def __repr__(self) -> str:
        return f"TimedGSS({self.inner!r})"


class GSSFactory:
    """
    Binds a GSS implementation class and a recorder. Creates wrapped instances and
    provides helper methods for merge_many with instrumentation.
    """
    def __init__(self, gss_class: Type[Any], recorder: TimingRecorder):
        self.gss_class = gss_class
        self.recorder = recorder

    def from_stacks(self, stacks: List[Tuple[List[Any], Any]]) -> TimedGSS:
        t0 = time.perf_counter_ns()
        try:
            inst = self.gss_class.from_stacks(stacks)
            return TimedGSS(inst, self.recorder, self)
        finally:
            self.recorder.record("from_stacks", time.perf_counter_ns() - t0)

    def empty(self) -> TimedGSS:
        # Using from_stacks([]) to ensure we consistently time construction.
        return self.from_stacks([])

    def merge_many(self, gss_list: Iterable[TimedGSS]) -> TimedGSS:
        t0 = time.perf_counter_ns()
        try:
            iterator = iter(gss_list)
            first = next(iterator, None)
            if first is None:
                return self.empty()
            acc = first
            for nxt in iterator:
                acc = acc.merge(nxt)
            return acc
        finally:
            self.recorder.record("merge_many", time.perf_counter_ns() - t0)

    def push_many(self, items: Iterable[Tuple[TimedGSS, Any]]) -> TimedGSS:
        # Materialize because we need two passes (push and then merge)
        items_list = list(items)
        t0 = time.perf_counter_ns()
        try:
            pushed = [g.push(v) for g, v in items_list]
            # Merge sequentially to capture per-merge costs
            result = pushed[0]
            for nxt in pushed[1:]:
                result = result.merge(nxt)
            return result
        finally:
            self.recorder.record("push_many", time.perf_counter_ns() - t0)
