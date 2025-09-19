from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    cast,
)

from .interface import GSS, Acc, T
from .reference_impl import ReferenceGSS

# Add a dummy profiler for when not running under kernprof
try:
    # This will be injected by the kernprof script.
    profile
except NameError:
    # If not running under kernprof, create a dummy decorator.
    def profile(func): return func

# ------------------------------
# Structural Trie (_A nodes)
#
# These nodes represent the shared structure of stacks as a prefix tree (trie).
# They are immutable and behave as value types. They do not contain accumulators.
# ------------------------------


@dataclass(frozen=True, eq=True)
class _A(ABC, Generic[T]):
    """Abstract base class for a structural stack node (prefix trie)."""

    __slots__ = ()

    @abstractmethod
    def _iter_stacks(self) -> Iterable[List[T]]:
        """Iterates through the stack lists represented by this structural node."""
        ...


@dataclass(frozen=True, eq=True)
class _ARoot(_A[T]):
    """Represents the end of a stack path (the empty stack suffix)."""

    __slots__ = ()

    def _iter_stacks(self) -> Iterable[List[T]]:
        yield []


_A_ROOT: _ARoot = _ARoot()


@dataclass(frozen=True, eq=True)
class _AInternal(_A[T], Generic[T]):
    """Represents an internal node in the prefix trie, with children."""

    # Children are stored as a sorted tuple to ensure hashability and canonical form.
    _children: Tuple[Tuple[T, _A[T]], ...]
    __slots__ = ("_children",)

    @classmethod
    def from_dict(cls, children: Dict[T, _A[T]]) -> _A[T]:
        """Creates a structural node from a dictionary of children."""
        if not children:
            return _A_ROOT
        # Sort items by key for a canonical representation.
        sorted_items = tuple(sorted(children.items()))
        return cls(sorted_items)

    def get_children(self) -> Dict[T, _A[T]]:
        return dict(self._children)

    def _iter_stacks(self) -> Iterable[List[T]]:
        for value, child_node in self._children:
            for stack_suffix in child_node._iter_stacks():
                yield [value] + stack_suffix


# ------------------------------
# LeveledGSS implementation
# ------------------------------


@dataclass(eq=False, frozen=True)
class LeveledGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A GSS implementation using a recursive, "leveled" representation.

    This class is a union of three variants, identified by `_kind`:
    1. EMPTY: Represents a GSS with no active stacks.
    2. GROUP: A "sucked-up" state where multiple stack structures (`_A` node)
       share a single, common accumulator. This is a compact leaf in the GSS tree.
    3. BRANCH: A "distributed" state representing a fork in the GSS. It holds
       an optional accumulator for the empty stack and a dictionary mapping
       stack prefixes (`T`) to sub-GSS instances.

    A canonical factory `_create` ensures that the most compact representation
    (GROUP) is used whenever possible by applying the "suck-up" logic automatically.
    """

    _kind: str
    # GROUP payload
    _node: Optional[_A[T]] = None
    _acc: Optional[Acc] = None
    # BRANCH payload
    _empty_acc: Optional[Acc] = None
    _children: Optional[Dict[T, LeveledGSS[T, Acc]]] = None

    # -------------------------
    # Construction and Canonicalization
    # -------------------------

    @classmethod
    def _empty(cls) -> LeveledGSS[T, Acc]:
        return cls(_kind="EMPTY")

    @classmethod
    def _group(cls, node: _A[T], acc: Acc) -> LeveledGSS[T, Acc]:
        return cls(_kind="GROUP", _node=node, _acc=acc)

    @classmethod
    def _branch(
        cls, empty_acc: Optional[Acc], children: Dict[T, LeveledGSS[T, Acc]]
    ) -> LeveledGSS[T, Acc]:
        return cls(_kind="BRANCH", _empty_acc=empty_acc, _children=children)

    # -------------------------

    @classmethod
    @profile
    def _create(
        cls, empty_acc: Optional[Acc], children: Dict[T, LeveledGSS[T, Acc]]
    ) -> LeveledGSS[T, Acc]:
        """
        Canonical factory for LeveledGSS. This is the sole entry point for
        creating instances, ensuring invariants are maintained.
        """
        # Filter out any empty children, as they don't represent any stacks.
        live_children = {
            v: c for v, c in children.items() if not c.is_empty()
        }

        if empty_acc is None and not live_children:
            return cls._empty()

        # "Suck-up" logic: If there's no empty stack accumulator and all children
        # are GROUP nodes with the same accumulator, merge them into a single GROUP.
        can_suck_up = empty_acc is None and live_children
        if can_suck_up:
            child_vals = list(live_children.values())
            first_child = child_vals[0]
            if first_child._kind == "GROUP":
                first_acc = first_child._acc
                if all(
                    c._kind == "GROUP" and c._acc == first_acc for c in child_vals[1:]
                ):
                    new_a_children = {
                        v: cast(_A[T], c._node) for v, c in live_children.items()
                    }
                    new_a_node = _AInternal.from_dict(new_a_children)
                    return cls._group(new_a_node, first_acc)

        return cls._branch(empty_acc, live_children)

    @profile
    def _distribute(self) -> LeveledGSS[T, Acc]:
        """Converts a GROUP node into its equivalent BRANCH representation."""
        if self._kind != "GROUP":
            return self

        node, acc = cast(_A[T], self._node), cast(Acc, self._acc)
        if isinstance(node, _ARoot):
            return self._create(acc, {})
        if isinstance(node, _AInternal):
            new_children = {
                v: self._group(c, acc) for v, c in node.get_children().items()
            }
            return self._create(None, new_children)
        return self._empty()  # Should be unreachable

    # -------------------------
    # GSS interface
    # -------------------------

    @classmethod
    def from_stacks(
        cls: Type[LeveledGSS], stacks: List[Tuple[List[T], Acc]]
    ) -> LeveledGSS[T, Acc]:
        if not stacks:
            return cls._empty()

        empty_acc: Optional[Acc] = None
        by_prefix: Dict[T, List[Tuple[List[T], Acc]]] = defaultdict(list)

        for stack, acc in stacks:
            if not stack:
                empty_acc = acc if empty_acc is None else empty_acc.merge(acc)
            else:
                by_prefix[stack[0]].append((stack[1:], acc))

        children = {
            prefix: cls.from_stacks(s_list) for prefix, s_list in by_prefix.items()
        }
        return cls._create(empty_acc, children)

    @profile
    def push(self, value: T) -> LeveledGSS[T, Acc]:
        if self.is_empty():
            return self._empty()
        return self._create(None, {value: self})

    @profile
    def pop(self) -> LeveledGSS[T, Acc]:
        if self._kind == "EMPTY":
            return self
        if self._kind == "GROUP":
            node = cast(_A[T], self._node)
            if isinstance(node, _ARoot):  # Popping the empty stack
                return self._empty()
            # Distribute and merge the children
            distributed = self._distribute()
            return LeveledGSS.merge_many(cast(Dict, distributed._children).values())
        # BRANCH
        return LeveledGSS.merge_many(cast(Dict, self._children).values())

    def is_empty(self) -> bool:
        """Checks if the GSS represents any stacks."""
        return self._kind == "EMPTY"

    @profile
    def isolate(self, value: Optional[T]) -> LeveledGSS[T, Acc]:
        gss = self._distribute()
        if gss._kind != "BRANCH":  # Can only be EMPTY
            return self._empty()

        if value is None:
            return self._create(gss._empty_acc, {})
        else:
            child = cast(Dict, gss._children).get(value)
            if child is None:
                return self._empty()
            return self._create(None, {value: child})

    @profile
    def apply(
        self, func: Callable[[Acc], Acc], *, _memo: Optional[Dict[int, Any]] = None
    ) -> LeveledGSS[T, Acc]:
        if _memo is None:
            _memo = {}
        if id(self) in _memo:
            return _memo[id(self)]

        if self._kind == "EMPTY":
            result = self
        elif self._kind == "GROUP":
            new_acc = func(cast(Acc, self._acc))
            if new_acc is self._acc:
                result = self
            else:
                result = self._group(cast(_A[T], self._node), new_acc)
        else:  # BRANCH
            new_empty_acc = (
                func(self._empty_acc) if self._empty_acc is not None else None
            )
            empty_acc_changed = new_empty_acc is not self._empty_acc

            children = cast(Dict, self._children)
            new_children = {}
            children_changed = False
            for v, c in children.items():
                new_c = c.apply(func, _memo=_memo)
                if new_c is not c:
                    children_changed = True
                new_children[v] = new_c

            if not empty_acc_changed and not children_changed:
                result = self
            else:
                result = self._create(new_empty_acc, new_children)

        _memo[id(self)] = result
        return result

    @profile
    def prune(self, predicate: Callable[[Acc], bool]) -> LeveledGSS[T, Acc]:
        if self._kind == "EMPTY":
            return self
        if self._kind == "GROUP":
            return self if predicate(cast(Acc, self._acc)) else self._empty()

        # BRANCH
        new_empty_acc = (
            self._empty_acc
            if self._empty_acc is not None and predicate(self._empty_acc)
            else None
        )
        empty_acc_changed = new_empty_acc is not self._empty_acc

        children = cast(Dict, self._children)
        new_children = {}
        children_changed = False
        for v, c in children.items():
            new_c = c.prune(predicate)
            if new_c is not c:
                children_changed = True
            new_children[v] = new_c

        if not empty_acc_changed and not children_changed:
            return self

        return self._create(new_empty_acc, new_children)

    @profile
    def merge(self, other: GSS[T, Acc]) -> LeveledGSS[T, Acc]:
        # Convert other to LeveledGSS if it isn't one.
        if not isinstance(other, LeveledGSS):
            ref_gss = other.to_reference_impl()
            # ref_gss._stacks has top-at-tail. LeveledGSS.from_stacks expects top-at-head.
            stacks_for_leveled = [(s[::-1], acc) for s, acc in ref_gss._stacks]
            other = LeveledGSS.from_stacks(stacks_for_leveled)

        # Now both are LeveledGSS.
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        gss1 = self._distribute()
        gss2 = other._distribute()

        # After distribution, they must be BRANCH kind (or EMPTY, handled above).
        # Merge empty accumulators
        merged_empty_acc: Optional[Acc] = None
        if gss1._empty_acc is not None and gss2._empty_acc is not None:
            merged_empty_acc = gss1._empty_acc.merge(gss2._empty_acc)
        elif gss1._empty_acc is not None:
            merged_empty_acc = gss1._empty_acc
        else:
            merged_empty_acc = gss2._empty_acc

        # Merge children
        gss1_children = gss1._children or {}
        gss2_children = gss2._children or {}
        all_keys = gss1_children.keys() | gss2_children.keys()

        merged_children: Dict[T, LeveledGSS[T, Acc]] = {}
        for k in all_keys:
            c1 = gss1_children.get(k)
            c2 = gss2_children.get(k)
            merged_children[k] = c1.merge(c2) if c1 and c2 else (c1 or c2)

        return LeveledGSS._create(merged_empty_acc, merged_children)

    def peek(self) -> Set[T]:
        gss = self._distribute()
        if gss._kind == "BRANCH":
            return set(cast(Dict, gss._children).keys())
        return set()

    def _iter_stacks(self) -> Iterable[Tuple[List[T], Acc]]:
        if self._kind == "EMPTY":
            return
        if self._kind == "GROUP":
            acc = cast(Acc, self._acc)
            for stack_list in cast(_A[T], self._node)._iter_stacks():
                yield stack_list, acc
        else:  # BRANCH
            if self._empty_acc is not None:
                yield [], self._empty_acc
            for value, child_gss in cast(Dict, self._children).items():
                for stack_suffix, acc in child_gss._iter_stacks():
                    yield [value] + stack_suffix, acc

    def reduce_acc(self) -> Optional[Acc]:
        accs = [acc for _, acc in self._iter_stacks()]
        if not accs:
            return None
        return reduce(lambda a, b: a.merge(b), accs)

    def to_reference_impl(self) -> ReferenceGSS[T, Acc]:
        # LeveledGSS internal representation is top-at-head for stack lists.
        # ReferenceGSS representation is top-at-tail.
        # We must reverse the lists to convert between them.
        stacks_for_ref = [(s[::-1], acc) for s, acc in self._iter_stacks()]
        return ReferenceGSS.from_stacks(stacks_for_ref)

    def _validate_invariants(self) -> None:
        """
        Recursively checks that this GSS node and all its descendants
        adhere to the canonical representation invariants.
        Raises AssertionError if an invariant is violated.
        """
        if self._kind == "BRANCH":
            # Invariant 1: A BRANCH node should not be "suck-up-able".
            # This would mean it's a non-canonical representation that should be a GROUP.
            children = cast(Dict, self._children)
            if self._empty_acc is None and children:

                def get_group_parts(
                    g: LeveledGSS[T, Acc],
                ) -> Optional[Tuple[_A[T], Acc]]:
                    """Helper to see if a GSS is equivalent to a simple GROUP."""
                    if g._kind == "GROUP":
                        return g._node, g._acc
                    # A BRANCH representing a single empty stack is equivalent to a GROUP.
                    if (
                        g._kind == "BRANCH"
                        and g._empty_acc is not None
                        and not g._children
                    ):
                        return _A_ROOT, g._empty_acc
                    return None

                child_vals = list(children.values())
                first_child_parts = get_group_parts(child_vals[0])

                if first_child_parts:
                    _, first_acc = first_child_parts

                    all_match = True
                    for c in child_vals[1:]:
                        parts = get_group_parts(c)
                        if not parts or parts[1] != first_acc:
                            all_match = False
                            break

                    if all_match:
                        raise AssertionError(
                            "Invariant violation: non-canonical BRANCH node that should be a GROUP node. "
                            f"Node: {self!r}"
                        )

            # Recursively validate children
            for child in cast(Dict, self._children).values():
                child._validate_invariants()

        elif self._kind == "GROUP":
            # Invariant 2: A GROUP node should not represent just an empty stack.
            # That should be a BRANCH node with an _empty_acc.
            if self._node == _A_ROOT:
                raise AssertionError(
                    "Invariant violation: GROUP node with _ARoot, should be a BRANCH. "
                    f"Node: {self!r}"
                )

    # -------------------------
    # Dunder methods
    # -------------------------

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, LeveledGSS):
            # Fast path for two canonical LeveledGSS instances
            return (
                self._kind == other._kind
                and self._node == other._node
                and self._acc == other._acc
                and self._empty_acc == other._empty_acc
                and self._children == other._children
            )
        if isinstance(other, GSS):
            return self.to_reference_impl() == other.to_reference_impl()
        return NotImplemented

    def __repr__(self) -> str:
        if self._kind == "EMPTY":
            return "LeveledGSS(EMPTY)"
        if self._kind == "GROUP":
            stacks = sorted(list(cast(_A[T], self._node)._iter_stacks()))
            return f"LeveledGSS(GROUP stacks={stacks!r}, acc={self._acc!r})"
        # BRANCH
        parts = []
        if self._empty_acc is not None:
            parts.append(f"[]: {self._empty_acc!r}")
        # Sort for deterministic output
        sorted_children = sorted(cast(Dict, self._children).items())
        for v, c in sorted_children:
            parts.append(f"{v!r}: {c!r}")
        return f"LeveledGSS(BRANCH {{{', '.join(parts)}}})"
