from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    FrozenSet,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from ..interface import GSS, T, Acc, NewAcc, Mergeable
from .reference_impl import ReferenceGSS

# -----------------------------------------------------------------------------
# Acc placeholder tokens and helpers
# -----------------------------------------------------------------------------

_AID_COUNTER = itertools.count(1)


def _new_token() -> int:
    return next(_AID_COUNTER)


# -----------------------------------------------------------------------------
# Node classes (structure is identical to LeveledGSS, but accumulators are
# stored as placeholder tokens (int) instead of real Acc values)
# -----------------------------------------------------------------------------

type Upper[T] = UpperBranch[T] | Interface[T]


@dataclass(frozen=True, eq=True)
class UpperBranch(Generic[T]):
    children: Dict[T, Dict[int, Upper[T]]]
    empty: Optional[int]  # token
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, "_max_depth", depth)

    def _all_children(self) -> Generator[Upper[T], None, None]:
        for children_at_depth in self.children.values():
            yield from children_at_depth.values()


@dataclass(frozen=True, eq=True)
class Interface(Generic[T]):
    children: Dict[T, Dict[int, "Lower[T]"]]
    acc: int  # token for the "body" accumulator
    empty: Optional[int]  # token for explicit empty at this level
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, "_max_depth", depth)

    def _all_children(self) -> Iterator["Lower[T]"]:
        for v_children in self.children.values():
            yield from v_children.values()


@dataclass(frozen=True, eq=True)
class Lower(Generic[T]):
    children: Dict[T, Dict[int, "Lower[T]"]]
    empty: bool
    _max_depth: int = field(init=False)

    def __post_init__(self):
        depth = max(child._max_depth for child in self._all_children()) + 1 if self.children else 0
        object.__setattr__(self, "_max_depth", depth)

    def _all_children(self) -> Iterator["Lower[T]"]:
        for v_children in self.children.values():
            yield from v_children.values()


# -----------------------------------------------------------------------------
# Structural utilities (token-aware)
# -----------------------------------------------------------------------------

def _merge_children_by_depth(
    c1: Dict[T, Dict[int, Any]],
    c2: Dict[T, Dict[int, Any]],
    merge_func: Callable[[Any, Any], Any],
) -> Dict[T, Dict[int, Any]]:
    if c1 is c2:
        return c1
    merged_children: Dict[T, Dict[int, Any]] = {}
    all_vals = c1.keys() | c2.keys()
    for v in all_vals:
        nodes_by_depth: DefaultDict[int, List[Any]] = DefaultDict(list)
        children_c1 = c1.get(v, {}).items()
        children_c2 = c2.get(v, {}).items()
        for depth, child in chain(children_c1, children_c2):
            nodes_by_depth[depth].append(child)
        if not nodes_by_depth:
            continue
        v_out = {
            (merged := reduce(merge_func, nodes))._max_depth: merged
            for nodes in nodes_by_depth.values()
        }
        merged_children[v] = v_out
    return merged_children


def merge_lower(l1: Lower[T], l2: Lower[T]) -> Lower[T]:
    if l1 is l2:
        return l1
    new_empty = l1.empty or l2.empty
    merged_children = _merge_children_by_depth(l1.children, l2.children, merge_lower)
    return Lower(children=merged_children, empty=new_empty)


def try_promote(node: UpperBranch[T]) -> Upper[T]:
    """Promotion based on token equality (not underlying Acc equality)."""
    all_children: List[Upper[T]] = list(node._all_children())
    if not all_children:
        # Leaf UpperBranch: explicit empty stack -> canonical Interface with no children.
        if node.empty is not None:
            return Interface(children={}, acc=node.empty, empty=node.empty)
        return node
    if not all(isinstance(c, Interface) for c in all_children):
        return node

    accs: Set[int] = set()
    if node.empty is not None:
        accs.add(node.empty)
    for c in all_children:
        ic: Interface[T] = c  # type: ignore[assignment]
        accs.add(ic.acc)
        if ic.empty is not None:
            accs.add(ic.empty)

    if len(accs) <= 1:
        the_acc: Optional[int] = next(iter(accs)) if accs else None
        if the_acc is None:
            return UpperBranch(children={}, empty=None)
        l_children: Dict[T, Dict[int, Lower[T]]] = {}
        for v, kids in node.children.items():
            v_map: Dict[int, Lower[T]] = {}
            for child in kids.values():
                ci: Interface[T] = child  # type: ignore[assignment]
                lower = Lower(children=ci.children, empty=(ci.empty is not None))
                v_map[lower._max_depth] = lower
            if v_map:
                l_children[v] = v_map
        return Interface(children=l_children, acc=the_acc, empty=node.empty)
    return node


def interface_to_upperbranch(it: Interface[T]) -> UpperBranch[T]:
    children: Dict[T, Dict[int, Upper[T]]] = {}
    for v, kids in it.children.items():
        v_map: Dict[int, Upper[T]] = {}
        for lchild in kids.values():
            ci = Interface(
                children=lchild.children,
                acc=it.acc,
                empty=(it.acc if lchild.empty else None),
            )
            v_map[ci._max_depth] = ci
        if v_map:
            children[v] = v_map
    new_empty = it.empty
    if not it.children and new_empty is None:
        new_empty = it.acc
    return UpperBranch(children=children, empty=new_empty)


# -----------------------------------------------------------------------------
# The main implementation
# -----------------------------------------------------------------------------

@dataclass(frozen=True, eq=True)
class LeveledAccTableGSS(GSS[T, Acc], Generic[T, Acc]):
    """
    A leveled-structure GSS that stores accumulators outside the nodes.

    - Nodes carry small integer tokens instead of real Acc values.
    - The instance carries a table mapping tokens -> Acc.
    - apply/prune/apply_and_prune operate on the token table only (no structural walk)
      and thus are very fast in the common case.
    - When pruned tokens exist, read APIs (to_stacks, peek, is_empty, reduce_acc)
      consult liveness (token presence in the table) and compute results with memoization.
    """

    inner: Upper[T]
    _values: Dict[int, Acc]  # token -> Acc (presence indicates liveness)
    _refs: FrozenSet[int] = field(init=False, compare=False, hash=False)  # tokens actually referenced by 'inner'

    # ------------------------------
    # Lifecycle and validation
    # ------------------------------
    def __post_init__(self):
        object.__setattr__(self, "_refs", frozenset(self._collect_refs(self.inner)))
        if os.environ.get("GSS_TESTER_VALIDATE"):
            self._validate()

    @staticmethod
    def _collect_refs(node: Upper[T] | Lower[T]) -> Set[int]:
        refs: Set[int] = set()

        def rec_upper(n: Upper[T]):
            if isinstance(n, UpperBranch):
                if n.empty is not None:
                    refs.add(n.empty)
                for kids in n.children.values():
                    for child in kids.values():
                        rec_upper(child)
            else:
                # Interface
                refs.add(n.acc)
                if n.empty is not None:
                    refs.add(n.empty)
                for kids in n.children.values():
                    for child in kids.values():
                        rec_lower(child)

        def rec_lower(n: Lower[T]):
            # Lower has no tokens, just recurse
            if n.empty:
                pass
            for kids in n.children.values():
                for child in kids.values():
                    rec_lower(child)

        if isinstance(node, (UpperBranch, Interface)):
            rec_upper(node)
        else:
            rec_lower(node)
        return refs

    def _validate(self):
        # Validate _max_depth depths for upper/lower nodes (same as leveled_impl but token-aware)
        def _validate_depths_upper(n: Upper[T]):
            if isinstance(n, Interface):
                def _validate_lower_recursively(m: Interface[T] | Lower[T]):
                    for children_at_depth in m.children.values():
                        for depth, child in children_at_depth.items():
                            if depth != child._max_depth:
                                raise ValueError(
                                    "LeveledAccTableGSS validation failed: incorrect max_depth for Lower child. "
                                    f"Expected {depth}, got {child._max_depth}. Node: {m}"
                                )
                            _validate_lower_recursively(child)
                _validate_lower_recursively(n)
                return
            for children_at_depth in n.children.values():
                for depth, child in children_at_depth.items():
                    if depth != child._max_depth:
                        raise ValueError(
                            "LeveledAccTableGSS validation failed: incorrect max_depth for Upper child. "
                            f"Expected {depth}, got {child._max_depth}. Node: {n}"
                        )
                    _validate_depths_upper(child)

        _validate_depths_upper(self.inner)

        # Structural non-empty constraints (same as leveled_impl)
        def _assert_populated(n: Upper[T] | Lower[T]) -> None:
            if isinstance(n, UpperBranch):
                if not n.children and n.empty is None:
                    raise ValueError("Invalid UpperBranch: no children and no empty in non-root position.")
                for kids in n.children.values():
                    for child in kids.values():
                        _assert_populated(child)
            elif isinstance(n, Interface):
                if not n.children and n.empty is None:
                    # Interface with no children and no explicit empty is okay only if it represents
                    # an implicit terminal via acc; we allow it.
                    pass
                for kids in n.children.values():
                    for child in kids.values():
                        _assert_populated(child)
            else:
                if not n.children and not n.empty:
                    raise ValueError("Invalid Lower: no children and empty=False.")
                for kids in n.children.values():
                    for child in kids.values():
                        _assert_populated(child)

        # Allow truly empty root
        if isinstance(self.inner, UpperBranch) and not self.inner.children and self.inner.empty is None:
            return
        _assert_populated(self.inner)

    # ------------------------------
    # Constructors
    # ------------------------------
    @classmethod
    def empty(cls: Type["LeveledAccTableGSS[T, Acc]"]) -> "LeveledAccTableGSS[T, Acc]":
        return cls(UpperBranch(children={}, empty=None), _values={})

    @classmethod
    def from_stacks(cls: Type["LeveledAccTableGSS[T, Acc]"], stacks: List[Tuple[List[T], Acc]]) -> "LeveledAccTableGSS[T, Acc]":
        # Canonicalize stacks and merge duplicates using ReferenceGSS.
        canonical_stacks = ReferenceGSS(stacks)._stacks  # merged and canonicalized

        empty_token: Optional[int] = None
        trie: Dict[T, Dict[str, Any]] = {}
        values: Dict[int, Acc] = {}

        for vals, acc in canonical_stacks:
            token = _new_token()
            values[token] = acc
            if not vals:
                empty_token = token
                continue
            node = trie
            for i, v in enumerate(reversed(vals)):
                entry = node.setdefault(v, {"end": None, "sub": {}})
                if i == len(vals) - 1:
                    entry["end"] = token
                else:
                    node = entry["sub"]

        def build(d: Dict[T, Dict[str, Any]], root_empty: Optional[int]) -> Upper[T]:
            children: Dict[T, Dict[int, Upper[T]]] = {}
            all_child_nodes: List[Upper[T]] = []
            for v, e in d.items():
                nodes_for_v: List[Upper[T]] = []
                end_tok = e.get("end")
                sub = e.get("sub", {})
                if end_tok is not None:
                    nodes_for_v.append(try_promote(UpperBranch(children={}, empty=end_tok)))
                if sub:
                    nodes_for_v.append(build(sub, None))
                if nodes_for_v:
                    children[v] = {n._max_depth: n for n in nodes_for_v}
                    all_child_nodes.extend(nodes_for_v)

            # Possible promotion to Interface if all children are Interfaces and they share the same token,
            # and possibly including root_empty.
            if all(isinstance(child, Interface) for child in all_child_nodes):
                accs: Set[int] = set()
                for c in all_child_nodes:
                    ic: Interface[T] = c  # type: ignore[assignment]
                    accs.add(ic.acc)
                    if ic.empty is not None:
                        accs.add(ic.empty)
                if root_empty is not None:
                    accs.add(root_empty)
                if len(accs) <= 1:
                    the_acc = next(iter(accs)) if accs else None

                    def build_lower(sub_d: Dict[T, Dict[str, Any]]) -> Lower[T]:
                        l_children: Dict[T, Dict[int, Lower[T]]] = {}
                        for v_l, e_l in sub_d.items():
                            sub_l = e_l.get("sub", {})
                            has_end = e_l.get("end") is not None
                            sub_lower = build_lower(sub_l) if sub_l else Lower(children={}, empty=False)
                            node_for_v = Lower(children=sub_lower.children, empty=has_end)
                            l_children[v_l] = {node_for_v._max_depth: node_for_v}
                        return Lower(children=l_children, empty=False)

                    if the_acc is None:
                        return UpperBranch(children={}, empty=None)
                    lower_tree = build_lower(d)
                    return Interface(children=lower_tree.children, acc=the_acc, empty=root_empty)

            return UpperBranch(children=children, empty=root_empty)

        inner = build(trie, empty_token)
        return cls(inner, values)

    def to_stacks(self) -> List[Tuple[List[T], Acc]]:
        res: List[Tuple[List[T], Acc]] = []

        def tok_live(tok: Optional[int]) -> bool:
            return tok is not None and tok in self._values

        def tok_val(tok: int) -> Acc:
            return self._values[tok]

        def dfs_lower(l: Lower[T], pref: List[T], acc_tok: int) -> None:
            if acc_tok not in self._values:
                return
            if l.empty:
                res.append((list(reversed(pref)), tok_val(acc_tok)))
            for v, kids in l.children.items():
                for child in kids.values():
                    dfs_lower(child, pref + [v], acc_tok)

        def dfs_upper(u: Upper[T], pref: List[T]) -> None:
            if isinstance(u, UpperBranch):
                if tok_live(u.empty):
                    res.append((list(reversed(pref)), tok_val(u.empty)))  # explicit empty
                for v, kids in u.children.items():
                    for child in kids.values():
                        dfs_upper(child, pref + [v])
            else:
                # Interface
                if tok_live(u.empty):
                    res.append((list(reversed(pref)), tok_val(u.empty)))  # explicit empty
                if not u.children and u.empty is None:
                    # implicit terminal by acc
                    if u.acc in self._values:
                        res.append((list(reversed(pref)), tok_val(u.acc)))
                else:
                    if u.acc in self._values:
                        for v, kids in u.children.items():
                            for child in kids.values():
                                dfs_lower(child, pref + [v], u.acc)

        dfs_upper(self.inner, [])

        # Canonical sorting/merging
        return ReferenceGSS(res).to_stacks()

    # ------------------------------
    # Core stack operations (structure-only transforms)
    # ------------------------------
    def push(self, value: T) -> "LeveledAccTableGSS[T, Acc]":
        if self.is_empty():
            return self
        if isinstance(self.inner, Interface):
            lower_node = Lower(children=self.inner.children, empty=self.inner.empty is not None)
            new_children = {value: {lower_node._max_depth: lower_node}}
            return LeveledAccTableGSS(Interface(children=new_children, acc=self.inner.acc, empty=None), dict(self._values))
        else:
            return LeveledAccTableGSS(UpperBranch(children={value: {self.inner._max_depth: self.inner}}, empty=None), dict(self._values))

    def pop(self) -> "LeveledAccTableGSS[T, Acc]":
        if isinstance(self.inner, Interface):
            all_children = list(self.inner._all_children())
            merged = reduce(merge_lower, all_children[1:], all_children[0]) if all_children else Lower(children={}, empty=False)
            merged_empty = self.inner.acc if merged.empty else None
            if merged_empty is None and not merged.children:
                return LeveledAccTableGSS(UpperBranch(children={}, empty=merged_empty), dict(self._values))
            else:
                return LeveledAccTableGSS(Interface(children=merged.children, acc=self.inner.acc, empty=merged_empty), dict(self._values))
        else:
            all_children = list(self.inner._all_children())
            merged_u = reduce(self._merge_upper_struct_only, all_children[1:], all_children[0]) if all_children else UpperBranch(children={}, empty=None)
            return LeveledAccTableGSS(try_promote(merged_u), dict(self._values))

    def popn(self, n: int) -> "LeveledAccTableGSS[T, Acc]":
        if n <= 0:
            return self
        if self.is_empty():
            return self

        memo_upper: Dict[Tuple[int, int], Upper[T]] = {}
        memo_lower: Dict[Tuple[int, int], Lower[T]] = {}

        def _popn_lower(node: Lower[T], k: int) -> Lower[T]:
            if k == 0:
                return node
            key = (id(node), k)
            if key in memo_lower:
                return memo_lower[key]
            all_children = list(node._all_children())
            if not all_children:
                res = Lower(children={}, empty=False)
                memo_lower[key] = res
                return res
            popped_children = [_popn_lower(child, k - 1) for child in all_children]
            res = reduce(merge_lower, popped_children[1:], popped_children[0])
            memo_lower[key] = res
            return res

        def _popn_upper(node: Upper[T], k: int) -> Upper[T]:
            if k == 0:
                return node
            key = (id(node), k)
            if key in memo_upper:
                return memo_upper[key]
            all_children = list(node._all_children())
            if not all_children:
                res = UpperBranch(children={}, empty=None)
                memo_upper[key] = res
                return res
            if isinstance(node, Interface):
                popped_lower_children = [_popn_lower(child, k - 1) for child in all_children]
                merged = reduce(merge_lower, popped_lower_children[1:], popped_lower_children[0])
                new_empty = node.acc if merged.empty else None
                if not merged.children and new_empty is None:
                    res = UpperBranch(children={}, empty=None)
                else:
                    res = Interface(children=merged.children, acc=node.acc, empty=new_empty)
            else:
                popped_upper_children = [_popn_upper(child, k - 1) for child in all_children]
                merged_u = reduce(self._merge_upper_struct_only, popped_upper_children[1:], popped_upper_children[0])
                res = try_promote(merged_u)
            memo_upper[key] = res
            return res

        return LeveledAccTableGSS(_popn_upper(self.inner, n), dict(self._values))

    def _merge_upper_struct_only(self, u1: Upper[T], u2: Upper[T]) -> Upper[T]:
        """Helper for structure-only merges (no token combining)."""
        if u1 is u2:
            return u1
        if isinstance(u1, Interface) and isinstance(u2, Interface):
            merged_children = _merge_children_by_depth(u1.children, u2.children, merge_lower)
            # Keep 'acc' token only if identical; otherwise go through UpperBranch path
            if u1.acc == u2.acc:
                new_empty = u1.empty if u1.empty == u2.empty else (u1.empty or u2.empty)
                # if empties differ (two tokens), we can't choose; but for structure-only
                # use UpperBranch route to avoid incorrect token conflation.
                if new_empty == (u1.empty or u2.empty) and (u1.empty is None or u2.empty is None or u1.empty == u2.empty):
                    return Interface(children=merged_children, acc=u1.acc, empty=new_empty)
        # Fallback through UpperBranch promotion path
        ub1 = u1 if isinstance(u1, UpperBranch) else interface_to_upperbranch(u1)
        ub2 = u2 if isinstance(u2, UpperBranch) else interface_to_upperbranch(u2)
        merged_children = _merge_children_by_depth(ub1.children, ub2.children, self._merge_upper_struct_only)
        return try_promote(UpperBranch(children=merged_children, empty=ub1.empty or ub2.empty))

    # ------------------------------
    # Accumulator transforms (token-table only)
    # ------------------------------
    def apply(self, func: Callable[[Acc], NewAcc], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        # Map only on tokens that are referenced by this graph and currently live.
        new_values: Dict[int, NewAcc] = {}
        for tok in self._refs:
            if tok in self._values:
                v = self._values[tok]
                if memo is not None:
                    k = id(v)
                    if k in memo:
                        nv = memo[k]
                    else:
                        nv = func(v)
                        memo[k] = nv
                else:
                    nv = func(v)
                new_values[tok] = nv  # keep same token, just change value domain
        return LeveledAccTableGSS(self.inner, new_values)  # type: ignore[type-var]

    def prune(self, predicate: Callable[[Acc], bool], memo: Optional[Dict[int, Any]] = None) -> "LeveledAccTableGSS[T, Acc]":
        if memo is None:
            memo = {}
        new_values: Dict[int, Acc] = {}
        for tok in self._refs:
            if tok in self._values:
                v = self._values[tok]
                k = id(v)
                if k in memo:
                    keep = memo[k]
                else:
                    keep = predicate(v)
                    memo[k] = keep
                if keep:
                    new_values[tok] = v
        return LeveledAccTableGSS(self.inner, new_values)

    def apply_and_prune(self, mutator: Callable[[Acc], Optional[NewAcc]], memo: Optional[Dict[int, Any]] = None) -> GSS[T, NewAcc]:
        cache: Dict[int, Optional[NewAcc]] = {} if memo is None else memo
        new_values: Dict[int, NewAcc] = {}
        for tok in self._refs:
            if tok in self._values:
                v = self._values[tok]
                k = id(v)
                if k in cache:
                    nv_opt = cache[k]
                else:
                    nv_opt = mutator(v)
                    cache[k] = nv_opt
                if nv_opt is not None:
                    new_values[tok] = nv_opt
        return LeveledAccTableGSS(self.inner, new_values)  # type: ignore[type-var]

    # ------------------------------
    # Merge
    # ------------------------------
    def merge(self, other: "LeveledAccTableGSS[T, Acc]") -> "LeveledAccTableGSS[T, Acc]":
        if self is other:
            return self
        if self.is_empty():
            return other
        if other.is_empty():
            return self

        # Canonical, semantics-first merge: build from explicit stacks.
        # This uses the same canonicalization as the reference implementation
        # and avoids subtle token-table edge cases during structural merge.
        combined = self.to_stacks() + other.to_stacks()
        return LeveledAccTableGSS.from_stacks(combined)

    # ------------------------------
    # Queries
    # ------------------------------
    def _all_tokens_live(self) -> bool:
        # If every referenced token is live, we can use fast structural answers.
        return len(self._values) >= len(self._refs) and all(tok in self._values for tok in self._refs)

    def is_empty(self) -> bool:
        # Fast structural path when no pruning has occurred
        if isinstance(self.inner, UpperBranch):
            if self._all_tokens_live():
                return not self.inner.children and self.inner.empty is None
        # Otherwise, consult liveness-aware recursion
        return not self._upper_active(self.inner, {})

    def _lower_has_terminal(self, node: Lower[T], memo: Dict[int, bool]) -> bool:
        nid = id(node)
        if nid in memo:
            return memo[nid]
        if node.empty:
            memo[nid] = True
            return True
        for kids in node.children.values():
            for child in kids.values():
                if self._lower_has_terminal(child, memo):
                    memo[nid] = True
                    return True
        memo[nid] = False
        return False

    def _upper_active(self, node: Upper[T], memo: Dict[int, bool]) -> bool:
        nid = id(node)
        if nid in memo:
            return memo[nid]
        if isinstance(node, UpperBranch):
            if node.empty is not None and node.empty in self._values:
                memo[nid] = True
                return True
            for kids in node.children.values():
                for child in kids.values():
                    if self._upper_active(child, memo):
                        memo[nid] = True
                        return True
            memo[nid] = False
            return False
        else:
            # Interface
            if node.empty is not None and node.empty in self._values:
                memo[nid] = True
                return True
            if not node.children and node.empty is None:
                # implicit terminal via acc
                memo[nid] = (node.acc in self._values)
                return memo[nid]
            if node.acc in self._values:
                # active iff any lower child has a terminal path
                lower_memo: Dict[int, bool] = {}
                for kids in node.children.values():
                    for child in kids.values():
                        if self._lower_has_terminal(child, lower_memo):
                            memo[nid] = True
                            return True
            memo[nid] = False
            return False

    def peek(self) -> Set[T]:
        # Fast structural path when all tokens are live
        if self._all_tokens_live():
            return set(self.inner.children.keys())
        # Otherwise, filter by active children
        out: Set[T] = set()
        for v, kids in self.inner.children.items():
            active = False
            for child in kids.values():
                if self._upper_active(child, {}):
                    active = True
                    break
            if active:
                out.add(v)
        return out

    def reduce_acc(self) -> Optional[Acc]:
        # Gather unique Acc objects across live tokens used by this graph
        unique_by_id: Dict[int, Acc] = {}
        for tok in self._refs:
            if tok in self._values:
                acc = self._values[tok]
                unique_by_id[id(acc)] = acc
        if not unique_by_id:
            return None
        accs = list(unique_by_id.values())
        if len(accs) == 1:
            return accs[0]
        return reduce(lambda a, b: a if a is b else a.merge(b), accs)  # type: ignore[return-value]

    # ------------------------------
    # Isolation
    # ------------------------------
    def isolate(self, value: Optional[T]) -> "LeveledAccTableGSS[T, Acc]":
        if value is None:
            # Keep only empty stacks at root (explicit empties at root node)
            if isinstance(self.inner, UpperBranch):
                new_empty = self.inner.empty
            else:
                new_empty = self.inner.empty
            new_root: Upper[T] = UpperBranch(children={}, empty=new_empty)
            return LeveledAccTableGSS(try_promote(new_root), dict(self._values))

        if isinstance(self.inner, UpperBranch):
            filtered_children = {value: self.inner.children[value]} if value in self.inner.children else {}
            return LeveledAccTableGSS(try_promote(UpperBranch(children=filtered_children, empty=None)), dict(self._values))
        else:
            if value not in self.inner.children:
                return LeveledAccTableGSS(UpperBranch(children={}, empty=None), dict(self._values))
            filtered_children = {value: self.inner.children[value]}
            return LeveledAccTableGSS(Interface(children=filtered_children, acc=self.inner.acc, empty=None), dict(self._values))

    def isolate_many(self, values: Iterable[Optional[T]]) -> "LeveledAccTableGSS[T, Acc]":
        vals = set(values)

        new_empty: Optional[int] = None
        if None in vals:
            if isinstance(self.inner, (UpperBranch, Interface)):
                new_empty = self.inner.empty

        if isinstance(self.inner, UpperBranch):
            filtered_children = {v: kids for v, kids in self.inner.children.items() if v in vals}
            new_inner = try_promote(UpperBranch(children=filtered_children, empty=new_empty))
            return LeveledAccTableGSS(new_inner, dict(self._values))
        else:
            filtered_children = {v: kids for v, kids in self.inner.children.items() if v in vals}
            if filtered_children:
                new_inner = Interface(children=filtered_children, acc=self.inner.acc, empty=new_empty)
                return LeveledAccTableGSS(new_inner, dict(self._values))
            else:
                new_inner = try_promote(UpperBranch(children={}, empty=new_empty))
                return LeveledAccTableGSS(new_inner, dict(self._values))


Leveled_acc_tableGSS = LeveledAccTableGSS