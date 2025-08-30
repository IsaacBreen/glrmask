from typing import Dict, List, Tuple, Optional, Iterable, Set, Deque
from common_interface import GraphProvider, RangeSet
import collections

# The alphabet for normalized graphs: labels are pairs (k, sid)
Label = Tuple[int, int]  # (k, state_id)
NormEdge = Tuple[Label, int]  # (label, dest_node_index)

class TokenNormalizer:
    """
    Builds token-specific normalized edges (epsilon-free) on demand.
    """
    def __init__(self, provider: GraphProvider, token: int, max_closure_expansions: int = 200000):
        self.provider = provider
        self.token = int(token)
        self._accept_cache: Dict[int, bool] = {}
        self._edges_cache: Dict[int, List[NormEdge]] = {}
        self._max_closure_expansions = max(1, int(max_closure_expansions))

    def accepting(self, node: int) -> bool:
        if node in self._accept_cache:
            return self._accept_cache[node]
        # BFS over None-edges only
        q: Deque[int] = collections.deque([node])
        seen: Set[int] = set()
        accepts = False
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            if self.provider.is_end(u):
                accepts = True
                break
            for pop, sid_opt, v in self.provider.iter_edges(u, self.token):
                if sid_opt is None:
                    q.append(v)
        self._accept_cache[node] = accepts
        return accepts

    def out_edges(self, node: int) -> List[NormEdge]:
        if node in self._edges_cache:
            return self._edges_cache[node]

        out: List[NormEdge] = []
        expansions = 0
        stack: List[Tuple[int, int]] = [(node, 0)]
        seen: Set[Tuple[int, int]] = set()

        while stack:
            u, ksum = stack.pop()
            key = (u, ksum)
            if key in seen:
                continue
            seen.add(key)

            for p, sid_opt, v in self.provider.iter_edges(u, self.token):
                if sid_opt is not None:
                    label: Label = (ksum + p, sid_opt)
                    out.append((label, v))

            for p, sid_opt, v in self.provider.iter_edges(u, self.token):
                if sid_opt is None:
                    new_ksum = ksum + p
                    stack.append((v, new_ksum))
                    expansions += 1
                    if expansions > self._max_closure_expansions:
                        raise RuntimeError(f"Exceeded maximum None-closure expansions for token {self.token}")

        dedup: Dict[Tuple[int, int, int], None] = {}
        norm_edges: List[NormEdge] = []
        for (label, dest) in out:
            k, sid = label
            key = (k, sid, dest)
            if key not in dedup:
                dedup[key] = None
                norm_edges.append((label, dest))

        self._edges_cache[node] = norm_edges
        return norm_edges

def nfa_equivalence_on_labels(
    normA: TokenNormalizer,
    rootA: int,
    normB: TokenNormalizer,
    rootB: int,
    max_product_states: int = 200000,
) -> Tuple[bool, Optional[List[Label]]]:
    def subset_accepts(nodes: Set[int], norm: TokenNormalizer) -> bool:
        return any(norm.accepting(n) for n in nodes)

    def next_subset(nodes: Set[int], norm: TokenNormalizer, label: Label) -> Set[int]:
        return {dest for n in nodes for lbl, dest in norm.out_edges(n) if lbl == label}

    def labels_from_subset(nodes: Set[int], norm: TokenNormalizer) -> Set[Label]:
        return {lbl for n in nodes for lbl, _ in norm.out_edges(n)}

    startA: frozenset = frozenset({rootA})
    startB: frozenset = frozenset({rootB})
    ParentKey = Tuple[frozenset, frozenset]
    parent: Dict[ParentKey, Tuple[Optional[ParentKey], Optional[Label]]] = {}

    def _reconstruct_path(to_key: ParentKey) -> List[Label]:
        seq: List[Label] = []
        cur: Optional[ParentKey] = to_key
        while cur is not None:
            par, edge_lab = parent.get(cur, (None, None))
            if edge_lab is not None:
                seq.append(edge_lab)
            cur = par
        seq.reverse()
        return seq
    parent[(startA, startB)] = (None, None)

    visited: Set[ParentKey] = set()
    q: Deque[ParentKey] = collections.deque([(startA, startB)])

    if subset_accepts(set(startA), normA) != subset_accepts(set(startB), normB):
        return (False, [])

    explored = 0
    while q:
        SA, SB = q.popleft()
        if (SA, SB) in visited:
            continue
        visited.add((SA, SB))
        explored += 1
        if explored > max_product_states:
            raise RuntimeError("Equivalence check exceeded product state limit")

        labels_A = labels_from_subset(set(SA), normA)
        labels_B = labels_from_subset(set(SB), normB)

        if labels_A != labels_B:
            diff_labels = (labels_A - labels_B) or (labels_B - labels_A)
            witness_step = next(iter(diff_labels))
            seq = _reconstruct_path((SA, SB))
            seq.append(witness_step)
            return (False, seq)

        for lab in labels_A:
            NSA = frozenset(next_subset(set(SA), normA, lab))
            NSB = frozenset(next_subset(set(SB), normB, lab))
            key = (NSA, NSB)
            if key not in parent:
                parent[key] = ((SA, SB), lab)

            if subset_accepts(set(NSA), normA) != subset_accepts(set(NSB), normB):
                return (False, _reconstruct_path(key))

            if key not in visited:
                q.append(key)

    return (True, None)

def collect_interesting_tokens(provider: GraphProvider, root: int, arena_nodes: Iterable[int]) -> List[int]:
    # For simplicity, return just [0] or a small set. Replace with full range boundary extraction if needed.
    return [0]


def are_equivalent_for_state(provider_a: GraphProvider, root_a: int, provider_b: GraphProvider, root_b: int,
                             tokens: Optional[List[int]] = None, verbose: bool = False) -> Tuple[bool, Optional[str]]:
    if tokens is None:
        # This is a placeholder. A real implementation would need a way to inspect all edge BVs.
        tokens = [0, 1, 10, 100, 1000]

    for token in tokens:
        try:
            norm_a = TokenNormalizer(provider_a, token)
            norm_b = TokenNormalizer(provider_b, token)
            eq, witness = nfa_equivalence_on_labels(norm_a, root_a, norm_b, root_b)
            if not eq:
                msg = f"Equivalence failed for token {token}"
                if witness is not None:
                    seq_str = " -> ".join(f"(k={k}, sid={sid})" for (k, sid) in witness) if witness else "(empty)"
                    msg += f". Counterexample label sequence: {seq_str}"
                return False, msg
        except Exception as e:
            return False, f"Equivalence check raised an exception for token {token}: {e}"
    return True, None
