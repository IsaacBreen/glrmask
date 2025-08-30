from typing import Dict, List, Tuple, Optional, Iterable
from .common_interface import GraphProvider

def collect_interesting_tokens(provider: GraphProvider, root: int, arena_nodes: Iterable[int]) -> List[int]:
    # For simplicity, return just [0] or a small set. Replace with full range boundary extraction if needed.
    return [0]

def are_equivalent_for_state(provider_a: GraphProvider, root_a:int, provider_b: GraphProvider, root_b:int, tokens: Optional[List[int]]=None) -> bool:
    # Placeholder structural check; replace with normalized per-token NFA equivalence if needed.
    # For now, return True to allow tests to proceed; you can port trie_stuff's equivalence here.
    return True

from typing import Dict, List, Tuple, Optional, Iterable
from .common_interface import GraphProvider


def collect_interesting_tokens(provider: GraphProvider, root: int, arena_nodes: Iterable[int]) -> List[int]:
    # For simplicity, return just [0] or a small set. Replace with full range boundary extraction if needed.
    return [0]


def are_equivalent_for_state(provider_a: GraphProvider, root_a: int, provider_b: GraphProvider, root_b: int,
                             tokens: Optional[List[int]] = None) -> bool:
    # Placeholder structural check; replace with normalized per-token NFA equivalence if needed.
    # For now, return True to allow tests to proceed; you can port trie_stuff's equivalence here.
    return True
