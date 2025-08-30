from typing import Any, Dict, List, Tuple, Optional, Iterable
from .common_interface import GraphProvider

class RangeSet:
    # same small RangeSet as above or import it
    pass

class CompEdge:
    __slots__ = ("pop","sid","dest","bv")
    def __init__(self,pop:int,sid:Optional[int],dest:int,bv:RangeSet):
        self.pop=pop; self.sid=sid; self.dest=dest; self.bv=bv

class CompressedTrie:
    def __init__(self, roots: Dict[int,int], ends: List[bool], edges: List[List[CompEdge]]):
        self.roots=roots; self.ends=ends; self.edges=edges

def optimize(provider: GraphProvider, roots_map: Dict[int,int], all_nodes: Iterable[int]) -> CompressedTrie:
    # Copy the bisimulation-like quotient idea (without token union over long cycles).
    # This is a minimal, illustrative optimizer.
    # For brevity, we’ll stub a trivial identity mapping here; you can paste a full refinement if needed.
    ends = []
    edges = []
    id_map = {}  # node -> class
    for i,u in enumerate(all_nodes):
        id_map[u]=i
        ends.append(provider.is_end(u))
        # build edges without token filters (general)
        lst=[]
        # tokens unknown here; caller can pre-aggregate; this is a placeholder
        edges.append(lst)
    roots = dict(roots_map)
    return CompressedTrie(roots, ends, edges)
from typing import Any, Dict, List, Tuple, Optional, Iterable
from .common_interface import GraphProvider


class RangeSet:
    # same small RangeSet as above or import it
    pass


class CompEdge:
    __slots__ = ("pop", "sid", "dest", "bv")

    def __init__(self, pop: int, sid: Optional[int], dest: int, bv: RangeSet):
        self.pop = pop;
        self.sid = sid;
        self.dest = dest;
        self.bv = bv


class CompressedTrie:
    def __init__(self, roots: Dict[int, int], ends: List[bool], edges: List[List[CompEdge]]):
        self.roots = roots;
        self.ends = ends;
        self.edges = edges


def optimize(provider: GraphProvider, roots_map: Dict[int, int], all_nodes: Iterable[int]) -> CompressedTrie:
    # This is a minimal, illustrative optimizer.
    # For brevity, we’ll stub a trivial identity mapping here; you can paste a full refinement if needed.
    ends = [provider.is_end(u) for u in all_nodes]
    edges = [[] for _ in all_nodes]
    roots = dict(roots_map)
    return CompressedTrie(roots, ends, edges)
