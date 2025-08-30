class GraphProvider:
    def get_root(self, state_id: int) -> int:
        raise NotImplementedError
    def is_end(self, node: int) -> bool:
        raise NotImplementedError
    def iter_edges(self, node: int, token: int):
        """
        Yield (pop_count: int, state_id_or_none: Optional[int], dest_node: int) for edges whose token filter passes.
        Implementations for precompute3 can prefilter with their token bitsets; for precompute2 leave filtering to caller.
        """
        raise NotImplementedError
