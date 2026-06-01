# Runtime template-DFA execution

Commit owns the online state transition. Template DFA execution is a semantic-preserving optimization of parser-stack advance only. Returning `None` is not failure; it means use the GLR table reference path.
