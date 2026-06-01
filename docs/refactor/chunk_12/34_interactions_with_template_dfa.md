# Interaction with template DFAs

Template DFAs are an acceleration for parser stack effects. They are not a separate semantics.

Commit calls `parser_advance.rs`, which can select template-DFA execution or reference parser-table advance. Validation modes should compare those results. The source split makes this relationship clearer because template execution is no longer buried in the same file as API methods and scanner pruning.
