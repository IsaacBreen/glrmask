//! L2+ equivalence analysis: full vocab equivalence for terminals with path length ≥ 2.
//!
//! Uses the existing combined equivalence analysis pipeline but filtered to
//! only track L2+ terminal groups in the DFA.
//!
//! TODO: Wire up L2+ group filtering using find_vocab_equivalence_classes_with_group_filter.
