use super::{BuildInput, LocalIdMapTerminalDwa};

pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    super::super::build_l1_id_map_and_terminal_dwa_production(
        input.partition_label,
        input.tokenizer,
        input.vocab,
        input.terminal_coloring,
        input.use_terminal_coloring,
        input.ignore_terminal,
        input.grammar,
        input.active_terminals,
        input.flat_trans,
        input.transitions_by_byte,
        input.initial_state_map,
        input.shared_generic_nfa_topology,
        input.shared_generic_nfa_trie,
        input.subset_parent_order,
    )
}
