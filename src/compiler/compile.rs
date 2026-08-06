pub(crate) use super::pipeline::{
    compile_owned,
    compile_owned_profiled_with_table_construction,
    compile_owned_with_table_construction,
    compile_profile_enabled,
    compile_top_profile_enabled,
    emit_compile_profile_summary,
};

pub(crate) fn prepare_vocab_for_compile(vocab: &crate::Vocab) {
    super::stages::id_map_and_terminal_dwa::prepare_vocab_for_terminal_dwa(vocab);
    super::constraint_possible_matches::prepare_vocab_for_possible_matches(vocab);
    super::constraint_possible_matches::prepare_vocab_for_dynamic_mask(vocab);
    super::vocab_suffix_index::prepare(vocab);
}
