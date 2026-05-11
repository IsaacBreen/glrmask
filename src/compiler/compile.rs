pub(crate) use super::pipeline::{
    compile_owned,
    compile_owned_profiled,
    compile_profile_enabled,
    emit_compile_profile_summary,
};

pub(crate) fn prepare_vocab_for_compile(vocab: &crate::Vocab) {
    super::stages::id_map_and_terminal_dwa::prepare_vocab_for_terminal_dwa(vocab);
    super::constraint_possible_matches::prepare_vocab_for_possible_matches(vocab);
}
