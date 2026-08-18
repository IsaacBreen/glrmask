pub(crate) use super::pipeline::{
    compile_owned,
    compile_owned_profiled_with_table_construction,
    compile_owned_with_table_construction,
    compile_profile_enabled,
    compile_top_profile_enabled,
    emit_compile_profile_summary,
};

pub(crate) fn prepare_vocab_for_compile(vocab: &crate::Vocab) {
    let profile = std::env::var_os("GLRMASK_PROFILE_VOCAB_PREPARE").is_some();
    let run = |name: &str, f: &mut dyn FnMut()| {
        let started = std::time::Instant::now();
        f();
        if profile {
            eprintln!(
                "[glrmask/profile][vocab_prepare] name={name} ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    };
    run("packed_token_bytes", &mut || {
        crate::runtime::prepare_shared_packed_token_bytes(vocab)
    });
    run("terminal_dwa", &mut || {
        super::stages::id_map_and_terminal_dwa::prepare_vocab_for_terminal_dwa(vocab)
    });
    run("possible_matches", &mut || {
        super::constraint_possible_matches::prepare_vocab_for_possible_matches(vocab)
    });
    run("dynamic_mask", &mut || {
        super::constraint_possible_matches::prepare_vocab_for_dynamic_mask(vocab)
    });
}
