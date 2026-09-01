pub(crate) use super::pipeline::{
    compile_owned,
    compile_owned_profiled_with_table_construction,
    compile_owned_with_table_construction,
    compile_owned_with_table_construction_and_protected_shift_terminal_names,
    compile_profile_enabled,
    compile_top_profile_enabled,
    emit_compile_profile_summary,
};

#[derive(Debug)]
struct VocabPackedTokenBytes {
    packed: std::sync::Arc<crate::runtime::PackedTokenBytes>,
}

impl glrmask_vocab::__private::VocabDerivedArtifact for VocabPackedTokenBytes {}

fn prepare_vocab_packed_token_bytes(
    vocab: &crate::Vocab,
) -> std::sync::Arc<crate::runtime::PackedTokenBytes> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabPackedTokenBytes>() {
        return std::sync::Arc::clone(&cached.packed);
    }
    let packed = std::sync::Arc::new(
        crate::runtime::PackedTokenBytes::from_runtime_entries(vocab.entries_map())
            .expect("vocabulary token bytes should form a valid indexed runtime vocabulary"),
    );
    vocab.vocab_derived_cache_set(std::sync::Arc::new(VocabPackedTokenBytes {
        packed: std::sync::Arc::clone(&packed),
    }));
    packed
}

pub(crate) fn vocab_packed_token_bytes(
    vocab: &crate::Vocab,
) -> std::sync::Arc<crate::runtime::PackedTokenBytes> {
    prepare_vocab_packed_token_bytes(vocab)
}

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
        let _ = prepare_vocab_packed_token_bytes(vocab);
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

#[cfg(test)]
mod tests {
    use super::vocab_packed_token_bytes;
    use crate::Vocab;
    use std::sync::Arc;

    #[test]
    fn packed_token_bytes_are_shared_across_vocab_clones() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (2, b"xyz".to_vec()),
        ]);
        let first = vocab_packed_token_bytes(&vocab);
        let clone = vocab.clone();
        let second = vocab_packed_token_bytes(&clone);

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.len(), 2);
        assert_eq!(first.get(0), Some(b"a".as_slice()));
        assert_eq!(first.get(2), Some(b"xyz".as_slice()));
    }
}
