pub(crate) use super::pipeline::{
    compile_owned,
    compile_owned_profiled_with_table_construction,
    try_compile_owned_with_table_construction,
    try_compile_owned_with_table_construction_and_protected_shift_terminal_names,
    compile_profile_enabled,
    compile_top_profile_enabled,
    emit_compile_profile_summary,
};

#[derive(Debug)]
struct VocabContentDigest {
    digest: [u8; 32],
}

impl glrmask_vocab::__private::VocabDerivedArtifact for VocabContentDigest {}

/// Strong content identity for a model vocabulary.
///
/// The digest is a pure vocabulary-derived artifact and is deliberately
/// prepared by `prepare_vocab_for_dynamic_compile`, which callers such as CFA
/// invoke outside per-schema timing.  Transfer artifacts can therefore verify
/// that the parent supplied the same vocabulary as the compile worker without
/// rescanning every token byte string for every schema load.
#[derive(Debug)]
struct VocabFingerprint {
    fingerprint: [u8; 32],
}

impl glrmask_vocab::__private::VocabDerivedArtifact for VocabFingerprint {}

/// Fingerprint of a vocabulary's complete public identity: every ID-to-bytes
/// mapping plus the special (exact-only) token ID set. Persisted artifacts store
/// this fingerprint instead of the vocabulary.
pub(crate) fn vocab_fingerprint(vocab: &crate::Vocab) -> [u8; 32] {
    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabFingerprint>() {
        return cached.fingerprint;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"glrmask-vocab-fingerprint-v1\0");
    hasher.update(&(vocab.len() as u64).to_le_bytes());
    for (token_id, bytes) in vocab.iter() {
        hasher.update(&token_id.to_le_bytes());
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    let fingerprint = *hasher.finalize().as_bytes();
    vocab.vocab_derived_cache_set(std::sync::Arc::new(VocabFingerprint { fingerprint }));
    fingerprint
}

pub(crate) fn vocab_content_digest(vocab: &crate::Vocab) -> [u8; 32] {    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabContentDigest>() {
        return cached.digest;
    }
    let buffered = crate::compiler::boundary_env::enabled("GLRMASK_BUFFERED_VOCAB_DIGEST");
    let digest = compute_vocab_content_digest(vocab, buffered);
    if buffered && crate::env::var_os("GLRMASK_VALIDATE_BUFFERED_VOCAB_DIGEST").is_some() {
        assert_eq!(digest, compute_vocab_content_digest(vocab, false),
            "buffered vocabulary digest changed the canonical byte transcript");
        eprintln!("[glrmask/validate][buffered_vocab_digest] exact=true tokens={}", vocab.len());
    }
    vocab.vocab_derived_cache_set(std::sync::Arc::new(VocabContentDigest { digest }));
    digest
}

/// Feed the existing byte transcript without one hasher dispatch for every
/// small token field. Buffering changes neither boundaries in the transcript
/// nor the cache key. Large token strings can be hashed directly.
struct BufferedDigest<'a> {
    hasher: &'a mut blake3::Hasher,
    bytes: [u8; 4096],
    used: usize,
}
impl BufferedDigest<'_> {
    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if self.used == 0 && input.len() >= self.bytes.len() {
                self.hasher.update(input);
                return;
            }
            let count = input.len().min(self.bytes.len() - self.used);
            self.bytes[self.used..self.used + count].copy_from_slice(&input[..count]);
            self.used += count;
            input = &input[count..];
            if self.used == self.bytes.len() { self.flush(); }
        }
    }
    fn flush(&mut self) {
        if self.used != 0 {
            self.hasher.update(&self.bytes[..self.used]);
            self.used = 0;
        }
    }
}

fn compute_vocab_content_digest(vocab: &crate::Vocab, buffered: bool) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    if buffered {
        let mut writer = BufferedDigest { hasher: &mut hasher, bytes: [0; 4096], used: 0 };
        writer.update(b"glrmask-vocab-content-v1\0");
        writer.update(&(vocab.len() as u64).to_le_bytes());
        for (token_id, bytes) in vocab.iter() {
            writer.update(&token_id.to_le_bytes());
            writer.update(&(bytes.len() as u64).to_le_bytes());
            writer.update(bytes);
        }
        writer.flush();
    } else {
        hasher.update(b"glrmask-vocab-content-v1\0");
        hasher.update(&(vocab.len() as u64).to_le_bytes());
        for (token_id, bytes) in vocab.iter() {
            hasher.update(&token_id.to_le_bytes());
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
    }
    *hasher.finalize().as_bytes()
}

/// Populate only vocabulary artifacts used by DynamicConstraint compilation/runtime.
///
/// O2 vocabulary partitioning reuses the terminal-DWA module's *pure vocabulary*
/// L1/partition caches (identity orders, bounded-analysis tries, char-type
/// sub-vocabs, finite vocab projections).  Those are model-vocabulary state, not
/// grammar/constraint state, so prepare them here rather than lazily charging
/// the first O2 constraint that happens to request a vocabulary partition.
///
/// This still deliberately excludes grammar-specific terminal-DWA construction
/// and static possible-match preparation.
pub(crate) fn prepare_vocab_for_dynamic_compile(vocab: &crate::Vocab) {
    let _ = vocab_content_digest(vocab);
    super::stages::id_map_and_terminal_dwa::prepare_vocab_for_terminal_dwa(vocab);
    super::constraint_possible_matches::prepare_vocab_for_dynamic_mask(vocab);
}

pub(crate) fn prepare_vocab_for_compile(vocab: &crate::Vocab) {
    let profile = crate::env::var_os("GLRMASK_PROFILE_VOCAB_PREPARE").is_some();
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
mod buffered_digest_tests {
    use super::*;

    #[test]
    fn buffered_digest_matches_original_and_independent_transcript() {
        let mut random = 761063u64;
        let mut next = || { random = random.wrapping_mul(6364136223846793005).wrapping_add(1); random >> 32 };
        for case in 0..96 {
            let mut entries = vec![(0, Vec::new()), (3, b"same".to_vec()), (71, b"same".to_vec()),
                (90001, vec![0, 128, 255])];
            for i in 0..64 {
                let len = match i % 8 { 0 => 4095, 1 => 4096, 2 => 4097, 3 => 20000, _ => (next() % 100) as usize };
                entries.push((100 + i * 17, (0..len).map(|_| next() as u8).collect()));
            }
            let vocab = crate::Vocab::from_entries_unchecked(entries);
            let mut transcript = b"glrmask-vocab-content-v1\0".to_vec();
            transcript.extend_from_slice(&(vocab.len() as u64).to_le_bytes());
            for (id, bytes) in vocab.iter() {
                transcript.extend_from_slice(&id.to_le_bytes());
                transcript.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                transcript.extend_from_slice(bytes);
            }
            let independent = *blake3::hash(&transcript).as_bytes();
            assert_eq!(compute_vocab_content_digest(&vocab, false), independent, "reference case={case}");
            assert_eq!(compute_vocab_content_digest(&vocab, true), independent, "buffered case={case}");
        }
        let empty = crate::Vocab::from_entries_unchecked(Vec::new());
        assert_eq!(compute_vocab_content_digest(&empty, true), compute_vocab_content_digest(&empty, false));
    }

    #[test]
    fn digest_buffer_preserves_arbitrary_piece_boundaries() {
        let input = (0..40000usize).map(|i| (i.wrapping_mul(17) ^ (i >> 4)) as u8).collect::<Vec<_>>();
        for size in [1, 2, 7, 63, 64, 65, 1023, 1024, 4095, 4096, 4097, 10000, 40000] {
            let mut hash = blake3::Hasher::new();
            let mut writer = BufferedDigest { hasher: &mut hash, bytes: [0; 4096], used: 0 };
            writer.update(&[]);
            for piece in input.chunks(size) { writer.update(piece); writer.update(&[]); }
            writer.flush(); writer.flush();
            assert_eq!(hash.finalize(), blake3::hash(&input), "piece size={size}");
        }
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::vocab_fingerprint;
    use crate::Vocab;

    #[test]
    fn fingerprint_is_stable_across_calls_and_clones_and_distinct_per_vocab() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let first = vocab_fingerprint(&vocab);
        let cloned = vocab.clone();
        assert_eq!(vocab_fingerprint(&cloned), first);
        assert_eq!(vocab_fingerprint(&vocab), first);

        let other = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"c".to_vec())]);
        assert_ne!(vocab_fingerprint(&other), first);
    }
}

#[cfg(test)]
mod empty_byte_summary_tests {
    use crate::Vocab;
    use std::sync::Barrier;

    #[test]
    fn empty_byte_summary_is_exact_shared_and_vocab_local() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (512, Vec::new()), (31, Vec::new()),
            (7, b"last spelling wins".to_vec()), (3, Vec::new()),
            (0, vec![0]), (2, b"a".to_vec()), (90, b"a".to_vec()),
        ]);
        assert_eq!(vocab.compiler_cache_entry_count(), 0);
        let first = vocab.entries_map().empty_byte_token_ids();
        assert_eq!(first, &[3, 31, 512]);
        // No grammar, tokenizer or vocabulary trie is prepared by this helper.
        assert_eq!(vocab.compiler_cache_entry_count(), 0);
        let cloned = vocab.clone();
        assert!(std::ptr::eq(first, cloned.entries_map().empty_byte_token_ids()));
        assert!(std::ptr::eq(first, vocab.entries_map().empty_byte_token_ids()));
        let other = Vocab::from_entries_unchecked(vec![
            (512, vec![0]), (31, b"a".to_vec()), (7, b"last spelling wins".to_vec()),
            (3, b"b".to_vec()), (0, Vec::new()), (2, b"a".to_vec()), (90, Vec::new()),
        ]);
        assert_eq!(other.entries_map().empty_byte_token_ids(), &[0, 90]);
        assert_eq!(vocab.entries_map().empty_byte_token_ids(), &[3, 31, 512]);
        assert!(Vocab::from_entries_unchecked(Vec::new()).entries_map().empty_byte_token_ids().is_empty());
    }

    #[test]
    fn empty_byte_summary_concurrent_clones_share_storage() {
        let vocab = Vocab::from_entries_unchecked((0..4096u32)
            .map(|id| (id * 3, if id % 19 == 0 { Vec::new() } else { vec![id as u8] }))
            .collect());
        let expected = vocab.iter().filter_map(|(id, bytes)| bytes.is_empty().then_some(id))
            .collect::<Vec<_>>();
        let barrier = Barrier::new(8);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..8 {
                let cloned = vocab.clone();
                let barrier = &barrier;
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    cloned
                }));
            }
            let ids = handles.into_iter().map(|h| h.join().unwrap()).collect::<Vec<_>>();
            for result in &ids {
                assert_eq!(result.entries_map().empty_byte_token_ids(), expected.as_slice());
                assert!(std::ptr::eq(ids[0].entries_map().empty_byte_token_ids(), result.entries_map().empty_byte_token_ids()));
            }
        });
        assert_eq!(vocab.compiler_cache_entry_count(), 0);
    }
}
