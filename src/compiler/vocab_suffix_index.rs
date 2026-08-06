use std::sync::Arc;

use glrmask_vocab::__private::VocabDerivedArtifact;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;

/// Vocabulary-only index of every non-empty suffix of every multi-byte token.
///
/// Boundary composition repeatedly asks which model tokens own a suffix that
/// belongs to a selected terminal-entry language. The suffix topology depends
/// only on token bytes, so building it inside every link is needless grammar
/// work. `Vocab::prepare_for_compile()` populates this cache before compilation
/// timing; composition then performs only the grammar-specific language test.
#[derive(Debug)]
pub(crate) struct VocabSuffixIndex {
    entries: Box<[VocabSuffixOwners]>,
}

#[derive(Debug)]
pub(crate) struct VocabSuffixOwners {
    suffix: Box<[u8]>,
    token_ids: Box<[u32]>,
}

impl VocabDerivedArtifact for VocabSuffixIndex {}

impl VocabSuffixIndex {
    pub(crate) fn entries(&self) -> &[VocabSuffixOwners] {
        &self.entries
    }
}

impl VocabSuffixOwners {
    pub(crate) fn suffix(&self) -> &[u8] {
        &self.suffix
    }

    pub(crate) fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

fn build(vocab: &Vocab) -> VocabSuffixIndex {
    let mut owners = FxHashMap::<&[u8], SmallVec<[u32; 2]>>::default();
    for (token_id, bytes) in vocab.iter() {
        if bytes.len() < 2 {
            continue;
        }
        for offset in 0..bytes.len() {
            owners.entry(&bytes[offset..]).or_default().push(token_id);
        }
    }

    let mut entries = owners
        .into_iter()
        .map(|(suffix, mut token_ids)| {
            token_ids.sort_unstable();
            token_ids.dedup();
            VocabSuffixOwners {
                suffix: suffix.to_vec().into_boxed_slice(),
                token_ids: token_ids.into_vec().into_boxed_slice(),
            }
        })
        .collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| left.suffix.cmp(&right.suffix));
    VocabSuffixIndex {
        entries: entries.into_boxed_slice(),
    }
}

pub(crate) fn get(vocab: &Vocab) -> Arc<VocabSuffixIndex> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabSuffixIndex>() {
        return cached;
    }
    let built = Arc::new(build(vocab));
    vocab.vocab_derived_cache_set(Arc::clone(&built));
    vocab
        .vocab_derived_cache_get::<VocabSuffixIndex>()
        .unwrap_or(built)
}

pub(crate) fn prepare(vocab: &Vocab) {
    drop(get(vocab));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_index_is_vocab_cached_and_tracks_alias_owners() {
        let vocab = Vocab::new(vec![
            (3, b"abc".to_vec()),
            (7, b"xbc".to_vec()),
            (9, b"z".to_vec()),
        ]);
        let first = get(&vocab);
        let second = get(&vocab);
        assert!(Arc::ptr_eq(&first, &second));
        let bc = first
            .entries()
            .iter()
            .find(|entry| entry.suffix() == b"bc")
            .expect("shared suffix");
        assert_eq!(bc.token_ids(), &[3, 7]);
        assert!(first.entries().iter().all(|entry| entry.suffix() != b"z"));
    }
}
