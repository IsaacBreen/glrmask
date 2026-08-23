#![deny(warnings)]
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

/// Model vocabulary used when compiling a grammar constraint.
///
/// Entries map model token IDs to their exact byte sequences. Token IDs may be
/// sparse; masks are indexed by the original model token IDs.
pub struct Vocab {
    entries: Arc<BTreeMap<u32, Vec<u8>>>,
    compiler_cache: Arc<VocabCompilerCache>,
    max_token_byte_len: OnceLock<usize>,
}

#[derive(Default)]
struct VocabCompilerCache {
    artifacts: Mutex<BTreeMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

#[derive(Debug)]
struct VocabRelevantBytes {
    bytes: Arc<[u8]>,
}

impl VocabDerivedArtifact for VocabRelevantBytes {}

mod derived_artifact {
    use std::any::Any;

    /// Marker for artifacts that are pure functions of a `Vocab`'s token bytes.
    ///
    /// Do not implement this for grammar-, tokenizer-, or constraint-specific
    /// artifacts. `Vocab` instances can be reused across many grammar compiles,
    /// so this cache must only contain data that remains valid for every grammar
    /// using the same token bytes.
    pub trait VocabDerivedArtifact: Any + Send + Sync {}
}

use derived_artifact::VocabDerivedArtifact;

impl fmt::Debug for VocabCompilerCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabCompilerCache")
            .field(
                "entries",
                &self
                    .artifacts
                    .lock()
                    .map(|artifacts| artifacts.len())
                    .unwrap_or(0),
            )
            .finish()
    }
}

impl fmt::Debug for Vocab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vocab")
            .field("entries", &self.entries)
            .finish()
    }
}

impl Clone for Vocab {
    fn clone(&self) -> Self {
        let max_token_byte_len = OnceLock::new();
        if let Some(&length) = self.max_token_byte_len.get() {
            let _ = max_token_byte_len.set(length);
        }
        Self {
            entries: Arc::clone(&self.entries),
            compiler_cache: Arc::clone(&self.compiler_cache),
            max_token_byte_len,
        }
    }
}

impl Vocab {
    /// Build a vocabulary from `(token_id, token_bytes)` pairs.
    pub fn new(entries: Vec<(u32, Vec<u8>)>) -> Self {
        let entries = Arc::new(entries.into_iter().collect::<BTreeMap<_, _>>());
        let max_token_byte_len = OnceLock::new();
        let _ = max_token_byte_len.set(entries.values().map(Vec::len).max().unwrap_or(0));
        Self {
            entries,
            compiler_cache: Arc::new(VocabCompilerCache::default()),
            max_token_byte_len,
        }
    }

    /// Maximum byte length of any token in this vocabulary.
    ///
    /// Fresh vocabularies compute this while being constructed. Deserialized
    /// vocabularies fill it lazily on first use, after which clones preserve the
    /// value instead of rescanning every token for every grammar compilation.
    #[doc(hidden)]
    pub fn max_token_byte_len(&self) -> usize {
        *self
            .max_token_byte_len
            .get_or_init(|| self.entries.values().map(Vec::len).max().unwrap_or(0))
    }

    /// Sorted byte alphabet observed anywhere in the vocabulary.
    #[doc(hidden)]
    pub fn relevant_bytes(&self) -> Arc<[u8]> {
        if let Some(cached) = self.vocab_derived_cache_get_internal::<VocabRelevantBytes>() {
            return Arc::clone(&cached.bytes);
        }
        let mut observed = [false; 256];
        for token in self.entries.values() {
            for &byte in token {
                observed[byte as usize] = true;
            }
        }
        let bytes = Arc::<[u8]>::from(
            observed
                .iter()
                .enumerate()
                .filter_map(|(byte, &present)| present.then_some(byte as u8))
                .collect::<Vec<_>>(),
        );
        self.vocab_derived_cache_set_internal(Arc::new(VocabRelevantBytes {
            bytes: Arc::clone(&bytes),
        }));
        bytes
    }

    /// Return the number of vocabulary entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the vocabulary contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the highest token ID, or `0` for an empty vocabulary.
    pub fn max_token_id(&self) -> u32 {
        self.entries
            .last_key_value()
            .map_or(0, |(&token_id, _)| token_id)
    }

    fn vocab_derived_cache_get_internal<T: VocabDerivedArtifact>(&self) -> Option<Arc<T>> {
        self.compiler_cache
            .artifacts
            .lock()
            .ok()?
            .get(&TypeId::of::<T>())
            .cloned()
            .and_then(|artifact| artifact.downcast::<T>().ok())
    }

    fn vocab_derived_cache_set_internal<T: VocabDerivedArtifact>(&self, artifact: Arc<T>) {
        let erased: Arc<dyn Any + Send + Sync> = artifact;
        if let Ok(mut artifacts) = self.compiler_cache.artifacts.lock() {
            artifacts.entry(TypeId::of::<T>()).or_insert(erased);
        }
    }

    /// Return the bytes associated with one token ID.
    pub fn get(&self, token_id: u32) -> Option<&[u8]> {
        self.entries.get(&token_id).map(Vec::as_slice)
    }

    /// Iterate over token IDs and their exact byte sequences in ID order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u32, &[u8])> {
        self.entries
            .iter()
            .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn entries_map(&self) -> &BTreeMap<u32, Vec<u8>> {
        &self.entries
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn entries_arc(&self) -> Arc<BTreeMap<u32, Vec<u8>>> {
        Arc::clone(&self.entries)
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn vocab_derived_cache_get<T: __private::VocabDerivedArtifact>(&self) -> Option<Arc<T>> {
        self.vocab_derived_cache_get_internal::<T>()
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn vocab_derived_cache_set<T: __private::VocabDerivedArtifact>(&self, artifact: Arc<T>) {
        self.vocab_derived_cache_set_internal(artifact);
    }

    #[cfg(feature = "internal-api")]
    #[doc(hidden)]
    pub fn compiler_cache_entry_count(&self) -> usize {
        self.compiler_cache
            .artifacts
            .lock()
            .map(|artifacts| artifacts.len())
            .unwrap_or(0)
    }
}

pub(crate) mod vocab_prefix_tree;

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub use super::derived_artifact::VocabDerivedArtifact;
    pub use super::Vocab;

    pub mod vocab_prefix_tree {
        pub use super::super::vocab_prefix_tree::*;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_pure_derived_artifact_cache() {
        let vocab = Vocab::new(vec![(0, b"ab".to_vec()), (1, b"bc".to_vec())]);
        assert_eq!(vocab.compiler_cache.artifacts.lock().unwrap().len(), 0);
        let _ = vocab.relevant_bytes();
        assert_eq!(vocab.compiler_cache.artifacts.lock().unwrap().len(), 1);

        let cloned = vocab.clone();
        assert!(Arc::ptr_eq(&vocab.compiler_cache, &cloned.compiler_cache));
        assert_eq!(cloned.compiler_cache.artifacts.lock().unwrap().len(), 1);
    }
}
