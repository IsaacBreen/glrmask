use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

/// Model vocabulary used when compiling a grammar constraint.
///
/// Entries map model token IDs to their exact byte sequences. Token IDs may be
/// sparse; masks are indexed by the original model token IDs.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Vocab {
    pub entries: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(skip)]
    compiler_cache: VocabCompilerCache,
    #[serde(skip)]
    max_token_byte_len: OnceLock<usize>,
}

#[derive(Default)]
struct VocabCompilerCache {
    artifacts: Mutex<BTreeMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

/// Marker for artifacts that are pure functions of a `Vocab`'s token bytes.
///
/// Do not implement this for grammar-, tokenizer-, or constraint-specific
/// artifacts. `Vocab` instances can be reused across many grammar compiles, so
/// this cache must only contain data that remains valid for every grammar using
/// the same token bytes.
pub(crate) trait VocabDerivedArtifact: Any + Send + Sync {}

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
            compiler_cache: VocabCompilerCache::default(),
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
            compiler_cache: VocabCompilerCache::default(),
            max_token_byte_len,
        }
    }

    /// Maximum byte length of any token in this vocabulary.
    ///
    /// Fresh vocabularies compute this while being constructed. Deserialized
    /// vocabularies fill it lazily on first use, after which clones preserve the
    /// value instead of rescanning every token for every grammar compilation.
    pub(crate) fn max_token_byte_len(&self) -> usize {
        *self
            .max_token_byte_len
            .get_or_init(|| self.entries.values().map(Vec::len).max().unwrap_or(0))
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

    pub(crate) fn vocab_derived_cache_get<T: VocabDerivedArtifact>(&self) -> Option<Arc<T>> {
        self.compiler_cache
            .artifacts
            .lock()
            .ok()?
            .get(&TypeId::of::<T>())
            .cloned()
            .and_then(|artifact| artifact.downcast::<T>().ok())
    }

    pub(crate) fn vocab_derived_cache_set<T: VocabDerivedArtifact>(&self, artifact: Arc<T>) {
        let erased: Arc<dyn Any + Send + Sync> = artifact;
        if let Ok(mut artifacts) = self.compiler_cache.artifacts.lock() {
            artifacts.entry(TypeId::of::<T>()).or_insert(erased);
        }
    }

    #[doc(hidden)]
    pub fn compiler_cache_entry_count(&self) -> usize {
        self.compiler_cache
            .artifacts
            .lock()
            .map(|artifacts| artifacts.len())
            .unwrap_or(0)
    }
}
