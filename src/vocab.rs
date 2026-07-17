use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Vocab {
    pub entries: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(skip)]
    compiler_cache: VocabCompilerCache,
}

#[derive(Default)]
struct VocabCompilerCache {
    artifacts: Mutex<BTreeMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

pub(crate) trait VocabDerivedArtifact: Any + Send + Sync {}

impl fmt::Debug for VocabCompilerCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabCompilerCache")
            .field("entries", &self.artifacts.lock().map(|a| a.len()).unwrap_or(0))
            .finish()
    }
}

impl fmt::Debug for Vocab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vocab").field("entries", &self.entries).finish()
    }
}

impl Clone for Vocab {
    fn clone(&self) -> Self {
        Self { entries: Arc::clone(&self.entries), compiler_cache: VocabCompilerCache::default() }
    }
}

impl Vocab {
    pub fn new(entries: Vec<(u32, Vec<u8>)>) -> Self {
        Self { entries: Arc::new(entries.into_iter().collect()), compiler_cache: VocabCompilerCache::default() }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn max_token_id(&self) -> u32 {
        self.entries.last_key_value().map_or(0, |(&id, _)| id)
    }

    pub(crate) fn vocab_derived_cache_get<T: VocabDerivedArtifact>(&self) -> Option<Arc<T>> {
        self.compiler_cache.artifacts.lock().ok()?.get(&TypeId::of::<T>()).cloned()
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
        self.compiler_cache.artifacts.lock().map(|a| a.len()).unwrap_or(0)
    }
}
