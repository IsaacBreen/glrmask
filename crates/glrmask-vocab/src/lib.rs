#![deny(warnings)]
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

/// Error returned when constructing a vocabulary-qualified exact-token value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactTokenError {
    message: String,
}

impl ExactTokenError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ExactTokenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ExactTokenError {}

/// Model vocabulary used when compiling a grammar constraint.
///
/// Entries map model token IDs to their exact byte sequences. Token IDs may be
/// sparse; masks are indexed by the original model token IDs.
pub struct Vocab {
    entries: Arc<BTreeMap<u32, Vec<u8>>>,
    /// Model token IDs which are valid for exact-token bindings but have no
    /// byte spelling in the grammar lexer (for example control/tool tokens).
    ///
    /// These IDs are deliberately absent from the byte entries: they must
    /// never become zero-byte alternatives in the vocabulary trie.
    exact_only_token_ids: Arc<BTreeSet<u32>>,
    compiler_cache: Arc<VocabCompilerCache>,
    exact_token_ranges: Arc<[(u32, u32)]>,
    max_token_byte_len: OnceLock<usize>,
}

/// An original token ID with no vocabulary identity. Only ID membership is checked against the target vocabulary; the same ID can refer to different bytes. Use `vocab.tokens([id])` for a vocabulary-bound singleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Token {
    id: u32,
}

impl Token {
    /// Membership is checked at compile/link binding.
    pub const fn new(id: u32) -> Self { Self { id } }

    pub const fn id(&self) -> u32 { self.id }
}

/// One token event chosen from a vocabulary-bound set. Vocabularies must have the same ID-to-byte mapping and exact-only IDs, even for a singleton. An empty set matches nothing.
#[derive(Debug, Clone)]
pub struct ExactTokens {
    vocab: Vocab,
    ranges: Arc<[(u32, u32)]>,
    // Materialized only when the caller explicitly requests individual IDs.
    ids: Arc<OnceLock<Box<[u32]>>>,
}

impl ExactTokens {
    /// The exact model token IDs, sorted in ascending order.
    ///
    /// This explicitly materializes the IDs. Compilation and linking use the
    /// compact ranges instead; use [`Self::ranges`] to inspect large sets.
    pub fn ids(&self) -> &[u32] {
        self.ids.get_or_init(|| self.ranges().flatten().collect())
    }

    /// Iterate over normalized, disjoint inclusive token-ID ranges.
    pub fn ranges(&self) -> impl ExactSizeIterator<Item = std::ops::RangeInclusive<u32>> + '_ {
        self.ranges.iter().map(|&(start, end)| start..=end)
    }

    /// Number of allowed token IDs (not the number of ranges).
    pub fn len(&self) -> usize {
        self.ranges.iter().map(|&(start, end)| (u64::from(end) - u64::from(start) + 1) as usize).sum()
    }

    pub fn is_empty(&self) -> bool { self.ranges.is_empty() }

    #[doc(hidden)]
    pub fn range_pairs(&self) -> &[(u32, u32)] { &self.ranges }

    /// Whether this value belongs to exactly vocab's token-ID/byte mapping.
    #[doc(hidden)]
    pub fn targets(&self, vocab: &Vocab) -> bool {
        self.vocab.same_model_vocab(vocab)
    }
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
            .field("exact_only_token_ids", &self.exact_only_token_ids)
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
            exact_only_token_ids: Arc::clone(&self.exact_only_token_ids),
            compiler_cache: Arc::clone(&self.compiler_cache),
            exact_token_ranges: Arc::clone(&self.exact_token_ranges),
            max_token_byte_len,
        }
    }
}

impl Vocab {
    /// Build a vocabulary from `(token_id, token_bytes)` pairs.
    pub fn new(entries: Vec<(u32, Vec<u8>)>) -> Self {
        Self::new_with_exact_token_ids(entries, std::iter::empty())
    }

    /// Build a vocabulary with additional model token IDs that have no byte
    /// spelling but are valid for exact-token bindings.
    ///
    /// This is primarily used by model integrations which can distinguish
    /// control/tool tokens from byte-backed text tokens.
    pub fn new_with_exact_token_ids(
        entries: Vec<(u32, Vec<u8>)>,
        exact_token_ids: impl IntoIterator<Item = u32>,
    ) -> Self {
        let entries = Arc::new(entries.into_iter().collect::<BTreeMap<_, _>>());
        let exact_only_token_ids = Arc::new(
            exact_token_ids
                .into_iter()
                .filter(|id| !entries.contains_key(id))
                .collect::<BTreeSet<_>>(),
        );
        let exact_token_ranges = range_set_blaze::RangeSetBlaze::from_iter(
            entries.keys().chain(exact_only_token_ids.iter()).copied(),
        ).ranges().map(|range| (*range.start(), *range.end())).collect::<Vec<_>>().into();
        let max_token_byte_len = OnceLock::new();
        let _ = max_token_byte_len.set(entries.values().map(Vec::len).max().unwrap_or(0));
        Self {
            entries,
            exact_only_token_ids,
            compiler_cache: Arc::new(VocabCompilerCache::default()),
            exact_token_ranges,
            max_token_byte_len,
        }
    }

    /// Match one of the supplied IDs, retaining vocabulary identity even for a singleton. Unknown or duplicate IDs are errors. Empty input matches nothing, not epsilon.
    pub fn tokens(
        &self,
        token_ids: impl IntoIterator<Item = u32>,
    ) -> Result<ExactTokens, ExactTokenError> {
        let mut ids = token_ids.into_iter().collect::<Vec<_>>();
        ids.sort_unstable();
        for pair in ids.windows(2) {
            if pair[0] == pair[1] {
                return Err(ExactTokenError::new(format!(
                    "exact-token binding contains duplicate token ID {}",
                    pair[0],
                )));
            }
        }
        if let Some(&missing) = ids.iter().find(|&&id| !self.contains_exact_token_id(id)) {
            return Err(ExactTokenError::new(format!(
                "token ID {missing} is not present in this vocabulary",
            )));
        }
        let ranges = range_set_blaze::RangeSetBlaze::from_iter(ids)
            .ranges().map(|range| (*range.start(), *range.end())).collect::<Vec<_>>();
        Ok(ExactTokens {
            vocab: self.clone(),
            ranges: ranges.into(),
            ids: Arc::new(OnceLock::new()),
        })
    }

    /// Union of inclusive ranges, normalized without enumerating IDs. Overlapping or adjacent ranges merge. Reversed ranges or unknown IDs are errors; empty input matches nothing. A match consumes one token, not a sequence.
    pub fn token_ranges(
        &self,
        ranges: impl IntoIterator<Item = std::ops::RangeInclusive<u32>>,
    ) -> Result<ExactTokens, ExactTokenError> {
        let mut pairs = Vec::new();
        for range in ranges {
            if range.is_empty() {
                return Err(ExactTokenError::new("exact-token range must not be reversed or empty"));
            }
            pairs.push((*range.start(), *range.end()));
        }
        pairs.sort_unstable();
        let mut normalized = Vec::<(u32, u32)>::with_capacity(pairs.len());
        for (start, end) in pairs {
            if let Some(previous) = normalized.last_mut()
                && u64::from(start) <= u64::from(previous.1) + 1
            {
                previous.1 = previous.1.max(end);
                continue;
            }
            normalized.push((start, end));
        }
        for &(start, end) in &normalized {
            if !self.contains_exact_token_range(start, end) {
                return Err(ExactTokenError::new(format!(
                    "token range {start}..={end} contains an ID not present in this vocabulary",
                )));
            }
        }
        Ok(ExactTokens {
            vocab: self.clone(),
            ranges: normalized.into(),
            ids: Arc::new(OnceLock::new()),
        })
    }

    /// Whether every ID in the inclusive range is defined (including exact-only IDs).
    #[doc(hidden)]
    /// All IDs that may be consumed as grammar events, including IDs without
    /// ordinary byte spellings. This view does not include root-only controls
    /// merely because they are present in a caller's termination policy.
    pub fn exact_token_ranges(&self) -> impl ExactSizeIterator<Item=std::ops::RangeInclusive<u32>> + '_ {
        self.exact_token_ranges.iter().map(|&(start,end)|start..=end)
    }

    pub fn contains_exact_token_range(&self, start: u32, end: u32) -> bool {
        if start > end { return false; }
        let index = self.exact_token_ranges.partition_point(|&(lower, _)| lower <= start);
        index > 0 && self.exact_token_ranges[index - 1].1 >= end
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

    /// Return the number of byte-backed vocabulary entries.
    ///
    /// Exact-only model token IDs are not counted because they do not
    /// participate in lexer/vocabulary-trie compilation.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the byte-backed vocabulary contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether this model token ID is known either as a byte-backed token or
    /// as an exact-only token.
    #[doc(hidden)]
    pub fn contains_exact_token_id(&self, token_id: u32) -> bool {
        self.entries.contains_key(&token_id) || self.exact_only_token_ids.contains(&token_id)
    }

    /// Highest defined model ID, including exact-only IDs.
    #[doc(hidden)]
    pub fn max_exact_token_id(&self) -> Option<u32> {
        self.exact_token_ranges.last().map(|&(_, end)| end)
    }

    /// Find a free ID at or above `start` without scanning each occupied ID.
    #[doc(hidden)]
    pub fn next_available_exact_token_id(&self, start: u32) -> Option<u32> {
        let index = self.exact_token_ranges.partition_point(|&(lower, _)| lower <= start);
        if index > 0 && self.exact_token_ranges[index - 1].1 >= start {
            self.exact_token_ranges[index - 1].1.checked_add(1)
        } else { Some(start) }
    }

    /// Whether at least one model ID intersects an inclusive range.
    #[doc(hidden)]
    pub fn intersects_exact_token_range(&self, start: u32, end: u32) -> bool {
        if start > end { return false; }
        let index = self.exact_token_ranges.partition_point(|&(_, upper)| upper < start);
        self.exact_token_ranges.get(index).is_some_and(|&(lower, _)| lower <= end)
    }

    /// Whether two vocabularies describe the same byte mapping and the same
    /// exact-only token domain.
    #[doc(hidden)]
    pub fn same_model_vocab(&self, other: &Vocab) -> bool {
        (Arc::ptr_eq(&self.entries, &other.entries) || self.entries == other.entries)
            && (Arc::ptr_eq(&self.exact_only_token_ids, &other.exact_only_token_ids)
                || self.exact_only_token_ids == other.exact_only_token_ids)
    }

    /// Exact-only token IDs in ascending order.
    #[doc(hidden)]
    pub fn exact_only_token_ids(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.exact_only_token_ids.iter().copied()
    }

    /// Highest byte-backed ID, or zero if none. Excludes exact-only controls; this is not an entry count.
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

    /// Exact byte spelling; returns `None` for unknown and exact-only IDs.
    pub fn get(&self, token_id: u32) -> Option<&[u8]> {
        self.entries.get(&token_id).map(Vec::as_slice)
    }

    /// Iterate over byte-backed token IDs and their exact byte sequences in ID
    /// order. Exact-only IDs are omitted.
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

    #[test]
    fn exact_tokens_validate_membership_and_duplicates() {
        let vocab = Vocab::new(vec![(2, b"a".to_vec()), (7, b"b".to_vec())]);
        assert_eq!(Token::new(2).id(), 2);
        assert_eq!(vocab.tokens([7, 2]).unwrap().ids(), &[2, 7]);
        assert_eq!(Token::new(3).id(), 3); // target membership is deferred
        let empty = vocab.tokens([]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert!(empty.ids().is_empty());
        assert!(empty.targets(&vocab));
        assert!(vocab.tokens([2, 2]).is_err());
    }

    #[test]
    fn exact_only_ids_are_bindable_but_not_byte_entries() {
        let vocab = Vocab::new_with_exact_token_ids(vec![(0, b"a".to_vec())], [77, 78]);
        assert_eq!(Token::new(77).id(), 77);
        assert_eq!(vocab.tokens([78, 77]).unwrap().ids(), &[77, 78]);
        assert_eq!(vocab.get(77), None);
        assert!(vocab.contains_exact_token_id(78));
        assert_eq!(vocab.max_token_id(), 0);

        let same = Vocab::new_with_exact_token_ids(vec![(0, b"a".to_vec())], [78, 77]);
        let missing_domain = Vocab::new(vec![(0, b"a".to_vec())]);
        assert!(vocab.same_model_vocab(&same));
        assert!(!vocab.same_model_vocab(&missing_domain));
        assert!(same.contains_exact_token_id(Token::new(77).id()));
        assert!(!missing_domain.contains_exact_token_id(Token::new(77).id()));
    }
}
#[cfg(test)]
mod compact_token_range_tests {
    use super::*;

    #[test]
    fn full_domain_ranges_do_not_materialize_ids_until_requested() {
        let vocab = Vocab::new_with_exact_token_ids(Vec::new(), 0..131_072);
        let tokens = vocab.token_ranges([0..=131_071]).unwrap();
        let shared = tokens.clone();
        assert_eq!(tokens.range_pairs(), &[(0, 131_071)]);
        assert_eq!(tokens.len(), 131_072);
        assert!(tokens.ids.get().is_none());
        assert!(shared.ids.get().is_none());
        assert!(Arc::ptr_eq(&tokens.ranges, &shared.ranges));
        assert!(tokens.targets(&vocab));
        assert_eq!(shared.ids().len(), 131_072);
        assert!(std::ptr::eq(tokens.ids(), shared.ids()));
    }

    #[test]
    fn coverage_validation_handles_gaps_and_u32_endpoint_without_expansion() {
        let vocab = Vocab::new_with_exact_token_ids(Vec::new(), [0, 1, 3, 4, u32::MAX]);
        assert!(vocab.token_ranges([0..=4]).is_err());
        let selected = vocab.token_ranges([3..=4, u32::MAX..=u32::MAX, 0..=1]).unwrap();
        assert_eq!(selected.len(), 5);
        assert_eq!(selected.range_pairs(), &[(0, 1), (3, 4), (u32::MAX, u32::MAX)]);
        assert!(selected.ids.get().is_none());
        assert_eq!(vocab.next_available_exact_token_id(0), Some(2));
        assert_eq!(vocab.next_available_exact_token_id(3), Some(5));
        assert_eq!(vocab.next_available_exact_token_id(u32::MAX), None);
        assert!(!vocab.intersects_exact_token_range(2, 2));
        assert!(vocab.intersects_exact_token_range(2, 3));
    }
}
