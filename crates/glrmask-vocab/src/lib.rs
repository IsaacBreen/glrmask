#![deny(warnings)]
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::{Range, RangeInclusive};
use std::sync::{Arc, Mutex, OnceLock};

mod error;
pub use error::{Error, ErrorKind};

const MAX_TOKEN_ID: u32 = (1 << 24) - 1;

/// A set of token IDs.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TokenSet {
    ranges: Arc<[(u32, u32)]>,
}

impl fmt::Debug for TokenSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TokenSet[")?;
        for (index, &(start, end)) in self.ranges.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{start}..={end}")?;
        }
        f.write_str("]")
    }
}

impl TokenSet {
    pub fn new(ids: impl IntoIterator<Item = u32>) -> Self {
        Self::from_pairs(ids.into_iter().map(|id| {
            assert!(id <= MAX_TOKEN_ID, "token ID {id} is outside the supported range");
            (id, id)
        }))
    }

    /// Builds a set from half-open ranges. Empty ranges contribute no IDs.
    pub fn from_ranges(ranges: impl IntoIterator<Item = Range<u32>>) -> Self {
        Self::from_pairs(ranges.into_iter().filter(|r| !r.is_empty()).map(|r| {
            let last = r.end - 1;
            assert!(last <= MAX_TOKEN_ID, "token ID {last} is outside the supported range");
            (r.start, last)
        }))
    }

    fn from_pairs(ranges: impl IntoIterator<Item = (u32, u32)>) -> Self {
        Self {
            ranges: RangeSetBlaze::from_iter(
                ranges.into_iter().map(|(start, end)| start..=end),
            )
            .ranges()
            .map(|range| (*range.start(), *range.end()))
            .collect::<Vec<_>>()
            .into(),
        }
    }

    /// The normalized, disjoint inclusive ranges. Internal representation.
    #[doc(hidden)]
    pub fn range_pairs(&self) -> &[(u32, u32)] {
        &self.ranges
    }
}


use range_set_blaze::RangeSetBlaze;

/// Each token ID and its bytes.
pub struct Vocab {
    entries: Arc<BTreeMap<u32, Vec<u8>>>,
    compiler_cache: Arc<VocabCompilerCache>,
    event_token_ranges: Arc<[(u32, u32)]>,
    byte_token_ranges: Arc<[(u32, u32)]>,
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
            event_token_ranges: Arc::clone(&self.event_token_ranges),
            byte_token_ranges: Arc::clone(&self.byte_token_ranges),
            max_token_byte_len,
        }
    }
}

impl Vocab {
    /// Fails on duplicate IDs.
    pub fn new(entries: impl IntoIterator<Item = (u32, Vec<u8>)>) -> Result<Self, Error> {
        let mut ids = BTreeSet::new();
        let mut validated = Vec::new();
        for (id, bytes) in entries {
            if id > MAX_TOKEN_ID {
                return Err(error::internal::new_error(ErrorKind::InvalidInput, format!("token ID {id} is outside the supported range")));
            }
            if !ids.insert(id) {
                return Err(error::internal::new_error(ErrorKind::InvalidInput, format!("duplicate vocabulary ID {id}")));
            }
            validated.push((id, bytes));
        }
        Ok(Self::from_entries_unchecked(validated))
    }

    #[doc(hidden)]
    pub fn from_entries_unchecked(entries: Vec<(u32, Vec<u8>)>) -> Self {
        let mut ids = BTreeSet::new();
        for (id, _) in &entries {
            if !ids.insert(*id) {
                panic!("duplicate vocabulary ID {id}");
            }
        }
        let entries: Arc<BTreeMap<u32, Vec<u8>>> = Arc::new(entries.into_iter().collect());
        let max_token_byte_len = OnceLock::new();
        let _ = max_token_byte_len.set(entries.values().map(Vec::len).max().unwrap_or(0));
        let mut vocab = Self {
            event_token_ranges: Arc::from(Vec::new()),
            byte_token_ranges: Arc::from(Vec::new()),
            entries,
            compiler_cache: Arc::new(VocabCompilerCache::default()),
            max_token_byte_len,
        };
        vocab.rebuild_event_ranges();
        vocab
    }

    fn rebuild_event_ranges(&mut self) {
        self.event_token_ranges = RangeSetBlaze::from_iter(self.entries.keys().copied())
            .ranges()
            .map(|range| (*range.start(), *range.end()))
            .collect::<Vec<_>>()
            .into();
        self.byte_token_ranges = RangeSetBlaze::from_iter(
            self.entries
                .iter()
                .filter_map(|(&id, bytes)| (!bytes.is_empty()).then_some(id)),
        )
        .ranges()
        .map(|range| (*range.start(), *range.end()))
        .collect::<Vec<_>>()
        .into();
    }

    /// The first ID in `set` that this vocabulary does not define.
    #[doc(hidden)]
    pub fn first_unknown_token_id(&self, set: &TokenSet) -> Option<u32> {
        set.range_pairs()
            .iter()
            .find_map(|&(start, end)| (start..=end).find(|&id| !self.contains_token_id(id)))
    }

    /// All IDs that may be consumed as grammar events, including special IDs
    /// without ordinary byte spellings. This view does not include root-only
    /// controls merely because they are present in a caller's termination
    /// policy.
    #[doc(hidden)]
    pub fn event_token_ranges(&self) -> impl ExactSizeIterator<Item = RangeInclusive<u32>> + '_ {
        self.event_token_ranges.iter().map(|&(start, end)| start..=end)
    }

    /// The IDs whose tokens have bytes, as inclusive ranges.
    #[doc(hidden)]
    pub fn byte_token_ranges(&self) -> impl ExactSizeIterator<Item = RangeInclusive<u32>> + '_ {
        self.byte_token_ranges.iter().map(|&(start, end)| start..=end)
    }


    /// Whether every ID in the inclusive range `start..=end` is defined,
    /// including special IDs.
    #[doc(hidden)]
    pub fn contains_token_range(&self, start: u32, end: u32) -> bool {
        if start > end {
            return false;
        }
        let index = self.event_token_ranges.partition_point(|&(lower, _)| lower <= start);
        index > 0 && self.event_token_ranges[index - 1].1 >= end
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

    /// The number of tokens with bytes; special tokens are not counted.
    #[doc(hidden)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no token has bytes.
    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether this model token ID is known either as a byte-backed token or
    /// as a special token.
    #[doc(hidden)]
    pub fn contains_token_id(&self, token_id: u32) -> bool {
        self.entries.contains_key(&token_id)
    }

    /// Highest defined model ID, including special IDs.
    #[doc(hidden)]
    pub fn max_event_token_id(&self) -> Option<u32> {
        self.event_token_ranges.last().map(|&(_, end)| end)
    }

    /// Find a free ID at or above `start` without scanning each occupied ID.
    #[doc(hidden)]
    pub fn next_available_token_id(&self, start: u32) -> Option<u32> {
        let index = self.event_token_ranges.partition_point(|&(lower, _)| lower <= start);
        if index > 0 && self.event_token_ranges[index - 1].1 >= start {
            self.event_token_ranges[index - 1].1.checked_add(1)
        } else {
            Some(start)
        }
    }

    /// Whether at least one model ID intersects an inclusive range.
    #[doc(hidden)]
    pub fn intersects_token_range(&self, start: u32, end: u32) -> bool {
        if start > end {
            return false;
        }
        let index = self.event_token_ranges.partition_point(|&(_, upper)| upper < start);
        self.event_token_ranges.get(index).is_some_and(|&(lower, _)| lower <= end)
    }

    /// Whether two vocabularies describe the same byte mapping and the same
    /// special-token domain.
    #[doc(hidden)]
    pub fn same_model_vocab(&self, other: &Vocab) -> bool {
        Arc::ptr_eq(&self.entries, &other.entries) || self.entries == other.entries
    }

    /// The highest byte-backed token ID, or zero if none; excludes special IDs.
    #[doc(hidden)]
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

    #[doc(hidden)]
    pub fn get(&self, token_id: u32) -> Option<&[u8]> {
        self.entries.get(&token_id).map(Vec::as_slice)
    }

    #[doc(hidden)]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u32, &[u8])> {
        self.entries
            .iter()
            .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
    }

    #[doc(hidden)]
    pub fn entries_map(&self) -> &BTreeMap<u32, Vec<u8>> {
        &self.entries
    }

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
    pub const MAX_TOKEN_ID: u32 = super::MAX_TOKEN_ID;
    pub use super::derived_artifact::VocabDerivedArtifact;
    pub use super::error::SourceSpan;
    pub use super::error::internal::{error_message, error_source_span, new_error, new_source_span, source_span_line_column, with_source_span};
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"ab".to_vec()), (1, b"bc".to_vec())]);
        assert_eq!(vocab.compiler_cache.artifacts.lock().unwrap().len(), 0);
        let _ = vocab.relevant_bytes();
        assert_eq!(vocab.compiler_cache.artifacts.lock().unwrap().len(), 1);

        let cloned = vocab.clone();
        assert!(Arc::ptr_eq(&vocab.compiler_cache, &cloned.compiler_cache));
        assert_eq!(cloned.compiler_cache.artifacts.lock().unwrap().len(), 1);
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let error = Vocab::new([(0, b"a".to_vec()), (0, b"b".to_vec())]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn token_sets_are_normalized_and_validate_membership() {
        let vocab = Vocab::from_entries_unchecked(vec![(2, b"a".to_vec()), (7, b"b".to_vec())]);
        assert_eq!(TokenSet::new([2u32]).range_pairs(), &[(2, 2)]);
        assert_eq!(TokenSet::new([7, 2, 7]).range_pairs(), &[(2, 2), (7, 7)]);
        assert_eq!(TokenSet::new([]).range_pairs(), &[]);
        assert_eq!(TokenSet::from_ranges(std::iter::once(3..6)).range_pairs(), &[(3, 5)]);
        assert_eq!(
            TokenSet::from_ranges([0..10, 5..13, 20..30]).range_pairs(),
            &[(0, 12), (20, 29)],
        );
        assert_eq!(
            TokenSet::from_ranges([0..12, 10..13, 20..30]).range_pairs(),
            &[(0, 12), (20, 29)],
        );
        // Overlapping and adjacent input round-trips to the same merged set.
        assert_eq!(
            TokenSet::from_ranges([0..10, 5..20, 20..30]),
            TokenSet::from_ranges(std::iter::once(0..30)),
        );
        assert_eq!(format!("{:?}", TokenSet::from_ranges([0..10, 20..30])),
            "TokenSet[0..=9, 20..=29]");
        assert_eq!(TokenSet::from_ranges(std::iter::once(std::ops::Range { start: 5, end: 3 })).range_pairs(), &[]);
        assert_eq!(TokenSet::from_ranges(std::iter::once(5..5)).range_pairs(), &[]);
        assert_eq!(vocab.first_unknown_token_id(&TokenSet::new([2, 7])), None);
        assert_eq!(vocab.first_unknown_token_id(&TokenSet::new([2, 9])), Some(9));
    }
}
#[cfg(test)]
mod compact_token_range_tests {
    use super::*;

    #[test]
    fn full_domain_ranges_do_not_materialize_ids_until_requested() {
        let vocab = Vocab::from_entries_unchecked((0..131_072).map(|i| (i, Vec::new())).collect());
        let tokens = TokenSet::from_ranges(std::iter::once(0..131_072));
        let shared = tokens.clone();
        assert_eq!(tokens.range_pairs(), &[(0, 131_071)]);
        assert!(Arc::ptr_eq(&tokens.ranges, &shared.ranges));
        assert_eq!(vocab.first_unknown_token_id(&tokens), None);
    }

    #[test]
    fn coverage_validation_handles_gaps_and_u32_endpoint_without_expansion() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, Vec::new()), (1, Vec::new()), (3, Vec::new()), (4, Vec::new()), (u32::MAX, Vec::new())]);
        assert_eq!(vocab.first_unknown_token_id(&TokenSet::from_ranges(std::iter::once(0..5))), Some(2));
        let selected = TokenSet::from_pairs([(3, 4), (u32::MAX, u32::MAX), (0, 1)]);
        assert_eq!(selected.range_pairs(), &[(0, 1), (3, 4), (u32::MAX, u32::MAX)]);
        assert_eq!(vocab.next_available_token_id(0), Some(2));
        assert_eq!(vocab.next_available_token_id(3), Some(5));
        assert_eq!(vocab.next_available_token_id(u32::MAX), None);
        assert!(!vocab.intersects_token_range(2, 2));
        assert!(vocab.intersects_token_range(2, 3));
        assert!(vocab.contains_token_range(0, 1));
        assert!(!vocab.contains_token_range(0, 2));
        assert!(!vocab.contains_token_range(4, 3));
    }
}
