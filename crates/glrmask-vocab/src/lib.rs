#![deny(warnings)]
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::{Bound, RangeBounds, RangeInclusive};
use std::sync::{Arc, Mutex, OnceLock};

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
    /// Creates a set from `ids`.
    pub fn new(ids: impl IntoIterator<Item = u32>) -> Self {
        Self::from_pairs(ids.into_iter().map(|id| (id, id)))
    }

    /// Creates a set from ranges of IDs, such as `0..10` or `0..=9`.
    pub fn from_ranges<R: RangeBounds<u32>>(
        ranges: impl IntoIterator<Item = R>,
    ) -> Self {
        let mut pairs = Vec::new();
        for range in ranges {
            let start = match range.start_bound() {
                Bound::Included(&start) => start,
                Bound::Excluded(&start) => match start.checked_add(1) {
                    Some(start) => start,
                    None => continue,
                },
                Bound::Unbounded => 0,
            };
            let end = match range.end_bound() {
                Bound::Included(&end) => end,
                Bound::Excluded(&end) => match end.checked_sub(1) {
                    Some(end) => end,
                    None => continue,
                },
                Bound::Unbounded => u32::MAX,
            };
            if start > end {
                continue;
            }
            pairs.push((start, end));
        }
        Self::from_pairs(pairs)
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

impl From<u32> for TokenSet {
    fn from(id: u32) -> Self {
        Self::from_pairs([(id, id)])
    }
}

use range_set_blaze::RangeSetBlaze;

/// A model's tokens and their bytes.
pub struct Vocab {
    entries: Arc<BTreeMap<u32, Vec<u8>>>,
    /// Model token IDs which are valid for token-set bindings but have no
    /// byte spelling in the grammar lexer (for example control/tool tokens).
    ///
    /// These IDs are deliberately absent from the byte entries: they must
    /// never become zero-byte alternatives in the vocabulary trie.
    special_token_ids: Arc<BTreeSet<u32>>,
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
            .field("special_token_ids", &self.special_token_ids)
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
            special_token_ids: Arc::clone(&self.special_token_ids),
            compiler_cache: Arc::clone(&self.compiler_cache),
            event_token_ranges: Arc::clone(&self.event_token_ranges),
            byte_token_ranges: Arc::clone(&self.byte_token_ranges),
            max_token_byte_len,
        }
    }
}

impl Vocab {
    /// Creates a vocabulary from `(id, bytes)` pairs.
    pub fn new(entries: Vec<(u32, Vec<u8>)>) -> Self {
        let entries: Arc<BTreeMap<u32, Vec<u8>>> = Arc::new(entries.into_iter().collect());
        let max_token_byte_len = OnceLock::new();
        let _ = max_token_byte_len.set(entries.values().map(Vec::len).max().unwrap_or(0));
        let mut vocab = Self {
            event_token_ranges: Arc::from(Vec::new()),
            byte_token_ranges: Arc::from(Vec::new()),
            entries,
            special_token_ids: Arc::new(BTreeSet::new()),
            compiler_cache: Arc::new(VocabCompilerCache::default()),
            max_token_byte_len,
        };
        vocab.rebuild_event_ranges();
        vocab
    }

    /// Adds tokens that have no bytes, such as end-of-sequence tokens. They can
    /// be used in token sets but never match text.
    pub fn with_special_tokens(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        {
            let entries = Arc::make_mut(&mut self.entries);
            let special = Arc::make_mut(&mut self.special_token_ids);
            for id in ids {
                if !entries.contains_key(&id) {
                    special.insert(id);
                }
            }
        }
        self.rebuild_event_ranges();
        self
    }

    fn rebuild_event_ranges(&mut self) {
        self.event_token_ranges = RangeSetBlaze::from_iter(
            self.entries.keys().chain(self.special_token_ids.iter()).copied(),
        )
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
        self.entries.contains_key(&token_id) || self.special_token_ids.contains(&token_id)
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
        (Arc::ptr_eq(&self.entries, &other.entries) || self.entries == other.entries)
            && (Arc::ptr_eq(&self.special_token_ids, &other.special_token_ids)
                || self.special_token_ids == other.special_token_ids)
    }

    /// Special token IDs in ascending order.
    #[doc(hidden)]
    pub fn special_token_ids(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.special_token_ids.iter().copied()
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

    /// The bytes of a token, or `None` for special and unknown IDs.
    pub fn get(&self, token_id: u32) -> Option<&[u8]> {
        self.entries.get(&token_id).map(Vec::as_slice)
    }

    /// Each token ID and its bytes, in ID order. Special tokens are skipped.
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
    fn token_sets_are_normalized_and_validate_membership() {
        let vocab = Vocab::new(vec![(2, b"a".to_vec()), (7, b"b".to_vec())]);
        assert_eq!(TokenSet::from(2u32).range_pairs(), &[(2, 2)]);
        assert_eq!(TokenSet::new([7, 2, 7]).range_pairs(), &[(2, 2), (7, 7)]);
        assert_eq!(TokenSet::new([]).range_pairs(), &[]);
        assert_eq!(TokenSet::from_ranges(std::iter::once(3..=5)).range_pairs(), &[(3, 5)]);
        assert_eq!(TokenSet::from_ranges(std::iter::once(3..6)).range_pairs(), &[(3, 5)]);
        assert_eq!(
            TokenSet::from_ranges([0..10, 5..13, 20..30]).range_pairs(),
            &[(0, 12), (20, 29)],
        );
        assert_eq!(
            TokenSet::from_ranges([0..=11, 10..=12, 20..=29]).range_pairs(),
            &[(0, 12), (20, 29)],
        );
        // Overlapping and adjacent input round-trips to the same merged set.
        assert_eq!(
            TokenSet::from_ranges([0..10, 5..20, 20..30]),
            TokenSet::from_ranges(std::iter::once(0..30)),
        );
        assert_eq!(format!("{:?}", TokenSet::from_ranges([0..10, 20..30])),
            "TokenSet[0..=9, 20..=29]");
        assert_eq!(TokenSet::from_ranges(std::iter::once(std::ops::RangeInclusive::new(5, 3))).range_pairs(), &[]);
        assert_eq!(TokenSet::from_ranges(std::iter::once(5..5)).range_pairs(), &[]);
        // An excluded start bound starts after that ID, not at it.
        assert_eq!(
            TokenSet::from_ranges([(Bound::Excluded(1), Bound::Included(3))]),
            TokenSet::from_ranges(std::iter::once(2..=3)),
        );
        assert_eq!(
            TokenSet::from_ranges([(Bound::Excluded(u32::MAX), Bound::Included(u32::MAX))])
                .range_pairs(),
            &[],
        );
        assert_eq!(vocab.first_unknown_token_id(&TokenSet::new([2, 7])), None);
        assert_eq!(vocab.first_unknown_token_id(&TokenSet::new([2, 9])), Some(9));
    }

    #[test]
    fn special_ids_are_bindable_but_not_byte_entries() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]).with_special_tokens([77, 78]);
        assert_eq!(TokenSet::new([78, 77]).range_pairs(), &[(77, 78)]);
        assert_eq!(vocab.get(77), None);
        assert!(vocab.contains_token_id(78));
        assert_eq!(vocab.max_token_id(), 0);

        let same = Vocab::new(vec![(0, b"a".to_vec())]).with_special_tokens([78, 77]);
        let missing_domain = Vocab::new(vec![(0, b"a".to_vec())]);
        assert!(vocab.same_model_vocab(&same));
        assert!(!vocab.same_model_vocab(&missing_domain));
        assert!(same.contains_token_id(77));
        assert!(!missing_domain.contains_token_id(77));
    }
}
#[cfg(test)]
mod compact_token_range_tests {
    use super::*;

    #[test]
    fn full_domain_ranges_do_not_materialize_ids_until_requested() {
        let vocab = Vocab::new(Vec::new()).with_special_tokens(0..131_072);
        let tokens = TokenSet::from_ranges(std::iter::once(0..131_072));
        let shared = tokens.clone();
        assert_eq!(tokens.range_pairs(), &[(0, 131_071)]);
        assert!(Arc::ptr_eq(&tokens.ranges, &shared.ranges));
        assert_eq!(vocab.first_unknown_token_id(&tokens), None);
    }

    #[test]
    fn coverage_validation_handles_gaps_and_u32_endpoint_without_expansion() {
        let vocab = Vocab::new(Vec::new()).with_special_tokens([0, 1, 3, 4, u32::MAX]);
        assert_eq!(vocab.first_unknown_token_id(&TokenSet::from_ranges(std::iter::once(0..=4))), Some(2));
        let selected = TokenSet::new([3, 4, u32::MAX, 0, 1]);
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
