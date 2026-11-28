#![deny(warnings)]
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::{Bound, RangeBounds, RangeInclusive};
use std::sync::{Arc, Mutex, OnceLock};

mod error;
pub use error::{Error, ErrorKind, SourceSpan};

/// The largest token ID GLRMask accepts: 16,777,215 (2^24 - 1).
pub const MAX_TOKEN_ID: u32 = (1 << 24) - 1;

/// A set of token IDs. IDs above [`MAX_TOKEN_ID`] are rejected when the set is used.
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
        Self::from_pairs(ids.into_iter().map(|id| (id, id)))
    }

    /// A range with no end, such as `10..`, ends at [`MAX_TOKEN_ID`].
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
                Bound::Unbounded => MAX_TOKEN_ID,
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


use range_set_blaze::RangeSetBlaze;

/// Immutable bytes and original-ID coordinate. A subset owns only its
/// sorted ID/span table; all views share the original byte allocation.
/// Dense IDs index directly, sparse IDs use the same sorted ID table.
#[doc(hidden)]
#[derive(Default)]
pub struct VocabEntries {
    ids: Box<[u32]>,
    spans: Box<[(usize, usize)]>,
    empty_ids: Box<[u32]>,
    data: Arc<Vec<u8>>,
    dense: bool,
}

impl VocabEntries {
    fn from_sorted<B: AsRef<[u8]>>(entries: impl IntoIterator<Item=(u32, B)>, count: usize, byte_count: usize) -> Self {
        let mut ids = Vec::with_capacity(count);
        let mut spans = Vec::with_capacity(count);
        let mut data = Vec::with_capacity(byte_count);
        for (id, bytes) in entries {
            assert!(ids.last().is_none_or(|&previous| previous < id), "duplicate or unordered vocabulary ID {id}");
            ids.push(id);
            let start = data.len();
            data.extend_from_slice(bytes.as_ref());
            spans.push((start, data.len()));
        }
        Self::from_parts(ids, spans, Arc::new(data))
    }

    fn from_parts(ids: Vec<u32>, spans: Vec<(usize, usize)>, data: Arc<Vec<u8>>) -> Self {
        let dense = ids.last().is_none_or(|&id| u64::from(id) + 1 == ids.len() as u64);
        let empty_ids = ids.iter().zip(&spans).filter_map(|(&id, &(start, end))|
            (start == end).then_some(id)).collect();
        Self { ids: ids.into_boxed_slice(), spans: spans.into_boxed_slice(), empty_ids, data, dense }
    }

    pub fn empty_byte_token_ids(&self) -> &[u32] { &self.empty_ids }

    pub fn len(&self) -> usize { self.ids.len() }
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }
    pub fn keys(&self) -> std::slice::Iter<'_, u32> { self.ids.iter() }

    #[inline]
    pub fn ordinal(&self, id: u32) -> Option<usize> {
        if self.dense {
            ((id as usize) < self.len()).then_some(id as usize)
        } else {
            self.ids.binary_search(&id).ok()
        }
    }

    #[inline]
    pub fn bytes_at(&self, ordinal: usize) -> &[u8] {
        let (start, end) = self.spans[ordinal];
        &self.data[start..end]
    }

    #[inline]
    pub fn get(&self, id: &u32) -> Option<&[u8]> {
        self.ordinal(*id).map(|ordinal| self.bytes_at(ordinal))
    }

    pub fn entry_at(&self, ordinal: usize) -> (u32, &[u8]) {
        (self.ids[ordinal], self.bytes_at(ordinal))
    }

    pub fn contains_key(&self, id: &u32) -> bool { self.ordinal(*id).is_some() }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item=(&u32, &[u8])> + ExactSizeIterator {
        self.ids.iter().enumerate().map(|(ordinal, id)| (id, self.bytes_at(ordinal)))
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item=&[u8]> + ExactSizeIterator {
        (0..self.len()).map(|ordinal| self.bytes_at(ordinal))
    }

    pub fn last_key_value(&self) -> Option<(&u32, &[u8])> { self.iter().next_back() }

    fn shared_word(&self, ordinal: usize) -> TokenBytes {
        TokenBytes { data: Arc::clone(&self.data), span: self.spans[ordinal] }
    }

    fn subset(&self, ids: impl IntoIterator<Item=u32>) -> Self {
        let mut ordinals = ids.into_iter().filter_map(|id| self.ordinal(id)).collect::<Vec<_>>();
        ordinals.sort_unstable();
        self.subset_ordinals(&ordinals)
    }

    fn subset_ordinals(&self, ordinals: &[usize]) -> Self {
        assert!(ordinals.windows(2).all(|pair| pair[0] < pair[1]), "unordered or duplicate vocabulary subset ordinal");
        let ids = ordinals.iter().map(|&i| self.ids[i]).collect();
        let spans = ordinals.iter().map(|&i| self.spans[i]).collect();
        Self::from_parts(ids, spans, Arc::clone(&self.data))
    }
}

impl PartialEq for VocabEntries {
    fn eq(&self, other: &Self) -> bool {
        self.ids == other.ids && (
            (Arc::ptr_eq(&self.data, &other.data) && self.spans == other.spans)
            || self.values().eq(other.values())
        )
    }
}
impl Eq for VocabEntries {}
impl fmt::Debug for VocabEntries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// A byte spelling backed by a shared immutable slab. Equality and order
/// depend on the spelling, not the identity or location of its allocation.
#[doc(hidden)]
#[derive(Clone)]
pub struct TokenBytes {
    data: Arc<Vec<u8>>,
    span: (usize, usize),
}

impl TokenBytes {
    /// Pack independent spellings into one allocation, for a standalone
    /// lexical tree not backed by a model vocabulary.
    pub fn pack<B: AsRef<[u8]>>(words: impl IntoIterator<Item=B>) -> Vec<Self> {
        let mut data = Vec::new();
        let mut spans = Vec::new();
        for word in words {
            let start = data.len();
            data.extend_from_slice(word.as_ref());
            spans.push((start, data.len()));
        }
        let data = Arc::new(data);
        spans.into_iter().map(|span| Self { data: Arc::clone(&data), span }).collect()
    }
}
impl AsRef<[u8]> for TokenBytes {
    fn as_ref(&self) -> &[u8] { &self.data[self.span.0..self.span.1] }
}
impl std::ops::Deref for TokenBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] { self.as_ref() }
}
impl PartialEq for TokenBytes {
    fn eq(&self, other: &Self) -> bool { self.as_ref() == other.as_ref() }
}
impl Eq for TokenBytes {}
impl PartialOrd for TokenBytes {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for TokenBytes {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.as_ref().cmp(other.as_ref()) }
}
impl fmt::Debug for TokenBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Debug::fmt(self.as_ref(), f) }
}

/// Each token ID and its bytes.
#[derive(Clone)]
pub struct Vocab(Arc<VocabData>);

struct VocabData {
    entries: VocabEntries,
    event_token_ranges: Box<[(u32, u32)]>,
    byte_token_ranges: Box<[(u32, u32)]>,
    max_token_byte_len: usize,
    byte_facts: OnceLock<Arc<VocabByteFacts>>,
    lexical: OnceLock<Arc<LexicalVocabulary>>,
    artifacts: Mutex<BTreeMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

#[derive(Debug)]
pub struct VocabByteFacts {
    bytes: Arc<[u8]>,
    words: [u64; 4],
    follow_words: Box<[[u64; 4]; 256]>,
    total_bytes: usize,
}

impl VocabByteFacts {
    pub fn byte_words(&self) -> [u64; 4] {
        self.words
    }

    pub fn follow_words(&self, byte: u8) -> [u64; 4] {
        self.follow_words[byte as usize]
    }
}

/// The immutable exact-spelling coordinate shared by vocabulary consumers.
/// IDs in each alias group are ordered by original ID; groups are ordered by
/// byte spelling. Grammar-dependent equivalence classes never enter this cache.
#[derive(Debug)]
pub struct LexicalVocabulary {
    pub original_slot_count: usize,
    pub ordered_to_originals: Arc<Vec<Vec<u32>>>,
    pub ordered_token_bytes: Arc<Vec<TokenBytes>>,
    entry_indices: Box<[u32]>,
    original_to_ordered: OnceLock<Arc<[u32]>>,
    original_order: OnceLock<Arc<[(u32, u32)]>>,
    trie: OnceLock<Arc<vocab_prefix_tree::VocabPrefixTree>>,
    split_offsets: OnceLock<Box<[usize]>>,
    suffix_index: OnceLock<Arc<suffix_index::VocabularySuffixIndex>>,
}

impl LexicalVocabulary {
    pub fn build(entries: &VocabEntries) -> Self {
        let mut sorted: Vec<u32> = (0..u32::try_from(entries.len()).expect("vocabulary entry count")).collect();
        sorted.sort_unstable_by(|&a, &b| entries.bytes_at(a as usize).cmp(entries.bytes_at(b as usize)).then(a.cmp(&b)));
        Self::from_entry_order(entries, sorted)
    }

    fn from_entry_order(entries: &VocabEntries, sorted: Vec<u32>) -> Self {
        let mut words: Vec<TokenBytes> = Vec::new();
        let mut aliases: Vec<Vec<u32>> = Vec::new();
        for &ordinal in &sorted {
            let ordinal = ordinal as usize;
            let id = entries.ids[ordinal];
            if words.last().is_some_and(|last| last.as_ref() == entries.bytes_at(ordinal)) {
                aliases.last_mut().expect("existing spelling").push(id);
            } else {
                words.push(entries.shared_word(ordinal));
                aliases.push(vec![id]);
            }
        }
        Self {
            original_slot_count: entries.last_key_value().map_or(0, |(&id, _)| id as usize + 1),
            ordered_to_originals: Arc::new(aliases),
            ordered_token_bytes: Arc::new(words),
            entry_indices: sorted.into_boxed_slice(),
            original_to_ordered: OnceLock::new(),
            original_order: OnceLock::new(),
            trie: OnceLock::new(),
            split_offsets: OnceLock::new(),
            suffix_index: OnceLock::new(),
        }
    }

    /// Original entry ordinals, ordered by exact spelling and then original ID.
    pub fn entry_indices(&self) -> &[u32] { &self.entry_indices }

    pub fn original_to_ordered(&self) -> &[u32] {
        self.original_to_ordered.get_or_init(|| {
            let mut map = vec![u32::MAX; self.original_slot_count];
            for (ordered, ids) in self.ordered_to_originals.iter().enumerate() {
                for &id in ids { map[id as usize] = ordered as u32; }
            }
            map.into()
        })
    }

    pub fn trie(&self) -> Arc<vocab_prefix_tree::VocabPrefixTree> {
        Arc::clone(self.trie.get_or_init(|| Arc::new(
            vocab_prefix_tree::VocabPrefixTree::from_ordered_tokens(Arc::clone(&self.ordered_token_bytes))
        )))
    }

    /// Reuse only an already prepared trie; a query must not force cold work.
    pub fn cached_trie(&self) -> Option<Arc<vocab_prefix_tree::VocabPrefixTree>> {
        self.trie.get().cloned()
    }

    pub fn len(&self) -> usize { self.ordered_token_bytes.len() }
    pub fn is_empty(&self) -> bool { self.ordered_token_bytes.is_empty() }
    pub fn bytes(&self, index: usize) -> &[u8] { &self.ordered_token_bytes[index] }
    pub fn originals(&self, index: usize) -> &[u32] { &self.ordered_to_originals[index] }

    /// End of the first byte child in a sorted interval with a common prefix.
    pub fn child_end(&self, start: usize, end: usize, offset: usize) -> usize {
        let byte = self.bytes(start)[offset];
        start + self.ordered_token_bytes[start..end].partition_point(|b| b[offset] == byte)
    }

    /// All positive proper-prefix split positions in lexical spelling order.
    pub fn split_offsets(&self) -> &[usize] {
        self.split_offsets.get_or_init(|| {
            let mut offsets = Vec::with_capacity(self.ordered_token_bytes.len() + 1);
            offsets.push(0usize);
            for bytes in self.ordered_token_bytes.iter() {
                offsets.push(offsets.last().copied().unwrap()
                    .checked_add(bytes.len().saturating_sub(1)).expect("vocabulary split count"));
            }
            offsets.into_boxed_slice()
        })
    }

    pub fn suffix_index(&self) -> Arc<suffix_index::VocabularySuffixIndex> {
        Arc::clone(self.suffix_index.get_or_init(|| Arc::new(
            suffix_index::VocabularySuffixIndex::build(&self.ordered_token_bytes)
        )))
    }

    pub fn original_order(&self) -> Arc<[(u32, u32)]> {
        Arc::clone(self.original_order.get_or_init(|| {
            let mut pairs = vec![(0, 0); self.entry_indices.len()];
            let entries = self.ordered_to_originals.iter().enumerate().flat_map(|(ordered, ids)|
                ids.iter().map(move |&id| (id, ordered as u32)));
            for (&ordinal, entry) in self.entry_indices.iter().zip(entries) {
                pairs[ordinal as usize] = entry;
            }
            pairs.into()
        }))
    }
}

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

impl fmt::Debug for Vocab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vocab")
            .field("entries", &self.0.entries)
            .finish()
    }
}

impl Vocab {
    /// Fails with `ErrorKind::InvalidInput` on a duplicate ID or an ID above `MAX_TOKEN_ID`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use glrmask_vocab::Vocab;
    /// let vocab = Vocab::new([(0, b"hello".as_slice()), (1, b"world".as_slice())])?;
    /// # Ok::<(), glrmask_vocab::Error>(())
    /// ```
    pub fn new<B: Into<Vec<u8>>>(entries: impl IntoIterator<Item = (u32, B)>) -> Result<Self, Error> {
        let mut ids = BTreeSet::new();
        let mut validated = Vec::new();
        for (id, bytes) in entries {
            if id > MAX_TOKEN_ID {
                return Err(error::internal::new_error(ErrorKind::InvalidInput, format!("token ID {id} is larger than the maximum token ID, {MAX_TOKEN_ID}")));
            }
            if !ids.insert(id) {
                return Err(error::internal::new_error(ErrorKind::InvalidInput, format!("duplicate vocabulary ID {id}")));
            }
            validated.push((id, bytes.into()));
        }
        Ok(Self::from_entries_unchecked(validated))
    }

    #[doc(hidden)]
    pub fn from_entries_unchecked(mut entries: Vec<(u32, Vec<u8>)>) -> Self {
        entries.sort_unstable_by_key(|entry| entry.0);
        let byte_count = entries.iter().map(|entry| entry.1.len()).sum();
        let count = entries.len();
        Self::from_storage(VocabEntries::from_sorted(entries, count, byte_count))
    }

    /// Restrict the original ID domain without duplicating token bytes.
    #[doc(hidden)]
    pub fn subset(&self, ids: impl IntoIterator<Item=u32>) -> Self {
        Self::from_storage(self.0.entries.subset(ids))
    }

    /// Restrict a sorted original-entry coordinate and propagate an existing
    /// lexical order without sorting or copying the selected byte spellings.
    /// A cold parent remains cold.
    #[doc(hidden)]
    pub fn subset_ordinals_with_cached_lexical(&self, ordinals: &[usize]) -> Self {
        let child = Self::from_storage(self.0.entries.subset_ordinals(ordinals));
        if let Some(parent_order) = self.cached_lexical_vocabulary() {
            let parent_original_order = parent_order.original_order();
            let mut sorted: Vec<u32> = (0..u32::try_from(ordinals.len()).expect("vocabulary entry count")).collect();
            sorted.sort_unstable_by_key(|&i| {
                let parent_ordinal = ordinals[i as usize];
                let (id, spelling) = parent_original_order[parent_ordinal];
                debug_assert_eq!(id, self.0.entries.ids[parent_ordinal]);
                (spelling, i)
            });
            let _ = child.0.lexical.set(Arc::new(LexicalVocabulary::from_entry_order(child.entries_map(), sorted)));
        }
        child
    }

    fn from_storage(entries: VocabEntries) -> Self {
        fn ranges(ids: impl IntoIterator<Item=u32>) -> Box<[(u32, u32)]> {
            let mut runs: Vec<(u32, u32)> = Vec::new();
            for id in ids {
                if let Some(last) = runs.last_mut()
                    && last.1.checked_add(1) == Some(id)
                {
                    last.1 = id;
                } else {
                    runs.push((id, id));
                }
            }
            runs.into_boxed_slice()
        }
        let event_token_ranges = ranges(entries.keys().copied());
        let byte_token_ranges = ranges(entries.iter()
            .filter_map(|(&id, bytes)| (!bytes.is_empty()).then_some(id)));
        let max_token_byte_len = entries.values().map(<[u8]>::len).max().unwrap_or(0);
        Self(Arc::new(VocabData {
            entries, event_token_ranges, byte_token_ranges, max_token_byte_len,
            byte_facts: OnceLock::new(),
            lexical: OnceLock::new(),
            artifacts: Mutex::default(),
        }))
    }

    /// A known original ID range containing the given ID, if any.
    fn event_range_at(&self, id: u32) -> Option<(u32, u32)> {
        let index = self.0.event_token_ranges.partition_point(|&(start, _)| start <= id);
        let range = *self.0.event_token_ranges.get(index.checked_sub(1)?)?;
        (range.1 >= id).then_some(range)
    }

    /// The first ID in `set` that this vocabulary does not define.
    #[doc(hidden)]
    pub fn first_unknown_token_id(&self, set: &TokenSet) -> Option<u32> {
        set.range_pairs().iter().find_map(|&(start, end)| {
            match self.event_range_at(start) {
                None => Some(start),
                Some((_, covered_end)) if covered_end < end => covered_end.checked_add(1),
                Some(_) => None,
            }
        })
    }

    /// All IDs that may be consumed as grammar events, including special IDs
    /// without ordinary byte spellings. This view does not include root-only
    /// controls merely because they are present in a caller's termination
    /// policy.
    #[doc(hidden)]
    pub fn event_token_ranges(&self) -> impl ExactSizeIterator<Item = RangeInclusive<u32>> + '_ {
        self.0.event_token_ranges.iter().map(|&(start, end)| start..=end)
    }

    /// The IDs whose tokens have bytes, as inclusive ranges.
    #[doc(hidden)]
    pub fn byte_token_ranges(&self) -> impl ExactSizeIterator<Item = RangeInclusive<u32>> + '_ {
        self.0.byte_token_ranges.iter().map(|&(start, end)| start..=end)
    }


    /// Whether every ID in the inclusive range `start..=end` is defined,
    /// including special IDs.
    #[doc(hidden)]
    pub fn contains_token_range(&self, start: u32, end: u32) -> bool {
        if start > end {
            return false;
        }
        self.event_range_at(start).is_some_and(|(_, upper)| upper >= end)
    }

    /// Maximum byte length of any token in this vocabulary.
    #[doc(hidden)]
    pub fn max_token_byte_len(&self) -> usize {
        self.0.max_token_byte_len
    }

    /// Sorted byte alphabet observed anywhere in the vocabulary.
    #[doc(hidden)]
    pub fn relevant_bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.byte_facts().bytes)
    }

    /// Membership in that same alphabet, in physical byte coordinates.
    #[doc(hidden)]
    pub fn relevant_byte_mask(&self) -> [bool; 256] {
        let words = self.byte_facts().words;
        std::array::from_fn(|byte| words[byte / 64] & (1u64 << (byte % 64)) != 0)
    }

    /// Number of bytes across the original vocabulary entries, before
    /// observational deduplication.
    #[doc(hidden)]
    pub fn total_token_bytes(&self) -> usize {
        self.byte_facts().total_bytes
    }

    /// Pure observations of byte support and adjacent pairs in this vocabulary.
    #[doc(hidden)]
    pub fn byte_facts(&self) -> Arc<VocabByteFacts> {
        Arc::clone(self.0.byte_facts.get_or_init(|| {
            let mut words = [0u64; 4];
            let mut follow_words = Box::new([[0u64; 4]; 256]);
            let mut total_bytes = 0usize;
            for token in self.0.entries.values() {
                total_bytes += token.len();
                let mut previous = None::<u8>;
                for &byte in token {
                    let index = byte as usize / 64;
                    let bit = 1u64 << (byte % 64);
                    words[index] |= bit;
                    if let Some(previous) = previous {
                        follow_words[previous as usize][index] |= bit;
                    }
                    previous = Some(byte);
                }
            }
            let bytes = (0..=u8::MAX)
                .filter(|&byte| words[byte as usize / 64] & (1u64 << (byte % 64)) != 0)
                .collect::<Vec<_>>()
                .into();
            Arc::new(VocabByteFacts { bytes, words, follow_words, total_bytes })
        }))
    }

    /// The number of tokens with bytes; special tokens are not counted.
    #[doc(hidden)]
    pub fn len(&self) -> usize {
        self.0.entries.len()
    }

    /// Whether no token has bytes.
    #[doc(hidden)]
    pub fn is_empty(&self) -> bool {
        self.0.entries.is_empty()
    }

    /// Whether this model token ID is known either as a byte-backed token or
    /// as a special token.
    #[doc(hidden)]
    pub fn contains_token_id(&self, token_id: u32) -> bool {
        self.0.entries.contains_key(&token_id)
    }

    /// Highest defined model ID, including special IDs.
    #[doc(hidden)]
    pub fn max_event_token_id(&self) -> Option<u32> {
        self.0.event_token_ranges.last().map(|&(_, end)| end)
    }

    /// Find a free ID at or above `start` without scanning each occupied ID.
    #[doc(hidden)]
    pub fn next_available_token_id(&self, start: u32) -> Option<u32> {
        self.event_range_at(start).map_or(Some(start), |(_, upper)| upper.checked_add(1))
    }

    /// Whether at least one model ID intersects an inclusive range.
    #[doc(hidden)]
    pub fn intersects_token_range(&self, start: u32, end: u32) -> bool {
        if start > end {
            return false;
        }
        let index = self.0.event_token_ranges.partition_point(|&(_, upper)| upper < start);
        self.0.event_token_ranges.get(index).is_some_and(|&(lower, _)| lower <= end)
    }

    /// Whether two vocabularies describe the same byte mapping and the same
    /// special-token domain.
    #[doc(hidden)]
    pub fn same_model_vocab(&self, other: &Vocab) -> bool {
        self.shares_storage_with(other) || self.0.entries == other.0.entries
    }

    /// The highest byte-backed token ID, or zero if none; excludes special IDs.
    #[doc(hidden)]
    pub fn max_token_id(&self) -> u32 {
        self.0.entries
            .last_key_value()
            .map_or(0, |(&token_id, _)| token_id)
    }

    #[doc(hidden)]
    pub fn lexical_vocabulary(&self) -> Arc<LexicalVocabulary> {
        Arc::clone(self.0.lexical.get_or_init(|| Arc::new(LexicalVocabulary::build(self.entries_map()))))
    }

    /// Borrow a prepared spelling order without initiating cold construction.
    #[doc(hidden)]
    pub fn cached_lexical_vocabulary(&self) -> Option<Arc<LexicalVocabulary>> {
        self.0.lexical.get().cloned()
    }

    fn vocab_derived_cache_get_internal<T: VocabDerivedArtifact>(&self) -> Option<Arc<T>> {
        self.0.artifacts
            .lock()
            .ok()?
            .get(&TypeId::of::<T>())
            .cloned()
            .and_then(|artifact| artifact.downcast::<T>().ok())
    }

    fn vocab_derived_cache_set_internal<T: VocabDerivedArtifact>(&self, artifact: Arc<T>) {
        let erased: Arc<dyn Any + Send + Sync> = artifact;
        if let Ok(mut artifacts) = self.0.artifacts.lock() {
            artifacts.entry(TypeId::of::<T>()).or_insert(erased);
        }
    }

    #[doc(hidden)]
    pub fn get(&self, token_id: u32) -> Option<&[u8]> {
        self.0.entries.get(&token_id)
    }

    #[doc(hidden)]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u32, &[u8])> {
        self.0.entries
            .iter()
            .map(|(&token_id, bytes)| (token_id, bytes))
    }

    #[doc(hidden)]
    pub fn entries_map(&self) -> &VocabEntries {
        &self.0.entries
    }

    #[doc(hidden)]
    pub fn shares_storage_with(&self, other: &Vocab) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
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
        self.0.artifacts
            .lock()
            .map(|artifacts| artifacts.len())
            .unwrap_or(0)
            + usize::from(self.0.lexical.get().is_some())
            + usize::from(self.0.byte_facts.get().is_some())
    }
}

pub(crate) mod vocab_prefix_tree;
mod suffix_index;

/// Implementation details shared by the GLRMask workspace.
///
/// This module is deliberately feature-gated and is not a stable API.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub use super::derived_artifact::VocabDerivedArtifact;
    pub use super::error::internal::{error_message, new_error, with_source_span};
    pub use super::{Vocab, VocabEntries, TokenBytes, LexicalVocabulary, VocabByteFacts};
    pub use super::suffix_index::VocabularySuffixIndex;

    pub mod vocab_prefix_tree {
        pub use super::super::vocab_prefix_tree::*;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_store_and_subsets_match_literal_mappings() {
        for entries in [
            vec![],
            vec![(0, b"a".to_vec()), (1, b"bc".to_vec()), (2, Vec::new())],
            vec![(1000, b"bc".to_vec()), (2, b"a".to_vec()), (9, b"bc".to_vec()), (33, Vec::new())],
        ] {
            let expected = entries.iter().cloned().collect::<BTreeMap<_, _>>();
            let vocab = Vocab::from_entries_unchecked(entries);
            let clone = vocab.clone();
            assert!(vocab.shares_storage_with(&clone));
            assert_eq!(vocab.iter().map(|(id, bytes)| (id, bytes.to_vec()))
                .collect::<BTreeMap<_, _>>(), expected);
            assert_eq!(vocab.max_event_token_id(), expected.keys().next_back().copied());
            for id in [0, 1, 2, 9, 33, 1000, 1001] {
                assert_eq!(vocab.get(id), expected.get(&id).map(Vec::as_slice));
            }
            let subset = vocab.subset([1000, 2, 0, 9]);
            let subset_expected = expected.iter().filter(|(id, _)| [1000, 2, 0, 9].contains(*id))
                .map(|(&id, bytes)| (id, bytes.clone())).collect::<BTreeMap<_, _>>();
            assert!(Arc::ptr_eq(&vocab.0.entries.data, &subset.0.entries.data));
            let independent = Vocab::from_entries_unchecked(subset_expected.into_iter().collect());
            assert!(subset.same_model_vocab(&independent));
            let lexical = subset.lexical_vocabulary();
            for (word, ids) in lexical.ordered_token_bytes.iter().zip(lexical.ordered_to_originals.iter()) {
                assert!(Arc::ptr_eq(&word.data, &vocab.0.entries.data));
                for &id in ids {
                    assert_eq!(word.as_ref(), expected[&id].as_slice());
                }
            }
        }
    }

    #[test]
    fn subset_inherits_exact_spelling_order_in_original_entry_coordinates() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (2, b"z".to_vec()), (19, b"a".to_vec()), (101, b"z".to_vec()),
            (1003, b"".to_vec()), (9_000_001, b"ab".to_vec()),
        ]);
        let ordinals = [0, 2, 3, 4];
        let cold = vocab.subset_ordinals_with_cached_lexical(&ordinals);
        assert!(cold.cached_lexical_vocabulary().is_none());
        let parent = vocab.lexical_vocabulary();
        assert_eq!(parent.entry_indices(), &[3, 1, 4, 0, 2]);
        let inherited = vocab.subset_ordinals_with_cached_lexical(&ordinals);
        let child = inherited.cached_lexical_vocabulary().expect("inherited order");
        assert_eq!(child.entry_indices(), &[2, 3, 0, 1]);
        assert_eq!(child.original_order().as_ref(), &[(2, 2), (101, 2), (1003, 0), (9_000_001, 1)]);
        assert_eq!(child.ordered_to_originals, cold.lexical_vocabulary().ordered_to_originals);
        assert!(parent.original_to_ordered.get().is_none(), "subset must not materialize sparse ID holes");
    }

    #[test]
    fn clones_share_pure_derived_artifact_cache() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"ab".to_vec()), (1, b"bc".to_vec())]);
        assert_eq!(vocab.compiler_cache_entry_count(), 0);
        let bytes = vocab.relevant_bytes();
        assert_eq!(bytes.as_ref(), b"abc");
        assert_eq!(vocab.total_token_bytes(), 4);
        assert_eq!(vocab.byte_facts().follow_words(b'a'), [0, 1u64 << (b'b' % 64), 0, 0]);
        assert_eq!(vocab.byte_facts().follow_words(b'c'), [0; 4]);
        assert_eq!(vocab.max_token_byte_len(), 2);
        assert!(vocab.relevant_byte_mask()[b'b' as usize]);
        assert!(!vocab.relevant_byte_mask()[b'z' as usize]);
        assert_eq!(vocab.compiler_cache_entry_count(), 1);

        let cloned = vocab.clone();
        assert!(vocab.shares_storage_with(&cloned));
        assert!(Arc::ptr_eq(&bytes, &cloned.relevant_bytes()));
        assert_eq!(cloned.compiler_cache_entry_count(), 1);
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let error = Vocab::new([(0, b"a".as_slice()), (0, b"b".as_slice())]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn token_sets_are_normalized_and_validate_membership() {
        let vocab = Vocab::from_entries_unchecked(vec![(2, b"a".to_vec()), (7, b"b".to_vec())]);
        assert_eq!(TokenSet::new([2u32]).range_pairs(), &[(2, 2)]);
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
