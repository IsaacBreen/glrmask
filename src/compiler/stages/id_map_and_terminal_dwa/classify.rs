//! Vocab and terminal classification utilities.

use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::Vocab;

use super::types::TerminalPathLength;

/// DFA-derived byte sets for terminal classification, identical across partitions.
///
/// `classify_terminal_path_lengths` scans the full DFA to compute per-terminal
/// byte sets. Since all partitions share the same tokenizer and terminal count,
/// this scan is redundant after the first call. Caching these byte sets via
/// `OnceLock` eliminates ~35ms of repeated DFA scanning per extra partition.
pub struct SharedClassifyBytesets {
    reachable_bytes: Vec<U8Set>,
    first_bytes: Vec<U8Set>,
    last_bytes: Vec<U8Set>,
    transitions_by_byte: Vec<u32>,
    sparse_transitions_by_byte: Vec<Vec<(u32, u32)>>,
    reverse_transitions_by_byte: Vec<ReverseByteTransitions>,
    matched_terminals_by_state: Arc<[Box<[u32]>]>,
    future_terminals_by_state: Arc<[Box<[u32]>]>,
    matched_states_by_terminal: Arc<[Vec<u32>]>,
    future_states_by_terminal: Arc<[Vec<u32>]>,
    has_matched_terminal_by_state: Vec<u8>,
    future_by_state_words: Vec<u64>,
    representative_future_terminal_by_state: Vec<u32>,
    words_per_terminal_set: usize,
    active_route_setup_cache: Mutex<HashMap<(BitSet, usize), Arc<ActiveL2pRouteSetup>>>,
}

impl SharedClassifyBytesets {
    pub(crate) fn ti_output_index(
        &self,
    ) -> Option<(Arc<[Box<[u32]>]>, Arc<[Box<[u32]>]>, Arc<[Vec<u32>]>, Arc<[Vec<u32>]>)> {
        let state_count = self.future_by_state_words.len() / self.words_per_terminal_set.max(1);
        (self.matched_terminals_by_state.len() == state_count
            && self.future_terminals_by_state.len() == state_count
            && self.future_states_by_terminal.len() == self.matched_states_by_terminal.len())
            .then(|| {
                (
                    Arc::clone(&self.matched_terminals_by_state),
                    Arc::clone(&self.future_terminals_by_state),
                    Arc::clone(&self.matched_states_by_terminal),
                    Arc::clone(&self.future_states_by_terminal),
                )
            })
    }
}

struct ActiveL2pRouteSetup {
    active_start_states: Arc<[u32]>,
    allowed_boundary_pairs: Box<[U8Set; 256]>,
    allowed_boundary_pair_words: Box<[u64; 1024]>,
    active_reachable_by_byte: Box<[u8; 256]>,
    active_suffix_start_by_byte: Box<[u8; 256]>,
}

#[derive(Default)]
struct ReverseByteTransitions {
    targets: Vec<u32>,
    source_offsets: Vec<u32>,
    sources: Vec<u32>,
}

fn build_reverse_transitions_by_byte(
    sparse_transitions_by_byte: &[Vec<(u32, u32)>],
    num_states: usize,
) -> Vec<ReverseByteTransitions> {
    let mut target_seen = vec![0u32; num_states];
    let mut target_index = vec![0u32; num_states];
    let mut stamp = 0u32;

    sparse_transitions_by_byte
        .iter()
        .map(|transitions| {
            if transitions.is_empty() {
                return ReverseByteTransitions::default();
            }
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                target_seen.fill(0);
                stamp = 1;
            }

            let mut targets = Vec::new();
            let mut counts = Vec::<u32>::new();
            for &(_, target) in transitions {
                let target = target as usize;
                if target_seen[target] != stamp {
                    target_seen[target] = stamp;
                    target_index[target] = targets.len() as u32;
                    targets.push(target as u32);
                    counts.push(0);
                }
                counts[target_index[target] as usize] += 1;
            }

            let mut source_offsets = Vec::with_capacity(targets.len() + 1);
            source_offsets.push(0);
            for &count in &counts {
                source_offsets.push(source_offsets.last().copied().unwrap() + count);
            }
            let mut next_source_offsets = source_offsets[..targets.len()].to_vec();
            let mut sources = vec![0u32; transitions.len()];
            for &(source, target) in transitions {
                let group = target_index[target as usize] as usize;
                let offset = &mut next_source_offsets[group];
                sources[*offset as usize] = source;
                *offset += 1;
            }

            ReverseByteTransitions {
                targets,
                source_offsets,
                sources,
            }
        })
        .collect()
}

/// Cache type for lazy `SharedClassifyBytesets` initialization across partitions.
pub type SharedClassifyCache = std::sync::OnceLock<SharedClassifyBytesets>;

pub(crate) struct L2pVocabBoundarySplit {
    boundary_token_ids: Vec<u32>,
    single_token_ids: Vec<u32>,
    pub(crate) adjacent_tokens: usize,
    pub(crate) boundary_tokens: usize,
    pub(crate) single_tokens: usize,
    pub(crate) irrelevant_tokens: usize,
}

impl L2pVocabBoundarySplit {
    fn materialize_vocab(vocab: &Vocab, token_ids: &[u32]) -> Vocab {
        let mut entries = Vec::with_capacity(token_ids.len());
        let mut token_ids = token_ids.iter().copied().peekable();
        for (&token_id, bytes) in vocab.entries.iter() {
            while token_ids.peek().is_some_and(|candidate| *candidate < token_id) {
                token_ids.next();
            }
            if token_ids.peek().is_some_and(|candidate| *candidate == token_id) {
                entries.push((token_id, bytes.clone()));
                token_ids.next();
            }
            if token_ids.peek().is_none() {
                break;
            }
        }
        Vocab::new(entries, None)
    }

    pub(crate) fn boundary_vocab(&self, vocab: &Vocab) -> Vocab {
        Self::materialize_vocab(vocab, &self.boundary_token_ids)
    }

    pub(crate) fn single_vocab(&self, vocab: &Vocab) -> Vocab {
        Self::materialize_vocab(vocab, &self.single_token_ids)
    }
}

fn merge_sorted_token_ids(mut left: Vec<u32>, right: Vec<u32>) -> Vec<u32> {
    if right.is_empty() {
        return left;
    }
    if left.is_empty() {
        return right;
    }

    let mut merged = Vec::with_capacity(left.len() + right.len());
    let mut right = right.into_iter().peekable();
    for token_id in left.drain(..) {
        while right.peek().is_some_and(|candidate| *candidate < token_id) {
            merged.push(right.next().unwrap());
        }
        merged.push(token_id);
    }
    merged.extend(right);
    merged
}

#[derive(Debug)]
struct VocabByteSet {
    bytes: U8Set,
}

impl crate::vocab::VocabDerivedArtifact for VocabByteSet {}

fn vocab_byte_set(vocab: &Vocab) -> U8Set {
    if let Some(cached) = vocab.vocab_derived_cache_get::<VocabByteSet>() {
        return cached.bytes;
    }

    let mut byteset = U8Set::empty();
    for bytes in vocab.entries.values() {
        for &byte in bytes {
            byteset.insert(byte);
        }
    }
    vocab.vocab_derived_cache_set(std::sync::Arc::new(VocabByteSet { bytes: byteset }));
    byteset
}

pub(crate) fn prepare_vocab_for_terminal_classification(vocab: &Vocab) {
    let _ = vocab_byte_set(vocab);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum L2pPartitionCostFn {
    Size,
    SizeLog,
    LogLog,
    UnionSize,
}

impl L2pPartitionCostFn {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Size => "size",
            Self::SizeLog => "size_log",
            Self::LogLog => "log_log",
            Self::UnionSize => "union_size",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum L2pPartitionObjective {
    Max,
    Sum,
}

impl L2pPartitionObjective {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Sum => "sum",
        }
    }
}

pub(crate) struct L2pCostPartitioning {
    pub(crate) partitions: Vec<Vec<u32>>,
    pub(crate) estimated_partition_costs: Vec<f64>,
    pub(crate) estimated_l2p_terminals: Vec<usize>,
    pub(crate) objective_score: f64,
}

#[derive(Clone)]
struct L2pTokenGroup {
    l2p_terminals: BitSet,
    token_ids: Vec<u32>,
}

#[derive(Clone)]
struct L2pPartitionBucket {
    l2p_intersection: Option<BitSet>,
    l2p_union: Option<BitSet>,
    token_ids: Vec<u32>,
}

impl L2pPartitionBucket {
    fn new() -> Self {
        Self {
            l2p_intersection: None,
            l2p_union: None,
            token_ids: Vec::new(),
        }
    }

    fn size(&self) -> usize {
        self.token_ids.len()
    }

    fn l2p_count(&self) -> usize {
        self.l2p_intersection.as_ref().map_or(0, BitSet::count_ones)
    }
}

impl SharedClassifyBytesets {
    #[inline]
    pub(crate) fn transitions_by_byte(&self) -> &[u32] {
        &self.transitions_by_byte
    }

    /// Scan the DFA to compute per-terminal byte sets.
    pub fn build(tokenizer: &Tokenizer, num_terminals: u32) -> Self {
        let nt = num_terminals as usize;
        let initial = tokenizer.start_state();

        // The old implementation expanded target terminal bitsets once per
        // transition. On a large tokenizer this is catastrophic: many
        // transitions reach states whose possible-future bitsets contain most
        // terminals. The result only depends on the transition byte, so first
        // union terminal bitsets into 256 byte buckets, then transpose those
        // buckets once into the terminal -> byte sets required by callers.
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let started_at = std::time::Instant::now();
        let words_per_terminal_set = nt.div_ceil(64);
        let mut reachable_by_byte = vec![0u64; 256 * words_per_terminal_set];
        let mut last_by_byte = vec![0u64; 256 * words_per_terminal_set];
        let num_states = tokenizer.num_states() as usize;
        let mut transitions_by_byte = vec![u32::MAX; 256 * num_states];
        let mut sparse_transitions_by_byte = vec![Vec::<(u32, u32)>::new(); 256];
        let build_ti_output_index =
            std::env::var_os("GLRMASK_DISABLE_CLASSIFY_TI_OUTPUT_INDEX").is_none();
        let mut matched_terminals_by_state =
            Vec::with_capacity(if build_ti_output_index { num_states } else { 0 });
        let mut future_terminals_by_state =
            Vec::with_capacity(if build_ti_output_index { num_states } else { 0 });
        let mut matched_states_by_terminal = vec![Vec::<u32>::new(); nt];
        let mut future_states_by_terminal = vec![Vec::<u32>::new(); nt];
        let mut has_matched_terminal_by_state = vec![0u8; num_states];
        let mut future_by_state_words = vec![0u64; num_states * words_per_terminal_set];
        let mut representative_future_terminal_by_state = vec![u32::MAX; num_states];
        let mut transition_count = 0usize;

        for state in 0..tokenizer.num_states() {
            let matched = tokenizer
                .matched_terminals_iter(state)
                .filter(|terminal| (*terminal as usize) < nt)
                .collect::<Vec<_>>();
            for &terminal in &matched {
                matched_states_by_terminal[terminal as usize].push(state);
                has_matched_terminal_by_state[state as usize] = 1;
            }
            if build_ti_output_index {
                matched_terminals_by_state.push(matched.into_boxed_slice());
            }

            let future_words = tokenizer.possible_future_terminals(state).words();
            future_by_state_words[state as usize * words_per_terminal_set
                ..(state as usize + 1) * words_per_terminal_set]
                .copy_from_slice(future_words);
            if build_ti_output_index {
                let future = tokenizer
                    .possible_future_terminals_iter(state)
                    .filter(|terminal| (*terminal as usize) < nt)
                    .collect::<Vec<_>>();
                for &terminal in &future {
                    future_states_by_terminal[terminal as usize].push(state);
                }
                future_terminals_by_state.push(future.into_boxed_slice());
            }
            if let Some((word_index, &word)) = future_words
                .iter()
                .take(words_per_terminal_set)
                .enumerate()
                .find(|(_, word)| **word != 0)
            {
                representative_future_terminal_by_state[state as usize] =
                    (word_index * 64 + word.trailing_zeros() as usize) as u32;
            }
            for (byte, target) in tokenizer.transitions_from(state) {
                transition_count += 1;
                transitions_by_byte[byte as usize * num_states + state as usize] = target;
                sparse_transitions_by_byte[byte as usize].push((state, target));
                let bucket_offset = byte as usize * words_per_terminal_set;
                let matched_words = tokenizer.matched_terminal_bitset(target).words();
                let future_words = tokenizer.possible_future_terminals(target).words();

                debug_assert!(matched_words.len() >= words_per_terminal_set);
                debug_assert!(future_words.len() >= words_per_terminal_set);
                for word_index in 0..words_per_terminal_set {
                    let matched_word = matched_words[word_index];
                    let future_word = future_words[word_index];
                    // Future terminals are strict and omit finalizers, so the
                    // reachable set is exactly their union. Fuse the two
                    // reachable-bucket writes into one pass.
                    reachable_by_byte[bucket_offset + word_index] |= matched_word | future_word;
                    last_by_byte[bucket_offset + word_index] |= matched_word;
                }
            }
        }
        let reverse_transitions_by_byte =
            build_reverse_transitions_by_byte(&sparse_transitions_by_byte, num_states);
        let scan_ms = started_at.elapsed().as_secs_f64() * 1000.0;

        let mut reachable_bytes = vec![U8Set::empty(); nt];
        let mut last_bytes = vec![U8Set::empty(); nt];
        for byte in 0u8..=u8::MAX {
            let bucket_offset = byte as usize * words_per_terminal_set;
            for word_index in 0..words_per_terminal_set {
                let base_terminal = word_index * 64;
                let mut reachable_word = reachable_by_byte[bucket_offset + word_index];
                while reachable_word != 0 {
                    let terminal = base_terminal + reachable_word.trailing_zeros() as usize;
                    if terminal < nt {
                        reachable_bytes[terminal].insert(byte);
                    }
                    reachable_word &= reachable_word - 1;
                }

                let mut last_word = last_by_byte[bucket_offset + word_index];
                while last_word != 0 {
                    let terminal = base_terminal + last_word.trailing_zeros() as usize;
                    if terminal < nt {
                        last_bytes[terminal].insert(byte);
                    }
                    last_word &= last_word - 1;
                }
            }
        }

        // first_bytes: only from initial state (single state, no parallelism needed).
        let mut first_bytes = vec![U8Set::empty(); nt];
        for (byte, target) in tokenizer.transitions_from(initial) {
            for terminal in tokenizer.matched_terminal_bitset(target).iter() {
                if terminal < nt {
                    first_bytes[terminal].insert(byte);
                }
            }
            for terminal in tokenizer.possible_future_terminals(target).iter() {
                let t = terminal as usize;
                if t < nt {
                    first_bytes[t].insert(byte);
                }
            }
        }

        if profile_enabled {
            eprintln!(
                "[glrmask/profile][classify_bytesets] terminals={} states={} transitions={} words_per_terminal_set={} scan_ms={:.3} transpose_and_first_ms={:.3} total_ms={:.3}",
                nt,
                tokenizer.num_states(),
                transition_count,
                words_per_terminal_set,
                scan_ms,
                started_at.elapsed().as_secs_f64() * 1000.0 - scan_ms,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }

        SharedClassifyBytesets {
            reachable_bytes,
            first_bytes,
            last_bytes,
            transitions_by_byte,
            sparse_transitions_by_byte,
            reverse_transitions_by_byte,
            matched_terminals_by_state: matched_terminals_by_state.into(),
            future_terminals_by_state: future_terminals_by_state.into(),
            matched_states_by_terminal: matched_states_by_terminal.into(),
            future_states_by_terminal: future_states_by_terminal.into(),
            has_matched_terminal_by_state,
            future_by_state_words,
            representative_future_terminal_by_state,
            words_per_terminal_set,
            active_route_setup_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
}

/// JSON structural characters used to keep tokens in the core non-alnum
/// partition (P0) rather than splitting them into the auxiliary P5.
const JSON_STRUCTURAL: &[u8] = b"\":[]{},";

/// `_` belongs with alphabetic bytes for vocabulary partitioning. This is a
/// routing convention only: it does not change lexer or grammar semantics.
fn is_partition_ascii_alpha(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

/// Characters whose sole repetition qualifies a non-alnum token for the
/// auxiliary P5 partition even if the token contains a structural byte.
const P5_REPEATED_CHARS: &[u8] = b"\n:{ ,";

/// Classifies a token's bytes by character type for vocab partitioning.
///
/// Returns:
/// - 0: non-alnum with JSON structural chars (multi-byte, not single-repeated)
/// - 1: mixed (contains both alnum and non-alnum)
/// - 2: ASCII word token with ≥1 alpha or `_`, optionally with leading space
/// - 3: pure digit, optionally with leading space
/// - 4: Unicode-only alpha (non-ASCII alphanumeric, e.g. CJK, Cyrillic,
///       Arabic, Hangul), optionally with leading space
/// - 5: non-alnum auxiliary short (no JSON structural, or single-char repeated,
///       or length 1; ≤ 8 bytes)
/// - 6: non-alnum auxiliary long (same criteria as 5, but > 8 bytes)
/// - 7: JSON literal-boundary tokens requiring structural treatment (leading-
///       space collisions, bracketed forms, and the special ` -` token)
/// - 8: quoted ASCII identifier-start tokens
///
/// Uses Unicode-aware classification so that non-Latin scripts are separated
/// into their own partition (4) instead of being lumped with ASCII punctuation (0)
/// or bloating the ASCII alpha partition (2).
///
/// P0/P5 split: non-alnum tokens containing JSON structural characters
/// (`":[]{},`) stay in P0 for efficient L2+ terminal processing, while
/// tokens without structural chars (or trivial single-char tokens) go to P5.
pub(crate) fn classify_vocab_char_type(bytes: &[u8]) -> u8 {
    if bytes.is_empty() {
        return 5;
    }
    // Bare ASCII word pieces that overlap a JSON literal spelling are ordinary
    // P2 material. Only their leading-space variants need to stay isolated at
    // the structural boundary.
    if !bytes.starts_with(b" ") && is_json_literal_collision(bytes) {
        return 2;
    }
    if is_quoted_identifier_boundary_token(bytes) {
        return 8;
    }
    if is_structural_boundary_lexical_token(bytes) {
        return 7;
    }
    // Strip optional leading ASCII space (GPT-2 BPE decodes Ġ → 0x20 before we see it)
    let content = if bytes[0] == b' ' {
        &bytes[1..]
    } else {
        bytes
    };
    if content.is_empty() {
        return 5; // Just a space marker → auxiliary non-alnum
    }
    if content.len() == 1 && matches!(content[0], b'+' | b'-') {
        return 1;
    }
    // Try to decode as UTF-8 for Unicode-aware classification.
    if let Ok(s) = std::str::from_utf8(content) {
        let all_word = s.chars().all(|c| c.is_alphanumeric() || c == '_');
        if all_word {
            let has_alpha = s.chars().any(|c| c.is_alphabetic() || c == '_');
            if has_alpha {
                let has_ascii_alpha = content.iter().copied().any(is_partition_ascii_alpha);
                if has_ascii_alpha {
                    return 2; // ASCII word token (may also contain non-ASCII alpha)
                }
                return 4; // Unicode-only alpha (CJK, Cyrillic, Arabic, etc.)
            }
            return 3; // Pure digit
        }
        // Check non-alphanumeric.
        if let Ok(full) = std::str::from_utf8(bytes) {
            if !full
                .chars()
                .any(|c| c.is_alphanumeric() || c == '_')
            {
                return classify_nonalnum(bytes);
            }
        }
        return 1; // Mixed
    }
    // Fallback: byte-level ASCII checks for invalid UTF-8.
    if content
        .iter()
        .copied()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        if content.iter().copied().any(is_partition_ascii_alpha) {
            return 2;
        }
        return 3;
    }
    if bytes
        .iter()
        .copied()
        .all(|byte| !byte.is_ascii_alphanumeric() && byte != b'_')
    {
        return classify_nonalnum(bytes);
    }
    1 // Mixed
}

fn is_json_literal_collision(content: &[u8]) -> bool {
    if content.is_empty() || !content.iter().all(|byte| byte.is_ascii_alphanumeric()) {
        return false;
    }

    [b"true".as_slice(), b"false".as_slice(), b"null".as_slice()]
        .iter()
        .any(|literal| literal.starts_with(content) || content.starts_with(literal))
}

fn is_structural_boundary_lexical_token(bytes: &[u8]) -> bool {
    if !structural_boundary_lexical_partition_enabled() {
        return false;
    }

    let content = bytes.strip_prefix(b" ").unwrap_or(bytes);
    if is_json_literal_collision(content) {
        return true;
    }
    if bytes == b" -" {
        return true;
    }
    if bytes.starts_with(b"[") && is_json_literal_collision(&bytes[1..]) {
        return true;
    }
    false
}

fn is_quoted_identifier_boundary_token(bytes: &[u8]) -> bool {
    structural_boundary_lexical_partition_enabled()
        && bytes
        .strip_prefix(b"\"")
        .is_some_and(|suffix| suffix.first().copied().is_some_and(is_partition_ascii_alpha))
}

fn structural_boundary_lexical_partition_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_STRUCTURAL_BOUNDARY_LEXICAL_PARTITION")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
            })
            .unwrap_or(true)
    })
}


/// Sub-classify a non-alphanumeric token into P0 (structural), P5 (short auxiliary),
/// or P6 (long auxiliary).
///
/// P5/P6 if: (a) no JSON structural char, (b) single repeated char from
/// `\n:{ ,`, or (c) length 1. Within that group, tokens > 8 bytes go to P6.
fn classify_nonalnum(bytes: &[u8]) -> u8 {
    // Length 1 → P5
    if bytes.len() <= 1 {
        return 5;
    }
    // Single repeated char from P5_REPEATED_CHARS → P5/P6
    if bytes.iter().all(|b| *b == bytes[0]) && P5_REPEATED_CHARS.contains(&bytes[0]) {
        return if bytes.len() > 8 { 6 } else { 5 };
    }
    // No JSON structural char → P5/P6
    if !bytes.iter().any(|b| JSON_STRUCTURAL.contains(b)) {
        return if bytes.len() > 8 { 6 } else { 5 };
    }
    0 // Structural non-alnum → P0
}

/// Classifies each terminal by the longest token-path length it can participate in.
///
/// - **Length 0**: No vocab byte from any tokenizer state can lead towards
///   matching this terminal.  The terminal is ignorable.
/// - **Length 1**: The terminal is matchable but never co-occurs with another
///   terminal inside a single vocab token.
/// - **Length 2+**: There exists a pair (t1, t2) of terminals with an allowed
///   follow relationship whose last/first bytes overlap the vocab byte bitset,
///   so a single token could span both.
pub(crate) fn classify_terminal_path_lengths(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    shared_classify_cache: Option<&SharedClassifyCache>,
) -> Vec<TerminalPathLength> {
    let nt = num_terminals as usize;

    // 1. Vocab byte bitset: all bytes appearing in any vocab token.
    let vocab_bytes = vocab_byte_set(vocab);

    // 2. Byte bitsets per terminal — use cache if available.
    let owned_bytesets: Option<SharedClassifyBytesets>;
    let bytesets: &SharedClassifyBytesets = if let Some(cache) = shared_classify_cache {
        cache.get_or_init(|| SharedClassifyBytesets::build(tokenizer, num_terminals))
    } else {
        owned_bytesets = Some(SharedClassifyBytesets::build(tokenizer, num_terminals));
        owned_bytesets.as_ref().unwrap()
    };
    let reachable_bytes = &bytesets.reachable_bytes;
    let first_bytes = &bytesets.first_bytes;
    let last_bytes = &bytesets.last_bytes;

    // 3. Mark terminals that may participate in paths of length ≥ 2.
    let mut is_two_plus = BitSet::new(nt);

    for t1 in 0..nt {
        if last_bytes[t1].is_disjoint(&vocab_bytes) {
            continue;
        }
        let disallowed = disallowed_follows.get(&(t1 as u32));
        for t2 in 0..nt {
            if first_bytes[t2].is_disjoint(&vocab_bytes) {
                continue;
            }
            if let Some(d) = disallowed {
                if d.contains(t2) {
                    continue;
                }
            }
            is_two_plus.set(t1);
            is_two_plus.set(t2);
        }
    }

    // 4. Final classification.
    let mut result = vec![TerminalPathLength::Zero; nt];
    for t in 0..nt {
        if reachable_bytes[t].is_disjoint(&vocab_bytes) {
            result[t] = TerminalPathLength::Zero;
        } else if is_two_plus.contains(t) {
            result[t] = TerminalPathLength::TwoPlus;
        } else {
            result[t] = TerminalPathLength::One;
        }
    }

    result
}

fn build_byte_terminal_reverse_index(
    bytesets: &SharedClassifyBytesets,
    num_terminals: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut byte_to_last: Vec<Vec<usize>> = vec![Vec::new(); 256];
    let mut byte_to_first: Vec<Vec<usize>> = vec![Vec::new(); 256];
    for terminal in 0..num_terminals {
        for byte in 0u8..=255 {
            if bytesets.last_bytes[terminal].contains(byte) {
                byte_to_last[byte as usize].push(terminal);
            }
            if bytesets.first_bytes[terminal].contains(byte) {
                byte_to_first[byte as usize].push(terminal);
            }
        }
    }
    (byte_to_last, byte_to_first)
}

fn token_l2p_terminals(
    bytes: &[u8],
    byte_to_last: &[Vec<usize>],
    byte_to_first: &[Vec<usize>],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: usize,
) -> BitSet {
    let mut seen = [false; 256];
    let mut last_set = BitSet::new(num_terminals);
    let mut first_set = BitSet::new(num_terminals);

    for &byte in bytes {
        if !seen[byte as usize] {
            seen[byte as usize] = true;
            for &terminal in &byte_to_last[byte as usize] {
                last_set.set(terminal);
            }
            for &terminal in &byte_to_first[byte as usize] {
                first_set.set(terminal);
            }
        }
    }

    let mut l2p_set = BitSet::new(num_terminals);
    for terminal_1 in last_set.iter() {
        let disallowed = disallowed_follows.get(&(terminal_1 as u32));
        for terminal_2 in first_set.iter() {
            if !disallowed.map_or(false, |blocked| blocked.contains(terminal_2)) {
                l2p_set.set(terminal_1);
                l2p_set.set(terminal_2);
            }
        }
    }

    l2p_set
}

fn token_has_active_l2p_boundary(bytes: &[u8], allowed_boundary_pairs: &[U8Set; 256]) -> bool {
    bytes
        .windows(2)
        .any(|pair| allowed_boundary_pairs[pair[0] as usize].contains(pair[1]))
}

fn token_has_active_l2p_boundary_words(
    bytes: &[u8],
    allowed_boundary_pair_words: &[u64; 1024],
) -> bool {
    bytes.windows(2).any(|pair| {
        let pair_index = ((pair[0] as usize) << 8) | pair[1] as usize;
        (allowed_boundary_pair_words[pair_index >> 6] & (1u64 << (pair_index & 63))) != 0
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenL2pRouteHint {
    Adjacent,
    Single,
    Irrelevant,
}

/// Classify the cheap, byte-level L2P route hint in one token pass.
///
/// The route scan formerly searched every token once for an allowed adjacent
/// pair and, when none was found, searched it again for an active reachable
/// byte. The adjacency result has precedence, so retaining the first byte while
/// tracking reachability produces exactly the same split with one traversal.
#[inline]
fn token_l2p_route_hint(
    bytes: &[u8],
    allowed_boundary_pair_words: &[u64; 1024],
    active_reachable_by_byte: &[u8; 256],
) -> TokenL2pRouteHint {
    let Some((&first, rest)) = bytes.split_first() else {
        return TokenL2pRouteHint::Irrelevant;
    };
    let mut previous = first;
    let mut reaches_active = active_reachable_by_byte[first as usize] != 0;
    for &byte in rest {
        let pair_index = ((previous as usize) << 8) | byte as usize;
        if (allowed_boundary_pair_words[pair_index >> 6] & (1u64 << (pair_index & 63)))
            != 0
        {
            return TokenL2pRouteHint::Adjacent;
        }
        reaches_active |= active_reachable_by_byte[byte as usize] != 0;
        previous = byte;
    }
    if reaches_active {
        TokenL2pRouteHint::Single
    } else {
        TokenL2pRouteHint::Irrelevant
    }
}

#[inline]
fn bitset_intersects_words(bitset: &BitSet, words: &[u64]) -> bool {
    bitset
        .words()
        .iter()
        .zip(words)
        .any(|(lhs, rhs)| (*lhs & *rhs) != 0)
}

#[inline]
fn state_future_intersects_words(
    bytesets: &SharedClassifyBytesets,
    state: u32,
    active_words: &[u64],
) -> bool {
    let offset = state as usize * bytesets.words_per_terminal_set;
    bytesets.future_by_state_words[offset..offset + bytesets.words_per_terminal_set]
        .iter()
        .zip(active_words)
        .any(|(state_word, active_word)| (*state_word & *active_word) != 0)
}

fn build_active_matched_by_state(
    bytesets: &SharedClassifyBytesets,
    active_bitset: &BitSet,
) -> Box<[u8]> {
    let mut matched = vec![
        0u8;
        bytesets.future_by_state_words.len() / bytesets.words_per_terminal_set
    ];
    for terminal in active_bitset.iter() {
        for &state in &bytesets.matched_states_by_terminal[terminal] {
            matched[state as usize] = 1;
        }
    }
    matched.into()
}

fn build_active_suffix_start_by_byte(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    active_words: &[u64],
) -> Box<[u8; 256]> {
    let mut can_start = Box::new([0u8; 256]);
    let initial = tokenizer.initial_state_id();
    if !state_future_intersects_words(bytesets, initial, active_words) {
        return can_start;
    }
    for byte in 0u8..=u8::MAX {
        let next = bytesets.transitions_by_byte
            [byte as usize * tokenizer.num_states() as usize + initial as usize];
        if next == u32::MAX {
            continue;
        }
        if state_future_intersects_words(bytesets, next, active_words)
            || bitset_intersects_words(tokenizer.matched_terminal_bitset(next), active_words)
        {
            can_start[byte as usize] = 1;
        }
    }
    can_start
}

#[inline]
fn suffix_can_reach_active_terminal(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    flat_trans: &[u32],
    suffix: &[u8],
    active_words: &[u64],
    active_matched_by_state: Option<&[u8]>,
) -> bool {
    let mut state = tokenizer.initial_state_id();
    for &byte in suffix {
        if !state_future_intersects_words(bytesets, state, active_words) {
            return false;
        }
        let next = flat_trans[state as usize * 256 + byte as usize];
        if next == u32::MAX {
            return false;
        }
        state = next;
        if active_matched_by_state.map_or_else(
            || bitset_intersects_words(tokenizer.matched_terminal_bitset(state), active_words),
            |matched| matched[state as usize] != 0,
        ) {
            return true;
        }
    }
    state_future_intersects_words(bytesets, state, active_words)
}

fn token_has_active_terminal_suffix(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    flat_trans: &[u32],
    bytes: &[u8],
    active_words: &[u64],
    active_matched_by_state: Option<&[u8]>,
    active_suffix_start_by_byte: &[u8; 256],
) -> bool {
    (1..bytes.len()).any(|suffix_start| {
        active_suffix_start_by_byte[bytes[suffix_start] as usize] != 0
            &&
        suffix_can_reach_active_terminal(
            tokenizer,
            bytesets,
            flat_trans,
            &bytes[suffix_start..],
            active_words,
            active_matched_by_state,
        )
    })
}

fn active_l2p_route_setup(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    active_bitset: &BitSet,
    disallowed_follows: &Arc<BTreeMap<u32, BitSet>>,
) -> Arc<ActiveL2pRouteSetup> {
    let cache_key = (
        active_bitset.clone(),
        Arc::as_ptr(disallowed_follows) as usize,
    );
    if let Some(cached) = bytesets
        .active_route_setup_cache
        .lock()
        .unwrap()
        .get(&cache_key)
        .cloned()
    {
        return cached;
    }

    let setup_started_at = std::time::Instant::now();
    let active_words = active_bitset.words();
    let active_count = active_bitset.count_ones();
    let use_representative_fast_path = active_count >= active_bitset.len().div_ceil(2);
    let mut active_start_states = Vec::new();
    let mut start_full_checks = 0usize;
    for state in 0..tokenizer.num_states() {
        if use_representative_fast_path {
            let representative = bytesets.representative_future_terminal_by_state[state as usize];
            if representative != u32::MAX && active_bitset.contains(representative as usize) {
                active_start_states.push(state);
                continue;
            }
        }
        {
            start_full_checks += 1;
            if state_future_intersects_words(bytesets, state, active_words) {
                active_start_states.push(state);
            }
        }
    }
    let start_states_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let boundary_pairs_started_at = std::time::Instant::now();
    let mut allowed_boundary_pairs = Box::new([U8Set::empty(); 256]);
    let mut active_reachable = U8Set::empty();
    let mut active_first_bytes = U8Set::empty();
    let mut unrestricted_last_bytes = U8Set::empty();
    let mut allowed_first_bytes_cache = HashMap::<&BitSet, U8Set>::new();

    for terminal in active_bitset.iter() {
        active_reachable = active_reachable.union(&bytesets.reachable_bytes[terminal]);
        active_first_bytes = active_first_bytes.union(&bytesets.first_bytes[terminal]);
    }
    let mut active_reachable_by_byte = Box::new([0u8; 256]);
    for byte in active_reachable.iter() {
        active_reachable_by_byte[byte as usize] = 1;
    }
    let active_suffix_start_by_byte =
        build_active_suffix_start_by_byte(tokenizer, bytesets, active_words);
    for terminal_1 in active_bitset.iter() {
        let last_bytes = bytesets.last_bytes[terminal_1];
        if last_bytes.is_empty() {
            continue;
        }
        let Some(blocked) = disallowed_follows
            .get(&(terminal_1 as u32))
            .filter(|blocked| !blocked.is_empty())
        else {
            unrestricted_last_bytes = unrestricted_last_bytes.union(&last_bytes);
            continue;
        };
        let allowed_first_bytes = *allowed_first_bytes_cache.entry(blocked).or_insert_with(|| {
            let mut allowed = U8Set::empty();
            for terminal_2 in active_bitset.iter() {
                if !blocked.contains(terminal_2) {
                    allowed = allowed.union(&bytesets.first_bytes[terminal_2]);
                }
            }
            allowed
        });
        for last_byte in last_bytes.iter() {
            allowed_boundary_pairs[last_byte as usize] =
                allowed_boundary_pairs[last_byte as usize].union(&allowed_first_bytes);
        }
    }
    for last_byte in unrestricted_last_bytes.iter() {
        allowed_boundary_pairs[last_byte as usize] =
            allowed_boundary_pairs[last_byte as usize].union(&active_first_bytes);
    }
    let mut allowed_boundary_pair_words = Box::new([0u64; 1024]);
    for (last_byte, first_bytes) in allowed_boundary_pairs.iter().enumerate() {
        for first_byte in first_bytes.iter() {
            let pair_index = (last_byte << 8) | first_byte as usize;
            allowed_boundary_pair_words[pair_index >> 6] |= 1u64 << (pair_index & 63);
        }
    }

    let setup = Arc::new(ActiveL2pRouteSetup {
        active_start_states: active_start_states.into(),
        allowed_boundary_pairs,
        allowed_boundary_pair_words,
        active_reachable_by_byte,
        active_suffix_start_by_byte,
    });
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p_route_setup] active={} starts={} start_full_checks={} start_states_ms={:.3} boundary_pairs_ms={:.3} total_ms={:.3}",
            active_bitset.count_ones(),
            setup.active_start_states.len(),
            start_full_checks,
            start_states_ms,
            boundary_pairs_started_at.elapsed().as_secs_f64() * 1000.0,
            setup_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let mut cache = bytesets.active_route_setup_cache.lock().unwrap();
    Arc::clone(cache.entry(cache_key).or_insert(setup))
}

#[derive(Clone, Copy, Debug)]
enum ExactL2pBoundaryFilterMode {
    Auto,
    Force(bool),
}

fn parse_exact_l2p_boundary_filter_mode(value: &str) -> ExactL2pBoundaryFilterMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => ExactL2pBoundaryFilterMode::Auto,
        "1" | "true" | "yes" | "on" => ExactL2pBoundaryFilterMode::Force(true),
        "0" | "false" | "no" | "off" => ExactL2pBoundaryFilterMode::Force(false),
        other => panic!(
            "invalid GLRMASK_EXACT_L2P_BOUNDARY_FILTER value `{other}`; expected auto, 1/true/on, or 0/false/off"
        ),
    }
}

fn exact_l2p_boundary_filter_mode() -> ExactL2pBoundaryFilterMode {
    static MODE: std::sync::OnceLock<ExactL2pBoundaryFilterMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("GLRMASK_EXACT_L2P_BOUNDARY_FILTER")
            .ok()
            .map(|value| parse_exact_l2p_boundary_filter_mode(&value))
            .unwrap_or(ExactL2pBoundaryFilterMode::Auto)
    })
}

fn exact_l2p_boundary_filter_work_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("GLRMASK_EXACT_L2P_BOUNDARY_FILTER_WORK_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(20_000_000)
    })
}

fn suffix_has_allowed_l2p_follow(
    tokenizer: &Tokenizer,
    terminal_1: usize,
    suffix: &[u8],
    active_bitset: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> bool {
    let allowed_after = disallowed_follows
        .get(&(terminal_1 as u32))
        .map_or_else(|| active_bitset.clone(), |blocked| active_bitset.difference(blocked));
    if allowed_after.is_empty() {
        return false;
    }

    let mut state = tokenizer.initial_state_id();
    for &byte in suffix {
        if tokenizer
            .possible_future_terminals(state)
            .is_disjoint(&allowed_after)
        {
            return false;
        }
        let Some(next) = tokenizer.step(state, byte) else {
            return false;
        };
        state = next;
        for terminal in tokenizer.matched_terminals_iter(state) {
            if allowed_after.contains(terminal as usize) {
                return true;
            }
        }
    }

    !tokenizer
        .possible_future_terminals(state)
        .is_disjoint(&allowed_after)
}

fn token_has_exact_active_l2p_boundary(
    tokenizer: &Tokenizer,
    bytes: &[u8],
    active_bitset: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_start_states: &[u32],
) -> bool {
    if bytes.len() < 2 {
        return false;
    }

    let mut current_states = active_start_states.to_vec();
    let mut next_states = Vec::<u32>::new();
    let mut seen = vec![0u32; tokenizer.num_states() as usize];
    let mut seen_stamp = 0u32;
    let mut suffix_cache = HashMap::<(usize, usize), bool>::new();

    for split_after in 0..bytes.len() - 1 {
        seen_stamp = seen_stamp.wrapping_add(1);
        if seen_stamp == 0 {
            seen.fill(0);
            seen_stamp = 1;
        }
        next_states.clear();

        for &state in &current_states {
            let Some(next) = tokenizer.step(state, bytes[split_after]) else {
                continue;
            };
            let slot = &mut seen[next as usize];
            if *slot == seen_stamp {
                continue;
            }
            *slot = seen_stamp;
            next_states.push(next);
        }

        if next_states.is_empty() {
            return false;
        }

        for &state in &next_states {
            for terminal_1 in tokenizer.matched_terminals_iter(state) {
                let terminal_1 = terminal_1 as usize;
                if !active_bitset.contains(terminal_1) {
                    continue;
                }
                let suffix_start = split_after + 1;
                let has_follow = *suffix_cache.entry((terminal_1, suffix_start)).or_insert_with(|| {
                    suffix_has_allowed_l2p_follow(
                        tokenizer,
                        terminal_1,
                        &bytes[suffix_start..],
                        active_bitset,
                        disallowed_follows,
                    )
                });
                if has_follow {
                    return true;
                }
            }
        }

        std::mem::swap(&mut current_states, &mut next_states);
    }

    false
}

#[derive(Default)]
struct ExactBoundaryPrefixNode {
    children: Vec<(u8, usize)>,
    allowed_follow_terminals: Option<BitSet>,
}

fn insert_exact_boundary_prefix(
    nodes: &mut Vec<ExactBoundaryPrefixNode>,
    parent: usize,
    byte: u8,
) -> usize {
    if let Some((_, child)) = nodes[parent]
        .children
        .iter()
        .find(|(candidate, _)| *candidate == byte)
    {
        return *child;
    }

    let child = nodes.len();
    nodes.push(ExactBoundaryPrefixNode::default());
    nodes[parent].children.push((byte, child));
    child
}

fn populate_exact_boundary_prefixes(
    tokenizer: &Tokenizer,
    transitions_by_byte: &[u32],
    sparse_transitions_by_byte: &[Vec<(u32, u32)>],
    reverse_transitions_by_byte: &[ReverseByteTransitions],
    active_bitset: &BitSet,
    nodes: &mut [ExactBoundaryPrefixNode],
    node: usize,
    current_states: &[u32],
    depth: usize,
    state_seen: &mut [u32],
    state_stamp: &mut u32,
    terminal_seen: &mut [u32],
    terminal_stamp: &mut u32,
    class_seen: &mut [u32],
    class_stamp: &mut u32,
    states_scanned: &mut usize,
    reached_states: &mut usize,
    finalizer_terminals_scanned: &mut usize,
    has_matched_terminal_by_state: &[u8],
    allowed_class_by_terminal: &[Option<usize>],
    allowed_follow_classes: &[BitSet],
    matched_classes: &mut Vec<usize>,
    state_buffers: &mut Vec<Vec<u32>>,
    source_seen_by_depth: &mut Vec<Vec<u32>>,
    source_stamps: &mut Vec<u32>,
) {
    let child_count = nodes[node].children.len();
    let frontier_is_dense = current_states.len() * 4 >= tokenizer.num_states() as usize;
    let source_stamp = if nodes[node]
        .children
        .iter()
        .any(|(byte, _)| {
            let sparse = &sparse_transitions_by_byte[*byte as usize];
            let reverse = &reverse_transitions_by_byte[*byte as usize];
            sparse.len() < current_states.len()
                || (frontier_is_dense && reverse.targets.len() * 2 < sparse.len())
        })
    {
        if source_seen_by_depth.len() <= depth {
            source_seen_by_depth.resize_with(depth + 1, || {
                vec![0u32; tokenizer.num_states() as usize]
            });
            source_stamps.resize(depth + 1, 0);
        }
        let source_seen = &mut source_seen_by_depth[depth];
        let source_stamp = &mut source_stamps[depth];
        *source_stamp = source_stamp.wrapping_add(1);
        if *source_stamp == 0 {
            source_seen.fill(0);
            *source_stamp = 1;
        }
        for &state in current_states {
            source_seen[state as usize] = *source_stamp;
        }
        Some(*source_stamp)
    } else {
        None
    };
    for child_index in 0..child_count {
        let (byte, child) = nodes[node].children[child_index];
        if state_buffers.len() <= depth {
            state_buffers.resize_with(depth + 1, Vec::new);
        }
        let mut next_states = std::mem::take(&mut state_buffers[depth]);
        next_states.clear();
        if next_states.capacity() < current_states.len() {
            next_states.reserve(current_states.len() - next_states.capacity());
        }
        let sparse_transitions = &sparse_transitions_by_byte[byte as usize];
        let reverse_transitions = &reverse_transitions_by_byte[byte as usize];
        let use_reverse_transitions = source_stamp.is_some()
            && frontier_is_dense
            && reverse_transitions.targets.len() * 2 < sparse_transitions.len();
        if use_reverse_transitions {
            let source_stamp = source_stamp.unwrap();
            let source_seen = &source_seen_by_depth[depth];
            for (target_index, &target) in reverse_transitions.targets.iter().enumerate() {
                let source_start = reverse_transitions.source_offsets[target_index] as usize;
                let source_end = reverse_transitions.source_offsets[target_index + 1] as usize;
                for &source in &reverse_transitions.sources[source_start..source_end] {
                    *states_scanned += 1;
                    if source_seen[source as usize] == source_stamp {
                        next_states.push(target);
                        break;
                    }
                }
            }
        } else {
            *state_stamp = state_stamp.wrapping_add(1);
            if *state_stamp == 0 {
                state_seen.fill(0);
                *state_stamp = 1;
            }
            if source_stamp.is_some() && sparse_transitions.len() < current_states.len() {
            *states_scanned += sparse_transitions.len();
            let source_stamp = source_stamp.unwrap();
            let source_seen = &source_seen_by_depth[depth];
            for &(source, next) in sparse_transitions {
                if source_seen[source as usize] != source_stamp {
                    continue;
                }
                let slot = &mut state_seen[next as usize];
                if *slot == *state_stamp {
                    continue;
                }
                *slot = *state_stamp;
                next_states.push(next);
            }
            } else {
            *states_scanned += current_states.len();
            let transition_column =
                &transitions_by_byte[byte as usize * tokenizer.num_states() as usize
                    ..(byte as usize + 1) * tokenizer.num_states() as usize];
            for &state in current_states {
                let next = transition_column[state as usize];
                if next == u32::MAX {
                    continue;
                }
                let slot = &mut state_seen[next as usize];
                if *slot == *state_stamp {
                    continue;
                }
                *slot = *state_stamp;
                next_states.push(next);
            }
            }
        }
        *reached_states += next_states.len();

        *terminal_stamp = terminal_stamp.wrapping_add(1);
        if *terminal_stamp == 0 {
            terminal_seen.fill(0);
            *terminal_stamp = 1;
        }
        *class_stamp = class_stamp.wrapping_add(1);
        if *class_stamp == 0 {
            class_seen.fill(0);
            *class_stamp = 1;
        }
        matched_classes.clear();
        for &state in &next_states {
            if has_matched_terminal_by_state[state as usize] == 0 {
                continue;
            }
            for terminal in tokenizer.matched_terminals_iter(state) {
                *finalizer_terminals_scanned += 1;
                let terminal = terminal as usize;
                if !active_bitset.contains(terminal)
                    || terminal_seen[terminal] == *terminal_stamp
                {
                    continue;
                }
                terminal_seen[terminal] = *terminal_stamp;
                if let Some(class) = allowed_class_by_terminal[terminal] {
                    if class_seen[class] != *class_stamp {
                        class_seen[class] = *class_stamp;
                        matched_classes.push(class);
                    }
                }
            }
        }
        let mut allowed_follow_terminals = None;
        for &class in matched_classes.iter() {
            allowed_follow_terminals
                .get_or_insert_with(|| BitSet::new(active_bitset.len()))
                .union_with(&allowed_follow_classes[class]);
        }
        nodes[child].allowed_follow_terminals = allowed_follow_terminals;

        if !next_states.is_empty() {
            populate_exact_boundary_prefixes(
                tokenizer,
                transitions_by_byte,
                sparse_transitions_by_byte,
                reverse_transitions_by_byte,
                active_bitset,
                nodes,
                child,
                &next_states,
                depth + 1,
                state_seen,
                state_stamp,
                terminal_seen,
                terminal_stamp,
                class_seen,
                class_stamp,
                states_scanned,
                reached_states,
                finalizer_terminals_scanned,
                has_matched_terminal_by_state,
                allowed_class_by_terminal,
                allowed_follow_classes,
                matched_classes,
                state_buffers,
                source_seen_by_depth,
                source_stamps,
            );
        }
        next_states.clear();
        state_buffers[depth] = next_states;
    }
}

fn tokens_have_exact_active_l2p_boundary(
    tokenizer: &Tokenizer,
    bytesets: &SharedClassifyBytesets,
    flat_trans: &[u32],
    transitions_by_byte: &[u32],
    tokens: &[&[u8]],
    active_bitset: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    active_start_states: &[u32],
) -> Vec<bool> {
    let total_started_at = std::time::Instant::now();
    let trie_started_at = std::time::Instant::now();
    let mut nodes = vec![ExactBoundaryPrefixNode::default()];
    let total_token_bytes = tokens
        .iter()
        .map(|bytes| bytes.len().saturating_sub(1))
        .sum();
    let mut token_path_offsets = Vec::with_capacity(tokens.len() + 1);
    let mut token_paths = Vec::with_capacity(total_token_bytes);
    token_path_offsets.push(0);
    for &bytes in tokens {
        let mut node = 0usize;
        for &byte in bytes.iter().take(bytes.len().saturating_sub(1)) {
            node = insert_exact_boundary_prefix(&mut nodes, node, byte);
            token_paths.push(node);
        }
        token_path_offsets.push(token_paths.len());
    }
    let trie_ms = trie_started_at.elapsed().as_secs_f64() * 1000.0;

    let populate_started_at = std::time::Instant::now();
    let mut state_seen = vec![0u32; tokenizer.num_states() as usize];
    let mut terminal_seen = vec![0u32; active_bitset.len()];
    let mut state_stamp = 0u32;
    let mut terminal_stamp = 0u32;
    let mut class_stamp = 0u32;
    let mut states_scanned = 0usize;
    let mut reached_states = 0usize;
    let mut finalizer_terminals_scanned = 0usize;
    let mut allowed_follow_classes = vec![active_bitset.clone()];
    let mut blocked_class_cache = HashMap::<&BitSet, usize>::new();
    let allowed_class_by_terminal = (0..active_bitset.len())
        .map(|terminal| {
            if !active_bitset.contains(terminal) {
                return None;
            }
            let Some(blocked) = disallowed_follows
                .get(&(terminal as u32))
                .filter(|blocked| !blocked.is_empty())
            else {
                return Some(0);
            };
            Some(*blocked_class_cache.entry(blocked).or_insert_with(|| {
                let class = allowed_follow_classes.len();
                allowed_follow_classes.push(active_bitset.difference(blocked));
                class
            }))
        })
        .collect::<Vec<_>>();
    let mut class_seen = vec![0u32; allowed_follow_classes.len()];
    let mut matched_classes = Vec::with_capacity(allowed_follow_classes.len());
    let mut state_buffers = Vec::<Vec<u32>>::new();
    let mut source_seen_by_depth = Vec::<Vec<u32>>::new();
    let mut source_stamps = Vec::<u32>::new();
    populate_exact_boundary_prefixes(
        tokenizer,
        transitions_by_byte,
        &bytesets.sparse_transitions_by_byte,
        &bytesets.reverse_transitions_by_byte,
        active_bitset,
        &mut nodes,
        0,
        active_start_states,
        0,
        &mut state_seen,
        &mut state_stamp,
        &mut terminal_seen,
        &mut terminal_stamp,
        &mut class_seen,
        &mut class_stamp,
        &mut states_scanned,
        &mut reached_states,
        &mut finalizer_terminals_scanned,
        &bytesets.has_matched_terminal_by_state,
        &allowed_class_by_terminal,
        &allowed_follow_classes,
        &mut matched_classes,
        &mut state_buffers,
        &mut source_seen_by_depth,
        &mut source_stamps,
    );
    let populate_ms = populate_started_at.elapsed().as_secs_f64() * 1000.0;

    let evaluate_started_at = std::time::Instant::now();
    let mut prefix_allowed_checks = 0usize;
    let mut suffixes_evaluated = 0usize;
    let mut suffixes_with_terminals = 0usize;
    let result = tokens
        .iter()
        .enumerate()
        .map(|(token_index, &bytes)| {
            token_paths[token_path_offsets[token_index]..token_path_offsets[token_index + 1]]
                .iter()
                .copied()
                .enumerate()
                .any(|(split_after, node)| {
                    let Some(allowed) = nodes[node].allowed_follow_terminals.as_ref() else {
                        return false;
                    };
                    prefix_allowed_checks += 1;
                    suffixes_evaluated += 1;
                    let suffix_start = split_after + 1;
                    let mut state = tokenizer.initial_state_id();
                    let mut consumed_suffix = true;
                    for &byte in &bytes[suffix_start..] {
                        if tokenizer.possible_future_terminals(state).is_disjoint(allowed) {
                            consumed_suffix = false;
                            break;
                        }
                        let next = flat_trans[state as usize * 256 + byte as usize];
                        if next == u32::MAX {
                            consumed_suffix = false;
                            break;
                        }
                        state = next;
                        for terminal in tokenizer.matched_terminals_iter(state) {
                            if allowed.contains(terminal as usize) {
                                suffixes_with_terminals += 1;
                                return true;
                            }
                        }
                    }
                    if consumed_suffix {
                        let suffix_has_allowed_terminal =
                            !tokenizer.possible_future_terminals(state).is_disjoint(allowed);
                        suffixes_with_terminals += usize::from(suffix_has_allowed_terminal);
                        return suffix_has_allowed_terminal;
                    }
                    false
                })
        })
        .collect();
    let evaluate_ms = evaluate_started_at.elapsed().as_secs_f64() * 1000.0;
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][exact_boundary_batch] tokens={} nodes={} follow_classes={} states_scanned={} reached_states={} finalizer_terminals_scanned={} prefix_allowed_checks={} suffixes_evaluated={} suffixes_with_terminals={} trie_ms={:.3} populate_ms={:.3} evaluate_ms={:.3} total_ms={:.3}",
            tokens.len(),
            nodes.len(),
            allowed_follow_classes.len(),
            states_scanned,
            reached_states,
            finalizer_terminals_scanned,
            prefix_allowed_checks,
            suffixes_evaluated,
            suffixes_with_terminals,
            trie_ms,
            populate_ms,
            evaluate_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

pub(crate) fn split_vocab_for_active_l2p_terminals(
    tokenizer: &Tokenizer,
    flat_trans: &[u32],
    vocab: &Vocab,
    disallowed_follows: &Arc<BTreeMap<u32, BitSet>>,
    num_terminals: u32,
    active_terminals: &[bool],
    shared_classify_cache: Option<&SharedClassifyCache>,
) -> L2pVocabBoundarySplit {
    let split_started_at = std::time::Instant::now();
    let owned_bytesets: Option<SharedClassifyBytesets>;
    let bytesets = if let Some(cache) = shared_classify_cache {
        cache.get_or_init(|| SharedClassifyBytesets::build(tokenizer, num_terminals))
    } else {
        owned_bytesets = Some(SharedClassifyBytesets::build(tokenizer, num_terminals));
        owned_bytesets.as_ref().unwrap()
    };

    let active: Vec<usize> = active_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, active)| active.then_some(terminal))
        .collect();
    let mut active_bitset = BitSet::new(num_terminals as usize);
    for &terminal in &active {
        active_bitset.set(terminal);
    }
    let active_setup_started_at = std::time::Instant::now();
    let route_setup =
        active_l2p_route_setup(tokenizer, bytesets, &active_bitset, disallowed_follows);
    let active_setup_ms = active_setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let boundary_pairs_ms = 0.0;

    let vocab_scan_started_at = std::time::Instant::now();
    let mut boundary_token_ids = Vec::<u32>::new();
    let mut single_token_ids = Vec::<u32>::with_capacity(vocab.entries.len());
    let mut irrelevant_tokens = 0usize;
    let mut adjacent_entries = Vec::new();
    for (&token_id, bytes) in vocab.entries.iter() {
        match token_l2p_route_hint(
            bytes,
            &route_setup.allowed_boundary_pair_words,
            &route_setup.active_reachable_by_byte,
        ) {
            TokenL2pRouteHint::Adjacent => adjacent_entries.push((token_id, bytes.as_slice())),
            TokenL2pRouteHint::Single => single_token_ids.push(token_id),
            TokenL2pRouteHint::Irrelevant => irrelevant_tokens += 1,
        }
    }
    let vocab_scan_ms = vocab_scan_started_at.elapsed().as_secs_f64() * 1000.0;
    let adjacent_candidate_count = adjacent_entries.len();
    let estimated_exact_work = route_setup
        .active_start_states
        .len()
        .saturating_mul(adjacent_candidate_count);
    let use_exact_boundary_filter = match exact_l2p_boundary_filter_mode() {
        ExactL2pBoundaryFilterMode::Force(enabled) => enabled,
        ExactL2pBoundaryFilterMode::Auto => {
            estimated_exact_work <= exact_l2p_boundary_filter_work_limit()
        }
    };

    if std::env::var_os("GLRMASK_PROFILE_EXACT_L2P_BOUNDARY_FILTER").is_some() {
        eprintln!(
            "[glrmask/profile][exact_l2p_boundary_filter] vocab_tokens={} active_terminals={} active_start_states={} adjacent_candidates={} estimated_work={} enabled={}",
            vocab.entries.len(),
            active.len(),
            route_setup.active_start_states.len(),
            adjacent_candidate_count,
            estimated_exact_work,
            use_exact_boundary_filter,
        );
    }

    let exact_started_at = std::time::Instant::now();
    let mut exact_prefilter_ms = 0.0;
    let mut exact_batch_ms = 0.0;
    let mut exact_viable_tokens = 0usize;
    let exact_boundary_matches = use_exact_boundary_filter.then(|| {
        let prefilter_started_at = std::time::Instant::now();
        let active_words = active_bitset.words();
        let active_matched_by_state = (adjacent_entries.len()
            > (tokenizer.num_states() as usize).div_ceil(16))
        .then(|| build_active_matched_by_state(bytesets, &active_bitset));
        let viable_indices = adjacent_entries
            .iter()
            .enumerate()
            .filter_map(|(index, (_, bytes))| {
                token_has_active_terminal_suffix(
                    tokenizer,
                    bytesets,
                    flat_trans,
                    bytes,
                    active_words,
                    active_matched_by_state.as_deref(),
                    &route_setup.active_suffix_start_by_byte,
                )
                .then_some(index)
            })
            .collect::<Vec<_>>();
        exact_prefilter_ms = prefilter_started_at.elapsed().as_secs_f64() * 1000.0;
        exact_viable_tokens = viable_indices.len();
        let tokens = viable_indices
            .iter()
            .map(|&index| adjacent_entries[index].1)
            .collect::<Vec<_>>();
        let batch_started_at = std::time::Instant::now();
        let viable_matches = tokens_have_exact_active_l2p_boundary(
            tokenizer,
            bytesets,
            flat_trans,
            &bytesets.transitions_by_byte,
            &tokens,
            &active_bitset,
            disallowed_follows.as_ref(),
            &route_setup.active_start_states,
        );
        exact_batch_ms = batch_started_at.elapsed().as_secs_f64() * 1000.0;
        let mut matches = vec![false; adjacent_entries.len()];
        for (index, exact_match) in viable_indices.into_iter().zip(viable_matches) {
            matches[index] = exact_match;
        }
        matches
    });
    let exact_ms = exact_started_at.elapsed().as_secs_f64() * 1000.0;
    let finalize_started_at = std::time::Instant::now();
    let mut adjacent_single_token_ids = Vec::<u32>::new();
    for (adjacent_index, &(token_id, bytes)) in adjacent_entries.iter().enumerate() {
        let exact_match = exact_boundary_matches
            .as_ref()
            .map_or(true, |matches| matches[adjacent_index]);
        if exact_match {
            boundary_token_ids.push(token_id);
        } else if bytes
            .iter()
            .any(|&byte| route_setup.active_reachable_by_byte[byte as usize] != 0)
        {
            adjacent_single_token_ids.push(token_id);
        } else {
            irrelevant_tokens += 1;
        }
    }

    single_token_ids = merge_sorted_token_ids(single_token_ids, adjacent_single_token_ids);
    let adjacent_tokens = adjacent_entries.len();
    let boundary_tokens = boundary_token_ids.len();
    let single_tokens = single_token_ids.len();
    let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if super::types::compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][l2p_vocab_route] tokens={} active={} starts={} adjacent={} exact_viable={} setup_ms={:.3} pairs_ms={:.3} scan_ms={:.3} exact_prefilter_ms={:.3} exact_batch_ms={:.3} exact_ms={:.3} finalize_ms={:.3} total_ms={:.3}",
            vocab.entries.len(),
            active.len(),
            route_setup.active_start_states.len(),
            adjacent_entries.len(),
            exact_viable_tokens,
            active_setup_ms,
            boundary_pairs_ms,
            vocab_scan_ms,
            exact_prefilter_ms,
            exact_batch_ms,
            exact_ms,
            finalize_ms,
            split_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    L2pVocabBoundarySplit {
        boundary_token_ids,
        single_token_ids,
        adjacent_tokens,
        boundary_tokens,
        single_tokens,
        irrelevant_tokens,
    }
}

fn compute_partition_cost(
    cost_fn: L2pPartitionCostFn,
    l2p_terminals: usize,
    partition_size: usize,
) -> f64 {
    if l2p_terminals == 0 || partition_size == 0 {
        return 0.0;
    }

    let num_l2p = l2p_terminals as f64;
    let size = partition_size as f64;
    match cost_fn {
        L2pPartitionCostFn::Size => num_l2p * size,
        L2pPartitionCostFn::SizeLog => num_l2p * size.ln(),
        L2pPartitionCostFn::LogLog => num_l2p.ln() * size.ln(),
        L2pPartitionCostFn::UnionSize => num_l2p * size,
    }
}

fn partition_metric_count(
    cost_fn: L2pPartitionCostFn,
    intersection_count: usize,
    union_count: usize,
) -> usize {
    match cost_fn {
        L2pPartitionCostFn::UnionSize => union_count,
        L2pPartitionCostFn::Size
        | L2pPartitionCostFn::SizeLog
        | L2pPartitionCostFn::LogLog => intersection_count,
    }
}

fn objective_score(objective: L2pPartitionObjective, costs: &[f64]) -> f64 {
    match objective {
        L2pPartitionObjective::Max => costs.iter().copied().fold(0.0, f64::max),
        L2pPartitionObjective::Sum => costs.iter().sum(),
    }
}

fn compute_token_l2p_map(
    vocab: &Vocab,
    bytesets: &SharedClassifyBytesets,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
) -> BTreeMap<u32, BitSet> {
    let num_terminals = num_terminals as usize;
    let (byte_to_last, byte_to_first) =
        build_byte_terminal_reverse_index(bytesets, num_terminals);

    let mut token_l2p_map = BTreeMap::<u32, BitSet>::new();
    for (&token_id, bytes) in vocab.entries.iter() {
        token_l2p_map.insert(
            token_id,
            token_l2p_terminals(
                bytes,
                &byte_to_last,
                &byte_to_first,
                disallowed_follows,
                num_terminals,
            ),
        );
    }
    token_l2p_map
}

pub(crate) fn partition_vocab_char_type_tokens(vocab: &Vocab) -> Vec<Vec<u32>> {
    let mut partitions: Vec<Vec<u32>> = (0..9).map(|_| Vec::new()).collect();
    for (&token_id, bytes) in vocab.entries.iter() {
        let idx = classify_vocab_char_type(bytes) as usize;
        partitions[idx].push(token_id);
    }
    partitions
}

pub(crate) fn estimate_l2p_objective_for_token_partitions(
    token_partitions: &[Vec<u32>],
    token_l2p_map: &BTreeMap<u32, BitSet>,
    cost_fn: L2pPartitionCostFn,
    objective: L2pPartitionObjective,
) -> (Vec<f64>, Vec<usize>, f64) {
    let mut costs = Vec::with_capacity(token_partitions.len());
    let mut l2p_counts = Vec::with_capacity(token_partitions.len());

    for token_ids in token_partitions {
        if token_ids.is_empty() {
            costs.push(0.0);
            l2p_counts.push(0);
            continue;
        }

        let mut intersection: Option<BitSet> = None;
        let mut union: Option<BitSet> = None;
        for &token_id in token_ids {
            if let Some(token_l2p) = token_l2p_map.get(&token_id) {
                if let Some(current) = &mut intersection {
                    current.intersect_with(token_l2p);
                } else {
                    intersection = Some(token_l2p.clone());
                }
                if let Some(current) = &mut union {
                    current.union_with(token_l2p);
                } else {
                    union = Some(token_l2p.clone());
                }
            }
        }

        let l2p_count = intersection.as_ref().map_or(0, BitSet::count_ones);
        let union_count = union.as_ref().map_or(0, BitSet::count_ones);
        l2p_counts.push(l2p_count);
        costs.push(compute_partition_cost(
            cost_fn,
            partition_metric_count(cost_fn, l2p_count, union_count),
            token_ids.len(),
        ));
    }

    let score = objective_score(objective, &costs);
    (costs, l2p_counts, score)
}

fn partition_token_l2p_map_by_cost(
    token_l2p_map: &BTreeMap<u32, BitSet>,
    num_partitions: usize,
    cost_fn: L2pPartitionCostFn,
    objective: L2pPartitionObjective,
) -> L2pCostPartitioning {
    let mut grouped_index = BTreeMap::<Vec<u64>, usize>::new();
    let mut groups: Vec<L2pTokenGroup> = Vec::new();

    for (&token_id, l2p_terminals) in token_l2p_map {
        let key = l2p_terminals.words().to_vec();
        if let Some(&group_idx) = grouped_index.get(&key) {
            groups[group_idx].token_ids.push(token_id);
        } else {
            let group_idx = groups.len();
            grouped_index.insert(key, group_idx);
            groups.push(L2pTokenGroup {
                l2p_terminals: l2p_terminals.clone(),
                token_ids: vec![token_id],
            });
        }
    }

    groups.sort_by(|left, right| {
        let left_weight = left.l2p_terminals.count_ones() * left.token_ids.len();
        let right_weight = right.l2p_terminals.count_ones() * right.token_ids.len();
        right_weight
            .cmp(&left_weight)
            .then_with(|| right.l2p_terminals.count_ones().cmp(&left.l2p_terminals.count_ones()))
            .then_with(|| right.token_ids.len().cmp(&left.token_ids.len()))
    });

    let mut buckets = vec![L2pPartitionBucket::new(); num_partitions.max(1)];
    let mut current_costs = vec![0.0; buckets.len()];

    for group in groups {
        let mut best_idx = 0usize;
        let mut best_score = f64::INFINITY;
        let mut best_cost = f64::INFINITY;
        let mut best_l2p_count = usize::MAX;
        let mut best_size = usize::MAX;

        for (idx, bucket) in buckets.iter().enumerate() {
            let candidate_intersection = if let Some(current) = &bucket.l2p_intersection {
                current.intersection(&group.l2p_terminals)
            } else {
                group.l2p_terminals.clone()
            };
            let candidate_union = if let Some(current) = &bucket.l2p_union {
                current.union(&group.l2p_terminals)
            } else {
                group.l2p_terminals.clone()
            };
            let candidate_l2p_count = candidate_intersection.count_ones();
            let candidate_union_count = candidate_union.count_ones();
            let candidate_size = bucket.size() + group.token_ids.len();
            let candidate_cost = compute_partition_cost(
                cost_fn,
                partition_metric_count(cost_fn, candidate_l2p_count, candidate_union_count),
                candidate_size,
            );

            let mut candidate_costs = current_costs.clone();
            candidate_costs[idx] = candidate_cost;
            let score = objective_score(objective, &candidate_costs);

            let better = score < best_score
                || (score == best_score
                    && (candidate_cost < best_cost
                        || (candidate_cost == best_cost
                            && (candidate_l2p_count < best_l2p_count
                                || (candidate_l2p_count == best_l2p_count
                                    && candidate_size < best_size)))));
            if better {
                best_idx = idx;
                best_score = score;
                best_cost = candidate_cost;
                best_l2p_count = candidate_l2p_count;
                best_size = candidate_size;
            }
        }

        let bucket = &mut buckets[best_idx];
        if let Some(current) = &mut bucket.l2p_intersection {
            current.intersect_with(&group.l2p_terminals);
        } else {
            bucket.l2p_intersection = Some(group.l2p_terminals.clone());
        }
        if let Some(current) = &mut bucket.l2p_union {
            current.union_with(&group.l2p_terminals);
        } else {
            bucket.l2p_union = Some(group.l2p_terminals.clone());
        }
        bucket.token_ids.extend(group.token_ids);
        current_costs[best_idx] = best_cost;
    }

    buckets.sort_by(|left, right| right.size().cmp(&left.size()));

    let estimated_partition_costs = buckets
        .iter()
        .map(|bucket| {
            let union_count = bucket.l2p_union.as_ref().map_or(0, BitSet::count_ones);
            compute_partition_cost(
                cost_fn,
                partition_metric_count(cost_fn, bucket.l2p_count(), union_count),
                bucket.size(),
            )
        })
        .collect::<Vec<_>>();
    let estimated_l2p_terminals = buckets
        .iter()
        .map(L2pPartitionBucket::l2p_count)
        .collect::<Vec<_>>();
    let partitions = buckets
        .into_iter()
        .map(|bucket| bucket.token_ids)
        .collect::<Vec<_>>();

    L2pCostPartitioning {
        objective_score: objective_score(objective, &estimated_partition_costs),
        partitions,
        estimated_partition_costs,
        estimated_l2p_terminals,
    }
}

pub(crate) fn partition_vocab_by_l2p_cost_with_token_map(
    vocab: &Vocab,
    bytesets: &SharedClassifyBytesets,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    num_partitions: usize,
    cost_fn: L2pPartitionCostFn,
    objective: L2pPartitionObjective,
) -> (L2pCostPartitioning, BTreeMap<u32, BitSet>) {
    let token_l2p_map = compute_token_l2p_map(vocab, bytesets, disallowed_follows, num_terminals);
    let partitioning =
        partition_token_l2p_map_by_cost(&token_l2p_map, num_partitions, cost_fn, objective);
    (partitioning, token_l2p_map)
}

pub(crate) fn partition_vocab_by_l2p_cost(
    vocab: &Vocab,
    bytesets: &SharedClassifyBytesets,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    num_partitions: usize,
    cost_fn: L2pPartitionCostFn,
    objective: L2pPartitionObjective,
) -> L2pCostPartitioning {
    partition_vocab_by_l2p_cost_with_token_map(
        vocab,
        bytesets,
        disallowed_follows,
        num_terminals,
        num_partitions,
        cost_fn,
        objective,
    )
    .0
}


#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};
    use std::sync::{Arc, Mutex};

    use super::{
        build_active_suffix_start_by_byte, build_reverse_transitions_by_byte,
        token_has_active_terminal_suffix,
    };
    use super::{
        classify_vocab_char_type, parse_exact_l2p_boundary_filter_mode,
        ExactL2pBoundaryFilterMode, SharedClassifyBytesets,
        TokenL2pRouteHint, state_future_intersects_words,
        token_has_active_l2p_boundary_words, token_l2p_route_hint,
    };
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::automata::lexer::Lexer;
    use crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table;
    use crate::ds::bitset::BitSet;
    use crate::ds::u8set::U8Set;

    fn reference_bytesets(tokenizer: &Tokenizer, num_terminals: u32) -> SharedClassifyBytesets {
        let nt = num_terminals as usize;
        let initial = tokenizer.start_state();
        let mut reachable_bytes = vec![U8Set::empty(); nt];
        let mut last_bytes = vec![U8Set::empty(); nt];

        for state in 0..tokenizer.num_states() {
            for (byte, target) in tokenizer.transitions_from(state) {
                for terminal in tokenizer.matched_terminal_bitset(target).iter() {
                    if terminal < nt {
                        reachable_bytes[terminal].insert(byte);
                        last_bytes[terminal].insert(byte);
                    }
                }
                for terminal in tokenizer.possible_future_terminals(target).iter() {
                    if terminal < nt {
                        reachable_bytes[terminal].insert(byte);
                    }
                }
            }
        }

        let mut first_bytes = vec![U8Set::empty(); nt];
        for (byte, target) in tokenizer.transitions_from(initial) {
            for terminal in tokenizer.matched_terminal_bitset(target).iter() {
                if terminal < nt {
                    first_bytes[terminal].insert(byte);
                }
            }
            for terminal in tokenizer.possible_future_terminals(target).iter() {
                if terminal < nt {
                    first_bytes[terminal].insert(byte);
                }
            }
        }

        SharedClassifyBytesets {
            reachable_bytes,
            first_bytes,
            last_bytes,
            transitions_by_byte: Vec::new(),
            sparse_transitions_by_byte: Vec::new(),
            reverse_transitions_by_byte: Vec::new(),
            matched_terminals_by_state: Arc::from(Vec::<Box<[u32]>>::new()),
            future_terminals_by_state: Arc::from(Vec::<Box<[u32]>>::new()),
            matched_states_by_terminal: Arc::from(Vec::<Vec<u32>>::new()),
            future_states_by_terminal: Arc::from(Vec::<Vec<u32>>::new()),
            has_matched_terminal_by_state: Vec::new(),
            future_by_state_words: Vec::new(),
            representative_future_terminal_by_state: Vec::new(),
            words_per_terminal_set: 0,
            active_route_setup_cache: Mutex::new(HashMap::new()),
        }
    }

    #[test]
    fn parse_exact_l2p_boundary_filter_mode_accepts_auto_and_forced_values() {
        assert!(matches!(parse_exact_l2p_boundary_filter_mode(""), ExactL2pBoundaryFilterMode::Auto));
        assert!(matches!(parse_exact_l2p_boundary_filter_mode("auto"), ExactL2pBoundaryFilterMode::Auto));
        assert!(matches!(parse_exact_l2p_boundary_filter_mode("1"), ExactL2pBoundaryFilterMode::Force(true)));
        assert!(matches!(parse_exact_l2p_boundary_filter_mode("true"), ExactL2pBoundaryFilterMode::Force(true)));
        assert!(matches!(parse_exact_l2p_boundary_filter_mode("on"), ExactL2pBoundaryFilterMode::Force(true)));
        assert!(matches!(parse_exact_l2p_boundary_filter_mode("0"), ExactL2pBoundaryFilterMode::Force(false)));
        assert!(matches!(parse_exact_l2p_boundary_filter_mode("false"), ExactL2pBoundaryFilterMode::Force(false)));
        assert!(matches!(parse_exact_l2p_boundary_filter_mode("off"), ExactL2pBoundaryFilterMode::Force(false)));
    }

    #[test]
    fn underscore_is_alphabetic_for_vocab_partitioning() {
        // `_` is treated like an ASCII alphabetic byte only for partition
        // routing. It must keep identifier-style tokens out of punctuation
        // partitions, including the quoted structural-boundary route.
        for bytes in [b"_".as_slice(), b" _", b"snake_case", b"123_456", b"__"] {
            assert_eq!(classify_vocab_char_type(bytes), 2, "bytes={bytes:?}");
        }
        assert_eq!(classify_vocab_char_type(b"123"), 3);
        assert_eq!(classify_vocab_char_type(b"_!"), 1);
        assert_eq!(classify_vocab_char_type(b"\"_field"), 8);
    }

    #[test]
    fn structural_boundary_lexical_tokens_split_literal_and_quoted_identifier_routes() {
        for bytes in [
            b" true".as_slice(),
            b" nullptr".as_slice(),
            b"[n".as_slice(),
            b" -".as_slice(),
        ] {
            assert_eq!(classify_vocab_char_type(bytes), 7, "bytes={bytes:?}");
        }
        for bytes in [
            b"t".as_slice(),
            b"true".as_slice(),
            b"falsehood".as_slice(),
            b"nullable".as_slice(),
        ] {
            assert_eq!(classify_vocab_char_type(bytes), 2, "bytes={bytes:?}");
        }
        for bytes in [
            b"\"name".as_slice(),
            b"\"_field".as_slice(),
            b"\"This".as_slice(),
        ] {
            assert_eq!(classify_vocab_char_type(bytes), 8, "bytes={bytes:?}");
        }
    }

    #[test]
    fn combined_l2p_route_hint_matches_two_pass_route_scan() {
        let mut active_reachable = U8Set::empty();
        active_reachable.insert(b'a');
        active_reachable.insert(b'z');
        let mut active_reachable_by_byte = [0u8; 256];
        for byte in active_reachable.iter() {
            active_reachable_by_byte[byte as usize] = 1;
        }
        let mut pairs = [0u64; 1024];
        let pair_index = ((b'x' as usize) << 8) | b'y' as usize;
        pairs[pair_index >> 6] |= 1u64 << (pair_index & 63);

        for bytes in [b"".as_slice(), b"a", b"q", b"xy", b"xya", b"qz", b"ax"] {
            let expected = if token_has_active_l2p_boundary_words(bytes, &pairs) {
                TokenL2pRouteHint::Adjacent
            } else if bytes.iter().any(|&byte| active_reachable.contains(byte)) {
                TokenL2pRouteHint::Single
            } else {
                TokenL2pRouteHint::Irrelevant
            };
            assert_eq!(
                token_l2p_route_hint(bytes, &pairs, &active_reachable_by_byte),
                expected,
                "bytes={bytes:?}"
            );
        }
    }

    #[test]
    fn reverse_byte_transition_index_preserves_frontier_targets() {
        let by_byte = vec![
            vec![(0, 2), (2, 2), (3, 5), (4, 2), (5, 5)],
            vec![(1, 4), (2, 4)],
        ];
        let reverse = build_reverse_transitions_by_byte(&by_byte, 6);

        assert_eq!(reverse[0].targets, vec![2, 5]);
        assert_eq!(reverse[0].source_offsets, vec![0, 3, 5]);
        assert_eq!(reverse[0].sources, vec![0, 2, 4, 3, 5]);

        let frontier = BTreeSet::from([2u32, 3, 5]);
        let direct_targets = by_byte[0]
            .iter()
            .filter_map(|&(source, target)| frontier.contains(&source).then_some(target))
            .collect::<BTreeSet<_>>();
        let reverse_targets = reverse[0]
            .targets
            .iter()
            .enumerate()
            .filter_map(|(index, &target)| {
                let start = reverse[0].source_offsets[index] as usize;
                let end = reverse[0].source_offsets[index + 1] as usize;
                reverse[0].sources[start..end]
                    .iter()
                    .any(|source| frontier.contains(source))
                    .then_some(target)
            })
            .collect::<BTreeSet<_>>();

        assert_eq!(reverse_targets, direct_targets);
    }

    #[test]
    fn byte_bucket_bytesets_match_reference_transition_scan() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"ab".to_vec()),
            Expr::Choice(vec![Expr::U8Seq(b"ac".to_vec()), Expr::U8Seq(b"b".to_vec())]),
            Expr::Seq(vec![
                Expr::U8Class(U8Set::from_bytes(b"xy")),
                Expr::Shared(Arc::new(Expr::U8Seq(b"z".to_vec()))),
            ]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        let actual = SharedClassifyBytesets::build(&tokenizer, tokenizer.num_terminals());
        let expected = reference_bytesets(&tokenizer, tokenizer.num_terminals());
        assert_eq!(actual.reachable_bytes, expected.reachable_bytes);
        assert_eq!(actual.first_bytes, expected.first_bytes);
        assert_eq!(actual.last_bytes, expected.last_bytes);
        for state in 0..tokenizer.num_states() {
            assert_eq!(
                actual.has_matched_terminal_by_state[state as usize] != 0,
                tokenizer
                    .matched_terminals_iter(state)
                    .any(|terminal| terminal < tokenizer.num_terminals()),
                "state={state}"
            );
        }

        let mut active_sets = vec![BitSet::new(tokenizer.num_terminals() as usize)];
        let mut all_active = BitSet::new(tokenizer.num_terminals() as usize);
        for terminal in 0..tokenizer.num_terminals() as usize {
            all_active.set(terminal);
            let mut single = BitSet::new(tokenizer.num_terminals() as usize);
            single.set(terminal);
            active_sets.push(single);
        }
        active_sets.push(all_active);
        let flat_trans = build_flat_transition_table(&tokenizer);
        let unrestricted_suffix_start = [1u8; 256];
        for active in &active_sets {
            let active_suffix_start =
                build_active_suffix_start_by_byte(&tokenizer, &actual, active.words());
            for bytes in [
                b"".as_slice(),
                b"a",
                b"ab",
                b"ac",
                b"xy",
                b"xyz",
                b"zz",
                b"bxyz",
            ] {
                assert_eq!(
                    token_has_active_terminal_suffix(
                        &tokenizer,
                        &actual,
                        &flat_trans,
                        bytes,
                        active.words(),
                        None,
                        &active_suffix_start,
                    ),
                    token_has_active_terminal_suffix(
                        &tokenizer,
                        &actual,
                        &flat_trans,
                        bytes,
                        active.words(),
                        None,
                        &unrestricted_suffix_start,
                    ),
                    "active={active:?} bytes={bytes:?}"
                );
            }
            for state in 0..tokenizer.num_states() {
                let representative = actual.representative_future_terminal_by_state[state as usize];
                let future = tokenizer.possible_future_terminals(state);
                assert_eq!(representative == u32::MAX, future.is_empty());
                if representative != u32::MAX {
                    assert!(future.contains(representative as usize));
                }
                let full = state_future_intersects_words(&actual, state, active.words());
                let fast = representative != u32::MAX && active.contains(representative as usize)
                    || (representative == u32::MAX || !active.contains(representative as usize))
                        && full;
                assert_eq!(fast, full, "state={state}");
            }
        }
    }
}
