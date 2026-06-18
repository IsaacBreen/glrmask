//! Vocab and terminal classification utilities.

use crate::automata::lexer::Lexer;
use std::collections::{BTreeMap, HashMap};

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
}

/// Cache type for lazy `SharedClassifyBytesets` initialization across partitions.
pub type SharedClassifyCache = std::sync::OnceLock<SharedClassifyBytesets>;

pub(crate) struct L2pVocabBoundarySplit {
    pub(crate) boundary_vocab: Vocab,
    pub(crate) single_vocab: Vocab,
    pub(crate) adjacent_tokens: usize,
    pub(crate) boundary_tokens: usize,
    pub(crate) single_tokens: usize,
    pub(crate) irrelevant_tokens: usize,
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
    /// Scan the DFA to compute per-terminal byte sets.
    ///
    /// Merges the reachable_bytes and last_bytes scans into a single parallel
    /// pass over all DFA transitions, eliminating the intermediate
    /// per-state incoming_bytes array.
    pub fn build(tokenizer: &Tokenizer, num_terminals: u32) -> Self {
        use rayon::prelude::*;

        let nt = num_terminals as usize;
        let initial = tokenizer.start_state();

        // Single parallel pass: compute reachable_bytes and last_bytes together.
        // reachable_bytes[t] = all bytes b where some transition (_, b, target) exists
        //   and target has terminal t in finalizers or possible_future_group_ids.
        // last_bytes[t] = all bytes b where some transition (_, b, target) exists
        //   and target has terminal t in finalizers.
        let (reachable_bytes, last_bytes) = (0..tokenizer.num_states())
            .into_par_iter()
            .fold(
                || (vec![U8Set::empty(); nt], vec![U8Set::empty(); nt]),
                |(mut reachable, mut last), state| {
                    for (byte, target) in tokenizer.transitions_from(state) {
                        for terminal in tokenizer.matched_terminals_iter(target) {
                            let t = terminal as usize;
                            if t < nt {
                                reachable[t].insert(byte);
                                last[t].insert(byte);
                            }
                        }
                        for terminal in tokenizer.possible_future_terminals_iter(target) {
                            let t = terminal as usize;
                            if t < nt {
                                reachable[t].insert(byte);
                            }
                        }
                    }
                    (reachable, last)
                },
            )
            .reduce(
                || (vec![U8Set::empty(); nt], vec![U8Set::empty(); nt]),
                |(mut ra, mut la), (rb, lb)| {
                    for i in 0..nt {
                        ra[i] = ra[i].union(&rb[i]);
                        la[i] = la[i].union(&lb[i]);
                    }
                    (ra, la)
                },
            );

        // first_bytes: only from initial state (single state, no parallelism needed).
        let mut first_bytes = vec![U8Set::empty(); nt];
        for (byte, target) in tokenizer.transitions_from(initial) {
            for terminal in tokenizer
                .matched_terminals_iter(target)
                .chain(tokenizer.possible_future_terminals_iter(target))
            {
                let t = terminal as usize;
                if t < nt {
                    first_bytes[t].insert(byte);
                }
            }
        }

        SharedClassifyBytesets {
            reachable_bytes,
            first_bytes,
            last_bytes,
        }
    }
}

/// JSON structural characters used to keep tokens in the core non-alnum
/// partition (P0) rather than splitting them into the auxiliary P5.
const JSON_STRUCTURAL: &[u8] = b"\":[]{},";

/// Characters whose sole repetition qualifies a non-alnum token for the
/// auxiliary P5 partition even if the token contains a structural byte.
const P5_REPEATED_CHARS: &[u8] = b"\n:{ ,";

/// Classifies a token's bytes by character type for vocab partitioning.
///
/// Returns:
/// - 0: non-alnum with JSON structural chars (multi-byte, not single-repeated)
/// - 1: mixed (contains both alnum and non-alnum)
/// - 2: ASCII alnum with ≥1 alpha, optionally with leading space
/// - 3: pure digit, optionally with leading space
/// - 4: Unicode-only alpha (non-ASCII alphanumeric, e.g. CJK, Cyrillic,
///       Arabic, Hangul), optionally with leading space
/// - 5: non-alnum auxiliary short (no JSON structural, or single-char repeated,
///       or length 1; ≤ 8 bytes)
/// - 6: non-alnum auxiliary long (same criteria as 5, but > 8 bytes)
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
        let all_alnum = s.chars().all(|c| c.is_alphanumeric());
        if all_alnum {
            let has_alpha = s.chars().any(|c| c.is_alphabetic());
            if has_alpha {
                let has_ascii_alpha = content.iter().any(|b| b.is_ascii_alphabetic());
                if has_ascii_alpha {
                    return 2; // ASCII alpha (may also contain non-ASCII alpha)
                }
                return 4; // Unicode-only alpha (CJK, Cyrillic, Arabic, etc.)
            }
            return 3; // Pure digit
        }
        // Check non-alphanumeric.
        if let Ok(full) = std::str::from_utf8(bytes) {
            if full.chars().all(|c| !c.is_alphanumeric()) {
                return classify_nonalnum(bytes);
            }
        }
        return 1; // Mixed
    }
    // Fallback: byte-level ASCII checks for invalid UTF-8.
    if content.iter().all(|b| b.is_ascii_alphanumeric()) {
        if content.iter().any(|b| b.is_ascii_alphabetic()) {
            return 2;
        }
        return 3;
    }
    if bytes.iter().all(|b| !b.is_ascii_alphanumeric()) {
        return classify_nonalnum(bytes);
    }
    1 // Mixed
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

fn exact_l2p_boundary_filter_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GLRMASK_EXACT_L2P_BOUNDARY_FILTER")
            .map(|value| {
                let trimmed = value.trim();
                trimmed.is_empty() || trimmed == "1" || trimmed.eq_ignore_ascii_case("true")
            })
            .unwrap_or(true)
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

pub(crate) fn split_vocab_for_active_l2p_terminals(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    num_terminals: u32,
    active_terminals: &[bool],
    shared_classify_cache: Option<&SharedClassifyCache>,
) -> L2pVocabBoundarySplit {
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
    let active_start_states: Vec<u32> = (0..tokenizer.num_states())
        .filter(|&state| {
            !tokenizer
                .possible_future_terminals(state)
                .is_disjoint(&active_bitset)
        })
        .collect();
    let mut allowed_boundary_pairs = [U8Set::empty(); 256];
    let mut active_reachable = U8Set::empty();

    for &terminal in &active {
        active_reachable = active_reachable.union(&bytesets.reachable_bytes[terminal]);
    }

    for &terminal_1 in &active {
        let last_bytes = bytesets.last_bytes[terminal_1];
        if last_bytes.is_empty() {
            continue;
        }
        let disallowed = disallowed_follows.get(&(terminal_1 as u32));
        for &terminal_2 in &active {
            if disallowed.map_or(false, |blocked| blocked.contains(terminal_2)) {
                continue;
            }
            let first_bytes = bytesets.first_bytes[terminal_2];
            if first_bytes.is_empty() {
                continue;
            }
            for last_byte in last_bytes.iter() {
                for first_byte in first_bytes.iter() {
                    allowed_boundary_pairs[last_byte as usize].insert(first_byte);
                }
            }
        }
    }

    let mut boundary_entries = Vec::<(u32, Vec<u8>)>::new();
    let mut single_entries = Vec::<(u32, Vec<u8>)>::new();
    let mut adjacent_tokens = 0usize;
    let mut irrelevant_tokens = 0usize;
    let use_exact_boundary_filter = exact_l2p_boundary_filter_enabled();

    for (&token_id, bytes) in vocab.entries.iter() {
        if token_has_active_l2p_boundary(bytes, &allowed_boundary_pairs) {
            adjacent_tokens += 1;
            if !use_exact_boundary_filter
                || token_has_exact_active_l2p_boundary(
                tokenizer,
                bytes,
                &active_bitset,
                disallowed_follows,
                &active_start_states,
            ) {
                boundary_entries.push((token_id, bytes.clone()));
            } else if bytes.iter().any(|&byte| active_reachable.contains(byte)) {
                single_entries.push((token_id, bytes.clone()));
            } else {
                irrelevant_tokens += 1;
            }
            continue;
        }

        if bytes.iter().any(|&byte| active_reachable.contains(byte)) {
            single_entries.push((token_id, bytes.clone()));
        } else {
            irrelevant_tokens += 1;
        }
    }

    let boundary_tokens = boundary_entries.len();
    let single_tokens = single_entries.len();
    L2pVocabBoundarySplit {
        boundary_vocab: Vocab::new(boundary_entries, None),
        single_vocab: Vocab::new(single_entries, None),
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
    let mut partitions: Vec<Vec<u32>> = (0..7).map(|_| Vec::new()).collect();
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
