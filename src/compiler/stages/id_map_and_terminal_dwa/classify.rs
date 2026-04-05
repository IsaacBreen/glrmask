//! Vocab and terminal classification utilities.

use std::collections::BTreeMap;

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
        let dfa_states = tokenizer.dfa.states();

        // Single parallel pass: compute reachable_bytes and last_bytes together.
        // reachable_bytes[t] = all bytes b where some transition (_, b, target) exists
        //   and target has terminal t in finalizers or possible_future_group_ids.
        // last_bytes[t] = all bytes b where some transition (_, b, target) exists
        //   and target has terminal t in finalizers.
        let (reachable_bytes, last_bytes) = dfa_states
            .par_iter()
            .fold(
                || (vec![U8Set::empty(); nt], vec![U8Set::empty(); nt]),
                |(mut reachable, mut last), state| {
                    for (byte, target) in state.transitions.iter() {
                        let target = *target;
                        let finalizers = tokenizer.dfa.finalizers(target);
                        let futures = tokenizer.dfa.possible_future_group_ids(target);
                        for t in finalizers.iter() {
                            if t < nt {
                                reachable[t].insert(byte);
                                last[t].insert(byte);
                            }
                        }
                        for t in futures.iter() {
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
        for (byte, target) in dfa_states[initial as usize].transitions.iter() {
            let target = *target;
            let finalizers = tokenizer.dfa.finalizers(target);
            let futures = tokenizer.dfa.possible_future_group_ids(target);
            for t in finalizers.iter().chain(futures.iter()) {
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
/// - 5: non-alnum auxiliary (no JSON structural, or single-char repeated,
///       or length 1)
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
    // Try to decode as UTF-8 for Unicode-aware classification.
    if let Ok(s) = std::str::from_utf8(content) {
        let all_alnum = s.chars().all(|c| c.is_alphanumeric());
        if all_alnum {
            let has_alpha = s.chars().any(|c| c.is_alphabetic());
            if has_alpha {
                // Distinguish ASCII-only alpha from Unicode alpha.
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

/// Sub-classify a non-alphanumeric token into P0 (structural) or P5 (auxiliary).
///
/// P5 if: (a) no JSON structural char, (b) single repeated char from
/// `\n:{ ,`, or (c) length 1.
fn classify_nonalnum(bytes: &[u8]) -> u8 {
    // Length 1 → P5
    if bytes.len() <= 1 {
        return 5;
    }
    // Single repeated char from P5_REPEATED_CHARS → P5
    if bytes.iter().all(|b| *b == bytes[0]) && P5_REPEATED_CHARS.contains(&bytes[0]) {
        return 5;
    }
    // No JSON structural char → P5
    if !bytes.iter().any(|b| JSON_STRUCTURAL.contains(b)) {
        return 5;
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
    let mut vocab_bytes = U8Set::empty();
    for bytes in vocab.entries.values() {
        for &b in bytes {
            vocab_bytes.insert(b);
        }
    }

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

    // Debug: collect the actual contributing pairs
    let debug_profile = super::types::debug_profile_enabled();
    let debug_verbose = super::types::debug_verbose_enabled();
    let collect_debug_pairs = debug_profile || debug_verbose;
    let mut l2p_pairs: Vec<(usize, usize)> = Vec::new();
    let mut all_l2p_pairs: Vec<(usize, usize)> = Vec::new();

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
            if collect_debug_pairs {
                all_l2p_pairs.push((t1, t2));
                if !is_two_plus.contains(t1) {
                    l2p_pairs.push((t1, t2));
                }
            }
            is_two_plus.set(t1);
            is_two_plus.set(t2);
        }
    }

    if debug_profile {
        let two_plus_count = (0..nt).filter(|&t| is_two_plus.contains(t)).count();
        eprintln!(
            "[glrmask/debug][classify_l2p] terminals_two_plus={} first_hit_pairs={} all_pairs={}",
            two_plus_count,
            l2p_pairs.len(),
            all_l2p_pairs.len(),
        );
    }

    if debug_verbose {
        // Dump L2+ terminals with their byte overlaps
        for t in 0..nt {
            if is_two_plus.contains(t) {
                let fb_overlap: Vec<u8> = (0..=255u8).filter(|b| first_bytes[t].contains(*b) && vocab_bytes.contains(*b)).collect();
                let lb_overlap: Vec<u8> = (0..=255u8).filter(|b| last_bytes[t].contains(*b) && vocab_bytes.contains(*b)).collect();
                let fb_str: String = fb_overlap.iter().map(|b| {
                    if b.is_ascii_graphic() || *b == b' ' { format!("{}", *b as char) } else { format!("\\x{:02x}", b) }
                }).collect::<Vec<_>>().join("");
                let lb_str: String = lb_overlap.iter().map(|b| {
                    if b.is_ascii_graphic() || *b == b' ' { format!("{}", *b as char) } else { format!("\\x{:02x}", b) }
                }).collect::<Vec<_>>().join("");
                eprintln!("[glrmask/debug][classify_l2p] terminal={} first_bytes_overlap=[{}] last_bytes_overlap=[{}]", t, fb_str, lb_str);
            }
        }
        // Dump the contributing pairs (first occurrence for each t1)
        for (t1, t2) in &l2p_pairs {
            let fb2: Vec<u8> = (0..=255u8).filter(|b| first_bytes[*t2].contains(*b) && vocab_bytes.contains(*b)).collect();
            let lb1: Vec<u8> = (0..=255u8).filter(|b| last_bytes[*t1].contains(*b) && vocab_bytes.contains(*b)).collect();
            let fb2_str: String = fb2.iter().map(|b| {
                if b.is_ascii_graphic() || *b == b' ' { format!("{}", *b as char) } else { format!("\\x{:02x}", b) }
            }).collect::<Vec<_>>().join("");
            let lb1_str: String = lb1.iter().map(|b| {
                if b.is_ascii_graphic() || *b == b' ' { format!("{}", *b as char) } else { format!("\\x{:02x}", b) }
            }).collect::<Vec<_>>().join("");
            eprintln!("[glrmask/debug][classify_l2p_pair] t1={} t2={} last_bytes_t1=[{}] first_bytes_t2=[{}]", t1, t2, lb1_str, fb2_str);
        }
        // Dump ALL contributing pairs (not just first per t1)
        for (t1, t2) in &all_l2p_pairs {
            let fb2: Vec<u8> = (0..=255u8).filter(|b| first_bytes[*t2].contains(*b) && vocab_bytes.contains(*b)).collect();
            let lb1: Vec<u8> = (0..=255u8).filter(|b| last_bytes[*t1].contains(*b) && vocab_bytes.contains(*b)).collect();
            let fb2_str: String = fb2.iter().map(|b| {
                if b.is_ascii_graphic() || *b == b' ' { format!("{}", *b as char) } else { format!("\\x{:02x}", b) }
            }).collect::<Vec<_>>().join("");
            let lb1_str: String = lb1.iter().map(|b| {
                if b.is_ascii_graphic() || *b == b' ' { format!("{}", *b as char) } else { format!("\\x{:02x}", b) }
            }).collect::<Vec<_>>().join("");
            eprintln!("[glrmask/debug][classify_l2p_pair_all] t1={} t2={} last_bytes_t1=[{}] first_bytes_t2=[{}]", t1, t2, lb1_str, fb2_str);
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
