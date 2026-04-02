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
    pub fn build(tokenizer: &Tokenizer, num_terminals: u32) -> Self {
        let nt = num_terminals as usize;
        let initial = tokenizer.start_state();
        let dfa_states = tokenizer.dfa.states();
        let num_states = tokenizer.num_states() as usize;

        let mut reachable_bytes = vec![U8Set::empty(); nt];
        for state in dfa_states {
            for (byte, target) in state.transitions.iter() {
                let target = *target;
                let finalizers = tokenizer.dfa.finalizers(target);
                let futures = tokenizer.dfa.possible_future_group_ids(target);
                for t in finalizers.iter().chain(futures.iter()) {
                    if t < nt {
                        reachable_bytes[t].insert(byte);
                    }
                }
            }
        }

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

        let mut incoming_bytes = vec![U8Set::empty(); num_states];
        for state in dfa_states {
            for (byte, target) in state.transitions.iter() {
                incoming_bytes[*target as usize].insert(byte);
            }
        }

        let mut last_bytes = vec![U8Set::empty(); nt];
        for (state_idx, _) in dfa_states.iter().enumerate() {
            let finalizers = tokenizer.dfa.finalizers(state_idx as u32);
            for t in finalizers.iter() {
                if t < nt {
                    last_bytes[t] = last_bytes[t].union(&incoming_bytes[state_idx]);
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

/// Classifies a token's bytes by character type for vocab partitioning.
/// Returns 0 (pure non-alnum), 1 (mixed), 2 (alnum with ≥1 alpha, optionally with leading space),
/// or 3 (pure digit, optionally with leading space).
pub(crate) fn classify_vocab_char_type(bytes: &[u8]) -> u8 {
    if bytes.is_empty() {
        return 0;
    }
    // Strip optional leading ASCII space (GPT-2 BPE decodes Ġ → 0x20 before we see it)
    let content = if bytes[0] == b' ' {
        &bytes[1..]
    } else {
        bytes
    };
    if content.is_empty() {
        return 0; // Just a space marker → non-alnum
    }
    if content.iter().all(|b| b.is_ascii_alphanumeric()) {
        if content.iter().any(|b| b.is_ascii_alphabetic()) {
            return 2; // Alnum with at least one alpha (optionally with leading space)
        }
        return 3; // Pure digit (optionally with leading space)
    }
    if bytes.iter().all(|b| !b.is_ascii_alphanumeric()) {
        return 0; // Pure non-alnum
    }
    1 // Mixed
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
