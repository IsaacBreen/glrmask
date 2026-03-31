//! Vocab and terminal classification utilities.

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::Vocab;

use super::types::TerminalPathLength;

/// Classifies a token's bytes by character type for vocab partitioning.
/// Returns 0 (pure non-alnum), 1 (mixed), or 2 (alnum, optionally with leading space).
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
        return 2; // Alnum (optionally with leading space)
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
) -> Vec<TerminalPathLength> {
    let nt = num_terminals as usize;

    // 1. Vocab byte bitset: all bytes appearing in any vocab token.
    let mut vocab_bytes = U8Set::empty();
    for bytes in vocab.entries.values() {
        for &b in bytes {
            vocab_bytes.insert(b);
        }
    }

    // 2. Byte bitsets per terminal.
    let num_states = tokenizer.num_states() as usize;
    let initial = tokenizer.start_state();
    let dfa_states = tokenizer.dfa.states();

    // reachable_bytes[t]: bytes from ANY state that lead towards matching
    // terminal t (finalized or in possible_future).  Used for L0 check.
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

    // first_bytes[t]: bytes from the INITIAL state leading towards terminal t.
    // After a terminal match the tokenizer resets to initial, so this is the
    // relevant set for "can t2 start after t1?".
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

    // last_bytes[t]: bytes on transitions arriving at states that finalize t.
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
