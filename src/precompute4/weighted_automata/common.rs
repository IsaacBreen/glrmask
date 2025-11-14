// src/precompute4/weighted_automata/common.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset;

pub(crate) const STOCHASTIC_DEBUG: bool = false;
pub(crate) const DETERMINIZE_DEBUG: bool = false;

pub type StateID = usize;
pub type Weight = SimpleBitset;
pub type NWAStateID = usize;

/// Format a non-negative i16 as a human-readable symbol code.
pub fn format_pos_code(code: i16) -> String {
    code.to_string()
}

/// Format an i16 as a symbol, distinguishing negative codes.
pub fn format_i16_char(code: i16) -> String {
    if code >= 0 {
        format_pos_code(code)
    } else {
        format!("neg({})", code.wrapping_sub(i16::MIN))
    }
}

/// Pretty-print a word as a list of codes.
pub fn format_word(word: &[i16]) -> String {
    let parts: Vec<String> = word.iter().map(|&c| format_i16_char(c)).collect();
    format!("[{}]", parts.join(", "))
}
