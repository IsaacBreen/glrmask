#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset;

pub(crate) const STOCHASTIC_DEBUG: bool = false;
pub(crate) const DETERMINIZE_DEBUG: bool = false;
pub(crate) const BENCHMARK_DEBUG: bool = false;
pub(crate) const OPTIMIZE_DEBUG: bool = false;

pub type StateID = usize;
pub type Weight = SimpleBitset;
pub type NWAStateID = usize;
pub type Label = i16;

pub fn format_pos_code(code: Label) -> String { code.to_string() }

pub fn format_i16_char(code: Label) -> String {
    if code >= 0 {
        format_pos_code(code)
    } else {
        format!("neg({})", code.wrapping_sub(Label::MIN))
    }
}

pub fn format_word(word: &[Label]) -> String {
    let parts: Vec<String> = word.iter().map(|&c| format_i16_char(c)).collect();
    format!("[{}]", parts.join(", "))
}
