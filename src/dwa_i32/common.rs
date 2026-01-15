#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::abstract_weight::AbstractWeight;
use range_set_blaze::RangeSetBlaze;

pub(crate) const STOCHASTIC_DEBUG: bool = false;
pub(crate) const DETERMINIZE_DEBUG: bool = false;
pub(crate) const BENCHMARK_DEBUG: bool = false;

pub(crate) fn optimize_debug() -> bool {
    std::env::var("TOKENIZERS_OPTIMIZE_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub type StateID = usize;
pub type Weight = AbstractWeight;
pub type NWAStateID = usize;
pub type Label = i32;

/// Create a weight representing the full universe (0..=usize::MAX).
pub fn weight_all() -> Weight {
    Weight::from_rsb(RangeSetBlaze::from_iter([0..=usize::MAX]))
}

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
