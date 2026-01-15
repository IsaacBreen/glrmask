#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::abstract_weight::AbstractWeight;
use super::get_weight_dimensions;

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

/// Create a weight representing the full universe for the current dimensions.
/// 
/// This is a convenience function that uses the global weight dimensions.
/// For explicit dimension control, use `Weight::all(dims)` instead.
pub fn weight_all() -> Weight {
    let dims = get_weight_dimensions();
    Weight::all(dims)
}

/// Create the complement of a weight relative to the full universe.
pub fn weight_complement(w: &Weight) -> Weight {
    &weight_all() - w
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
