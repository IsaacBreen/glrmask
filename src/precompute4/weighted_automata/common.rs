// src/precompute4/weighted_automata/common.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset;

pub(crate) const STOCHASTIC_DEBUG: bool = false; // Set to false by default to avoid heavy stochastic validation on large automata
pub(crate) const DETERMINIZE_DEBUG: bool = false;

pub type StateID = usize;
pub type Weight = SimpleBitset;
pub type NWAStateID = usize;

pub fn format_pos_code(code: i16) -> String {
    format!("{}", code)
    // let u = code as u16;
    // if let Some(c) = char::from_u32(u as u32) {
    //     if c.is_ascii_graphic() || c == ' ' {
    //         format!("'{}'", c)
    //     } else {
    //         format!("{}", u)
    //     }
    // }
    // } else {
    //     format!("{}", u)
    // }
}
pub fn format_i16_char(code: i16) -> String {
    if code >= 0 {
        format_pos_code(code)
    } else {
        format!("neg({})", code.wrapping_sub(i16::MIN))
    }
}
pub fn format_word(word: &[i16]) -> String {
    let parts: Vec<String> = word.iter().map(|&c| format_i16_char(c)).collect();
    format!("[{}]", parts.join(", "))
}
