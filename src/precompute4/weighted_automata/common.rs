// src/precompute4/weighted_automata/common.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset;
use std::collections::BTreeMap;

pub(crate) const STOCHASTIC_DEBUG: bool = true; // Set to false by default to avoid heavy stochastic validation on large automata

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct I16Map<T> {
    pub exceptions: BTreeMap<i16, T>,
    pub default: Option<T>,
}

impl<T> I16Map<T> {
    pub fn new() -> Self {
        Self { exceptions: BTreeMap::new(), default: None }
    }
    pub fn with_default(default_value: T) -> Self {
        Self { exceptions: BTreeMap::new(), default: Some(default_value) }
    }
    pub fn get(&self, key: i16) -> Option<&T> {
        self.exceptions.get(&key).or(self.default.as_ref())
    }
    pub fn iter_exceptions(&self) -> impl Iterator<Item = (&i16, &T)> {
        self.exceptions.iter()
    }
    pub fn get_default(&self) -> Option<&T> {
        self.default.as_ref()
    }
}

pub type StateID = usize;
pub type Weight = SimpleBitset;
pub type NWAStateID = usize;

pub fn format_pos_code(code: i16) -> String {
    let u = code as u16;
    if let Some(c) = char::from_u32(u as u32) {
        if c.is_ascii_graphic() || c == ' ' {
            format!("'{}'", c)
        } else {
            format!("{}", u)
        }
    } else {
        format!("{}", u)
    }
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
