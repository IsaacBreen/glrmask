//! Parser-state label encoding shared across the compiler and runtime.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

/// Compact fallback label used for parser-side default transitions.
pub const DEFAULT_LABEL: i32 = i32::MAX - 1;

pub(crate) fn encode_positive_label(state: u32) -> i32 {
    unimplemented!()
}

pub(crate) fn encode_negative_label(state: u32) -> i32 {
    unimplemented!()
}

pub(crate) fn is_negative_label(label: i32) -> bool {
    unimplemented!()
}

pub(crate) fn negative_to_positive_label(label: i32) -> i32 {
    unimplemented!()
}
