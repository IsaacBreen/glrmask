//! Parser-state label encoding shared across the compiler and runtime.

/// Compact fallback label used for parser-side default transitions.
pub const DEFAULT_LABEL: i32 = i32::MAX - 1;

pub(crate) fn encode_positive_label(state: u32) -> i32 {
    i32::try_from(state).expect("parser state ID does not fit in i32")
}

pub(crate) fn encode_negative_label(state: u32) -> i32 {
    -encode_positive_label(state) - 1
}

pub(crate) fn is_negative_label(label: i32) -> bool {
    label < 0 && label != DEFAULT_LABEL
}

pub(crate) fn negative_to_positive_label(label: i32) -> i32 {
    debug_assert!(is_negative_label(label));
    -label - 1
}
