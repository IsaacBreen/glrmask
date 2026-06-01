use super::*;
use crate::sets::bitset::BitSet;
use rustc_hash::FxHasher;
use super::options::table_options_from_env;
use std::hash::{Hash, Hasher};

fn stack_shift_predecessor_canonicalization_enabled() -> bool {
    table_options_from_env().stack_shift_predecessor_canonicalization
}

fn recognizer_suffix_quotient_enabled() -> bool {
    table_options_from_env().recognizer_suffix_quotient
}

fn max_guarded_stack_effects() -> Option<usize> {
    table_options_from_env().max_guarded_stack_effects
}

