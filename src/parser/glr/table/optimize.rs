//! GLR table optimization passes.
//!
//! This module is intentionally a textual facade over smaller pass files.  The
//! files are included into one module scope so the first refactor preserves the
//! old private helper relationships while making the mathematical passes visible
//! to readers.  A later compile-repair chunk can convert these textual includes
//! into true Rust submodules once the helper visibility boundaries have been
//! made explicit.

use super::*;
use crate::ds::bitset::BitSet;
use rustc_hash::FxHasher;
use super::options::table_options_from_env;
use std::hash::{Hash, Hasher};

include!("optimize/policy_adapter.rs");
include!("optimize/stack_effect_keys.rs");
include!("optimize/table_passes.rs");
include!("optimize/suffix_quotient.rs");
include!("optimize/merged_state_quotient.rs");
include!("optimize/guarded/frame_model.rs");
include!("optimize/guarded/reduce_frame.rs");
include!("optimize/guarded/action_exploration.rs");
include!("optimize/guarded/action_materialize.rs");
include!("optimize/guarded/stack_shift_canonicalization.rs");
include!("optimize/guarded/stack_shift_canonicalization_tests.rs");
include!("optimize/unit_reductions.rs");
include!("optimize/same_core_merge.rs");
