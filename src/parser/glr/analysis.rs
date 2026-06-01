use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

use crate::ds::bitset::BitSet;
use crate::grammar::flat::{GrammarDef, NonterminalID, Rule, Symbol, TerminalID};

pub const EOF: TerminalID = u32::MAX;

include!("analysis/options.rs");
include!("analysis/profile.rs");
include!("analysis/model.rs");
include!("analysis/right_recursion.rs");
include!("analysis/null_production_inline.rs");
include!("analysis/nullable_run_compress.rs");
include!("analysis/left_recursion.rs");
include!("analysis/reachability_unit_dedup.rs");
include!("analysis/normalize.rs");
include!("analysis/fixed_point_sets.rs");
include!("analysis/tests.rs");
