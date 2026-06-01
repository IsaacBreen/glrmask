//! Commit runtime.
//!
//! Commit consumes token bytes or raw bytes, lets the tokenizer emit every
//! completed grammar terminal boundary, and advances the active parser stacks
//! through the GLR transition relation for those completed terminals. If bytes
//! stop in the middle of a terminal, the state records the partial tokenizer
//! state so the next commit can complete it.
//!
//! Mathematically, Commit is the transition relation on live constraint states:
//! a byte fragment is scanned into zero or more completed terminals plus a
//! residual tokenizer state, and every completed terminal is interpreted as a
//! parser-stack effect. This module is intentionally just the routing layer; the
//! implementation is split by sub-relation below.

mod acceptance;
mod api;
mod fast_path;
mod general;
mod initial_scan;
mod mask_assert;
mod options;
mod parser_advance;
mod profiled;
mod pruning;
pub(crate) mod profile;
mod queue;
mod single_top;
mod template_advance;
mod terminal_advance;
pub(crate) mod tokenizer_scan;
mod token_lookup;
mod types;

use std::collections::{BTreeMap, BTreeSet};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::{TokenizerExecResult, TokenizerMatch};
use crate::parser::glr::accumulator::TerminalsDisallowed;
use crate::parser::glr::advance::{
    AdvanceProfile,
    ParserGSS,
    apply_guarded_stack_shifts_fast,
    stack_can_advance_on,
    stack_can_advance_on_any,
};
use crate::parser::glr::table::{Action, GLRTable};
use crate::runtime::constraint::Constraint;
use crate::runtime::state::{CommitBuffers, ConstraintState};

use self::acceptance::*;
use self::fast_path::*;
use self::general::*;
use self::initial_scan::*;
use self::mask_assert::{assert_mask_commit_equivalence, snapshot_mask_membership};
use self::options::template_advance_enabled;
use self::parser_advance::{
    advance_parser_stacks,
    advance_parser_stacks_owned,
    advance_parser_stacks_profiled,
};
use self::profile::{
    CommitProfile,
    PerAdvanceEntry,
    apply_advance_profile,
    fast_action_advance_profile,
};
use self::profiled::*;
use self::pruning::*;
use self::queue::*;
use self::single_top::*;
use self::terminal_advance::*;
use self::token_lookup::token_bytes_for_id;
use self::tokenizer_scan::{InitialCommitScan, execute_tokenizer_from_state_small};
use self::types::*;
