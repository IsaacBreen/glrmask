use super::accumulator::TerminalsDisallowed;
use super::analysis::EOF;
use super::table::{
    Action,
    GLRTable,
    GuardedShiftCellIndex,
    GuardedStackShift,
    StackShift,
    StackShiftGuard,
};
use crate::sets::bitset::BitSet;
use crate::parser::gss::{LeveledGSS, VirtualStack};
use crate::grammar::flat::TerminalID;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
mod options;
mod profile;

pub(crate) use options::ParserAdvanceOptions;

pub use profile::{
    AdvanceTrace,
    AdvanceTraceGoto,
    AdvanceProfile,
    AdvanceTraceReduce,
    AdvanceTraceStep,
    AdvanceTraceWave,
};

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

type ReduceSources = SmallVec<[(u32, ParserGSS); 4]>;
type ReduceBranches = SmallVec<[(ParserGSS, u32, bool); 4]>;
type FloorCrossShift = (u32, u32, bool);

const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 64;
const GUARDED_STACK_TO_STACKS_MAX_DEPTH: usize = 64;
const SMALL_REDUCE_FANOUT_COLLAPSE_MAX_BRANCHES: usize = 8;

fn advance_options() -> &'static ParserAdvanceOptions {
    ParserAdvanceOptions::global()
}

fn guarded_stack_to_stacks_fallback_disabled() -> bool {
    advance_options().disable_guarded_stack_to_stacks_fallback
}

fn stack_effect_to_stacks_fallback_disabled() -> bool {
    advance_options().disable_stack_effect_to_stacks_fallback
}

fn advance_trace_enabled() -> bool {
    advance_options().trace_enabled
}

include!("profile_trace.rs");
include!("entry_points.rs");
include!("fast_paths.rs");
include!("guarded_shifts.rs");
include!("reduce_sources.rs");
include!("deterministic_vstack.rs");
include!("deterministic_profiled.rs");
include!("nondeterministic_profiled.rs");
include!("nondeterministic.rs");
include!("deterministic.rs");
include!("applicability.rs");
include!("tests.rs");
include!("applicability_any.rs");
