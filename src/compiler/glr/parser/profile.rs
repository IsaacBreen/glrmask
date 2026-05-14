#[derive(Clone, Debug, Default)]
pub struct AdvanceTrace {
    pub det_steps: Vec<AdvanceTraceStep>,
    pub nondet_waves: Vec<AdvanceTraceWave>,
}

#[derive(Clone, Debug, Default)]
pub struct AdvanceTraceWave {
    pub wave_index: u32,
    pub frontier_states: Vec<u32>,
    pub branches: Vec<AdvanceTraceStep>,
}

#[derive(Clone, Debug, Default)]
pub struct AdvanceTraceStep {
    pub source_state: u32,
    pub action_kind: String,
    pub shift_target: Option<u32>,
    pub shift_replace: Option<bool>,
    pub reduces: Vec<AdvanceTraceReduce>,
}

#[derive(Clone, Debug, Default)]
pub struct AdvanceTraceReduce {
    pub lhs_nt: u32,
    pub lhs_name: Option<String>,
    pub pop_len: u32,
    pub goto_sources: Vec<u32>,
    pub goto_targets: Vec<AdvanceTraceGoto>,
}

#[derive(Clone, Debug, Default)]
pub struct AdvanceTraceGoto {
    pub source_state: u32,
    pub target_state: u32,
    pub replace: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AdvanceProfile {
    pub pure_shift: bool,
    pub deterministic_entered: bool,
    pub deterministic_finished: bool,
    pub nondeterministic_entered: bool,
    pub vstack_len: u32,
    pub n_reduces_above_floor: u32,
    pub n_floor_crossings: u32,
    pub n_nondet_waves: u32,
    pub n_nondet_branches: u32,
    pub top_states: u32,
    pub gss_depth: u32,
    pub total_ns: u64,
    pub clone_ns: u64,
    pub fast_path_ns: u64,
    pub stack_shift_apply_ns: u64,
    pub det_ns: u64,
    pub nondet_ns: u64,
    pub nondet_det_ns: u64,
    pub det_exit_reason: u32,
    pub det_exit_state: u32,
    pub n_det_action_lookups: u32,
    pub n_det_goto_lookups: u32,
    pub n_det_popn_ops: u32,
    pub n_nondet_reduce_ops: u32,
    pub n_nondet_merges: u32,
    pub n_nondet_isolates: u32,
    pub trace: Option<AdvanceTrace>,
}
