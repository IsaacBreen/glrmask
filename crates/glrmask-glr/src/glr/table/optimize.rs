use super::*;
use crate::ds::bitset::BitSet;
use glrmask_invariant::__private::charge_template_build_budget as charge_build_budget;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec};
use rustc_hash::FxHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

const DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION_ENV: &str =
    "GLRMASK_DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION";
const DISABLE_RECOGNIZER_SUFFIX_QUOTIENT_ENV: &str =
    "GLRMASK_DISABLE_RECOGNIZER_SUFFIX_QUOTIENT";
const RECOGNIZER_SUFFIX_QUOTIENT_MAX_STATES_ENV: &str =
    "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_STATES";
const RECOGNIZER_SUFFIX_QUOTIENT_MAX_ALTS_ENV: &str =
    "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_ALTS";
const RECOGNIZER_SUFFIX_QUOTIENT_MAX_WIDTH_ENV: &str =
    "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_WIDTH";
const RECOGNIZER_SUFFIX_QUOTIENT_MAX_DEPTH_ENV: &str =
    "GLRMASK_RECOGNIZER_SUFFIX_QUOTIENT_MAX_DEPTH";
const MAX_GUARDED_STACK_EFFECTS_ENV: &str = "GLRMASK_MAX_GUARDED_STACK_EFFECTS";
const UNIT_INLINE_WORK_MAX_WALL_MS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_MS";
const UNIT_INLINE_WORK_MAX_ITERATIONS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_ITERATIONS";
const UNIT_INLINE_WORK_MAX_CELLS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_CELLS";
const UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES_ENV: &str =
    "GLRMASK_UNIT_REDUCTION_INLINE_MAX_SYNTHETIC_STATES";
const UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_STACK_VISITS";
const UNIT_INLINE_STATE_MAX_STACK_EFFECT_VISITS_ENV: &str =
    "GLRMASK_UNIT_REDUCTION_INLINE_STATE_MAX_STACK_VISITS";
const DEFAULT_UNIT_INLINE_WORK_MAX_WALL_MS: u128 = 5_000;
const DEFAULT_UNIT_INLINE_WORK_MAX_ITERATIONS: usize = 64;
const DEFAULT_UNIT_INLINE_WORK_MAX_CELLS: usize = 2_000_000;
const DEFAULT_UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES: usize = 4_096;
const DEFAULT_UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS: usize = 100_000;
const DEFAULT_UNIT_INLINE_STATE_MAX_STACK_EFFECT_VISITS: usize = 512;
const UNIT_INLINE_COMPLETED_ABORT_ENV: &str =
    "GLRMASK_UNIT_INLINE_COMPLETED_WORK_ABORT";

fn completed_work_abort_eligible(construction: GlrTableConstruction) -> bool {
    // The measured benefit is in the exact core-merged builder. Keep legacy
    // and LALR scheduling untouched instead of charging successful static
    // passes for a certificate they have not been shown to benefit from.
    construction == GlrTableConstruction::ExperimentalCoreMerged
}

// Most unit-inlining states are singleton constituents. Keeping those
// inline avoids thousands of tiny BTreeSet allocations before any synthetic
// merged state is actually needed. Subsets remain sorted and deduplicated.
type StateSubset = SmallVec<[u32; 2]>;
// Runtime predecessors are always kept sorted and deduplicated. Most parser
// states have at most a couple of predecessors, so an inline vector avoids
// the per-state tree allocation and pointer chasing of BTreeSet.
type PredecessorSet = SmallVec<[u32; 2]>;
type RuntimePredecessors = Vec<PredecessorSet>;

fn stack_shift_predecessor_canonicalization_enabled() -> bool {
    !env_flag_enabled(DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION_ENV, false)
}

fn recognizer_suffix_quotient_enabled() -> bool {
    !env_flag_enabled(DISABLE_RECOGNIZER_SUFFIX_QUOTIENT_ENV, false)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_u128(name: &str, default: u128) -> u128 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(default)
}

fn unit_inline_state_max_stack_effect_visits() -> usize {
    env_usize(
        UNIT_INLINE_STATE_MAX_STACK_EFFECT_VISITS_ENV,
        DEFAULT_UNIT_INLINE_STATE_MAX_STACK_EFFECT_VISITS,
    )
}

fn max_guarded_stack_effects() -> Option<usize> {
    std::env::var(MAX_GUARDED_STACK_EFFECTS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackEffectVisitKey {
    state: u32,
    tid: TerminalID,
    action_tag: u8,
    frame: StackEffectFrame,
}

#[derive(Clone)]
struct StackEffectResult {
    effects: Vec<GuardedStackShift>,
    origin_dependent: bool,
}

#[derive(Debug, Clone)]
pub(super) struct UnitReductionInliningReport {
    pub(super) aborted: bool,
    pub(super) reason: Option<&'static str>,
    pub(super) iterations: usize,
    pub(super) cells: usize,
    pub(super) synthetic_states: usize,
    pub(super) stack_effect_visits: usize,
    pub(super) elapsed_ms: f64,
    pub(super) changed_original_states: Vec<u32>,
}

/// Number of real control-elimination runs process-wide. No-op calls on
/// control-free tables do not count. The prepared static linker's
/// signed-transfer boundary compiler asserts this does not move across a link.
static CONTROL_ELIMINATION_RUNS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Process-wide count of real `eliminate_control_terminals_exact` runs.
pub fn control_elimination_run_count() -> u64 {
    CONTROL_ELIMINATION_RUNS.load(std::sync::atomic::Ordering::SeqCst)
}

#[derive(Debug, Clone)]
pub struct ControlEliminationReport {
    pub states: usize,
    pub controls: usize,
    pub compiled_cells: usize,
    pub closure_frames: usize,
    pub stack_effect_visits: usize,
    /// Real terminal columns whose consuming action changed while compiling
    /// zero-width control closure. Component-local templates remain exact for
    /// every other terminal and can therefore be transported directly.
    pub changed_terminals: Vec<TerminalID>,
    pub elapsed_ms: f64,
}

struct UnitInlineUndo {
    original_num_states: u32,
    original_action_len: usize,
    original_goto_len: usize,
    original_advance_len: usize,
    original_cells: FxHashMap<(usize, TerminalID), Option<Action>>,
    original_advance_rows: FxHashMap<usize, BitSet>,
}

impl UnitInlineUndo {
    fn new(table: &GLRTable) -> Self {
        Self {
            original_num_states: table.num_states,
            original_action_len: table.action.len(),
            original_goto_len: table.goto.len(),
            original_advance_len: table.advance.len(),
            original_cells: FxHashMap::default(),
            original_advance_rows: FxHashMap::default(),
        }
    }

    fn record_cell(&mut self, table: &GLRTable, state: usize, tid: TerminalID) {
        if state >= self.original_num_states as usize {
            return;
        }
        self.original_cells
            .entry((state, tid))
            .or_insert_with(|| table.action[state].get(&tid).cloned());
        if self.original_advance_len == self.original_action_len {
            self.original_advance_rows
                .entry(state)
                .or_insert_with(|| table.advance[state].clone());
        }
    }

    fn changed_original_states(&self) -> Vec<u32> {
        let mut states = self
            .original_cells
            .keys()
            .map(|&(state, _)| state as u32)
            .collect::<Vec<_>>();
        states.sort_unstable();
        states.dedup();
        states
    }

    fn rollback(self, table: &mut GLRTable) {
        table.action.truncate(self.original_action_len);
        table.goto.truncate(self.original_goto_len);
        table.advance.truncate(self.original_advance_len);
        table.num_states = self.original_num_states;
        for ((state, tid), action) in self.original_cells {
            match action {
                Some(action) => {
                    table.action[state].insert(tid, action);
                }
                None => {
                    table.action[state].remove(&tid);
                }
            }
        }
        for (state, advance) in self.original_advance_rows {
            table.advance[state] = advance;
        }
    }
}

struct UnitInlineBudget {
    started_at: std::time::Instant,
    max_ms: u128,
    max_iterations: usize,
    max_cells: usize,
    max_synthetic_states: usize,
    max_stack_effect_visits: usize,
    iterations: std::sync::atomic::AtomicUsize,
    cells: std::sync::atomic::AtomicUsize,
    synthetic_states: std::sync::atomic::AtomicUsize,
    stack_effect_visits: std::sync::atomic::AtomicUsize,
    // 0 = not aborted; otherwise a small code identifying the reason. Using an
    // atomic lets the budget be shared across the parallel stack-effect
    // precompute below while keeping serial behaviour byte-for-byte identical.
    abort_code: std::sync::atomic::AtomicU8,
}

/// A lower bound on work the serial fold will charge, not a shared child
/// budget. Only completed jobs publish here, so a scheduling-dependent partial
/// analysis can never cause a successful reference optimization to be lost.
/// No production table changes until the normal serial phase or rollback.
struct CompletedStackEffectWork {
    visits: std::sync::atomic::AtomicUsize,
    limit: usize,
}

impl CompletedStackEffectWork {
    fn new(budget: &UnitInlineBudget) -> Option<Self> {
        // In particular, leave the uncapped correctness oracle unchanged.
        (budget.max_stack_effect_visits != usize::MAX).then(|| Self {
            visits: std::sync::atomic::AtomicUsize::new(budget.stack_effect_visits()),
            limit: budget.max_stack_effect_visits,
        })
    }

    // Keep fetch_update for older Rust while accepting its newer try_update
    // rename without changing the atomic update or its ordering.
    #[allow(deprecated)]
    fn record_completed(&self, visits: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        // One uncontended/local-budget total per job, not an atomic per visit.
        // Saturation preserves the exceeded predicate even at usize::MAX-1.
        let _ = self.visits.fetch_update(Relaxed, Relaxed, |total| {
            Some(total.saturating_add(visits).min(self.limit + 1))
        });
    }

    fn exceeded(&self) -> bool {
        self.visits.load(std::sync::atomic::Ordering::Relaxed) > self.limit
    }
}

const ABORT_NONE: u8 = 0;
const ABORT_ELAPSED: u8 = 1;
const ABORT_ITERATIONS: u8 = 2;
const ABORT_CELLS: u8 = 3;
const ABORT_SYNTHETIC_STATES: u8 = 4;
const ABORT_STACK_EFFECT_VISITS: u8 = 5;
// A dirty synthetic row could not be rebuilt. Abort the optional optimization
// instead of keeping a stale row after originals change.
const ABORT_REFRESH_REBUILD: u8 = 6;

/// Convergence caps for exact control elimination (see
/// `exact_control_elimination`). Legitimate tables use a handful of visits
/// (small segmented fixtures: 8–13) and sub-millisecond walls; the
/// pathological cyclic case burns millions of visits without converging, so
/// 100k visits (~0.6 s observed) fails fast with >1000x headroom.
const EXACT_CONTROL_ELIMINATION_MAX_STACK_EFFECT_VISITS: usize = 100_000;
const EXACT_CONTROL_ELIMINATION_MAX_WALL_MS: u128 = 10_000;

fn abort_reason_str(code: u8) -> Option<&'static str> {
    match code {
        ABORT_ELAPSED => Some("elapsed_ms"),
        ABORT_ITERATIONS => Some("iterations"),
        ABORT_CELLS => Some("cells"),
        ABORT_SYNTHETIC_STATES => Some("synthetic_states"),
        ABORT_STACK_EFFECT_VISITS => Some("stack_effect_visits"),
        ABORT_REFRESH_REBUILD => Some("refresh_rebuild"),
        _ => None,
    }
}

impl UnitInlineBudget {
    fn from_env() -> Self {
        use std::sync::atomic::{AtomicU8, AtomicUsize};
        Self {
            started_at: std::time::Instant::now(),
            max_ms: env_u128(UNIT_INLINE_WORK_MAX_WALL_MS_ENV, DEFAULT_UNIT_INLINE_WORK_MAX_WALL_MS),
            max_iterations: env_usize(
                UNIT_INLINE_WORK_MAX_ITERATIONS_ENV,
                DEFAULT_UNIT_INLINE_WORK_MAX_ITERATIONS,
            ),
            max_cells: env_usize(UNIT_INLINE_WORK_MAX_CELLS_ENV, DEFAULT_UNIT_INLINE_WORK_MAX_CELLS),
            max_synthetic_states: env_usize(
                UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES_ENV,
                DEFAULT_UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES,
            ),
            max_stack_effect_visits: env_usize(
                UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS_ENV,
                DEFAULT_UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS,
            ),
            iterations: AtomicUsize::new(0),
            cells: AtomicUsize::new(0),
            synthetic_states: AtomicUsize::new(0),
            stack_effect_visits: AtomicUsize::new(0),
            abort_code: AtomicU8::new(ABORT_NONE),
        }
    }

    /// Exact control elimination is a semantic compilation pass, not an
    /// optional optimizer. It must not succeed or fail according to the
    /// environment limits used to bound speculative unit-reduction inlining,
    /// so the convergence caps below are hardcoded, not env-tunable.
    ///
    /// The caps exist because the symbolic traversal is not guaranteed to
    /// terminate: on cyclic reduce graphs every lap accumulates a distinct
    /// guard set, so the backtracking `visiting` set never repeats a key and
    /// the DFS tree grows without bound (observed: a 7947-state composed
    /// table burning 170k visits/s with no end). The visits cap bounds total
    /// traversal work deterministically; the wall backstop bounds residual
    /// non-visit work (frame cloning, cache-hit closure steps). Exhaustion
    /// reports `reason=stack_effect_visits` / `reason=elapsed_ms`; callers
    /// detect it with `control_elimination_budget_exhausted` and fall back
    /// to dynamic boundary shards (static linking is an accelerator — the
    /// dynamic backend always remains exact).
    fn exact_control_elimination() -> Self {
        use std::sync::atomic::{AtomicU8, AtomicUsize};
        Self {
            started_at: std::time::Instant::now(),
            max_ms: EXACT_CONTROL_ELIMINATION_MAX_WALL_MS,
            max_iterations: usize::MAX,
            max_cells: usize::MAX,
            max_synthetic_states: usize::MAX,
            max_stack_effect_visits: EXACT_CONTROL_ELIMINATION_MAX_STACK_EFFECT_VISITS,
            iterations: AtomicUsize::new(0),
            cells: AtomicUsize::new(0),
            synthetic_states: AtomicUsize::new(0),
            stack_effect_visits: AtomicUsize::new(0),
            abort_code: AtomicU8::new(ABORT_NONE),
        }
    }

    fn iterations(&self) -> usize {
        self.iterations.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn cells(&self) -> usize {
        self.cells.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn synthetic_states(&self) -> usize {
        self.synthetic_states.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn stack_effect_visits(&self) -> usize {
        self.stack_effect_visits.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn is_aborted(&self) -> bool {
        self.abort_code.load(std::sync::atomic::Ordering::Relaxed) != ABORT_NONE
    }

    fn report(&self) -> UnitReductionInliningReport {
        let code = self.abort_code.load(std::sync::atomic::Ordering::Relaxed);
        UnitReductionInliningReport {
            aborted: code != ABORT_NONE,
            reason: abort_reason_str(code),
            iterations: self.iterations(),
            cells: self.cells(),
            synthetic_states: self.synthetic_states(),
            stack_effect_visits: self.stack_effect_visits(),
            elapsed_ms: self.started_at.elapsed().as_secs_f64() * 1000.0,
            changed_original_states: Vec::new(),
        }
    }

    fn abort(&self, code: u8) {
        let _ = self.abort_code.compare_exchange(
            ABORT_NONE,
            code,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    fn check_elapsed(&self) -> bool {
        if self.is_aborted() {
            return false;
        }
        if self.started_at.elapsed().as_millis() > self.max_ms {
            self.abort(ABORT_ELAPSED);
            return false;
        }
        true
    }

    fn record_iteration(&self) -> bool {
        charge_build_budget("glr_table_collapse", 1_000);
        let value = self.iterations.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if value > self.max_iterations {
            self.abort(ABORT_ITERATIONS);
            return false;
        }
        self.check_elapsed()
    }

    fn record_cell(&self) -> bool {
        charge_build_budget("glr_table_collapse", 1);
        let value = self.cells.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if value > self.max_cells {
            self.abort(ABORT_CELLS);
            return false;
        }
        self.check_elapsed()
    }

    fn record_synthetic_state(&self) -> bool {
        let value = self.synthetic_states.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if value > self.max_synthetic_states {
            self.abort(ABORT_SYNTHETIC_STATES);
            return false;
        }
        self.check_elapsed()
    }

    fn record_stack_effect_visit(&self) -> bool {
        charge_build_budget("glr_table_collapse", 8);
        let value = self.stack_effect_visits.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if value > self.max_stack_effect_visits {
            self.abort(ABORT_STACK_EFFECT_VISITS);
            return false;
        }
        if value & 0x3ff == 0 {
            return self.check_elapsed();
        }
        !self.is_aborted()
    }

    /// Create a budget that shares this one's limits and start instant but owns
    /// independent counters. Used by the parallel stack-effect precompute so
    /// each task increments private (uncontended) atomics; counts are folded
    /// back into the parent afterwards. The shared `started_at`/`max_ms`
    /// preserves the global wall-clock cap across the parallel phase.
    fn child(&self) -> UnitInlineBudget {
        use std::sync::atomic::{AtomicU8, AtomicUsize};
        UnitInlineBudget {
            started_at: self.started_at,
            max_ms: self.max_ms,
            max_iterations: self.max_iterations,
            max_cells: self.max_cells,
            max_synthetic_states: self.max_synthetic_states,
            max_stack_effect_visits: self.max_stack_effect_visits,
            iterations: AtomicUsize::new(0),
            cells: AtomicUsize::new(0),
            synthetic_states: AtomicUsize::new(0),
            stack_effect_visits: AtomicUsize::new(0),
            abort_code: AtomicU8::new(ABORT_NONE),
        }
    }

    /// Create one independently bounded parser-state analysis. A state that
    /// exceeds this local stack-effect limit is skipped, while sibling states
    /// and the enclosing fixed-point iteration may still complete.
    fn child_with_stack_effect_visit_limit(&self, limit: usize) -> UnitInlineBudget {
        let child = self.child();
        UnitInlineBudget {
            max_stack_effect_visits: child.max_stack_effect_visits.min(limit),
            ..child
        }
    }

    fn fold_in_counts(&self, child: &UnitInlineBudget) {
        use std::sync::atomic::Ordering::Relaxed;
        self.synthetic_states
            .fetch_add(child.synthetic_states(), Relaxed);
        let total = self
            .stack_effect_visits
            .fetch_add(child.stack_effect_visits(), Relaxed)
            + child.stack_effect_visits();
        if total > self.max_stack_effect_visits {
            self.abort(ABORT_STACK_EFFECT_VISITS);
        }
    }

    /// Fold a child budget's consumption back into this one and re-check the
    /// global caps, so total work across the parallel tasks is still bounded.
    fn fold_in(&self, child: &UnitInlineBudget) {
        use std::sync::atomic::Ordering::Relaxed;
        self.fold_in_counts(child);
        let child_code = child.abort_code.load(Relaxed);
        if child_code != ABORT_NONE {
            self.abort(child_code);
        }
    }

    /// Fold one core-merged parser-state analysis. Local stack-effect
    /// exhaustion rejects only that state, but its work still contributes to
    /// the aggregate cap. Other abort causes remain global.
    fn fold_in_isolated_state(&self, child: &UnitInlineBudget) {
        use std::sync::atomic::Ordering::Relaxed;
        self.fold_in_counts(child);
        let child_code = child.abort_code.load(Relaxed);
        if child_code != ABORT_NONE && child_code != ABORT_STACK_EFFECT_VISITS {
            self.abort(child_code);
        }
    }
}

/// Certify that zero-width control closure never needs to discover an unknown
/// parser state below the represented stack prefix.
///
/// Reductions are the only consuming actions whose effect depends on goto
/// predecessors. A control may also expose an unknown predecessor when it pops
/// without pushing a replacement top. In every other case the post-control top
/// is explicit in the action, so exact elimination can use an empty predecessor
/// relation.
fn control_elimination_has_known_tops(
    table: &GLRTable,
    controls: &BTreeSet<TerminalID>,
) -> bool {
    let mut queue = VecDeque::<u32>::new();
    let mut visited = vec![false; table.num_states as usize];
    for (state, row) in table.action.iter().enumerate() {
        if controls.iter().any(|control| row.contains_key(control)) {
            queue.push_back(state as u32);
        }
    }

    while let Some(state) = queue.pop_front() {
        let Some(visited_state) = visited.get_mut(state as usize) else {
            return false;
        };
        if *visited_state {
            continue;
        }
        *visited_state = true;
        let Some(row) = table.action.get(state as usize) else {
            return false;
        };

        for (terminal, action) in row.iter() {
            if !controls.contains(&terminal) {
                if action.reduce_count() != 0 {
                    return false;
                }
                continue;
            }

            match action {
                Action::Shift(target, _) => queue.push_back(*target),
                Action::ReplaceShifts(targets) => queue.extend(targets.iter().copied()),
                Action::StackShifts(shifts) => {
                    for shift in shifts {
                        if let Some(&target) = shift.pushes.last() {
                            queue.push_back(target);
                        } else if shift.pop != 0 {
                            return false;
                        } else {
                            queue.push_back(state);
                        }
                    }
                }
                Action::GuardedStackShifts(shifts) => {
                    for shift in shifts {
                        if let Some(&target) = shift.pushes.last() {
                            queue.push_back(target);
                        } else if shift.pop != 0 {
                            return false;
                        } else {
                            queue.push_back(state);
                        }
                    }
                }
                Action::Split { shift, reduces, .. } => {
                    if !reduces.is_empty() {
                        return false;
                    }
                    if let Some((target, _)) = shift {
                        queue.push_back(*target);
                    }
                }
                Action::Skip => queue.push_back(state),
                Action::Accept => {}
                Action::Reduce(..) => return false,
            }
        }
    }
    true
}

impl GLRTable {
    pub(super) fn canonicalize_stack_shift_predecessors(&mut self) {
        let protected = BitSet::new(self.num_terminals as usize);
        self.canonicalize_stack_shift_predecessors_except(&protected);
    }

    pub(super) fn canonicalize_stack_shift_predecessors_except(&mut self, protected: &BitSet) {
        self.canonicalize_stack_shift_predecessors_with_enabled(
            stack_shift_predecessor_canonicalization_enabled(),
            protected,
        );
    }

    fn canonicalize_stack_shift_predecessors_with_enabled(
        &mut self,
        enabled: bool,
        protected: &BitSet,
    ) {
        if !enabled {
            return;
        }

        for state in 0..self.num_states as usize {
            let terminals: Vec<TerminalID> = self.action[state].keys().collect();
            for terminal in terminals {
                if protected.get(terminal as usize) {
                    continue;
                }
                let Some(Action::StackShifts(shifts)) = self.action[state].get(&terminal).cloned() else {
                    continue;
                };

                let mut shifts = shifts;
                canonicalize_stack_shift_predecessors_by_goto_superset(self, &mut shifts);
                let Some(action) = stack_shift_action(shifts) else {
                    self.action[state].remove(&terminal);
                    if self.advance.len() == self.num_states as usize
                        && let Some(bit) = self.terminal_bit(terminal)
                    {
                        self.advance[state].clear(bit);
                    }
                    continue;
                };
                self.action[state].insert(terminal, action);
            }
        }
    }

    /// Least coarsening generated by identical rows under successive target
    /// quotients. Unlike an optimistic bisimulation, mutually recursive but
    /// initially distinguishable states are not merged merely by assumption.
    pub(super) fn merge_identical_rows(&mut self) {
        self.coarsen_identical_rows(None);
    }

    /// The caller has already established the quotient before the local
    /// mutation. Only changed rows and reverse dependencies of later merges
    /// can newly enter an equality class.
    pub(super) fn merge_identical_rows_from_dirty(&mut self, dirty: &[u32]) -> bool {
        if dirty.is_empty() || dirty.iter().any(|&state| state >= self.num_states) {
            return false;
        }
        self.coarsen_identical_rows(Some(dirty))
    }

    fn coarsen_identical_rows(&mut self, dirty: Option<&[u32]>) -> bool {
        if self.num_states <= 1 {
            return false;
        }
        let profile = std::env::var("GLRMASK_PROFILE_GLR_ROW_MERGE_DETAIL")
            .is_ok_and(|value| value == "1");
        let started = profile.then(std::time::Instant::now);
        let n = self.num_states as usize;
        let quotient = RowCoarsener::new(self).solve(dirty);
        let Some((representatives, mapping)) = quotient else {
            return false;
        };
        let after = representatives.len();
        let mut keep = vec![false; n];
        for state in representatives {
            keep[state as usize] = true;
        }
        self.publish_state_mapping(&mapping, &keep);
        if let Some(started) = started {
            eprintln!("[glrmask/profile][row_merge] states_before={n} states_after={after} incremental={} total_ms={:.3}",
                dirty.is_some(), started.elapsed().as_secs_f64() * 1000.0);
        }
        true
    }

    /// Move retained rows into their final IDs once. The same publication
    /// handles row quotients, same-core quotients and reachability pruning.
    /// A missing ID is represented by MAX only for discarded states.
    fn publish_state_mapping(&mut self, mapping: &[u32], keep: &[bool]) {
        let n = self.num_states as usize;
        let has_advance = self.advance.len() == n;
        let kept = keep.iter().filter(|&&keep| keep).count();
        let action = std::mem::take(&mut self.action);
        let goto = std::mem::take(&mut self.goto);
        let advance = has_advance.then(|| std::mem::take(&mut self.advance));
        self.action = Vec::with_capacity(kept);
        self.goto = Vec::with_capacity(kept);
        for (state, (mut action, mut goto)) in action.into_iter().zip(goto).enumerate() {
            if keep[state] {
                debug_assert_eq!(mapping[state] as usize, self.action.len());
                remap_action_row_targets_in_place(&mut action, mapping);
                remap_goto_row_targets_in_place(&mut goto, mapping);
                self.action.push(action);
                self.goto.push(goto);
            }
        }
        if let Some(advance) = advance {
            self.advance = advance.into_iter().zip(keep)
                .filter_map(|(row, &keep)| keep.then_some(row)).collect();
        }
        self.forwarded_shifts = self.forwarded_shifts.iter().filter_map(|&(state, terminal)| {
            let target = mapping[state as usize];
            (target != u32::MAX).then_some((target, terminal))
        }).collect();
        self.direct_regular_wide_frontiers.retain_mut(|descriptor| {
            let source = mapping[descriptor.source_state as usize];
            if source == u32::MAX || descriptor.target_states.iter().any(|&state| mapping[state as usize] == u32::MAX) {
                return false;
            }
            descriptor.source_state = source;
            for target in &mut descriptor.target_states {
                *target = mapping[*target as usize];
            }
            descriptor.target_states.sort_unstable();
            descriptor.target_states.dedup();
            true
        });
        // These caches describe physical rows, not logical state IDs.
        // Final table publication rebuilds both after all rewrites finish.
        self.guarded_shift_index.clear();
        self.unconditional_advance.clear();
        self.num_states = kept as u32;
    }

    pub(super) fn prune_unreachable_states(&mut self) {
        let _ = self.prune_unreachable_states_impl(false);
    }

    /// Remove unreachable states and return the old-state -> new-state map.
    /// Unreachable states map to `u32::MAX`.
    ///
    /// Composition-time structural sharing needs this map to transport the
    /// already-compiled parser-DWA label relation through a table compaction.
    pub fn prune_unreachable_states_with_mapping(&mut self) -> Vec<u32> {
        self.prune_unreachable_states_impl(true)
            .unwrap_or_default()
    }

    fn prune_unreachable_states_impl(&mut self, return_mapping: bool) -> Option<Vec<u32>> {
        if self.num_states == 0 {
            return return_mapping.then(Vec::new);
        }

        let mut reachable = vec![false; self.num_states as usize];
        let mut stack = vec![0u32];
        reachable[0] = true;

        while let Some(state) = stack.pop() {
            for action in self.action[state as usize].values() {
                push_action_targets(action, &mut reachable, &mut stack);
            }
            for &(target, _) in self.goto[state as usize].values() {
                push_reachable_state(target, &mut reachable, &mut stack);
            }
        }

        if reachable.iter().all(|&is_reachable| is_reachable) {
            return return_mapping.then(|| (0..self.num_states).collect());
        }

        let mut mapping = vec![u32::MAX; self.num_states as usize];
        let mut kept = Vec::new();
        for (state, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                mapping[state] = kept.len() as u32;
                kept.push(state as u32);
            }
        }

        self.publish_state_mapping(&mapping, &reachable);
        return_mapping.then_some(mapping)
    }

    /// Replace several context-distinguishable LR states by one physical LR
    /// state while retaining the old source identity as a guard on the state
    /// immediately below it on the parser stack.
    ///
    /// This is an exact quotient, not an approximation. `groups` are merely
    /// candidates. A group is accepted only when every member has a non-empty,
    /// pairwise-disjoint set of possible immediate predecessors, those
    /// predecessor states remain unmerged, and predecessor provenance is known
    /// complete (no complex stack-effect edge can enter the member). Each
    /// member action is compiled to an equivalent macro stack effect with a
    /// depth-one predecessor guard. Consequently, on every reachable stack,
    /// exactly the original member's behavior remains enabled after the top
    /// state is replaced by the shared state.
    ///
    /// Returns an old-state -> final-state map covering the state domain that
    /// existed on entry. A no-op returns the identity map.
    pub fn share_context_distinguishable_states_exact(
        &mut self,
        groups: &[Vec<u32>],
    ) -> Vec<u32> {
        let debug_context_share =
            std::env::var_os("GLRMASK_DEBUG_CONTEXT_STATE_SHARING").is_some();
        let original_num_states = self.num_states;
        if original_num_states <= 1
            || groups.is_empty()
            || self.admission_policy != super::AdmissionPolicy::ExactSimulation
        {
            return (0..original_num_states).collect();
        }

        let original_len = original_num_states as usize;
        let constituent_sets: Vec<StateSubset> = (0..original_num_states)
            .map(|state| smallvec![state])
            .collect();
        let Ok(predecessors) = build_runtime_state_predecessors(self, &constituent_sets) else {
            return (0..original_num_states).collect();
        };

        // The predecessor recurrence above is exact for ordinary Shift/Split
        // shift and goto edges. If a state can be entered by a precompiled
        // arbitrary stack effect, its immediate predecessor may depend on a
        // deeper stack value that recurrence intentionally does not model.
        // Mark such targets unsafe, then propagate unsafety through replace
        // edges (which inherit their source predecessor). Non-replace edges
        // establish their source state as the new immediate predecessor and
        // therefore stop the propagation.
        let mut unsafe_predecessor = vec![false; original_len];
        for row in &self.action[..original_len] {
            for action in row.values() {
                match action {
                    Action::StackShifts(shifts) => {
                        for shift in shifts {
                            for &target in &shift.pushes {
                                if let Some(slot) = unsafe_predecessor.get_mut(target as usize) {
                                    *slot = true;
                                }
                            }
                        }
                    }
                    Action::GuardedStackShifts(shifts) => {
                        for shift in shifts {
                            for &target in &shift.pushes {
                                if let Some(slot) = unsafe_predecessor.get_mut(target as usize) {
                                    *slot = true;
                                }
                            }
                        }
                    }
                    Action::ReplaceShifts(targets) => {
                        for &target in targets.iter() {
                            if let Some(slot) = unsafe_predecessor.get_mut(target as usize) {
                                *slot = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for source in 0..original_len {
                if !unsafe_predecessor[source] {
                    continue;
                }
                for action in self.action[source].values() {
                    let target = match action {
                        Action::Shift(target, true) => Some(*target),
                        Action::Split {
                            shift: Some((target, true)),
                            ..
                        } => Some(*target),
                        _ => None,
                    };
                    if let Some(target) = target
                        && let Some(slot) = unsafe_predecessor.get_mut(target as usize)
                        && !*slot
                    {
                        *slot = true;
                        changed = true;
                    }
                }
                for &(target, replace) in self.goto[source].values() {
                    if replace
                        && let Some(slot) = unsafe_predecessor.get_mut(target as usize)
                        && !*slot
                    {
                        *slot = true;
                        changed = true;
                    }
                }
            }
        }

        let mut protected = vec![false; original_len];
        protected[0] = true;
        // Existing guards observe concrete LR-state identity. Keep every state
        // they can inspect distinct; otherwise remapping a guard set could
        // broaden its predicate.
        for row in &self.action[..original_len] {
            for action in row.values() {
                if let Action::GuardedStackShifts(shifts) = action {
                    for shift in shifts {
                        for guard in &shift.guards {
                            for &state in &guard.states {
                                if let Some(slot) = protected.get_mut(state as usize) {
                                    *slot = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        // These descriptors are runtime observations with their own state
        // semantics. Preserve them unchanged rather than extending this first
        // exact quotient with another proof obligation.
        for descriptor in &self.direct_regular_wide_frontiers {
            if let Some(slot) = protected.get_mut(descriptor.source_state as usize) {
                *slot = true;
            }
            for &state in &descriptor.target_states {
                if let Some(slot) = protected.get_mut(state as usize) {
                    *slot = true;
                }
            }
        }
        // Reduction macro compilation may need every LR state having a goto
        // for a reduced nonterminal after the reconstructed suffix is popped
        // completely. Build that exact inverse index once; repeatedly scanning
        // all goto rows inside each speculative reduction is O(groups × rows)
        // and dominated large subgrammar compositions.
        let mut goto_sources_by_nonterminal =
            vec![Vec::<u32>::new(); self.nonterminal_display_names.len()];
        for (state, row) in self.goto[..original_len].iter().enumerate() {
            for (&nonterminal, _) in row.iter() {
                if nonterminal as usize >= goto_sources_by_nonterminal.len() {
                    goto_sources_by_nonterminal
                        .resize_with(nonterminal as usize + 1, Vec::new);
                }
                goto_sources_by_nonterminal[nonterminal as usize].push(state as u32);
            }
        }

        // Cheap first pass: any state used as immediate-predecessor provenance
        // by a structurally viable candidate cannot itself be shared. Compute
        // that necessary protection before compiling any suffix-effect macro
        // rows. This may conservatively protect a predecessor for a candidate
        // whose later macro compilation would fail, but can only lose an
        // optimization; it cannot make an invalid merge admissible. In the
        // common duplicated-subgrammar case this removes a large fraction of
        // speculative interior/caller groups before the expensive exact row
        // compilation.
        let mut early_provenance_protected = vec![false; original_len];
        for input_group in groups {
            let mut group = input_group
                .iter()
                .copied()
                .filter(|&state| state < original_num_states)
                .collect::<Vec<_>>();
            group.sort_unstable();
            group.dedup();
            if group.len() < 2
                || group.iter().any(|&state| {
                    protected[state as usize]
                        || unsafe_predecessor[state as usize]
                        || predecessors[state as usize].is_empty()
                })
            {
                continue;
            }
            let Some(plan) = context_share_provenance_plan(&group, &predecessors) else {
                continue;
            };
            let first_goto = &self.goto[group[0] as usize];
            let goto_ok = group[1..].iter().all(|&state| {
                let row = &self.goto[state as usize];
                row.len() == first_goto.len()
                    && first_goto
                        .iter()
                        .all(|(&nonterminal, &target)| row.get(&nonterminal) == Some(&target))
            });
            if !goto_ok {
                continue;
            }
            // Accept cannot be represented by a predecessor-guarded macro, so
            // reject such groups before allowing them to reserve provenance.
            if group.iter().any(|&state| {
                self.action[state as usize]
                    .values()
                    .any(|action| matches!(action, Action::Accept | Action::Split { accept: true, .. }))
            }) {
                continue;
            }
            for member_predecessors in &plan.predecessors_by_member {
                for &predecessor in member_predecessors {
                    early_provenance_protected[predecessor as usize] = true;
                }
            }
        }
        // First establish which proposed classes are individually viable.
        // Only *those* classes are allowed to reserve their predecessor IDs as
        // provenance tags. Reserving predecessors for every speculative class
        // is overly conservative: an unmergeable return/caller class can use a
        // perfectly mergeable interior state as its predecessor and would then
        // freeze the useful state for no semantic reason.
        let mut viable = Vec::<ContextStateSharePlan>::new();
        for input_group in groups {
            let mut group = input_group
                .iter()
                .copied()
                .filter(|&state| state < original_num_states)
                .collect::<Vec<_>>();
            group.sort_unstable();
            group.dedup();
            let blocked_member = group.iter().copied().find(|&state| {
                protected[state as usize]
                    || unsafe_predecessor[state as usize]
                    || predecessors[state as usize].is_empty()
            });
            if group.len() < 2 || blocked_member.is_some() {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=member group={group:?} blocked={blocked_member:?} protected={:?} unsafe={:?} predecessors={:?}",
                        group
                            .iter()
                            .map(|&state| protected[state as usize])
                            .collect::<Vec<_>>(),
                        group
                            .iter()
                            .map(|&state| unsafe_predecessor[state as usize])
                            .collect::<Vec<_>>(),
                        group
                            .iter()
                            .map(|&state| predecessors[state as usize].to_vec())
                            .collect::<Vec<_>>(),
                    );
                }
                continue;
            }
            let Some(mut plan) = context_share_provenance_plan(&group, &predecessors) else {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=overlapping_predecessors group={group:?}",
                    );
                }
                continue;
            };

            // Gotos are not terminal actions and therefore cannot carry the
            // provenance guards. Require the concrete goto rows to be exactly
            // equal; a disjoint goto key would otherwise become spuriously
            // available from every source represented by the shared state.
            let first_goto = &self.goto[group[0] as usize];
            let goto_ok = group[1..].iter().all(|&state| {
                let row = &self.goto[state as usize];
                row.len() == first_goto.len()
                    && first_goto
                        .iter()
                        .all(|(&nonterminal, &target)| row.get(&nonterminal) == Some(&target))
            });
            if !goto_ok {
                if debug_context_share {
                    eprintln!("[glrmask/debug][context_state_share] reject=goto group={group:?}");
                }
                continue;
            }

            if group
                .iter()
                .any(|&state| early_provenance_protected[state as usize])
            {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=early_provenance group={group:?}",
                    );
                }
                continue;
            }

            // Build the exact row against the unmodified table. Each initial
            // frame reconstructs the concrete member only when the immediate
            // predecessor belongs to that member's certified, disjoint
            // predecessor set. The existing suffix-effect compiler then
            // executes reductions to the eventual consuming effect.
            let mut terminals = BTreeSet::<TerminalID>::new();
            for &state in &group {
                terminals.extend(self.action[state as usize].keys());
            }
            let mut action_row = ActionRow::default();
            let mut row_ok = true;
            for terminal in terminals {
                let mut effects = Vec::<GuardedStackShift>::new();
                let mut accepts = 0usize;
                for (member_index, &state) in group.iter().enumerate() {
                    let Some(action) = self.action[state as usize].get(&terminal) else {
                        continue;
                    };
                    let frame = context_share_frame_for_member(
                        state,
                        &plan.predecessors_by_member[member_index],
                    );
                    if collect_context_share_effects_from_frame(
                        self,
                        terminal,
                        state,
                        action,
                        frame,
                        &goto_sources_by_nonterminal,
                        &mut effects,
                        &mut accepts,
                    )
                    .is_err()
                    {
                        row_ok = false;
                        break;
                    }
                }
                // Accept has no guard-bearing representation. Keep any group
                // containing an accepting action separate.
                if !row_ok || accepts != 0 {
                    row_ok = false;
                    break;
                }
                normalize_guarded_effects_for_suffix_quotient(&mut effects);
                for effect in &effects {
                    for guard in &effect.guards {
                        plan.generated_guard_states.extend_from_slice(&guard.states);
                    }
                }
                if !effects.is_empty() {
                    action_row.insert(terminal, Action::GuardedStackShifts(effects));
                }
            }
            if !row_ok {
                if debug_context_share {
                    eprintln!("[glrmask/debug][context_state_share] reject=row group={group:?}");
                }
                continue;
            }

            if debug_context_share {
                eprintln!(
                    "[glrmask/debug][context_state_share] viable group={group:?} predecessors={:?}",
                    plan.predecessors_by_member,
                );
            }
            plan.generated_guard_states.sort_unstable();
            plan.generated_guard_states.dedup();

            // State 0 is the table's distinguished subgrammar entry state. A
            // later composition overlays that row onto each caller instead of
            // copying state 0 as an ordinary child state. Therefore an
            // internal macro action that observes state 0 in a stack guard is
            // exact for this standalone table but is not representable by the
            // current subgrammar-splice ABI without expanding the observation
            // to caller states (and preserving caller correlation). Keep the
            // structural quotient closed under further composition by simply
            // rejecting such an optimization. This can only leave states
            // unmerged; it cannot change the accepted language.
            if plan.generated_guard_states.binary_search(&0).is_ok() {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=compositional_start_guard group={group:?}",
                    );
                }
                continue;
            }

            plan.action_row = action_row;
            viable.push(plan);
        }

        // Every predecessor state used by a provenance guard is a runtime tag.
        // Give that role priority over sharing the tag itself. This also makes
        // all guard predicates stable under the later old->shared state map.
        let mut provenance_protected = vec![false; original_len];
        for plan in &viable {
            for member_predecessors in &plan.predecessors_by_member {
                for &guard_state in member_predecessors {
                    if let Some(slot) = provenance_protected.get_mut(guard_state as usize) {
                        *slot = true;
                    }
                }
            }
            for &guard_state in &plan.generated_guard_states {
                if let Some(slot) = provenance_protected.get_mut(guard_state as usize) {
                    *slot = true;
                }
            }
        }
        let mut claimed = vec![false; original_len];
        let mut accepted = Vec::<ContextStateSharePlan>::new();
        for plan in viable {
            let group = &plan.group;
            if group.iter().any(|&state| {
                provenance_protected[state as usize] || claimed[state as usize]
            }) {
                if debug_context_share {
                    eprintln!(
                        "[glrmask/debug][context_state_share] reject=provenance group={group:?} protected={:?}",
                        group
                            .iter()
                            .map(|&state| provenance_protected[state as usize])
                            .collect::<Vec<_>>(),
                    );
                }
                continue;
            }
            for &state in group {
                claimed[state as usize] = true;
            }
            accepted.push(plan);
        }

        if accepted.is_empty() {
            return (0..original_num_states).collect();
        }

        // The exact macro rows were already compiled during viability. Once
        // the accepted classes are fixed, transport their embedded state IDs
        // through the final sharing map rather than recomputing the same
        // reduction/suffix closure a second time.
        let mut share_map = (0..original_num_states).collect::<Vec<_>>();
        let shared_base = original_num_states;
        for (index, plan) in accepted.iter().enumerate() {
            let shared = shared_base + index as u32;
            for &state in &plan.group {
                share_map[state as usize] = shared;
            }
        }
        share_map.extend((shared_base..shared_base + accepted.len() as u32).collect::<Vec<_>>());

        let had_advance_rows = self.advance.len() == original_len;
        let mut shared_rows = Vec::<(ActionRow, GotoRow, Option<BitSet>)>::new();
        for plan in &accepted {
            let group = &plan.group;
            let mut action_row = plan.action_row.clone();
            remap_action_row_targets_in_place(&mut action_row, &share_map);
            action_row.for_each_value_mut(|action| {
                if let Action::GuardedStackShifts(effects) = action {
                    normalize_guarded_effects_for_suffix_quotient(effects);
                }
            });

            let goto_row = self.goto[group[0] as usize]
                .iter()
                .map(|(&nonterminal, &(target, replace))| {
                    (nonterminal, (share_map[target as usize], replace))
                })
                .collect();
            let advance = had_advance_rows.then(|| union_advance_rows(self, &group.iter().copied().collect()));
            shared_rows.push((action_row, goto_row, advance));
        }

        // Redirect every incoming table edge before pruning the now-unreachable
        // concrete member states.
        for row in &mut self.action[..original_len] {
            remap_action_row_targets_in_place(row, &share_map);
        }
        for row in &mut self.goto[..original_len] {
            remap_goto_row_targets_in_place(row, &share_map);
        }
        for (action, goto, advance) in shared_rows {
            self.action.push(action);
            self.goto.push(goto);
            if let Some(advance) = advance {
                self.advance.push(advance);
            }
            self.num_states += 1;
        }
        self.forwarded_shifts = self
            .forwarded_shifts
            .iter()
            .filter_map(|&(state, terminal)| {
                (share_map[state as usize] == state).then_some((state, terminal))
            })
            .collect();

        let prune_map = self.prune_unreachable_states_with_mapping();
        let mut final_map = vec![u32::MAX; original_len];
        for old in 0..original_len {
            let shared = share_map[old] as usize;
            final_map[old] = prune_map.get(shared).copied().unwrap_or(u32::MAX);
        }
        debug_assert!(final_map.iter().all(|&state| state != u32::MAX));
        self.rebuild_guarded_shift_index();
        self.compress_default_action_rows();
        final_map
    }

    /// Partially evaluate linker-internal zero-width controls into ordinary
    /// consuming parser actions.
    ///
    /// For every source LR state and every real terminal reachable after zero
    /// or more control actions, this compiles the exact guarded stack relation
    /// `control* ; terminal` into one `StackShifts`/`GuardedStackShifts` cell.
    /// The next consuming cell performs its own leading closure, so no runtime
    /// control closure is required between tokens. EOF is represented by row
    /// presence, matching `stacks_finished` semantics.
    pub fn eliminate_control_terminals_exact(
        &mut self,
    ) -> Result<ControlEliminationReport, String> {
        if self.control_terminals.is_empty() {
            return Ok(ControlEliminationReport {
                states: self.num_states as usize,
                controls: 0,
                compiled_cells: 0,
                closure_frames: 0,
                stack_effect_visits: 0,
                changed_terminals: Vec::new(),
                elapsed_ms: 0.0,
            });
        }
        // Proof invariant for the prepared static linker: the new signed-transfer
        // boundary compiler must never invoke control elimination. Count every
        // real elimination run (no-op calls on control-free tables do not count)
        // so link tests can assert the counter does not move across the link.
        CONTROL_ELIMINATION_RUNS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let started_at = std::time::Instant::now();
        let source = self.clone();
        let controls = source.control_terminals.clone();
        let predecessor_started_at = std::time::Instant::now();
        let predecessor_free = control_elimination_has_known_tops(&source, &controls);
        let predecessors = if predecessor_free {
            Vec::new()
        } else if std::env::var_os("GLRMASK_DISABLE_CONTROL_PREDECESSOR_DEMAND").is_none() {
            let origins = (0..source.num_states)
                .filter(|&state| controls.iter().any(|control| source.action[state as usize].contains_key(control)))
                .collect::<Vec<_>>();
            build_control_elimination_predecessors_demand(&source, &origins)?
        } else {
            build_control_elimination_predecessors(&source)?
        };
        let predecessor_ms = predecessor_started_at.elapsed().as_secs_f64() * 1000.0;
        let budget = UnitInlineBudget::exact_control_elimination();
        let mut compiled_rows = Vec::with_capacity(source.num_states as usize);
        let mut compiled_cells = 0usize;
        let mut closure_frames = 0usize;
        let mut copied_row_count = 0usize;
        let mut symbolic_row_count = 0usize;
        let mut changed_terminals = BTreeSet::<TerminalID>::new();
        let mut slowest_rows = Vec::<(u64, u32, usize, usize)>::new();
        let profile_control = std::env::var_os("GLRMASK_PROFILE_CONTROL_ELIMINATION").is_some();

        for origin_state in 0..source.num_states {
            let row_started_at = profile_control.then(std::time::Instant::now);
            let visits_before = budget.stack_effect_visits();
            let source_row = &source.action[origin_state as usize];
            if !controls
                .iter()
                .any(|control| source_row.contains_key(control))
            {
                // A leading zero-width control closure can only begin with an
                // action in the current top-state row. With no control cell,
                // the closure is exactly identity and every ordinary action is
                // already the required compiled result. Copy it verbatim;
                // symbolically recompiling every child-table row dominates
                // realistic multi-schema composition for no semantic gain.
                compiled_cells = compiled_cells.saturating_add(source_row.len());
                compiled_rows.push(source_row.clone());
                closure_frames = closure_frames.saturating_add(1);
                copied_row_count += 1;
                continue;
            }
            symbolic_row_count += 1;

            let mut depth_cache = FxHashMap::default();
            let closure = control_effect_closure(
                &source,
                &predecessors,
                origin_state,
                &controls,
                StackEffectFrame {
                    pop: 0,
                    pushes: Vec::new(),
                    guards: Vec::new(),
                },
                &mut depth_cache,
                &budget,
            )?;
            closure_frames = closure_frames.saturating_add(closure.len());

            let mut candidates = BTreeSet::<TerminalID>::new();
            for frame in &closure {
                for (top_state, _) in frame_top_branches(
                    &predecessors,
                    origin_state,
                    frame,
                    &mut depth_cache,
                    &budget,
                )? {
                    candidates.extend(
                        source.action[top_state as usize]
                            .keys()
                            .filter(|terminal| !controls.contains(terminal)),
                    );
                }
            }

            let mut row = ActionRow::default();
            for terminal in candidates {
                if terminal == EOF {
                    if matches!(source.action(origin_state, EOF), Some(Action::Accept)) {
                        row.insert(EOF, Action::Accept);
                        compiled_cells += 1;
                        continue;
                    }
                    let mut effects = Vec::<GuardedStackShift>::new();
                    for frame in &closure {
                        for (top_state, top_frame) in frame_top_branches(
                            &predecessors,
                            origin_state,
                            frame,
                            &mut depth_cache,
                            &budget,
                        )? {
                            let Some(action) = source.action[top_state as usize].get(&EOF) else {
                                continue;
                            };
                            effects.extend(eof_accept_effects_for_action(
                                &source,
                                &predecessors,
                                origin_state,
                                top_state,
                                action,
                                top_frame,
                                &mut depth_cache,
                                &mut FxHashSet::default(),
                                &budget,
                            )?);
                        }
                    }
                    if let Some(action) = control_stack_effect_action(&source, effects) {
                        row.insert(EOF, action);
                        compiled_cells += 1;
                    }
                    continue;
                }

                let mut effects = Vec::<GuardedStackShift>::new();
                for frame in &closure {
                    for (top_state, top_frame) in frame_top_branches(
                        &predecessors,
                        origin_state,
                        frame,
                        &mut depth_cache,
                        &budget,
                    )? {
                        let Some(action) = source.action[top_state as usize].get(&terminal) else {
                            continue;
                        };
                        let Some(result) = stack_effects_for_action(
                            &source,
                            &predecessors,
                            origin_state,
                            terminal,
                            top_state,
                            action,
                            top_frame,
                            &mut depth_cache,
                            &mut FxHashSet::default(),
                            &mut Vec::new(),
                            &budget,
                        ) else {
                            return Err(control_elimination_failure(
                                &budget,
                                origin_state,
                                terminal,
                                "real-terminal stack-effect traversal failed",
                            ));
                        };
                        effects.extend(result.effects);
                    }
                }

                if let Some(action) = control_stack_effect_action(&source, effects) {
                    row.insert(terminal, action);
                    compiled_cells += 1;
                }
            }
            for terminal in source_row.keys().chain(row.keys()) {
                if controls.contains(&terminal) {
                    continue;
                }
                if source_row.get(&terminal) != row.get(&terminal) {
                    changed_terminals.insert(terminal);
                }
            }
            compiled_rows.push(row);
            if let Some(row_started_at) = row_started_at {
                let elapsed_ns = row_started_at.elapsed().as_nanos() as u64;
                let visits = budget.stack_effect_visits().saturating_sub(visits_before);
                slowest_rows.push((elapsed_ns, origin_state, closure.len(), visits));
            }
        }

        if budget.is_aborted() {
            return Err(control_elimination_failure(
                &budget,
                0,
                EOF,
                "control elimination exceeded its compile budget",
            ));
        }

        self.action = compiled_rows;
        self.control_terminals.clear();
        self.forwarded_shifts.clear();
        self.admission_policy = AdmissionPolicy::RowPresenceExact;
        self.rebuild_advance_rows_from_actions();
        self.rebuild_unconditional_advance_rows();
        self.rebuild_guarded_shift_index();

        if profile_control {
            slowest_rows.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
            slowest_rows.truncate(20);
            eprintln!(
                "[glrmask/profile][control_elimination] states={} controls={} copied_rows={} symbolic_rows={} compiled_cells={} closure_frames={} stack_effect_visits={} predecessor_free={} predecessor_ms={predecessor_ms:.3} total_ms={:.3} slowest_rows={:?}",
                source.num_states,
                controls.len(),
                copied_row_count,
                symbolic_row_count,
                compiled_cells,
                closure_frames,
                budget.stack_effect_visits(),
                predecessor_free,
                started_at.elapsed().as_secs_f64() * 1000.0,
                slowest_rows,
            );
        }

        Ok(ControlEliminationReport {
            states: self.num_states as usize,
            controls: controls.len(),
            compiled_cells,
            closure_frames,
            stack_effect_visits: budget.stack_effect_visits(),
            changed_terminals: changed_terminals.into_iter().collect(),
            elapsed_ms: started_at.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Collapse unit reductions by inlining their destination actions.
    ///
    /// When inlining produces multiple shift destinations, create a synthetic
    /// merged state whose row is the union of its constituents' rows. This
    /// keeps the parser representation unchanged: every action cell still has
    /// at most one shift slot, but that shift target may be a merged state.
    #[cfg(test)]
    pub(super) fn collapse_sr_unit_reductions_for_correctness_oracle(
        &mut self,
    ) -> UnitReductionInliningReport {
        use std::sync::atomic::{AtomicU8, AtomicUsize};

        let budget = UnitInlineBudget {
            started_at: std::time::Instant::now(),
            max_ms: u128::MAX,
            max_iterations: usize::MAX,
            max_cells: usize::MAX,
            max_synthetic_states: usize::MAX,
            max_stack_effect_visits: usize::MAX,
            iterations: AtomicUsize::new(0),
            cells: AtomicUsize::new(0),
            synthetic_states: AtomicUsize::new(0),
            stack_effect_visits: AtomicUsize::new(0),
            abort_code: AtomicU8::new(ABORT_NONE),
        };
        let mut undo = UnitInlineUndo::new(self);
        let protected = BitSet::new(self.num_terminals as usize);
        self.collapse_sr_unit_reductions_with_compatible_gotos_inner(
            &budget,
            &mut undo,
            &protected,
        );
        let mut report = budget.report();
        if report.aborted {
            undo.rollback(self);
        } else {
            report.changed_original_states = undo.changed_original_states();
        }
        report
    }

    pub(super) fn collapse_sr_unit_reductions_with_compatible_gotos(
        &mut self,
    ) -> UnitReductionInliningReport {
        let protected = BitSet::new(self.num_terminals as usize);
        self.collapse_sr_unit_reductions_with_compatible_gotos_except(&protected)
    }

    pub(super) fn collapse_sr_unit_reductions_with_compatible_gotos_except(
        &mut self,
        protected_terminals: &BitSet,
    ) -> UnitReductionInliningReport {
        debug_assert_eq!(protected_terminals.len(), self.num_terminals as usize);
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let journal_started_at = profile_enabled.then(std::time::Instant::now);
        let mut undo = UnitInlineUndo::new(self);
        let journal_ms = journal_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        let inner_started_at = profile_enabled.then(std::time::Instant::now);
        let budget = UnitInlineBudget::from_env();
        self.collapse_sr_unit_reductions_with_compatible_gotos_inner(
            &budget,
            &mut undo,
            protected_terminals,
        );
        let mut report = budget.report();
        let inner_ms = inner_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        if report.aborted {
            undo.rollback(self);
        } else {
            report.changed_original_states = undo.changed_original_states();
        }
        if let (Some(journal_ms), Some(inner_ms)) = (journal_ms, inner_ms) {
            eprintln!(
                "[glrmask/profile][unit_reduction_inlining] outcome={} reason={} iterations={} cells={} synthetic_states={} stack_effect_visits={} journal_ms={:.3} inner_ms={:.3} elapsed_ms={:.3} max_ms={} max_iterations={} max_cells={} max_synthetic_states={} max_stack_effect_visits={}",
                if report.aborted { "aborted" } else { "committed" },
                report.reason.unwrap_or("none"),
                report.iterations,
                report.cells,
                report.synthetic_states,
                report.stack_effect_visits,
                journal_ms,
                inner_ms,
                report.elapsed_ms,
                budget.max_ms,
                budget.max_iterations,
                budget.max_cells,
                budget.max_synthetic_states,
                budget.max_stack_effect_visits,
            );
        }
        report
    }

    fn collapse_sr_unit_reductions_with_compatible_gotos_inner(
        &mut self,
        budget: &UnitInlineBudget,
        undo: &mut UnitInlineUndo,
        protected_terminals: &BitSet,
    ) {
        self.collapse_sr_unit_reductions_with_completed_work_abort(
            budget,
            undo,
            protected_terminals,
            completed_work_abort_eligible(self.construction)
                && env_flag_enabled(UNIT_INLINE_COMPLETED_ABORT_ENV, true),
        );
    }

    fn collapse_sr_unit_reductions_with_completed_work_abort(
        &mut self,
        budget: &UnitInlineBudget,
        undo: &mut UnitInlineUndo,
        protected_terminals: &BitSet,
        stop_proven_abort_work: bool,
    ) {
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let original_num_states = self.num_states;
        let mut constituent_sets: Vec<StateSubset> = (0..self.num_states)
            .map(|state| smallvec![state])
            .collect();
        // Singleton subsets map directly to their state ID, so only synthetic
        // (multi-state) subsets need hash-map entries.
        let mut subset_to_state: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
        let mut failed_subsets: FxHashSet<Vec<u32>> = FxHashSet::default();
        let mut dirty_original_states: BTreeSet<u32> = BTreeSet::new();
        let mut cached_predecessors: Option<RuntimePredecessors> = None;
        let mut next_worklist: Option<Vec<UnitInlineCell>> = None;
        let mut dependencies = UnitInlineDependencies::default();

        loop {
            if !budget.record_iteration() {
                break;
            }
            let iteration = budget.iterations();
            let worklist = next_worklist.take();
            let using_worklist = worklist.is_some();
            let cells_before = budget.cells();
            let visits_before = budget.stack_effect_visits();
            let refresh_started_at = profile_enabled.then(std::time::Instant::now);
            if !using_worklist && !dirty_original_states.is_empty() {
                refresh_merged_states_depending_on(
                    self,
                    original_num_states,
                    &mut constituent_sets,
                    &mut subset_to_state,
                    &mut failed_subsets,
                    &dirty_original_states,
                    budget,
                );
                if budget.is_aborted() {
                    break;
                }
                dirty_original_states.clear();
            }

            let refresh_ms = refresh_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let predecessors_started_at = profile_enabled.then(std::time::Instant::now);
            if !using_worklist {
                let Ok(predecessors) = build_runtime_state_predecessors(self, &constituent_sets) else {
                    budget.abort(ABORT_REFRESH_REBUILD);
                    break;
                };
                cached_predecessors = Some(predecessors);
            }
            let predecessors = cached_predecessors
                .as_ref()
                .expect("first unit-inlining pass constructs predecessors");
            let predecessors_ms = if using_worklist {
                0.0
            } else {
                predecessors_started_at
                    .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                    .unwrap_or(0.0)
            };
            // The scan below computes updates against a stable snapshot of the
            // current original rows for this iteration. Delayed synthetic
            // states may be appended, but existing rows are not mutated until
            // after the scan.
            let nstates = original_num_states as usize;
            let worklist_cells = worklist.as_ref().map(|cells| {
                cells.iter().copied().collect::<FxHashSet<UnitInlineCell>>()
            });
            let synthetic_states_before_scan = self.num_states;
            let mut pending_updates: Vec<(usize, TerminalID, CellUpdate)> = Vec::new();
            let scan_started_at = profile_enabled.then(std::time::Instant::now);

            // Phase 1 (parallel, read-only): stack-shift inlining is a pure
            // function of the stable table snapshot and the predecessors graph.
            // Schedule only states that actually have a candidate cell. The old
            // implementation spawned a task, a local cache, and a child budget
            // for every state, even though most rows contain only shifts.
            let candidate_states = if let Some(cells) = &worklist_cells {
                let mut states = cells
                    .iter()
                    .map(|&(state, _)| state as usize)
                    .collect::<Vec<_>>();
                states.sort_unstable();
                states.dedup();
                states
            } else {
                (0..nstates)
                    .filter(|&state| {
                        self.action[state]
                            .iter()
                            .any(|(terminal, action)| {
                                !protected_terminals.contains(terminal as usize)
                                    && action_has_inlinable_reductions(action)
                            })
                    })
                    .collect::<Vec<_>>()
            };
            let stack_phase_started_at = profile_enabled.then(std::time::Instant::now);
            let isolate_state_stack_effect_budget =
                self.construction == GlrTableConstruction::ExperimentalCoreMerged;
            let state_stack_effect_visit_limit = unit_inline_state_max_stack_effect_visits();
            // Core-merged children stop at the local cap plus one failed visit.
            // If even every child exhausting that cap cannot exceed the parent
            // limit, no certificate is possible in this phase: pay no atomics.
            let certificate_possible = !isolate_state_stack_effect_budget
                || candidate_states.len().saturating_mul(
                    state_stack_effect_visit_limit
                        .min(budget.max_stack_effect_visits)
                        .saturating_add(1),
                ) > budget.max_stack_effect_visits
                    .saturating_sub(budget.stack_effect_visits());
            let completed_work = (stop_proven_abort_work && certificate_possible)
                .then(|| CompletedStackEffectWork::new(budget))
                .flatten();
            let per_state: Vec<(
                usize,
                Vec<(TerminalID, StackInlineAttempt)>,
                Vec<(UnitInlineCell, UnitInlineCell)>,
                UnitInlineBudget,
            )> = {
                let table: &GLRTable = self;
                let predecessors = &predecessors;
                candidate_states
                    .par_iter()
                    .map(|&state| {
                        let local_budget = if isolate_state_stack_effect_budget {
                            budget.child_with_stack_effect_visit_limit(
                                state_stack_effect_visit_limit,
                            )
                        } else {
                            budget.child()
                        };
                        if completed_work.as_ref().is_some_and(|work| work.exceeded()) {
                            // A complete result slot retains indexed collection
                            // order; this empty job contributes no actual work.
                            return (state, Vec::new(), Vec::new(), local_budget);
                        }
                        let mut local_depth_cache: FxHashMap<(u32, u32), Option<StateSubset>> =
                            FxHashMap::default();
                        let mut reads = Vec::new();
                        let rows = table.action[state]
                            .iter()
                            .filter(|(tid, action)| {
                                !protected_terminals.contains(*tid as usize)
                                    && action_has_inlinable_reductions(action)
                                    && worklist_cells.as_ref().is_none_or(|cells| {
                                        cells.contains(&(state as u32, *tid))
                                    })
                            })
                            .map(|(tid, action)| {
                                let inlined = if local_budget.is_aborted() {
                                    StackInlineAttempt {
                                        action: None,
                                        outcome: StackInlineOutcomeKind::TraversalFailed,
                                    }
                                } else if is_trivial_origin_dependent_one_push(
                                    table,
                                    predecessors,
                                    state as u32,
                                    tid,
                                    action,
                                    &mut reads,
                                ) {
                                    StackInlineAttempt {
                                        action: None,
                                        outcome: StackInlineOutcomeKind::TrivialOriginDependentOnePush,
                                    }
                                } else {
                                    try_inline_action_to_stack_shifts_detailed(
                                        table,
                                        predecessors,
                                        state as u32,
                                        tid,
                                        action,
                                        &mut local_depth_cache,
                                        &mut reads,
                                        &local_budget,
                                    )
                                };
                                (tid, inlined)
                            })
                            .collect();
                        if let Some(work) = &completed_work {
                            work.record_completed(local_budget.stack_effect_visits());
                        }
                        (state, rows, reads, local_budget)
                    })
                    .collect()
            };

            let stack_phase_ms = stack_phase_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            if completed_work.as_ref().is_some_and(|work| work.exceeded()) {
                // Every completed child's nonnegative count is charged by both
                // reference fold policies, including locally exhausted states.
                // The reference therefore must abort before committing this
                // iteration. Account for all completed work and use the same
                // outer undo journal to restore earlier iterations as well.
                for (_, _, _, child) in &per_state {
                    budget.fold_in_counts(child);
                }
                budget.abort(ABORT_STACK_EFFECT_VISITS);
                if profile_enabled {
                    eprintln!(
                        "[glrmask/profile][unit_reduction_completed_abort] candidates={} nonempty_results={} stack_phase_ms={stack_phase_ms:.3} completed_stack_visits={} max_stack_visits={}",
                        candidate_states.len(),
                        per_state.iter().filter(|(_, rows, _, _)| !rows.is_empty()).count(),
                        budget.stack_effect_visits(),
                        budget.max_stack_effect_visits,
                    );
                }
                break;
            }
            let unit_phase_started_at = profile_enabled.then(std::time::Instant::now);
            let mut stack_effect_candidates = 0usize;
            let mut unit_candidates = 0usize;
            let mut stack_effect_resolved = 0usize;
            let mut stack_effect_updates = 0usize;
            let mut unit_updates = 0usize;
            let mut candidate_kind_counts = [0usize; StackInlineCandidateKind::COUNT];
            let mut resolved_kind_counts = [0usize; StackInlineCandidateKind::COUNT];
            let mut direct_unit_candidates = 0usize;
            let mut direct_unit_resolved = 0usize;
            let mut stack_outcome_counts = [0usize; StackInlineOutcomeKind::COUNT];
            // Phase 2 (serial): apply stack-shift results in canonical state
            // order and run unit-reduction inlining (which may synthesize
            // shared states). `candidate_states` is sorted and Rayon preserves
            // indexed-iterator collection order.
            if !budget.is_aborted() {
                'scan: for (state, row_stack_shifts, reads, local_budget) in per_state {
                    if isolate_state_stack_effect_budget {
                        budget.fold_in_isolated_state(&local_budget);
                    } else {
                        budget.fold_in(&local_budget);
                    }
                    dependencies.record_reads(reads);
                    if budget.is_aborted() {
                        break 'scan;
                    }
                    for (tid, stack_attempt) in row_stack_shifts {
                        let StackInlineAttempt {
                            action: stack_inlined,
                            outcome: stack_outcome,
                        } = stack_attempt;
                        let Some(action) = self.action[state].get(&tid).cloned() else {
                            continue;
                        };
                        stack_effect_candidates += 1;
                        stack_outcome_counts[stack_outcome.index()] += 1;
                        let candidate_kind = stack_inline_candidate_kind(&action);
                        candidate_kind_counts[candidate_kind.index()] += 1;
                        let is_unit_candidate = action_needs_unit_reduction_inlining(&action);
                        unit_candidates += usize::from(is_unit_candidate);
                        let is_direct_unit_candidate = profile_enabled
                            && matches!(
                                &action,
                                Action::Reduce(nonterminal, 1)
                                    if unit_reduce_destination(
                                        self,
                                        predecessors,
                                        state as u32,
                                        *nonterminal,
                                    )
                                    .is_some()
                            );
                        direct_unit_candidates += usize::from(is_direct_unit_candidate);
                        let resolved_by_stack_effect = stack_inlined.is_some();
                        direct_unit_resolved += usize::from(
                            is_direct_unit_candidate && resolved_by_stack_effect,
                        );
                        stack_effect_resolved += usize::from(resolved_by_stack_effect);
                        if resolved_by_stack_effect {
                            resolved_kind_counts[candidate_kind.index()] += 1;
                        }
                        if !budget.record_cell() {
                            break 'scan;
                        }

                        let update = if let Some(inlined) = stack_inlined {
                            Some(CellUpdate::Set(inlined))
                        } else if is_unit_candidate {
                            let mut visiting = BTreeSet::new();
                            match try_inline_unit_reductions_for_cell_inner(
                                self,
                                predecessors,
                                state as u32,
                                tid,
                                &action,
                                &mut constituent_sets,
                                &mut subset_to_state,
                                &mut failed_subsets,
                                &mut visiting,
                                (state as u32, tid),
                                &mut dependencies,
                                budget,
                            ) {
                                Ok(update) => update,
                                Err(()) => continue,
                            }
                        } else {
                            None
                        };
                        if budget.is_aborted() {
                            break 'scan;
                        }

                        match update {
                            Some(CellUpdate::Set(new_action)) if new_action != action => {
                                if resolved_by_stack_effect {
                                    stack_effect_updates += 1;
                                } else {
                                    unit_updates += 1;
                                }
                                pending_updates.push((state, tid, CellUpdate::Set(new_action)));
                            }
                            Some(CellUpdate::Remove) => {
                                if resolved_by_stack_effect {
                                    stack_effect_updates += 1;
                                } else {
                                    unit_updates += 1;
                                }
                                pending_updates.push((state, tid, CellUpdate::Remove));
                            }
                            _ => {}
                        }
                    }
                }
            }


            let unit_phase_ms = unit_phase_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            if pending_updates.is_empty() || budget.is_aborted() {
                if profile_enabled {
                    eprintln!(
                        "[glrmask/profile][unit_reduction_inlining_iteration] iteration={} refresh_ms={:.3} predecessors_ms={:.3} stack_phase_ms={:.3} unit_phase_ms={:.3} scan_ms={:.3} apply_ms=0.000 candidates={} unit_candidates={} stack_resolved={} stack_updates={} unit_updates={} kinds={:?} resolved_kinds={:?} direct_unit_candidates={} direct_unit_resolved={} outcomes={:?} scanned_cells={} stack_effect_visits={} pending_updates={} terminal={}",
                        iteration,
                        refresh_ms,
                        predecessors_ms,
                        stack_phase_ms,
                        unit_phase_ms,
                        scan_started_at
                            .expect("profile start exists when profiling is enabled")
                            .elapsed()
                            .as_secs_f64()
                            * 1000.0,
                        stack_effect_candidates,
                        unit_candidates,
                        stack_effect_resolved,
                        stack_effect_updates,
                        unit_updates,
                        candidate_kind_counts,
                        resolved_kind_counts,
                        direct_unit_candidates,
                        direct_unit_resolved,
                        stack_outcome_counts,
                        budget.cells() - cells_before,
                        budget.stack_effect_visits() - visits_before,
                        pending_updates.len(),
                        if budget.is_aborted() { "abort" } else { "stable" },
                    );
                }
                break;
            }

            let update_count = pending_updates.len();
            let scan_ms = scan_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let apply_started_at = profile_enabled.then(std::time::Instant::now);
            let mut changed_cells = Vec::with_capacity(update_count);
            let mut predecessor_edges_changed = 0usize;
            for (state, tid, update) in pending_updates {
                undo.record_cell(self, state, tid);
                let old_action = self.action[state].get(&tid).cloned();
                match update {
                    CellUpdate::Set(new_action) => {
                        predecessor_edges_changed += usize::from(
                            old_action
                                .as_ref()
                                .and_then(action_predecessor_signature)
                                != action_predecessor_signature(&new_action),
                        );
                        self.action[state].insert(tid, new_action);
                    }
                    CellUpdate::Remove => {
                        predecessor_edges_changed += usize::from(
                            old_action
                                .as_ref()
                                .and_then(action_predecessor_signature)
                                .is_some(),
                        );
                        self.action[state].remove(&tid);
                        if self.advance.len() == self.num_states as usize
                            && let Some(bit) = self.terminal_bit(tid)
                        {
                            self.advance[state].clear(bit);
                        }
                    }
                }
                dirty_original_states.insert(state as u32);
                changed_cells.push((state as u32, tid));
            }
            let can_use_worklist = predecessor_edges_changed == 0
                && self.num_states == synthetic_states_before_scan
                && self.num_states == original_num_states;
            if can_use_worklist {
                next_worklist = Some(dependencies.worklist_for_changes(changed_cells.iter().copied()));
            } else {
                cached_predecessors = None;
                next_worklist = None;
            }
            if profile_enabled {
                eprintln!(
                    "[glrmask/profile][unit_reduction_inlining_iteration] iteration={} refresh_ms={:.3} predecessors_ms={:.3} stack_phase_ms={:.3} unit_phase_ms={:.3} scan_ms={:.3} apply_ms={:.3} candidates={} unit_candidates={} stack_resolved={} stack_updates={} unit_updates={} kinds={:?} resolved_kinds={:?} direct_unit_candidates={} direct_unit_resolved={} outcomes={:?} scanned_cells={} stack_effect_visits={} pending_updates={} terminal=continue",
                    iteration,
                    refresh_ms,
                    predecessors_ms,
                    stack_phase_ms,
                    unit_phase_ms,
                    scan_ms,
                    apply_started_at
                        .expect("profile start exists when profiling is enabled")
                        .elapsed()
                        .as_secs_f64()
                        * 1000.0,
                    stack_effect_candidates,
                    unit_candidates,
                    stack_effect_resolved,
                    stack_effect_updates,
                    unit_updates,
                    candidate_kind_counts,
                    resolved_kind_counts,
                    direct_unit_candidates,
                    direct_unit_resolved,
                    stack_outcome_counts,
                    budget.cells() - cells_before,
                    budget.stack_effect_visits() - visits_before,
                    update_count,
                );
            }
        }
    }



    /// Collapse recognizer-equivalent stack-effect alternatives by replacing a
    /// set of pushed LR stack suffixes with one synthetic suffix state.
    ///
    /// A synthetic suffix state denotes a finite union of concrete LR stack
    /// suffixes over the same lower stack. Its rows are compiled from those
    /// concrete suffixes into ordinary `StackShifts`/`GuardedStackShifts`, so the
    /// runtime still consumes the same table representation. The pass is purely
    /// table-level: it does not change import, grammar lowering, or parser
    /// runtime behavior.
    pub(super) fn quotient_recognizer_stack_suffixes(&mut self) {
        let protected = BitSet::new(self.num_terminals as usize);
        self.quotient_recognizer_stack_suffixes_except(&protected);
    }

    pub(super) fn quotient_recognizer_stack_suffixes_except(&mut self, protected: &BitSet) {
        if !recognizer_suffix_quotient_enabled() {
            return;
        }

        let mut quotient = SuffixQuotient::new();
        let original_states = self.num_states as usize;
        let mut changed = false;

        for state in 0..original_states {
            let terminals: Vec<TerminalID> = self.action[state].keys().collect();
            for terminal in terminals {
                if protected.get(terminal as usize) {
                    continue;
                }
                let Some(action) = self.action[state].get(&terminal).cloned() else {
                    continue;
                };
                let Some(new_action) = quotient.normalize_action(self, action.clone()) else {
                    continue;
                };
                if new_action != action {
                    self.action[state].insert(terminal, new_action);
                    changed = true;
                }
            }
        }

        if changed || quotient.created_states > 0 {
            self.extend_advance_rows_from_actions();
            self.validate_structure("recognizer suffix quotient before prune");
            self.prune_unreachable_states();
            self.merge_identical_rows();
            self.validate_structure("recognizer suffix quotient after merge");
        }
    }
}

#[derive(Debug)]
struct SuffixQuotient {
    suffix_to_state: FxHashMap<Vec<Vec<u32>>, u32>,
    failed_suffixes: FxHashSet<Vec<Vec<u32>>>,
    max_states: usize,
    max_alts: usize,
    max_width: usize,
    max_depth: usize,
    build_depth: usize,
    created_states: usize,
}

impl SuffixQuotient {
    fn new() -> Self {
        Self {
            suffix_to_state: FxHashMap::default(),
            failed_suffixes: FxHashSet::default(),
            // Do not lower this default to mask correctness or crash bugs.
            // If a schema fails only above this cap, fix the quotient/table
            // invariant that fails at scale instead of hiding the failure.
            max_states: env_usize(RECOGNIZER_SUFFIX_QUOTIENT_MAX_STATES_ENV, 4096),
            max_alts: env_usize(RECOGNIZER_SUFFIX_QUOTIENT_MAX_ALTS_ENV, 64),
            max_width: env_usize(RECOGNIZER_SUFFIX_QUOTIENT_MAX_WIDTH_ENV, 8),
            // Synthetic rows can recursively require further synthetic rows.
            // The quotient is optional and transactional, so aborting a deeply
            // nested attempt preserves the exact unquotiented table while
            // avoiding a native stack overflow on adversarial grammars.
            max_depth: env_usize(RECOGNIZER_SUFFIX_QUOTIENT_MAX_DEPTH_ENV, 128),
            build_depth: 0,
            created_states: 0,
        }
    }

    fn normalize_action(&mut self, table: &mut GLRTable, action: Action) -> Option<Action> {
        match action {
            Action::StackShifts(shifts) => {
                let mut effects = shifts
                    .into_iter()
                    .map(|shift| GuardedStackShift {
                        guards: Vec::new(),
                        pop: shift.pop,
                        pushes: shift.pushes,
                    })
                    .collect::<Vec<_>>();
                let normalized = self.quotient_effect_groups(table, &mut effects).ok()?;
                if Self::action_has_multi_stack_shifts(&normalized) {
                    return None;
                }
                Some(normalized)
            }
            Action::GuardedStackShifts(mut effects) => {
                let normalized = self.quotient_effect_groups(table, &mut effects).ok()?;
                if Self::action_has_multi_stack_shifts(&normalized) {
                    return None;
                }
                Some(normalized)
            }
            Action::Split {
                shift,
                reduces,
                accept,
            } if reduces.is_empty() && !accept => {
                shift.map(|(target, replace)| Action::Shift(target, replace))
            }
            _ => None,
        }
    }

    fn quotient_effect_groups(
        &mut self,
        table: &mut GLRTable,
        effects: &mut Vec<GuardedStackShift>,
    ) -> Result<Action, ()> {
        normalize_guarded_effects_for_suffix_quotient(effects);
        if effects.is_empty() {
            return Err(());
        }

        let mut groups: BTreeMap<(Vec<StackShiftGuard>, u32), Vec<Vec<u32>>> = BTreeMap::new();
        for effect in effects.iter() {
            groups
                .entry((effect.guards.clone(), effect.pop))
                .or_default()
                .push(effect.pushes.clone());
        }

        let mut out = Vec::new();
        for ((guards, pop), mut suffixes) in groups {
            normalize_suffixes(&mut suffixes);
            if suffixes.len() > 1
                && suffixes.iter().all(|suffix| !suffix.is_empty())
                && let Ok(target) = self.ensure_suffix_state(table, suffixes.clone())
            {
                out.push(GuardedStackShift {
                    guards,
                    pop,
                    pushes: vec![target],
                });
            } else {
                for pushes in suffixes {
                    out.push(GuardedStackShift {
                        guards: guards.clone(),
                        pop,
                        pushes,
                    });
                }
            }
        }

        normalize_guarded_effects_for_suffix_quotient(&mut out);
        if out.iter().all(|effect| effect.guards.is_empty()) {
            let shifts = out
                .into_iter()
                .map(|effect| StackShift {
                    pop: effect.pop,
                    pushes: effect.pushes,
                })
                .collect();
            return stack_shift_action(shifts).ok_or(());
        }
        Ok(Action::GuardedStackShifts(out))
    }

    fn ensure_suffix_state(
        &mut self,
        table: &mut GLRTable,
        mut suffixes: Vec<Vec<u32>>,
    ) -> Result<u32, ()> {
        normalize_suffixes(&mut suffixes);
        if suffixes.is_empty() || suffixes.iter().any(|suffix| suffix.is_empty()) {
            return Err(());
        }
        if suffixes.len() > self.max_alts || suffixes.iter().any(|suffix| suffix.len() > self.max_width) {
            return Err(());
        }
        if suffixes.len() == 1 && suffixes[0].len() == 1 {
            return Ok(suffixes[0][0]);
        }
        if let Some(&state) = self.suffix_to_state.get(&suffixes) {
            return Ok(state);
        }
        if self.failed_suffixes.contains(&suffixes) || self.created_states >= self.max_states {
            return Err(());
        }
        if self.build_depth >= self.max_depth {
            return Err(());
        }

        let rollback_state = table.num_states;
        let rollback_created_states = self.created_states;
        let had_advance_rows = table.advance.len() == table.num_states as usize;
        let state = rollback_state;
        table.num_states += 1;
        table.action.push(ActionRow::default());
        table.goto.push(GotoRow::default());
        if had_advance_rows {
            table
                .advance
                .push(super::action_presence_row(&ActionRow::default(), table.num_terminals));
        }
        self.created_states += 1;
        self.suffix_to_state.insert(suffixes.clone(), state);
        self.build_depth += 1;
        let built = (|| {
            let action = self.build_suffix_action_row(table, &suffixes)?;
            let goto = self.build_suffix_goto_row(table, &suffixes)?;
            Ok::<_, ()>((action, goto))
        })();
        self.build_depth -= 1;

        match built {
            Ok((action, goto)) => {
                if Self::action_row_has_multi_stack_shifts(&action) {
                    self.suffix_to_state.retain(|_, mapped_state| *mapped_state < rollback_state);
                    table.action.truncate(rollback_state as usize);
                    table.goto.truncate(rollback_state as usize);
                    if had_advance_rows {
                        table.advance.truncate(rollback_state as usize);
                    }
                    table.num_states = rollback_state;
                    self.created_states = rollback_created_states;
                    self.failed_suffixes.insert(suffixes);
                    return Err(());
                }
                table.action[state as usize] = action;
                table.goto[state as usize] = goto;
                if had_advance_rows {
                    table.advance[state as usize] =
                        super::action_presence_row(&table.action[state as usize], table.num_terminals);
                }
                Ok(state)
            }
            Err(()) => {
                self.suffix_to_state.retain(|_, mapped_state| *mapped_state < rollback_state);
                table.action.truncate(rollback_state as usize);
                table.goto.truncate(rollback_state as usize);
                if had_advance_rows {
                    table.advance.truncate(rollback_state as usize);
                }
                table.num_states = rollback_state;
                self.created_states = rollback_created_states;
                self.failed_suffixes.insert(suffixes);
                Err(())
            }
        }
    }

    fn build_suffix_action_row(
        &mut self,
        table: &mut GLRTable,
        suffixes: &[Vec<u32>],
    ) -> Result<ActionRow, ()> {
        let mut terminals = BTreeSet::new();
        for suffix in suffixes {
            let top = *suffix.last().ok_or(())?;
            for terminal in table.action[top as usize].keys() {
                terminals.insert(terminal);
            }
        }

        let mut row = ActionRow::default();
        for terminal in terminals {
            let mut effects = Vec::new();
            let mut accepts = 0usize;
            for suffix in suffixes {
                let top = *suffix.last().ok_or(())?;
                let Some(action) = table.action[top as usize].get(&terminal).cloned() else {
                    continue;
                };
                self.collect_effects_for_suffix_action(table, suffix, terminal, &action, &mut effects, &mut accepts)?;
            }

            if accepts > 0 {
                if !effects.is_empty() {
                    return Err(());
                }
                row.insert(terminal, Action::Accept);
                continue;
            }

            if effects.is_empty() {
                continue;
            }
            let action = self.quotient_effect_groups(table, &mut effects)?;
            row.insert(terminal, action);
        }
        Ok(row)
    }

    fn build_suffix_goto_row(
        &mut self,
        table: &mut GLRTable,
        suffixes: &[Vec<u32>],
    ) -> Result<GotoRow, ()> {
        let mut nts = BTreeSet::new();
        for suffix in suffixes {
            let top = *suffix.last().ok_or(())?;
            for &nt in table.goto[top as usize].keys() {
                nts.insert(nt);
            }
        }

        let mut row = GotoRow::default();
        for nt in nts {
            let mut result_suffixes = Vec::new();
            for suffix in suffixes {
                let top = *suffix.last().ok_or(())?;
                let Some(&(target, replace)) = table.goto[top as usize].get(&nt) else {
                    continue;
                };
                result_suffixes.push(apply_goto_to_suffix(suffix, target, replace));
            }
            normalize_suffixes(&mut result_suffixes);
            if result_suffixes.is_empty() {
                continue;
            }
            let target = self.ensure_suffix_target(table, result_suffixes)?;
            row.insert(nt, (target, true));
        }
        Ok(row)
    }


fn action_row_has_multi_stack_shifts(row: &ActionRow) -> bool {
    row.values().any(|action| matches!(action, Action::StackShifts(shifts) if shifts.len() > 1))
}

fn action_has_multi_stack_shifts(action: &Action) -> bool {
    matches!(action, Action::StackShifts(shifts) if shifts.len() > 1)
}
    fn ensure_suffix_target(
        &mut self,
        table: &mut GLRTable,
        mut suffixes: Vec<Vec<u32>>,
    ) -> Result<u32, ()> {
        normalize_suffixes(&mut suffixes);
        match suffixes.as_slice() {
            [only] if only.len() == 1 => Ok(only[0]),
            _ => self.ensure_suffix_state(table, suffixes),
        }
    }

    fn collect_effects_for_suffix_action(
        &mut self,
        table: &GLRTable,
        suffix: &[u32],
        terminal: TerminalID,
        action: &Action,
        effects: &mut Vec<GuardedStackShift>,
        accepts: &mut usize,
    ) -> Result<(), ()> {
        if suffix.is_empty() {
            return Err(());
        }
        let Some(&state) = suffix.last() else {
            return Err(());
        };
        let frame = StackEffectFrame {
            pop: 1,
            pushes: suffix.to_vec(),
            guards: Vec::new(),
        };
        collect_suffix_effects_from_frame(
            table,
            terminal,
            state,
            action,
            frame,
            None,
            &mut FxHashSet::default(),
            effects,
            accepts,
        )
    }
}

fn normalize_suffixes(suffixes: &mut Vec<Vec<u32>>) {
    suffixes.sort();
    suffixes.dedup();
}

fn normalize_guarded_effects_for_suffix_quotient(effects: &mut Vec<GuardedStackShift>) {
    effects.retain_mut(|effect| {
        let mut guards = std::mem::take(&mut effect.guards);
        guards.sort_unstable_by_key(|guard| guard.pop);
        for mut guard in guards {
            guard.states.sort_unstable();
            guard.states.dedup();
            if let Some(last) = effect.guards.last_mut()
                && last.pop == guard.pop
            {
                last.states.retain(|state| guard.states.binary_search(state).is_ok());
                if last.states.is_empty() {
                    return false;
                }
            } else {
                if guard.states.is_empty() {
                    return false;
                }
                effect.guards.push(guard);
            }
        }
        true
    });
    effects.sort();
    effects.dedup();
}

fn apply_goto_to_suffix(suffix: &[u32], target: u32, replace: bool) -> Vec<u32> {
    let mut out = suffix.to_vec();
    if replace {
        if let Some(top) = out.last_mut() {
            *top = target;
        } else {
            out.push(target);
        }
    } else {
        out.push(target);
    }
    out
}

/// Interpret one consuming alternative as a partial relation on a stack
/// suffix. The unknown prefix is represented by the original-pop coordinate;
/// known pushed states discharge guards without consulting predecessors.
#[derive(Clone, Copy)]
enum RewriteError {
    GuardRange,
    Overflow,
}

fn compose_stack_rewrite(
    mut frame: StackEffectFrame,
    guards: &[StackShiftGuard],
    pop: u32,
    pushes: &[u32],
) -> Result<Option<StackEffectFrame>, RewriteError> {
    let pushed_len = u32::try_from(frame.pushes.len()).map_err(|_| RewriteError::Overflow)?;
    for guard in guards {
        if guard.states.is_empty() {
            return Ok(None);
        }
        if guard.pop < pushed_len {
            let known = frame.pushes[(pushed_len - 1 - guard.pop) as usize];
            if guard.states.binary_search(&known).is_err() {
                return Ok(None);
            }
        } else {
            let depth = frame.pop.checked_add(guard.pop - pushed_len).ok_or(RewriteError::Overflow)?;
            if !add_guard_to_frame(&mut frame, depth, guard.states.iter().copied()) {
                return Ok(None);
            }
        }
    }
    if guards.iter().any(|guard| guard.pop > pop) {
        return Err(RewriteError::GuardRange);
    }
    pop_frame(&mut frame, pop).map_err(|()| RewriteError::Overflow)?;
    frame.pushes.extend_from_slice(pushes);
    Ok(Some(frame))
}

/// The consuming alternatives of a cell are stack relations, independent of
/// how the caller treats reduction continuations or acceptance.
fn compose_action_rewrites(
    table: &GLRTable,
    state: u32,
    terminal: TerminalID,
    action: &Action,
    frame: &StackEffectFrame,
    skip_invalid_guards: bool,
) -> Result<SmallVec<[StackEffectFrame; 2]>, ()> {
    let forwarded = matches!(action, Action::Shift(..) | Action::Split { .. })
        && table.forwarded_shifts.contains(&(state, terminal));
    let mut out = SmallVec::new();
    let mut invalid = false;
    action.for_each_stack_effect(|guards, pop, pushes| {
        if invalid {
            return;
        }
        match compose_stack_rewrite(frame.clone(), guards, if forwarded { 0 } else { pop }, pushes) {
            Ok(Some(next)) => out.push(next),
            Ok(None) => {}
            Err(RewriteError::GuardRange) if skip_invalid_guards => {}
            Err(_) => invalid = true,
        }
    });
    if invalid { Err(()) } else { Ok(out) }
}

fn collect_suffix_effects_from_frame(
    table: &GLRTable,
    terminal: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    goto_sources_by_nonterminal: Option<&[Vec<u32>]>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    effects: &mut Vec<GuardedStackShift>,
    accepts: &mut usize,
) -> Result<(), ()> {
    StackWalk {
        table,
        terminal,
        endpoint: StackEndpoint::Suffix,
        domain: StackWalkDomain::Suffix(goto_sources_by_nonterminal),
        visiting,
        effects,
        accepts,
        origin_dependent: false,
    }.walk(state, action, frame)
}

/// Nonrecursive cells need neither a predecessor domain nor a visit set.
fn collect_context_share_effects_from_frame(
    table: &GLRTable,
    terminal: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    goto_sources_by_nonterminal: &[Vec<u32>],
    effects: &mut Vec<GuardedStackShift>,
    accepts: &mut usize,
) -> Result<(), ()> {
    if !matches!(action, Action::Reduce(..) | Action::Accept | Action::Split { .. })
        || matches!(action, Action::Split { reduces, accept: false, .. } if reduces.is_empty())
    {
        effects.extend(compose_action_rewrites(table, state, terminal, action, &frame, false)?
            .into_iter().map(frame_to_guarded_shift));
        return Ok(());
    }
    collect_suffix_effects_from_frame(
        table, terminal, state, action, frame, Some(goto_sources_by_nonterminal),
        &mut FxHashSet::default(), effects, accepts,
    )
}

fn reduce_suffix_frame(
    table: &GLRTable,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
    goto_sources_by_nonterminal: Option<&[Vec<u32>]>,
) -> Result<Vec<StackEffectFrame>, ()> {
    pop_frame(&mut frame, len)?;
    if let Some(&state) = frame.pushes.last() {
        return reduce_exposed_frame(table, frame, nt, &[state], false);
    }
    let sources;
    let candidates = if let Some(index) = goto_sources_by_nonterminal {
        index.get(nt as usize).map(Vec::as_slice).unwrap_or(&[])
    } else {
        sources = table.goto.iter().enumerate()
            .filter(|(_, row)| row.contains_key(&nt))
            .map(|(state, _)| state as u32)
            .collect::<Vec<_>>();
        &sources
    };
    // Without an origin-specific predecessor certificate, every possible
    // goto source must be tested against the original stack.
    reduce_exposed_frame(table, frame, nt, candidates, true)
}

#[inline]
fn unordered_entry_hash<T: Hash>(entry: &T) -> u64 {
    let mut hasher = FxHasher::default();
    entry.hash(&mut hasher);
    hasher.finish()
}

/// Stable, order-independent bucket hash for a sparse row.
///
/// `SparseRow` can iterate inline entries or an `FxHashMap`; neither iteration
/// order is a semantic property. We therefore combine independently hashed
/// entries commutatively instead of allocating and sorting every row. This hash
/// only selects equality candidates: `rows_equal` remains the exact criterion.
#[inline]
fn unordered_row_hash<K: Hash, V: Hash>(entries: impl Iterator<Item = (K, V)>, len: usize) -> (u64, u64) {
    let mut sum = 0u64;
    let mut mixed_xor = 0u64;
    for (key, value) in entries {
        let entry_hash = unordered_entry_hash(&(key, value));
        sum = sum.wrapping_add(entry_hash);
        mixed_xor ^= entry_hash.rotate_left((entry_hash >> 58) as u32);
    }
    (sum, mixed_xor ^ (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

#[derive(PartialEq, Eq, Hash)]
enum ObservedAction<'a> {
    Shift(u32, bool),
    Reduce(NonterminalID, u32),
    Split(Option<(u32, bool)>, &'a [(NonterminalID, u32)], bool),
    Accept,
    Skip,
    // Stack alternatives use exactly the canonical publication transform.
    Stack(Action),
}

impl<'a> ObservedAction<'a> {
    fn new(action: &'a Action, mapping: &[u32], normalize: bool) -> Self {
        match action {
            Action::Shift(target, replace) => Self::Shift(mapping[*target as usize], *replace),
            Action::Reduce(nt, len) => Self::Reduce(*nt, *len),
            Action::Split { shift, reduces, accept } => Self::Split(
                shift.map(|(target, replace)| (mapping[target as usize], replace)),
                reduces, *accept,
            ),
            Action::Accept => Self::Accept,
            Action::Skip => Self::Skip,
            _ => {
                let mut mapped = action.clone();
                remap_action_state_references(&mut mapped, mapping, normalize);
                Self::Stack(mapped)
            }
        }
    }
}

#[cfg(test)]
fn row_fingerprint(
    action_row: &ActionRow,
    goto_row: &GotoRow,
    advance_row: Option<&BitSet>,
) -> u64 {
    let (action_sum, action_xor) =
        unordered_row_hash(action_row.iter(), action_row.len());
    let (goto_sum, goto_xor) = unordered_row_hash(
        goto_row.iter().map(|(nonterminal, target)| (*nonterminal, target)),
        goto_row.len(),
    );
    let mut hasher = FxHasher::default();
    action_row.len().hash(&mut hasher);
    action_sum.hash(&mut hasher);
    action_xor.hash(&mut hasher);
    goto_row.len().hash(&mut hasher);
    goto_sum.hash(&mut hasher);
    goto_xor.hash(&mut hasher);
    if let Some(advance_row) = advance_row {
        advance_row.hash(&mut hasher);
    }
    hasher.finish()
}

fn remapped_row_fingerprint(
    action_row: &ActionRow,
    goto_row: &GotoRow,
    advance_row: Option<&BitSet>,
    state_to_group: &[u32],
    normalize: bool,
) -> u64 {
    let (action_sum, action_xor) = unordered_row_hash(
        action_row.iter().map(|(terminal, action)| {
            (terminal, ObservedAction::new(action, state_to_group, normalize))
        }),
        action_row.len(),
    );
    let (goto_sum, goto_xor) = unordered_row_hash(
        goto_row.iter().map(|(nonterminal, &(target, replace))| {
            (*nonterminal, (state_to_group[target as usize], replace))
        }),
        goto_row.len(),
    );
    let mut hasher = FxHasher::default();
    action_row.len().hash(&mut hasher);
    action_sum.hash(&mut hasher);
    action_xor.hash(&mut hasher);
    goto_row.len().hash(&mut hasher);
    goto_sum.hash(&mut hasher);
    goto_xor.hash(&mut hasher);
    if let Some(advance_row) = advance_row {
        advance_row.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
fn rows_equal(
    action_a: &ActionRow,
    goto_a: &GotoRow,
    advance_a: Option<&BitSet>,
    action_b: &ActionRow,
    goto_b: &GotoRow,
    advance_b: Option<&BitSet>,
) -> bool {
    advance_a == advance_b
        && action_a.len() == action_b.len()
        && goto_a.len() == goto_b.len()
        && action_a
            .iter()
            .all(|(terminal, action)| action_b.get(&terminal) == Some(action))
        && goto_a
            .iter()
            .all(|(&nonterminal, target)| goto_b.get(&nonterminal) == Some(target))
}

fn rows_equal_mapped(
    action_a: &ActionRow,
    goto_a: &GotoRow,
    advance_a: Option<&BitSet>,
    action_b: &ActionRow,
    goto_b: &GotoRow,
    advance_b: Option<&BitSet>,
    state_to_group: &[u32],
    normalize: bool,
) -> bool {
    advance_a == advance_b
        && action_a.len() == action_b.len()
        && goto_a.len() == goto_b.len()
        && action_a.iter().all(|(terminal, action)| {
            action_b
                .get(&terminal)
                .is_some_and(|other| ObservedAction::new(action, state_to_group, normalize) == ObservedAction::new(other, state_to_group, normalize))
        })
        && goto_a.iter().all(|(&nonterminal, &(target, replace))| {
            goto_b.get(&nonterminal).is_some_and(|&(other_target, other_replace)| {
                replace == other_replace
                    && state_to_group[target as usize] == state_to_group[other_target as usize]
            })
        })
}

#[cfg(test)]
fn rows_equal_after_remap(
    action_a: &ActionRow, goto_a: &GotoRow, advance_a: Option<&BitSet>,
    action_b: &ActionRow, goto_b: &GotoRow, advance_b: Option<&BitSet>,
    mapping: &[u32],
) -> bool {
    rows_equal_mapped(action_a, goto_a, advance_a, action_b, goto_b, advance_b, mapping, true)
}

/// Every guard membership is an observable predicate on an ORIGINAL
/// stack symbol. Equal outgoing LR rows alone do not authorize identifying
/// two symbols tested differently by any such predicate.
#[derive(Default, Hash, PartialEq, Eq)]
struct RowMarkers {
    forwarded: SmallVec<[TerminalID; 1]>,
    guard_roles: SmallVec<[usize; 1]>,
    pinned: Option<u32>,
}

struct RowObservations<'a> {
    table: &'a GLRTable,
    markers: Vec<RowMarkers>,
    has_advance: bool,
    has_guard_roles: bool,
}

impl<'a> RowObservations<'a> {
    fn new(table: &'a GLRTable) -> Self {
        let n = table.num_states as usize;
        let mut markers = (0..n).map(|_| RowMarkers::default()).collect::<Vec<_>>();
        for &(state, terminal) in &table.forwarded_shifts {
            markers[state as usize].forwarded.push(terminal);
        }
        for marker in &mut markers {
            marker.forwarded.sort_unstable();
        }
        let mut role = 0usize;
        for row in &table.action {
            for action in row.values() {
                if let Action::GuardedStackShifts(shifts) = action {
                    for shift in shifts {
                        for guard in &shift.guards {
                            for &state in &guard.states {
                                let roles = &mut markers[state as usize].guard_roles;
                                if roles.last() != Some(&role) {
                                    roles.push(role);
                                }
                            }
                            role += 1;
                        }
                    }
                }
            }
        }
        // Wide frontier certificates describe specific physical LR
        // symbols. Until a quotient-aware certificate exists, retain them.
        for descriptor in &table.direct_regular_wide_frontiers {
            for state in std::iter::once(descriptor.source_state)
                .chain(descriptor.target_states.iter().copied())
            {
                markers[state as usize].pinned = Some(state);
            }
        }
        Self { table, markers, has_advance: table.advance.len() == n, has_guard_roles: role != 0 }
    }

    fn fingerprint(&self, state: u32, mapping: &[u32], normalize: bool) -> u64 {
        let state = state as usize;
        unordered_entry_hash(&(
            remapped_row_fingerprint(
                &self.table.action[state], &self.table.goto[state],
                self.has_advance.then(|| &self.table.advance[state]), mapping, normalize,
            ),
            &self.markers[state],
        ))
    }

    fn equal(&self, left: u32, right: u32, mapping: &[u32], normalize: bool) -> bool {
        let (left, right) = (left as usize, right as usize);
        self.markers[left] == self.markers[right]
            && rows_equal_mapped(
                &self.table.action[left], &self.table.goto[left],
                self.has_advance.then(|| &self.table.advance[left]),
                &self.table.action[right], &self.table.goto[right],
                self.has_advance.then(|| &self.table.advance[right]), mapping, normalize,
            )
    }
}

/// First-representative exact partition of a fingerprint bucket. The
/// fingerprint has no authority to merge states. This works for both
/// bottom-up coarsening and top-down splitting of an initial core class.
fn exact_row_bucket(
    mut states: SmallVec<[u32; 1]>,
    mut equal: impl FnMut(u32, u32) -> bool,
    pairs: &mut Vec<(u32, u32)>,
) {
    states.sort_unstable();
    let mut representatives = SmallVec::<[u32; 2]>::new();
    for state in states {
        if let Some(&rep) = representatives.iter().find(|&&rep| equal(rep, state)) {
            pairs.push((rep, state));
        } else {
            representatives.push(state);
        }
    }
}

/// Exact collision-checked row coarsening, with a canonical first-state
/// representative and eager original-to-representative coordinates.
/// A hash bucket is processed once per round, not once per active member.
struct RowCoarsener<'a> {
    table: &'a GLRTable,
    mapping: Vec<u32>,
    members: Vec<SmallVec<[u32; 1]>>,
    observations: RowObservations<'a>,
    fingerprints: Vec<u64>,
    buckets: FxHashMap<u64, SmallVec<[u32; 1]>>,
    reverse: Option<Vec<SmallVec<[u32; 2]>>>,
}

impl<'a> RowCoarsener<'a> {
    fn new(table: &'a GLRTable) -> Self {
        let mut this = Self {
            table,
            mapping: (0..table.num_states).collect(),
            members: (0..table.num_states).map(|state| smallvec![state]).collect(),
            observations: RowObservations::new(table),
            fingerprints: Vec::new(),
            buckets: FxHashMap::default(),
            reverse: None,
        };
        this.fingerprints = (0..table.num_states).into_par_iter()
            .map(|state| this.fingerprint(state)).collect();
        for (state, &hash) in this.fingerprints.iter().enumerate() {
            this.buckets.entry(hash).or_default().push(state as u32);
        }
        this
    }

    fn fingerprint(&self, state: u32) -> u64 {
        self.observations.fingerprint(state, &self.mapping, true)
    }

    fn equal(&self, left: u32, right: u32) -> bool {
        self.observations.equal(left, right, &self.mapping, true)
    }

    fn remove_bucket_member(&mut self, state: u32) {
        let hash = self.fingerprints[state as usize];
        let bucket = self.buckets.get_mut(&hash).expect("representative has fingerprint");
        bucket.retain(|&member| member != state);
        if bucket.is_empty() {
            self.buckets.remove(&hash);
        }
    }

    fn refresh(&mut self, active: &[u32]) {
        for &state in active {
            let hash = self.fingerprint(state);
            if hash != self.fingerprints[state as usize] {
                self.remove_bucket_member(state);
                self.fingerprints[state as usize] = hash;
                self.buckets.entry(hash).or_default().push(state);
            }
        }
    }

    fn reverse_dependencies(&self) -> Vec<SmallVec<[u32; 2]>> {
        let mut reverse = vec![SmallVec::new(); self.mapping.len()];
        let mut targets = Vec::new();
        for state in 0..self.mapping.len() {
            targets.clear();
            for action in self.table.action[state].values() {
                collect_action_targets(action, &mut targets);
            }
            targets.extend(self.table.goto[state].values().map(|&(target, _)| target));
            targets.sort_unstable();
            targets.dedup();
            for &target in &targets {
                reverse[target as usize].push(state as u32);
            }
        }
        reverse
    }

    fn matching_pairs(&self, active: Option<&[u32]>) -> Vec<(u32, u32)> {
        let hashes = if let Some(active) = active {
            active.iter().map(|&state| self.fingerprints[state as usize]).collect::<FxHashSet<_>>()
        } else {
            self.buckets.iter().filter_map(|(&hash, bucket)| (bucket.len() > 1).then_some(hash)).collect()
        };
        let mut pairs = Vec::new();
        for hash in hashes {
            let bucket = &self.buckets[&hash];
            if bucket.len() <= 1 {
                continue;
            }
            exact_row_bucket(bucket.clone(), |left, right| self.equal(left, right), &mut pairs);
        }
        pairs
    }

    fn solve(mut self, dirty: Option<&[u32]>) -> Option<(Vec<u32>, Vec<u32>)> {
        // Guard-role observations are global: adding/removing a guard in
        // one cell can change another state's role without changing its
        // own outgoing row. Use the full initial schedule on guarded input.
        let mut active = if self.observations.has_guard_roles {
            None
        } else {
            dirty.map(|states| states.to_vec())
        };
        let mut merged = false;
        loop {
            if let Some(active) = &mut active {
                for state in active.iter_mut() {
                    *state = self.mapping[*state as usize];
                }
                active.sort_unstable();
                active.dedup();
                self.refresh(active);
            }
            let pairs = self.matching_pairs(active.as_deref());
            if pairs.is_empty() {
                break;
            }
            merged = true;
            if self.reverse.is_none() {
                self.reverse = Some(self.reverse_dependencies());
            }
            let mut next = Vec::new();
            // Pairs are a disjoint collection of stars from the stable
            // partition. All comparisons precede any change of coordinates.
            for (winner, loser) in pairs {
                self.remove_bucket_member(loser);
                let members = std::mem::take(&mut self.members[loser as usize]);
                for member in members {
                    self.mapping[member as usize] = winner;
                    next.extend_from_slice(&self.reverse.as_ref().unwrap()[member as usize]);
                    self.members[winner as usize].push(member);
                }
            }
            active = Some(next);
        }
        if !merged {
            return None;
        }
        let representatives = (0..self.table.num_states)
            .filter(|&state| self.mapping[state as usize] == state).collect::<Vec<_>>();
        let mut compact = vec![u32::MAX; self.mapping.len()];
        for (id, &state) in representatives.iter().enumerate() {
            compact[state as usize] = id as u32;
        }
        for state in &mut self.mapping {
            *state = compact[*state as usize];
        }
        Some((representatives, self.mapping))
    }
}

fn push_reachable_state(state: u32, reachable: &mut [bool], stack: &mut Vec<u32>) {
    let Some(slot) = reachable.get_mut(state as usize) else {
        return;
    };
    if !*slot {
        *slot = true;
        stack.push(state);
    }
}

fn collect_action_targets(action: &Action, out: &mut Vec<u32>) {
    action.for_each_state_reference(|state| out.push(state));
}

fn push_action_targets(action: &Action, reachable: &mut [bool], stack: &mut Vec<u32>) {
    action.for_each_state_reference(|state| push_reachable_state(state, reachable, stack));
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CellUpdate {
    Set(Action),
    Remove,
}

/// Only actions that contain at least one non-accepting reduction can change
/// under stack-effect or unit-reduction inlining. In particular, a plain shift
/// has no reduction to resolve and the recursive unit inliner must return it
/// unchanged, so visiting it is pure overhead.
fn action_has_inlinable_reductions(action: &Action) -> bool {
    match action {
        Action::Reduce(_, _) => true,
        Action::Split {
            reduces,
            accept: false,
            ..
        } => !reduces.is_empty(),
        _ => false,
    }
}

/// The unit-reduction pass adds work beyond stack-effect inlining only when an
/// action has both a shift and a unit reduction. A reduction-only action is
/// either handled by the stack-effect pass or cannot gain a shift by this
/// phase; this matches the old dispatch exactly.
#[derive(Clone, Copy)]
enum StackInlineCandidateKind {
    ReduceZero,
    ReduceOne,
    ReduceMany,
    Split,
}

impl StackInlineCandidateKind {
    const COUNT: usize = 4;

    fn index(self) -> usize {
        match self {
            Self::ReduceZero => 0,
            Self::ReduceOne => 1,
            Self::ReduceMany => 2,
            Self::Split => 3,
        }
    }
}

fn stack_inline_candidate_kind(action: &Action) -> StackInlineCandidateKind {
    match action {
        Action::Reduce(_, 0) => StackInlineCandidateKind::ReduceZero,
        Action::Reduce(_, 1) => StackInlineCandidateKind::ReduceOne,
        Action::Reduce(_, _) => StackInlineCandidateKind::ReduceMany,
        Action::Split { .. } => StackInlineCandidateKind::Split,
        _ => unreachable!("only stack-inlining candidates are classified"),
    }
}

fn action_needs_unit_reduction_inlining(action: &Action) -> bool {
    matches!(
        action,
        Action::Split {
            shift: Some(_),
            reduces,
            accept: false,
        } if !reduces.is_empty()
    )
}

type UnitInlineCell = (u32, TerminalID);

#[derive(Default)]
struct UnitInlineDependencies {
    reverse: FxHashMap<UnitInlineCell, Vec<UnitInlineCell>>,
}

impl UnitInlineDependencies {
    fn record_read(&mut self, owner: UnitInlineCell, read: UnitInlineCell) {
        if owner != read {
            self.reverse.entry(read).or_default().push(owner);
        }
    }

    fn record_reads(&mut self, reads: impl IntoIterator<Item = (UnitInlineCell, UnitInlineCell)>) {
        for (owner, read) in reads {
            self.record_read(owner, read);
        }
    }

    fn worklist_for_changes(&self, changed: impl IntoIterator<Item = UnitInlineCell>) -> Vec<UnitInlineCell> {
        let mut queued = FxHashSet::default();
        for changed_cell in changed {
            queued.insert(changed_cell);
            if let Some(dependents) = self.reverse.get(&changed_cell) {
                queued.extend(dependents.iter().copied());
            }
        }
        let mut worklist = queued.into_iter().collect::<Vec<_>>();
        worklist.sort_unstable();
        worklist
    }
}

/// A cheap sufficient test for the common rejected stack-effect shape:
/// reducing one symbol and then taking only non-forwarded replace shifts. The
/// general stack-effect evaluator would classify this as origin-dependent with
/// one pushed state and deliberately leave the reduce untouched, so skipping
/// it is semantically exact. Every action row inspected here is recorded for
/// fixed-point invalidation because a later rewrite can make the shortcut
/// inapplicable.
fn is_trivial_origin_dependent_one_push(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    tid: TerminalID,
    action: &Action,
    dependencies: &mut Vec<(UnitInlineCell, UnitInlineCell)>,
) -> bool {
    let Action::Reduce(nonterminal, 1) = action else {
        return false;
    };
    let owner = (state, tid);
    let mut saw_effect = false;
    for &predecessor in &predecessors[state as usize] {
        let Some((next_state, _)) = table.goto[predecessor as usize].get(nonterminal).copied() else {
            continue;
        };
        let Some(next_action) = table.action[next_state as usize].get(&tid) else {
            continue;
        };
        dependencies.push((owner, (next_state, tid)));
        let is_replace_shift = match next_action {
            Action::Shift(_, replace) => {
                *replace && !table.forwarded_shifts.contains(&(next_state, tid))
            }
            Action::Split {
                shift: Some((_, replace)),
                reduces,
                accept: false,
            } => {
                reduces.is_empty()
                    && *replace
                    && !table.forwarded_shifts.contains(&(next_state, tid))
            }
            _ => false,
        };
        if !is_replace_shift {
            return false;
        }
        saw_effect = true;
    }
    saw_effect
}

fn action_predecessor_signature(action: &Action) -> Option<&Action> {
    match action {
        Action::Shift(..)
        | Action::Split { shift: Some(_), .. }
        | Action::ReplaceShifts(_)
        | Action::StackShifts(_)
        | Action::GuardedStackShifts(_) => Some(action),
        _ => None,
    }
}

fn build_runtime_state_predecessors(
    table: &GLRTable,
    constituent_sets: &[StateSubset],
) -> Result<RuntimePredecessors, String> {
    let equations = PredecessorEquations::extract(table, Some(constituent_sets))?;
    PredecessorSolver::new(&equations).solve(None)
}

fn control_elimination_failure(
    budget: &UnitInlineBudget,
    state: u32,
    terminal: TerminalID,
    message: &str,
) -> String {
    let reason = budget.report().reason.unwrap_or("semantic traversal");
    format!(
        "{message}: state={state} terminal={terminal} reason={reason} stack_effect_visits={}",
        budget.stack_effect_visits(),
    )
}

/// True when a control-elimination error reports convergence-budget
/// exhaustion (`reason=stack_effect_visits` / `reason=elapsed_ms`) rather
/// than a semantic traversal failure. Budget exhaustion is a *decline*:
/// the table stays controlled and the caller falls back to dynamic
/// boundary shards. Any other elimination error is a hard failure.
pub fn control_elimination_budget_exhausted(error: &str) -> bool {
    error.contains("reason=stack_effect_visits") || error.contains("reason=elapsed_ms")
}

#[cfg(test)]
fn for_each_action_stack_rewrite(action: &Action, mut visit: impl FnMut(u32, &[u32])) {
    // Predecessor construction observes potential stack rewrites, including
    // guarded alternatives, but a Skip contributes no predecessor edge.
    if !matches!(action, Action::Skip) {
        action.for_each_stack_effect(|_, pop, pushes| visit(pop, pushes));
    }
}

#[cfg(test)]
fn action_stack_rewrites(action: &Action) -> Vec<(u32, Vec<u32>)> {
    let mut rewrites = Vec::new();
    for_each_action_stack_rewrite(action, |pop, pushes| {
        rewrites.push((pop, pushes.to_vec()));
    });
    rewrites
}

fn predecessor_states_at_depth(
    predecessors: &[PredecessorSet],
    origin: u32,
    depth: u32,
) -> PredecessorSet {
    let mut states = smallvec![origin];
    for _ in 0..depth {
        let mut next = PredecessorSet::new();
        for state in states {
            if let Some(preds) = predecessors.get(state as usize) {
                next.extend_from_slice(preds);
            }
        }
        next.sort_unstable();
        next.dedup();
        if next.is_empty() {
            return next;
        }
        states = next;
    }
    states
}

/// Original-state sets in a copy-SCC row. Small sparse rows never allocate
/// a full state-width bitmap; dense rows keep word-wise union throughput.
/// Promotion is monotone and leaves the lattice unchanged.
#[derive(Clone)]
enum PredecessorBits {
    Sparse(PredecessorSet),
    Dense(Vec<u64>),
}

impl PredecessorBits {
    fn empty() -> Self {
        Self::Sparse(PredecessorSet::new())
    }

    fn from_seeds(seeds: &[u32], width: usize) -> Result<Self, String> {
        let mut bits = Self::Sparse(PredecessorSet::from_slice(seeds));
        bits.promote(width)?;
        Ok(bits)
    }

    fn promote(&mut self, width: usize) -> Result<(), String> {
        let Self::Sparse(states) = self else { return Ok(()); };
        let words = width.div_ceil(64);
        if states.len() <= words.saturating_mul(2).max(8) {
            return Ok(());
        }
        let mut bits = Vec::new();
        bits.try_reserve_exact(words)
            .map_err(|_| "control predecessor bitmap exceeds available memory".to_owned())?;
        bits.resize(words, 0u64);
        for &state in states.iter() {
            bits[state as usize / 64] |= 1u64 << (state % 64);
        }
        *self = Self::Dense(bits);
        Ok(())
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Sparse(states) => states.is_empty(),
            // Dense rows are only created by promotion of nonempty sets.
            Self::Dense(_) => false,
        }
    }

    fn visit(&self, width: usize, mut visit: impl FnMut(u32)) {
        match self {
            Self::Sparse(states) => states.iter().copied().for_each(visit),
            Self::Dense(bits) => for_each_set_bit(bits, width, &mut visit),
        }
    }

    fn union(&mut self, source: &Self, width: usize) -> Result<bool, String> {
        if source.is_empty() {
            return Ok(false);
        }
        if matches!(source, Self::Dense(_)) && let Self::Sparse(states) = self {
            let mut bits = Vec::new();
            let words = width.div_ceil(64);
            bits.try_reserve_exact(words)
                .map_err(|_| "control predecessor bitmap exceeds available memory".to_owned())?;
            bits.resize(words, 0u64);
            for &state in states.iter() {
                bits[state as usize / 64] |= 1u64 << (state % 64);
            }
            *self = Self::Dense(bits);
        }
        let changed = match (&mut *self, source) {
            (Self::Dense(dst), Self::Dense(src)) => {
                let mut changed = false;
                for (slot, &word) in dst.iter_mut().zip(src) {
                    let merged = *slot | word;
                    changed |= merged != *slot;
                    *slot = merged;
                }
                changed
            }
            (Self::Dense(dst), Self::Sparse(src)) => {
                let mut changed = false;
                for &state in src {
                    let slot = &mut dst[state as usize / 64];
                    let mask = 1u64 << (state % 64);
                    changed |= *slot & mask == 0;
                    *slot |= mask;
                }
                changed
            }
            (Self::Sparse(dst), Self::Sparse(src)) => {
                // Insert small lists in place; avoid allocating a merge row
                // for the overwhelmingly common singleton copy.
                let mut changed = false;
                for &state in src {
                    if let Err(index) = dst.binary_search(&state) {
                        dst.insert(index, state);
                        changed = true;
                    }
                }
                changed
            }
            (Self::Sparse(_), Self::Dense(_)) => unreachable!("dense source promotes destination"),
        };
        self.promote(width)?;
        Ok(changed)
    }

    fn storage_bytes(&self) -> usize {
        match self {
            Self::Sparse(states) => states.len() * std::mem::size_of::<u32>(),
            Self::Dense(bits) => bits.len() * std::mem::size_of::<u64>(),
        }
    }
}

struct PredecessorRows {
    bit_len: usize,
    rows: Vec<Option<PredecessorBits>>,
}

impl PredecessorRows {
    fn new(components: usize, bit_len: usize) -> Self {
        Self { bit_len, rows: vec![None; components] }
    }

    fn active(&self, component: u32) -> bool {
        self.rows[component as usize].is_some()
    }

    fn activate(&mut self, component: u32, seeds: &[u32]) -> Result<bool, String> {
        let slot = &mut self.rows[component as usize];
        if slot.is_some() {
            return Ok(false);
        }
        *slot = Some(PredecessorBits::from_seeds(seeds, self.bit_len)?);
        Ok(true)
    }

    fn row(&self, component: u32) -> &PredecessorBits {
        self.rows[component as usize].as_ref().expect("active predecessor row")
    }

    fn union_row(&mut self, source: u32, destination: u32) -> Result<bool, String> {
        let (source, destination) = (source as usize, destination as usize);
        if source == destination {
            return Ok(false);
        }
        let (src, dst) = if source < destination {
            let (left, right) = self.rows.split_at_mut(destination);
            (&left[source], &mut right[0])
        } else {
            let (left, right) = self.rows.split_at_mut(source);
            (&right[0], &mut left[destination])
        };
        dst.as_mut().unwrap().union(src.as_ref().unwrap(), self.bit_len)
    }

    fn union_bits(&mut self, destination: u32, bits: &PredecessorBits) -> Result<bool, String> {
        self.rows[destination as usize].as_mut().unwrap().union(bits, self.bit_len)
    }

    fn storage_bytes(&self) -> usize {
        self.rows.iter().flatten().map(PredecessorBits::storage_bytes).sum()
    }
}

fn for_each_set_bit(words: &[u64], bit_len: usize, mut visit: impl FnMut(u32)) {
    for (word_index, &word) in words.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let state = word_index * 64 + bit;
            if state < bit_len {
                visit(state as u32);
            }
            remaining &= remaining - 1;
        }
    }
}

fn condense_pop_one_predecessor_graph(
    adjacency: &[PredecessorSet],
) -> (Vec<u32>, Vec<PredecessorSet>, Vec<u32>) {
    let state_count = adjacency.len();
    let mut visited = vec![false; state_count];
    let mut finish_order = Vec::with_capacity(state_count);

    for start in 0..state_count {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((state, next_index)) = stack.last_mut() {
            if *next_index < adjacency[*state].len() {
                let destination = adjacency[*state][*next_index] as usize;
                *next_index += 1;
                if !visited[destination] {
                    visited[destination] = true;
                    stack.push((destination, 0));
                }
            } else {
                finish_order.push(*state);
                stack.pop();
            }
        }
    }

    let mut reverse = vec![Vec::<u32>::new(); state_count];
    for (source, destinations) in adjacency.iter().enumerate() {
        for &destination in destinations {
            reverse[destination as usize].push(source as u32);
        }
    }

    visited.fill(false);
    let mut state_to_component = vec![0u32; state_count];
    let mut component_count = 0u32;
    for &start in finish_order.iter().rev() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start as u32];
        while let Some(state) = stack.pop() {
            state_to_component[state as usize] = component_count;
            for &predecessor in &reverse[state as usize] {
                if !visited[predecessor as usize] {
                    visited[predecessor as usize] = true;
                    stack.push(predecessor);
                }
            }
        }
        component_count += 1;
    }

    let mut component_edges = vec![PredecessorSet::new(); component_count as usize];
    for (source, destinations) in adjacency.iter().enumerate() {
        let source_component = state_to_component[source];
        for &destination in destinations {
            let destination_component = state_to_component[destination as usize];
            if source_component != destination_component {
                component_edges[source_component as usize].push(destination_component);
            }
        }
    }
    for edges in &mut component_edges {
        edges.sort_unstable();
        edges.dedup();
    }

    let mut indegree = vec![0usize; component_count as usize];
    for edges in &component_edges {
        for &destination in edges {
            indegree[destination as usize] += 1;
        }
    }
    let mut queue = VecDeque::new();
    for (component, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(component as u32);
        }
    }
    let mut topological_order = Vec::with_capacity(component_count as usize);
    while let Some(component) = queue.pop_front() {
        topological_order.push(component);
        for &destination in &component_edges[component as usize] {
            indegree[destination as usize] -= 1;
            if indegree[destination as usize] == 0 {
                queue.push_back(destination);
            }
        }
    }
    debug_assert_eq!(topological_order.len(), component_count as usize);

    (state_to_component, component_edges, topological_order)
}

/// One graph of monotone predecessor equations:
///     P[destination] contains P^pop[source].
/// Stack adjacency inside a pushed sequence, and pop-zero edges, are seeds.
/// Pop-one copy SCCs have identical least-fixed-point rows, but the bits in
/// those rows always name ORIGINAL parser states (not SCC identities).
struct PredecessorEquations {
    state_to_component: Vec<u32>,
    seeds: Vec<PredecessorSet>,
    copy_destinations: Vec<PredecessorSet>,
    copy_sources: Vec<PredecessorSet>,
    deep_rules: Vec<(u32, u32, u32)>,
    deep_sources: Vec<PredecessorSet>,
    topological_order: Vec<u32>,
    rewrite_count: usize,
}

impl PredecessorEquations {
    fn extract(table: &GLRTable, constituents: Option<&[StateSubset]>) -> Result<Self, String> {
        let n = table.num_states as usize;
        let mut direct = vec![PredecessorSet::new(); n];
        let mut copies = vec![PredecessorSet::new(); n];
        let mut deep_rules = Vec::<(u32, u32, u32)>::new();
        let mut rewrite_count = 0usize;
        let mut invalid = false;
        let seed = |row: &mut PredecessorSet, state: u32| {
            row.push(state);
            if let Some(members) = constituents.and_then(|sets| sets.get(state as usize)) {
                row.extend_from_slice(members);
            }
        };
        for source in 0..table.num_states {
            let mut record = |pop: u32, pushes: &[u32]| {
                rewrite_count += 1;
                if pushes.iter().any(|&state| state as usize >= n) {
                    invalid = true;
                    return;
                }
                let Some(&destination) = pushes.first() else { return; };
                for pair in pushes.windows(2) {
                    seed(&mut direct[pair[1] as usize], pair[0]);
                }
                match pop {
                    0 => seed(&mut direct[destination as usize], source),
                    1 => copies[source as usize].push(destination),
                    _ => deep_rules.push((source, pop, destination)),
                }
            };
            for (terminal, action) in &table.action[source as usize] {
                let forwarded = matches!(action, Action::Shift(..) | Action::Split { .. })
                    && table.forwarded_shifts.contains(&(source, terminal));
                action.for_each_stack_effect(|_, pop, pushes| {
                    record(if forwarded { 0 } else { pop }, pushes);
                });
            }
            for &(target, replace) in table.goto[source as usize].values() {
                record(u32::from(replace), std::slice::from_ref(&target));
            }
        }
        if invalid {
            return Err("control predecessor rewrite has an out-of-domain stack state".into());
        }
        for row in &mut copies {
            row.sort_unstable();
            row.dedup();
        }
        let (state_to_component, copy_destinations, topological_order) =
            condense_pop_one_predecessor_graph(&copies);
        let components = topological_order.len();
        let mut seeds = vec![PredecessorSet::new(); components];
        for (state, row) in direct.into_iter().enumerate() {
            seeds[state_to_component[state] as usize].extend(row);
        }
        let mut copy_sources = vec![PredecessorSet::new(); components];
        for (source, row) in copy_destinations.iter().enumerate() {
            for &destination in row {
                copy_sources[destination as usize].push(source as u32);
            }
        }
        for (source, _, destination) in &mut deep_rules {
            *source = state_to_component[*source as usize];
            *destination = state_to_component[*destination as usize];
        }
        deep_rules.sort_unstable();
        deep_rules.dedup();
        let mut deep_sources = vec![PredecessorSet::new(); components];
        for &(source, _, destination) in &deep_rules {
            deep_sources[destination as usize].push(source);
        }
        for row in seeds.iter_mut().chain(deep_sources.iter_mut()) {
            row.sort_unstable();
            row.dedup();
        }
        Ok(Self {
            state_to_component, seeds, copy_destinations, copy_sources,
            deep_rules, deep_sources, topological_order, rewrite_count,
        })
    }
}

struct PredecessorSolver<'a> {
    equations: &'a PredecessorEquations,
    rows: PredecessorRows,
    active: Vec<u32>,
    pending: VecDeque<u32>,
    dirty: Vec<bool>,
    depth_marks: Vec<u32>,
    generation: u32,
    changed_copies: usize,
    changed_deep: usize,
    rounds: usize,
}

impl<'a> PredecessorSolver<'a> {
    fn new(equations: &'a PredecessorEquations) -> Self {
        let components = equations.seeds.len();
        Self {
            equations,
            rows: PredecessorRows::new(components, equations.state_to_component.len()),
            active: Vec::new(),
            pending: VecDeque::new(),
            dirty: vec![false; components],
            depth_marks: vec![0; components],
            generation: 0,
            changed_copies: 0,
            changed_deep: 0,
            rounds: 0,
        }
    }

    fn demand(&mut self, component: u32) -> Result<bool, String> {
        if !self.rows.activate(component, &self.equations.seeds[component as usize])? {
            return Ok(false);
        }
        self.active.push(component);
        self.pending.push_back(component);
        self.dirty[component as usize] = true;
        Ok(true)
    }

    fn activate_dependencies(&mut self) -> Result<(), String> {
        let equations = self.equations;
        while let Some(destination) = self.pending.pop_front() {
            for &source in &equations.copy_sources[destination as usize] {
                self.demand(source)?;
                // The destination may have become active after this source's
                // last propagation. Replay that copy even if its bits did
                // not change.
                self.dirty[source as usize] = true;
            }
            for &source in &equations.deep_sources[destination as usize] {
                self.demand(source)?;
            }
        }
        Ok(())
    }

    fn propagate_copies(&mut self) -> Result<(), String> {
        for &source in &self.equations.topological_order {
            if !std::mem::replace(&mut self.dirty[source as usize], false) {
                continue;
            }
            for &destination in &self.equations.copy_destinations[source as usize] {
                if self.rows.active(destination) && self.rows.union_row(source, destination)? {
                    self.changed_copies += 1;
                    self.dirty[destination as usize] = true;
                }
            }
        }
        Ok(())
    }

    /// Walk the ORIGINAL predecessor relation at an arbitrary pop depth,
    /// reading equal copy-SCC rows only once per layer. Newly discovered
    /// component dependencies are seeded immediately, and their pending
    /// equations force another monotone round before publication.
    fn at_depth(&mut self, origin: u32, depth: u32) -> Result<PredecessorBits, String> {
        debug_assert!(depth >= 1);
        self.demand(origin)?;
        let mut current = self.rows.row(origin).clone();
        let mut components = Vec::<u32>::new();
        for _ in 1..depth {
            if current.is_empty() {
                break;
            }
            let mut next = PredecessorBits::empty();
            components.clear();
            self.generation = self.generation.wrapping_add(1);
            if self.generation == 0 {
                self.depth_marks.fill(0);
                self.generation = 1;
            }
            let generation = self.generation;
            let state_to_component = &self.equations.state_to_component;
            let marks = &mut self.depth_marks;
            current.visit(self.rows.bit_len, |state| {
                let component = state_to_component[state as usize];
                let mark = &mut marks[component as usize];
                if *mark != generation {
                    *mark = generation;
                    components.push(component);
                }
            });
            for &component in &components {
                self.demand(component)?;
                next.union(self.rows.row(component), self.rows.bit_len)?;
            }
            current = next;
        }
        Ok(current)
    }

    fn close_demand_under_predecessors(&mut self) -> Result<(), String> {
        let active = self.active.clone();
        let mut missing = Vec::new();
        for component in active {
            self.rows.row(component).visit(self.rows.bit_len, |state| {
                let predecessor = self.equations.state_to_component[state as usize];
                if !self.rows.active(predecessor) {
                    missing.push(predecessor);
                }
            });
        }
        missing.sort_unstable();
        missing.dedup();
        for component in missing {
            self.demand(component)?;
        }
        Ok(())
    }

    fn solve(&mut self, origins: Option<&[u32]>) -> Result<RuntimePredecessors, String> {
        let equations = self.equations;
        if let Some(origins) = origins {
            for &origin in origins {
                let component = *equations.state_to_component.get(origin as usize)
                    .ok_or_else(|| "control predecessor origin lies outside table".to_owned())?;
                self.demand(component)?;
            }
        } else {
            for component in 0..equations.seeds.len() {
                self.demand(component as u32)?;
            }
        }
        // Every nonfinal round grows at least one component row or activates
        // a new component. Copy propagation is exact within its DAG.
        let pass_limit = equations.seeds.len()
            .saturating_mul(self.rows.bit_len.saturating_add(1)).saturating_add(2);
        for _ in 0..pass_limit {
            self.rounds += 1;
            self.activate_dependencies()?;
            self.propagate_copies()?;
            let mut changed = false;
            let mut cache = FxHashMap::<(u32, u32), PredecessorBits>::default();
            for &(source, pop, destination) in &equations.deep_rules {
                if !self.rows.active(destination) {
                    continue;
                }
                let bases = match cache.entry((source, pop)) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(self.at_depth(source, pop)?)
                    }
                };
                if self.rows.union_bits(destination, bases)? {
                    changed = true;
                    self.changed_deep += 1;
                    self.dirty[destination as usize] = true;
                }
            }
            if changed || !self.pending.is_empty() {
                continue;
            }
            if origins.is_some() {
                // A control can inspect deeper than any single equation.
                // Every state below a demanded origin must have its exact
                // row available, not just rows used by equation evaluation.
                self.close_demand_under_predecessors()?;
                if !self.pending.is_empty() {
                    continue;
                }
            }
            let result = equations.state_to_component.iter().map(|&component| {
                let mut row = PredecessorSet::new();
                if self.rows.active(component) {
                    self.rows.row(component).visit(self.rows.bit_len, |state| row.push(state));
                }
                row
            }).collect();
            return Ok(result);
        }
        Err(format!("control predecessor equations did not converge: states={}", self.rows.bit_len))
    }
}

fn solve_control_predecessors(table: &GLRTable, origins: Option<&[u32]>) -> Result<RuntimePredecessors, String> {
    let started = std::time::Instant::now();
    let equations = PredecessorEquations::extract(table, None)?;
    let mut solver = PredecessorSolver::new(&equations);
    let result = solver.solve(origins)?;
    if origins.is_some() && std::env::var_os("GLRMASK_VALIDATE_CONTROL_PREDECESSOR_DEMAND").is_some() {
        let reference = PredecessorSolver::new(&equations).solve(None)?;
        for (state, &component) in equations.state_to_component.iter().enumerate() {
            if solver.rows.active(component) {
                assert_eq!(result[state], reference[state], "demand predecessor row differs for state {state}");
            }
        }
    }
    if std::env::var_os("GLRMASK_PROFILE_CONTROL_ELIMINATION").is_some() {
        eprintln!("[glrmask/profile][control_predecessors] demand={} states={} components={} active_components={} rewrites={} deep_rules={} rounds={} changed_copies={} changed_deep={} predecessor_edges={} max_predecessors={} matrix_mib={:.3} total_ms={:.3}",
            origins.is_some(), table.num_states, equations.seeds.len(), solver.active.len(),
            equations.rewrite_count, equations.deep_rules.len(), solver.rounds,
            solver.changed_copies, solver.changed_deep, result.iter().map(|row| row.len()).sum::<usize>(),
            result.iter().map(|row| row.len()).max().unwrap_or(0),
            solver.rows.storage_bytes() as f64 / (1024.0 * 1024.0),
            started.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(result)
}

fn build_control_elimination_predecessors_demand(
    table: &GLRTable,
    origins: &[u32],
) -> Result<RuntimePredecessors, String> {
    solve_control_predecessors(table, Some(origins))
}

fn build_control_elimination_predecessors(table: &GLRTable) -> Result<RuntimePredecessors, String> {
    solve_control_predecessors(table, None)
}

#[cfg(test)]
fn build_control_elimination_predecessors_synchronous_reference(
    table: &GLRTable,
) -> Result<RuntimePredecessors, String> {
    let mut predecessors = vec![PredecessorSet::new(); table.num_states as usize];
    let pass_limit = (table.num_states as usize)
        .saturating_mul(table.num_states as usize)
        .saturating_add(2);

    for _ in 0..pass_limit {
        let snapshot = predecessors.clone();
        let mut changed = false;

        for source in 0..table.num_states {
            let mut rewrites = Vec::<(u32, Vec<u32>)>::new();
            for (terminal, action) in &table.action[source as usize] {
                let forwarded = matches!(action, Action::Shift(..) | Action::Split { .. })
                    && table.forwarded_shifts.contains(&(source, terminal));
                for (pop, pushes) in action_stack_rewrites(action) {
                    rewrites.push((if forwarded { 0 } else { pop }, pushes));
                }
            }
            rewrites.extend(
                table.goto[source as usize]
                    .values()
                    .map(|&(target, replace)| (u32::from(replace), vec![target])),
            );

            for (pop, pushes) in rewrites {
                if pushes.is_empty() {
                    continue;
                }
                for pair in pushes.windows(2) {
                    let destination = pair[1] as usize;
                    let before = predecessors[destination].len();
                    predecessors[destination].push(pair[0]);
                    predecessors[destination].sort_unstable();
                    predecessors[destination].dedup();
                    changed |= predecessors[destination].len() != before;
                }

                let bases = if pop == 0 {
                    smallvec![source]
                } else {
                    predecessor_states_at_depth(&snapshot, source, pop)
                };
                let first = pushes[0] as usize;
                let before = predecessors[first].len();
                predecessors[first].extend_from_slice(&bases);
                predecessors[first].sort_unstable();
                predecessors[first].dedup();
                changed |= predecessors[first].len() != before;
            }
        }

        if !changed {
            return Ok(predecessors);
        }
    }

    Err(format!(
        "reference control-elimination predecessor propagation did not converge: states={}",
        table.num_states,
    ))
}

fn frame_top_branches(
    predecessors: &[PredecessorSet],
    origin_state: u32,
    frame: &StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
) -> Result<Vec<(u32, StackEffectFrame)>, String> {
    if let Some(&state) = frame.pushes.last() {
        return Ok(vec![(state, frame.clone())]);
    }

    let Some(states) = states_at_depth(
        predecessors,
        origin_state,
        frame.pop,
        states_at_depth_cache,
        budget,
    )
    .cloned()
    else {
        if budget.is_aborted() {
            return Err(control_elimination_failure(
                budget,
                origin_state,
                EOF,
                "top-state traversal failed",
            ));
        }
        return Ok(Vec::new());
    };

    if states.len() == 1 {
        return Ok(vec![(states[0], frame.clone())]);
    }

    let mut branches = Vec::with_capacity(states.len());
    for state in states {
        let mut branch = frame.clone();
        if add_guard_to_frame(&mut branch, frame.pop, [state]) {
            branches.push((state, branch));
        }
    }
    Ok(branches)
}

fn control_effect_closure(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    controls: &BTreeSet<TerminalID>,
    seed: StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
) -> Result<Vec<StackEffectFrame>, String> {
    let mut seen = BTreeSet::<StackEffectFrame>::new();
    let mut queue = VecDeque::<StackEffectFrame>::new();
    seen.insert(seed.clone());
    queue.push_back(seed);
    let frame_limit = (table.num_states as usize)
        .saturating_mul(controls.len().max(1))
        .saturating_mul(64)
        .saturating_add(256);

    let mut closure_pops: u64 = 0;
    while let Some(frame) = queue.pop_front() {
        closure_pops += 1;
        // Cache-hit closure steps burn no visits; check the wall backstop
        // directly so a giant sparse-control closure cannot stall.
        if closure_pops.is_multiple_of(16_384) && !budget.check_elapsed() {
            return Err(control_elimination_failure(
                budget,
                origin_state,
                EOF,
                "control stack-effect closure exceeded its compile budget",
            ));
        }
        for (top_state, top_frame) in frame_top_branches(
            predecessors,
            origin_state,
            &frame,
            states_at_depth_cache,
            budget,
        )? {
            for &control in controls {
                let Some(action) = table.action[top_state as usize].get(&control) else {
                    continue;
                };
                let Some(result) = stack_effects_for_action(
                    table,
                    predecessors,
                    origin_state,
                    control,
                    top_state,
                    action,
                    top_frame.clone(),
                    states_at_depth_cache,
                    &mut FxHashSet::default(),
                    &mut Vec::new(),
                    budget,
                ) else {
                    return Err(control_elimination_failure(
                        budget,
                        origin_state,
                        control,
                        "control stack-effect traversal failed",
                    ));
                };
                for effect in result.effects {
                    let next = StackEffectFrame {
                        pop: effect.pop,
                        pushes: effect.pushes,
                        guards: effect.guards,
                    };
                    if seen.insert(next.clone()) {
                        if seen.len() > frame_limit {
                            return Err(format!(
                                "control stack-effect closure did not converge: origin_state={origin_state} controls={controls:?} frames={} limit={frame_limit}",
                                seen.len(),
                            ));
                        }
                        queue.push_back(next);
                    }
                }
            }
        }
    }

    Ok(seen.into_iter().collect())
}

/// EOF rewrites are zero-width: follow them and reductions until Accept.
/// A repeated completion frame contributes no additional accepting effect.
fn eof_accept_effects_for_action(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    budget: &UnitInlineBudget,
) -> Result<Vec<GuardedStackShift>, String> {
    let mut effects = Vec::new();
    StackWalk {
        table,
        terminal: EOF,
        endpoint: StackEndpoint::Eof,
        domain: StackWalkDomain::Origin {
            state: origin_state, predecessors, states_at_depth_cache,
            dependencies: &mut Vec::new(), budget,
        },
        visiting,
        effects: &mut effects,
        accepts: &mut 0,
        origin_dependent: false,
    }.walk(state, action, frame).map_err(|()| control_elimination_failure(
        budget, origin_state, EOF, "EOF stack-effect traversal failed",
    ))?;
    effects.sort();
    effects.dedup();
    Ok(effects)
}

fn control_stack_effect_action(
    table: &GLRTable,
    mut effects: Vec<GuardedStackShift>,
) -> Option<Action> {
    normalize_guarded_effects_for_suffix_quotient(&mut effects);
    if effects.is_empty() {
        return None;
    }

    if effects.iter().all(|effect| effect.guards.is_empty()) {
        let mut shifts = effects
            .into_iter()
            .map(|effect| StackShift {
                pop: effect.pop,
                pushes: effect.pushes,
            })
            .collect::<Vec<_>>();
        normalize_stack_shifts(&mut shifts);
        if shifts.len() == 1 && shifts[0].pop == 0 && shifts[0].pushes.is_empty() {
            return Some(Action::Skip);
        }
        if stack_shift_predecessor_canonicalization_enabled() {
            canonicalize_stack_shift_predecessors_by_goto_superset(table, &mut shifts);
        }
        return stack_shift_action(shifts);
    }

    // This is an exact linker compilation pass, not an optional table-size
    // optimization. Environment caps used by generic reduction inlining must
    // never turn a live subgrammar path into a missing action.
    Some(Action::GuardedStackShifts(effects))
}

fn subset_key(subset: &StateSubset) -> Vec<u32> {
    subset.iter().copied().collect()
}

fn union_state_subsets(
    states: impl IntoIterator<Item = u32>,
    constituent_sets: &[StateSubset],
) -> StateSubset {
    let mut out = StateSubset::new();
    for state in states {
        out.extend_from_slice(&constituent_sets[state as usize]);
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn merge_shift_into_pending(
    pending: &mut PendingAction,
    target: u32,
    replace: bool,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<(), ()> {
    match pending.shift {
        None => {
            pending.shift = Some((target, replace));
            Ok(())
        }
        Some((existing_target, existing_replace)) => {
            if existing_target == target {
                return if existing_replace == replace { Ok(()) } else { Err(()) };
            }
            if existing_replace != replace {
                return Err(());
            }
            let merged_subset = union_state_subsets([existing_target, target], constituent_sets);
            let merged_target = ensure_subset_state(
                table,
                &merged_subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
                budget,
            )?;
            pending.shift = Some((merged_target, replace));
            Ok(())
        }
    }
}

fn merge_action_into_pending(
    pending: &mut PendingAction,
    action: &Action,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<(), ()> {
    match action {
        Action::Shift(target, replace) => merge_shift_into_pending(
            pending,
            *target,
            *replace,
            table,
            constituent_sets,
            subset_to_state,
            failed_subsets,
            budget,
        ),
        Action::StackShifts(_) | Action::ReplaceShifts(_) => Err(()),
        Action::GuardedStackShifts(_) | Action::Skip => Err(()),
        Action::Reduce(nt, len) => {
            pending.push_reduce(*nt, *len);
            Ok(())
        }
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                merge_shift_into_pending(
                    pending,
                    *target,
                    *replace,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    budget,
                )?;
            }
            for &(nt, len) in reduces {
                pending.push_reduce(nt, len);
            }
            if *accept {
                pending.push_accept();
            }
            Ok(())
        }
        Action::Accept => {
            pending.push_accept();
            Ok(())
        }
    }
}

fn build_merged_action_row(
    table: &mut GLRTable,
    subset: &StateSubset,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<ActionRow, ()> {
    let mut terminals = BTreeSet::new();
    for &state in subset {
        for tid in table.action[state as usize].keys() {
            terminals.insert(tid);
        }
    }

    let mut row = ActionRow::default();
    for tid in terminals {
        if !budget.record_cell() {
            return Err(());
        }
        let mut pending = PendingAction::default();
        for &state in subset {
            if let Some(action) = table.action[state as usize].get(&tid).cloned() {
                merge_action_into_pending(
                    &mut pending,
                    &action,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    budget,
                )?;
            }
        }
        if let Some(action) = pending.maybe_finish() {
            row.insert(tid, action);
        }
    }

    Ok(row)
}

fn build_merged_goto_row(
    table: &mut GLRTable,
    subset: &StateSubset,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<GotoRow, ()> {
    let mut nts = BTreeSet::new();
    for &state in subset {
        for &nt in table.goto[state as usize].keys() {
            nts.insert(nt);
        }
    }

    let mut row = GotoRow::default();
    for nt in nts {
        if !budget.record_cell() {
            return Err(());
        }
        let mut replace: Option<bool> = None;
        let mut target_subset = StateSubset::new();
        let mut saw_target = false;

        for &state in subset {
            if let Some(&(target, is_replace)) = table.goto[state as usize].get(&nt) {
                saw_target = true;
                match replace {
                    None => replace = Some(is_replace),
                    Some(existing) if existing == is_replace => {}
                    Some(_) => return Err(()),
                }
                target_subset.extend_from_slice(&constituent_sets[target as usize]);
            }
        }

        if !saw_target {
            continue;
        }

        target_subset.sort_unstable();
        target_subset.dedup();
        target_subset.sort_unstable();
        target_subset.dedup();
        let merged_target = ensure_subset_state(
            table,
            &target_subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
            budget,
        )?;
        row.insert(nt, (merged_target, replace.unwrap()));
    }

    Ok(row)
}

fn union_advance_rows(table: &GLRTable, subset: &StateSubset) -> BitSet {
    let mut out = BitSet::new(table.num_terminals as usize + 1);
    if table.advance.len() == table.num_states as usize {
        for &state in subset {
            out.union_with(&table.advance[state as usize]);
        }
    } else {
        for &state in subset {
            for terminal in table.action[state as usize].keys() {
                let bit = if terminal == EOF {
                    table.num_terminals as usize
                } else if terminal < table.num_terminals {
                    terminal as usize
                } else {
                    continue;
                };
                out.set(bit);
            }
        }
    }
    out
}

fn ensure_subset_state(
    table: &mut GLRTable,
    subset: &StateSubset,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &UnitInlineBudget,
) -> Result<u32, ()> {
    debug_assert!(!subset.is_empty());
    if subset.len() == 1 {
        return Ok(subset[0]);
    }

    // Forwarded replace-flagged shifts actually APPEND in the runtime.
    // A synthetic row copies a raw shift flag but has no per-terminal
    // forwarded_shifts metadata. Even OR-ing constituent flags cannot
    // represent a mixture of effective append and replace stack effects.
    // Decline this optional merge until it can represent both exactly.
    if subset.iter().any(|&member| {
        table.action[member as usize].iter().any(|(tid, action)| {
            let is_replace_shift = match action {
                Action::Shift(_, replace) => *replace,
                Action::Split { shift: Some((_, replace)), .. } => *replace,
                _ => false,
            };
            is_replace_shift && table.forwarded_shifts.contains(&(member, tid))
        })
    }) {
        return Err(());
    }

    let key = subset_key(subset);
    if let Some(&state) = subset_to_state.get(&key) {
        return Ok(state);
    }
    if failed_subsets.contains(&key) {
        return Err(());
    }

    if !budget.record_synthetic_state() {
        return Err(());
    }
    let had_advance_rows = table.advance.len() == table.num_states as usize;
    let advance_row = had_advance_rows.then(|| union_advance_rows(table, subset));

    let state = table.num_states;
    table.num_states += 1;
    table.action.push(ActionRow::default());
    table.goto.push(GotoRow::default());
    if let Some(advance_row) = advance_row {
        table.advance.push(advance_row);
    }
    constituent_sets.push(subset.clone());
    subset_to_state.insert(key.clone(), state);

    let built = (|| {
        let action_row = build_merged_action_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
            budget,
        )?;
        let goto_row = build_merged_goto_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
            budget,
        )?;
        Ok::<_, ()>((action_row, goto_row))
    })();

    match built {
        Ok((action_row, goto_row)) => {
            table.action[state as usize] = action_row;
            table.goto[state as usize] = goto_row;
            Ok(state)
        }
        Err(()) => {
            // The row builders can recursively create nested synthetic states.
            // A single pop would discard the last child but retain this parent's
            // provisional row and stale child mappings. Roll back every state
            // allocated since the parent reserved its ID, including descendants.
            subset_to_state.retain(|_, mapped| *mapped < state);
            failed_subsets.insert(key);
            table.action.truncate(state as usize);
            table.goto.truncate(state as usize);
            if had_advance_rows {
                table.advance.truncate(state as usize);
            }
            table.num_states = state;
            constituent_sets.truncate(state as usize);
            Err(())
        }
    }
}

fn refresh_merged_states_depending_on(
    table: &mut GLRTable,
    original_num_states: u32,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    dirty_original_states: &BTreeSet<u32>,
    budget: &UnitInlineBudget,
) {
    let mut state = original_num_states as usize;
    while state < table.num_states as usize {
        if !budget.check_elapsed() {
            break;
        }
        // Merged-state rows depend only on their flattened original
        // constituents, so only subsets intersecting changed original rows
        // need to be rebuilt.
        if !constituent_sets[state]
            .iter()
            .any(|member| dirty_original_states.contains(member))
        {
            state += 1;
            continue;
        }

        let subset = constituent_sets[state].clone();
        // Rebuilding an existing synthetic row can allocate *new* child
        // subset states before a later merge fails. Keep refresh atomic.
        let synthetic_savepoint = table.num_states;
        let had_advance_rows = table.advance.len() == table.num_states as usize;
        let rebuilt = (|| {
            let action_row = build_merged_action_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
                budget,
            )?;
            let goto_row = build_merged_goto_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
                budget,
            )?;
            Ok::<_, ()>((action_row, goto_row))
        })();

        match rebuilt {
            Ok((action_row, goto_row)) => {
                table.action[state] = action_row;
                table.goto[state] = goto_row;
                if table.advance.len() == table.num_states as usize {
                    table.advance[state] = union_advance_rows(table, &subset);
                }
            }
            Err(()) => {
                // The original row is still committed because assignment is
                // deferred until both rebuilt rows succeed. Undo every fresh
                // child state and mapping created by this failed refresh.
                subset_to_state.retain(|_, mapped| *mapped < synthetic_savepoint);
                table.action.truncate(synthetic_savepoint as usize);
                table.goto.truncate(synthetic_savepoint as usize);
                if had_advance_rows {
                    table.advance.truncate(synthetic_savepoint as usize);
                }
                table.num_states = synthetic_savepoint;
                constituent_sets.truncate(synthetic_savepoint as usize);
                // The original state rows have already changed. Retaining an
                // old synthetic row here would expose a stale merged action
                // or goto language, even though fresh child IDs were cleaned.
                // The caller's UnitInlineUndo restores original table cells.
                budget.abort(ABORT_REFRESH_REBUILD);
                return;
            }
        }

        state += 1;
    }
}

fn unit_reduce_destination(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    lhs: NonterminalID,
) -> Option<u32> {
    let preds = &predecessors[state as usize];
    if preds.is_empty() {
        return None;
    }

    // The fallback's unguarded specialization is sound only if every possible
    // origin has the required goto. Mixed present/missing gotos need guarded
    // effects (or a declined inline), not silent elimination of dead origins.
    if preds.iter().any(|&pred| !table.goto[pred as usize].contains_key(&lhs)) {
        return None;
    }
    let relevant_preds: Vec<u32> = preds.iter().copied().collect();

    let mut reduce_dst: Option<u32> = None;
    for pred in relevant_preds {
        let (dst, is_replace) = table.goto[pred as usize][&lhs];
        if is_replace {
            return None;
        }
        if table.goto[dst as usize] != table.goto[state as usize] {
            return None;
        }
        match reduce_dst {
            None => reduce_dst = Some(dst),
            Some(existing) if existing == dst => {}
            Some(_) => return None,
        }
    }

    reduce_dst
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StackEffectFrame {
    pop: u32,
    pushes: Vec<u32>,
    guards: Vec<StackShiftGuard>,
}

#[derive(Debug, Clone)]
struct ContextStateSharePlan {
    group: Vec<u32>,
    predecessors_by_member: Vec<Vec<u32>>,
    generated_guard_states: Vec<u32>,
    /// Exact macro row compiled against the original table during viability.
    /// Once the final sharing map is known, remapping its embedded state IDs is
    /// sufficient; recompiling the same suffix effects would be redundant.
    action_row: ActionRow,
}

fn context_share_provenance_plan(
    group: &[u32],
    predecessors: &[StateSubset],
) -> Option<ContextStateSharePlan> {
    let predecessors_by_member = group
        .iter()
        .map(|&state| predecessors.get(state as usize).map(|set| set.to_vec()))
        .collect::<Option<Vec<_>>>()?;
    if predecessors_by_member.iter().any(Vec::is_empty) {
        return None;
    }
    let mut seen = BTreeSet::<u32>::new();
    for member_predecessors in &predecessors_by_member {
        for &predecessor in member_predecessors {
            if !seen.insert(predecessor) {
                return None;
            }
        }
    }
    Some(ContextStateSharePlan {
        group: group.to_vec(),
        predecessors_by_member,
        generated_guard_states: Vec::new(),
        action_row: ActionRow::default(),
    })
}

fn context_share_frame_for_member(state: u32, predecessors: &[u32]) -> StackEffectFrame {
    StackEffectFrame {
        pop: 1,
        pushes: vec![state],
        guards: vec![StackShiftGuard {
            pop: 1,
            states: predecessors.to_vec(),
        }],
    }
}

enum ReduceFrameResult {
    Dead,
    Frames {
        frames: Vec<StackEffectFrame>,
        origin_dependent: bool,
    },
}

/// Union one predecessor frontier without sorting a potentially much larger
/// multiset of repeated targets. Only representation changes: results retain
/// the same ascending IDs, and the caller still charges one visit per depth.
fn predecessor_frontier_union(
    predecessors: &[PredecessorSet],
    states: &[u32],
    bitmap: &mut Vec<u64>,
    bitmap_enabled: bool,
) -> Option<StateSubset> {
    fn sorted_union(predecessors: &[PredecessorSet], states: &[u32]) -> Option<StateSubset> {
        let mut next = StateSubset::new();
        for &state in states {
            next.extend_from_slice(predecessors.get(state as usize)?);
        }
        next.sort_unstable();
        next.dedup();
        Some(next)
    }

    const MAX_BITMAP_STATES: usize = 65_536;
    if !bitmap_enabled || states.len() < 4 || predecessors.len() > MAX_BITMAP_STATES {
        return sorted_union(predecessors, states);
    }
    let words = predecessors.len().div_ceil(64);
    let mut edges = 0usize;
    for &state in states {
        edges = edges.saturating_add(predecessors.get(state as usize)?.len());
    }
    // Sparse unions keep the original small-vector path. The threshold pays
    // for clearing/enumerating the bounded bitmap as well as the second scan.
    if edges < words.saturating_mul(4).max(64) {
        return sorted_union(predecessors, states);
    }
    bitmap.resize(words, 0);
    bitmap.fill(0);
    for &state in states {
        for &target in &predecessors[state as usize] {
            let Some(word) = bitmap.get_mut(target as usize / 64) else {
                // A malformed graph can contain a dangling target that the
                // reference retains at its last depth. Never silently discard
                // it or assume every ID fits the valid-state bitmap.
                return sorted_union(predecessors, states);
            };
            *word |= 1u64 << (target % 64);
        }
    }
    let mut next = StateSubset::new();
    for (index, &word) in bitmap.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            next.push((index * 64 + word.trailing_zeros() as usize) as u32);
            word &= word - 1;
        }
    }
    Some(next)
}


fn states_at_depth<'a>(
    predecessors: &[PredecessorSet],
    origin_state: u32,
    depth: u32,
    cache: &'a mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
) -> Option<&'a StateSubset> {
    states_at_depth_with_bitmap(predecessors, origin_state, depth, cache, budget, true)
}

fn states_at_depth_with_bitmap<'a>(
    predecessors: &[PredecessorSet],
    origin_state: u32,
    depth: u32,
    cache: &'a mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
    bitmap_enabled: bool,
) -> Option<&'a StateSubset> {
    let cache_key = (origin_state, depth);
    if !cache.contains_key(&cache_key) {
        let mut states = smallvec![origin_state];
        let mut bitmap = Vec::new();
        for _ in 0..depth {
            if !budget.record_stack_effect_visit() {
                return None;
            }
            let next = predecessor_frontier_union(predecessors, &states, &mut bitmap, bitmap_enabled)?;
            if next.is_empty() {
                cache.insert(cache_key, None);
                return None;
            }
            states = next;
        }

        cache.insert(cache_key, Some(states));
    }

    cache.get(&cache_key).and_then(Option::as_ref)
}

fn normalize_states(mut states: Vec<u32>) -> Vec<u32> {
    states.sort_unstable();
    states.dedup();
    states
}

fn add_guard_to_frame(
    frame: &mut StackEffectFrame,
    pop: u32,
    states: impl IntoIterator<Item = u32>,
) -> bool {
    let states = normalize_states(states.into_iter().collect());
    if states.is_empty() {
        return false;
    }

    if let Some(existing) = frame.guards.iter_mut().find(|guard| guard.pop == pop) {
        let wanted: BTreeSet<u32> = states.into_iter().collect();
        existing.states.retain(|state| wanted.contains(state));
        return !existing.states.is_empty();
    }

    frame.guards.push(StackShiftGuard { pop, states });
    frame.guards.sort_by_key(|guard| guard.pop);
    true
}

fn pop_frame(frame: &mut StackEffectFrame, pop: u32) -> Result<(), ()> {
    if pop as usize <= frame.pushes.len() {
        frame.pushes.truncate(frame.pushes.len() - pop as usize);
    } else {
        frame.pop = frame.pop.checked_add(pop - frame.pushes.len() as u32).ok_or(())?;
        frame.pushes.clear();
    }
    Ok(())
}

fn push_transition_to_frame(frame: &mut StackEffectFrame, target: u32, replace: bool) -> Result<(), ()> {
    pop_frame(frame, u32::from(replace))?;
    frame.pushes.push(target);
    Ok(())
}

fn frame_to_guarded_shift(frame: StackEffectFrame) -> GuardedStackShift {
    GuardedStackShift {
        guards: frame.guards,
        pop: frame.pop,
        pushes: frame.pushes,
    }
}

fn stack_effect_action_tag(action: &Action) -> u8 {
    match action {
        Action::Shift(..) => 0,
        Action::StackShifts(_) => 1,
        Action::GuardedStackShifts(_) => 2,
        Action::Reduce(..) => 3,
        Action::Split { .. } => 4,
        Action::Accept => 5,
        Action::ReplaceShifts(_) => 6,
        Action::Skip => 7,
    }
}

/// Reduce after exposing the goto source. A certified candidate set may
/// omit a guard only if every candidate has exactly the same goto effect.
/// A suffix-state expansion lacks that certificate and must guard every
/// unknown source, even when the goto row happens to be unique.
fn reduce_exposed_frame(
    table: &GLRTable,
    frame: StackEffectFrame,
    nt: NonterminalID,
    candidates: &[u32],
    require_guard: bool,
) -> Result<Vec<StackEffectFrame>, ()> {
    let mut by_target = BTreeMap::<(u32, bool), Vec<u32>>::new();
    let mut missing = false;
    for &source in candidates {
        if let Some(&(target, replace)) = table.goto.get(source as usize).ok_or(())?.get(&nt) {
            by_target.entry((target, replace)).or_default().push(source);
        } else {
            missing = true;
        }
    }
    let needs_guard = require_guard || missing || by_target.len() > 1;
    let guard_pop = frame.pop;
    let mut frames = Vec::new();
    for ((target, replace), froms) in by_target {
        let mut next = frame.clone();
        if needs_guard && !add_guard_to_frame(&mut next, guard_pop, froms) {
            continue;
        }
        push_transition_to_frame(&mut next, target, replace)?;
        frames.push(next);
    }
    frames.sort();
    frames.dedup();
    Ok(frames)
}

fn apply_reduce_to_frame(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    budget: &UnitInlineBudget,
) -> Option<ReduceFrameResult> {
    pop_frame(&mut frame, len).ok()?;
    let origin_dependent = frame.pushes.is_empty();
    let direct;
    let candidates = if let Some(&state) = frame.pushes.last() {
        direct = [state];
        &direct[..]
    } else {
        states_at_depth(
            predecessors, origin_state, frame.pop, states_at_depth_cache, budget,
        )?
    };
    let frames = reduce_exposed_frame(table, frame, nt, candidates, false).ok()?;
    if frames.is_empty() {
        Some(ReduceFrameResult::Dead)
    } else {
        Some(ReduceFrameResult::Frames { frames, origin_dependent })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StackEndpoint {
    /// A cell must consume the symbol; an Accept branch is not representable.
    Consume,
    /// A reconstructed suffix may accept, but its caller must establish
    /// whether accepting and consuming alternatives can coexist.
    Suffix,
    /// All rewrites are zero-width until canonical acceptance is reached.
    Eof,
}

enum StackWalkDomain<'a> {
    Origin {
        state: u32,
        predecessors: &'a [PredecessorSet],
        states_at_depth_cache: &'a mut FxHashMap<(u32, u32), Option<StateSubset>>,
        dependencies: &'a mut Vec<(UnitInlineCell, UnitInlineCell)>,
        budget: &'a UnitInlineBudget,
    },
    Suffix(Option<&'a [Vec<u32>]>),
}

/// One interpreter for a partial relation over the unknown original stack
/// prefix and a known, written suffix. Domain selection determines which
/// exposed goto sources need guards; endpoint selection determines when
/// to stop. Neither changes the composition of stack rewrites themselves.
struct StackWalk<'a> {
    table: &'a GLRTable,
    terminal: TerminalID,
    endpoint: StackEndpoint,
    domain: StackWalkDomain<'a>,
    visiting: &'a mut FxHashSet<StackEffectVisitKey>,
    effects: &'a mut Vec<GuardedStackShift>,
    accepts: &'a mut usize,
    origin_dependent: bool,
}

enum StackWalkTask<'a> {
    Enter { state: u32, action: &'a Action, frame: StackEffectFrame, dependency: bool },
    Exit(StackEffectVisitKey),
    Continue(StackEffectFrame),
    Reduce { nt: NonterminalID, len: u32, frame: StackEffectFrame, charge: bool },
}

impl<'a> StackWalk<'a> {
    fn charge(&self) -> Result<(), ()> {
        match &self.domain {
            StackWalkDomain::Origin { budget, .. } if !budget.record_stack_effect_visit() => Err(()),
            _ => Ok(()),
        }
    }

    fn reduce(
        &mut self,
        frame: StackEffectFrame,
        nt: NonterminalID,
        len: u32,
    ) -> Result<Vec<StackEffectFrame>, ()> {
        match &mut self.domain {
            StackWalkDomain::Origin { state, predecessors, states_at_depth_cache, budget, .. } => {
                match apply_reduce_to_frame(
                    self.table, predecessors, *state, frame, nt, len, states_at_depth_cache, budget,
                ).ok_or(())? {
                    ReduceFrameResult::Dead => Ok(Vec::new()),
                    ReduceFrameResult::Frames { frames, origin_dependent } => {
                        self.origin_dependent |= origin_dependent;
                        Ok(frames)
                    }
                }
            }
            StackWalkDomain::Suffix(sources) => reduce_suffix_frame(
                self.table, frame, nt, len, *sources,
            ),
        }
    }

    fn schedule_continuation<'w>(
        &mut self, frame: StackEffectFrame, agenda: &mut Vec<StackWalkTask<'w>>,
    ) -> Result<(), ()> where 'a: 'w {
        let table = self.table;
        let terminal = self.terminal;
        if self.endpoint == StackEndpoint::Eof {
            let StackWalkDomain::Origin { state, predecessors, states_at_depth_cache, budget, .. }
                = &mut self.domain else { return Err(()); };
            let tops = frame_top_branches(
                predecessors, *state, &frame, states_at_depth_cache, budget,
            ).map_err(|_| ())?;
            for (state, frame) in tops.into_iter().rev() {
                if let Some(action) = table.action.get(state as usize).ok_or(())?.get(&terminal) {
                    agenda.push(StackWalkTask::Enter { state, action, frame, dependency: false });
                }
            }
        } else if let Some(&state) = frame.pushes.last()
            && let Some(action) = table.action.get(state as usize).ok_or(())?.get(&terminal)
        {
            agenda.push(StackWalkTask::Enter { state, action, frame, dependency: true });
        }
        Ok(())
    }

    /// Explicit DFS continuations retain path-local cycle detection and
    /// the old depth-first branch order without borrowing the native call
    /// stack. Speculative suffix expansion may decline on resource limits;
    /// it never publishes a partially interpreted relation.
    fn walk<'w>(
        &mut self, state: u32, action: &'w Action, frame: StackEffectFrame,
    ) -> Result<(), ()> where 'a: 'w {
        let mut agenda = vec![StackWalkTask::Enter { state, action, frame, dependency: false }];
        let mut suffix_visits = 0usize;
        let result = (|| {
            while let Some(task) = agenda.pop() {
                match task {
                    StackWalkTask::Exit(key) => {
                        self.visiting.remove(&key);
                    }
                    StackWalkTask::Continue(frame) => self.schedule_continuation(frame, &mut agenda)?,
                    StackWalkTask::Reduce { nt, len, frame, charge } => {
                        if charge {
                            self.charge()?;
                        }
                        for frame in self.reduce(frame, nt, len)?.into_iter().rev() {
                            agenda.push(StackWalkTask::Continue(frame));
                        }
                    }
                    StackWalkTask::Enter { state, action, frame, dependency } => {
                        if dependency
                            && let StackWalkDomain::Origin { state: origin, dependencies, .. } = &mut self.domain
                        {
                            dependencies.push(((*origin, self.terminal), (state, self.terminal)));
                        }
                        self.charge()?;
                        if self.endpoint == StackEndpoint::Suffix {
                            suffix_visits += 1;
                            if suffix_visits > EXACT_CONTROL_ELIMINATION_MAX_STACK_EFFECT_VISITS {
                                return Err(());
                            }
                        }
                        let key = StackEffectVisitKey {
                            state, tid: self.terminal,
                            action_tag: stack_effect_action_tag(action),
                            frame: frame.clone(),
                        };
                        if !self.visiting.insert(key.clone()) {
                            if self.endpoint == StackEndpoint::Eof {
                                continue;
                            }
                            return Err(());
                        }
                        agenda.push(StackWalkTask::Exit(key));
                        self.schedule_cell(state, action, frame, &mut agenda)?;
                    }
                }
                if self.endpoint == StackEndpoint::Suffix
                    && agenda.len() > EXACT_CONTROL_ELIMINATION_MAX_STACK_EFFECT_VISITS
                {
                    return Err(());
                }
            }
            Ok(())
        })();
        // An early decline must unwind the same path-local keys a recursive
        // interpreter would remove while returning through its callers.
        for task in agenda {
            if let StackWalkTask::Exit(key) = task {
                self.visiting.remove(&key);
            }
        }
        result
    }

    fn schedule_cell<'w>(
        &mut self, state: u32, action: &Action, frame: StackEffectFrame,
        agenda: &mut Vec<StackWalkTask<'w>>,
    ) -> Result<(), ()> {
        if matches!(action, Action::Accept | Action::Split { accept: true, .. }) {
            match self.endpoint {
                StackEndpoint::Consume => return Err(()),
                StackEndpoint::Suffix => *self.accepts += 1,
                StackEndpoint::Eof => self.effects.push(frame_to_guarded_shift(frame.clone())),
            }
        }
        let split = matches!(action, Action::Split { .. });
        if split && self.endpoint != StackEndpoint::Eof
            && matches!(action, Action::Split { shift: Some(_), .. })
        {
            self.charge()?;
        }
        let rewritten = compose_action_rewrites(
            self.table, state, self.terminal, action, &frame,
            self.endpoint == StackEndpoint::Eof,
        )?;
        let mut reductions = SmallVec::<[(NonterminalID, u32); 2]>::new();
        action.for_each_reduce(|nt, len| reductions.push((nt, len)));
        // The agenda is LIFO: consuming/completion alternatives precede
        // reduction alternatives in their original row order.
        for (nt, len) in reductions.into_iter().rev() {
            agenda.push(StackWalkTask::Reduce { nt, len, frame: frame.clone(), charge: split });
        }
        if self.endpoint == StackEndpoint::Eof {
            for next in rewritten.into_iter().rev() {
                agenda.push(StackWalkTask::Continue(next));
            }
        } else {
            self.effects.extend(rewritten.into_iter().map(frame_to_guarded_shift));
        }
        Ok(())
    }
}

fn stack_effects_for_action(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    origin_state: u32,
    tid: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    dependencies: &mut Vec<(UnitInlineCell, UnitInlineCell)>,
    budget: &UnitInlineBudget,
) -> Option<StackEffectResult> {
    let mut effects = Vec::new();
    let mut accepts = 0;
    let mut walk = StackWalk {
        table,
        terminal: tid,
        endpoint: StackEndpoint::Consume,
        domain: StackWalkDomain::Origin {
            state: origin_state, predecessors, states_at_depth_cache, dependencies, budget,
        },
        visiting,
        effects: &mut effects,
        accepts: &mut accepts,
        origin_dependent: false,
    };
    walk.walk(state, action, frame).ok()?;
    let origin_dependent = walk.origin_dependent;
    effects.sort();
    effects.dedup();
    Some(StackEffectResult { effects, origin_dependent })
}

fn normalize_guarded_effects(effects: &mut Vec<GuardedStackShift>) {
    effects.retain(|effect| !effect.pushes.is_empty());
    normalize_guarded_effects_for_suffix_quotient(effects);
}

fn stack_effect_action(table: &GLRTable, mut effects: Vec<GuardedStackShift>) -> Option<Action> {
    normalize_guarded_effects(&mut effects);
    if effects.is_empty() {
        return None;
    }
    // Core-merged predecessor graphs can encode guarded effects very
    // compactly in the LR table but expand them into runtime guard scans. The
    // reduction interpreter is cheaper for those origin-dependent actions.
    // Unguarded effects are deterministic stack rewrites and remain profitable.
    if table.construction == GlrTableConstruction::ExperimentalCoreMerged
        && effects.iter().any(|effect| !effect.guards.is_empty())
    {
        return None;
    }
    if effects.iter().all(|effect| effect.guards.is_empty()) {
        let mut shifts: Vec<_> = effects
            .into_iter()
            .map(|effect| StackShift {
                pop: effect.pop,
                pushes: effect.pushes,
            })
            .collect();
        if stack_shift_predecessor_canonicalization_enabled() {
            canonicalize_stack_shift_predecessors_by_goto_superset(table, &mut shifts);
        }
        return stack_shift_action(shifts);
    }
    // Opt-in diagnostic knob only. Do not use this to hide correctness or
    // compile-time bugs in guarded stack-effect lowering.
    if max_guarded_stack_effects().is_some_and(|limit| effects.len() > limit) {
        return None;
    }
    Some(Action::GuardedStackShifts(effects))
}

#[derive(Clone, Copy, Debug)]
enum StackInlineOutcomeKind {
    NotCandidate,
    TraversalFailed,
    EmptyEffects,
    OriginDependentOnePush,
    TrivialOriginDependentOnePush,
    CannotRepresent,
    Inlined,
}

impl StackInlineOutcomeKind {
    const COUNT: usize = 7;

    fn index(self) -> usize {
        match self {
            Self::NotCandidate => 0,
            Self::TraversalFailed => 1,
            Self::EmptyEffects => 2,
            Self::OriginDependentOnePush => 3,
            Self::TrivialOriginDependentOnePush => 4,
            Self::CannotRepresent => 5,
            Self::Inlined => 6,
        }
    }
}

struct StackInlineAttempt {
    action: Option<Action>,
    outcome: StackInlineOutcomeKind,
}

fn try_inline_action_to_stack_shifts_detailed(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    tid: TerminalID,
    action: &Action,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    dependencies: &mut Vec<(UnitInlineCell, UnitInlineCell)>,
    budget: &UnitInlineBudget,
) -> StackInlineAttempt {
    let has_reductions = match action {
        Action::Reduce(..) => true,
        Action::Split {
            reduces,
            accept: false,
            ..
        } => !reduces.is_empty(),
        _ => false,
    };
    if !has_reductions {
        return StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::NotCandidate,
        };
    }

    let Some(result) = stack_effects_for_action(
        table,
        predecessors,
        state,
        tid,
        state,
        action,
        StackEffectFrame {
            pop: 0,
            pushes: Vec::new(),
            guards: Vec::new(),
        },
        states_at_depth_cache,
        &mut FxHashSet::default(),
        dependencies,
        budget,
    ) else {
        return StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::TraversalFailed,
        };
    };
    let effects = result.effects;
    if result.origin_dependent
        && matches!(action, Action::Reduce(_, 1))
        && !effects.is_empty()
        && effects.iter().all(|effect| effect.pushes.len() == 1)
    {
        return StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::OriginDependentOnePush,
        };
    }
    if effects.is_empty() {
        return StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::EmptyEffects,
        };
    }
    match stack_effect_action(table, effects) {
        Some(action) => StackInlineAttempt {
            action: Some(action),
            outcome: StackInlineOutcomeKind::Inlined,
        },
        None => StackInlineAttempt {
            action: None,
            outcome: StackInlineOutcomeKind::CannotRepresent,
        },
    }
}

fn try_inline_action_to_stack_shifts(
    table: &GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    tid: TerminalID,
    action: &Action,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<StateSubset>>,
    dependencies: &mut Vec<(UnitInlineCell, UnitInlineCell)>,
    budget: &UnitInlineBudget,
) -> Option<Action> {
    try_inline_action_to_stack_shifts_detailed(
        table,
        predecessors,
        state,
        tid,
        action,
        states_at_depth_cache,
        dependencies,
        budget,
    )
    .action
}

fn normalize_stack_shifts(shifts: &mut Vec<StackShift>) {
    shifts.sort_by(|a, b| a.pop.cmp(&b.pop).then_with(|| a.pushes.cmp(&b.pushes)));
    shifts.dedup();
}

fn canonicalize_stack_shift_predecessors_by_goto_superset(table: &GLRTable, shifts: &mut [StackShift]) {
    for idx in 0..shifts.len() {
        if shifts[idx].pushes.len() < 2 {
            continue;
        }

        for pos in 0..shifts[idx].pushes.len() - 1 {
            for rep_idx in 0..idx {
                if shifts[idx].pop != shifts[rep_idx].pop
                    || shifts[idx].pushes.len() != shifts[rep_idx].pushes.len()
                    || shifts[idx].pushes[..pos] != shifts[rep_idx].pushes[..pos]
                    || shifts[idx].pushes[pos + 1..] != shifts[rep_idx].pushes[pos + 1..]
                {
                    continue;
                }

                // This pushed state is buried below an identical pushed suffix
                // and below the current top state. Once buried, it can only be
                // observed by a later reduction querying its goto row, so
                // prefer a predecessor whose goto row is a compatible superset
                // and let the otherwise identical stack paths merge.
                let pred = shifts[idx].pushes[pos];
                let rep = shifts[rep_idx].pushes[pos];
                if goto_row_is_target_compatible_subset(table, pred, rep) {
                    shifts[idx].pushes[pos] = rep;
                    break;
                }
                if goto_row_is_target_compatible_subset(table, rep, pred) {
                    shifts[rep_idx].pushes[pos] = pred;
                    break;
                }
            }
        }
    }
}

fn goto_row_is_target_compatible_subset(table: &GLRTable, subset: u32, superset: u32) -> bool {
    let subset_row = &table.goto[subset as usize];
    let superset_row = &table.goto[superset as usize];
    !subset_row.is_empty()
        && subset_row.iter().all(|(&nt, &target)| {
            superset_row
                .get(&nt)
                .is_some_and(|&superset_target| superset_target == target)
        })
}

fn stack_shift_action(mut shifts: Vec<StackShift>) -> Option<Action> {
    normalize_stack_shifts(&mut shifts);
    if shifts.is_empty() {
        return None;
    }
    if shifts.len() == 1 {
        let shift = &shifts[0];
        if shift.pushes.len() == 1 {
            match shift.pop {
                0 => return Some(Action::Shift(shift.pushes[0], false)),
                1 => return Some(Action::Shift(shift.pushes[0], true)),
                _ => {}
            }
        }
    }
    Some(Action::StackShifts(shifts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::accumulator::TerminalsDisallowed;
    use crate::glr::parser::{ParserGSS, advance_stacks};
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        name: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let previous = std::env::var(name).ok();
            unsafe {
                std::env::set_var(name, value);
            }
            Self { name, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                if let Some(previous) = &self.previous {
                    std::env::set_var(self.name, previous);
                } else {
                    std::env::remove_var(self.name);
                }
            }
        }
    }

    fn default_unit_inline_budget() -> UnitInlineBudget {
        UnitInlineBudget {
            started_at: std::time::Instant::now(),
            max_ms: DEFAULT_UNIT_INLINE_WORK_MAX_WALL_MS,
            max_iterations: DEFAULT_UNIT_INLINE_WORK_MAX_ITERATIONS,
            max_cells: DEFAULT_UNIT_INLINE_WORK_MAX_CELLS,
            max_synthetic_states: DEFAULT_UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES,
            max_stack_effect_visits: DEFAULT_UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS,
            iterations: std::sync::atomic::AtomicUsize::new(0),
            cells: std::sync::atomic::AtomicUsize::new(0),
            synthetic_states: std::sync::atomic::AtomicUsize::new(0),
            stack_effect_visits: std::sync::atomic::AtomicUsize::new(0),
            abort_code: std::sync::atomic::AtomicU8::new(ABORT_NONE),
        }
    }

    #[test]
    fn bitmap_predecessor_frontiers_match_literal_sort_with_duplicates_and_dangling_ids() {
        fn reference(predecessors: &[PredecessorSet], states: &[u32]) -> Option<StateSubset> {
            let mut out = StateSubset::new();
            for &state in states { out.extend_from_slice(predecessors.get(state as usize)?); }
            out.sort_unstable();
            out.dedup();
            Some(out)
        }
        let mut seed = 948752_u64;
        let mut random = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };
        let mut bitmap = Vec::new();
        for n in [0, 1, 2, 7, 63, 64, 65, 257, 1024] {
            for _ in 0..12 {
                let mut predecessors = vec![PredecessorSet::new(); n];
                for row in &mut predecessors {
                    for _ in 0..(random() as usize % 80) {
                        row.push((random() as usize % n) as u32);
                    }
                }
                let mut states: Vec<u32> = (0..n.min(20) as u32).rev().collect();
                for _ in 0..2 {
                    assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true),
                               reference(&predecessors, &states));
                    states.reverse();
                }
                if n > 0 {
                    // A dangling target is returned at the last depth, not
                    // dropped even when another branch uses the bitmap.
                    predecessors[0].extend_from_slice(&[n as u32, u32::MAX, 0, 0]);
                    assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true),
                               reference(&predecessors, &states));
                }
                states.push(n as u32);
                assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true), None);
            }
        }
        let mut predecessors = vec![PredecessorSet::new(); 65];
        for row in predecessors.iter_mut().take(8) {
            row.extend((0..65).rev().chain(0..65));
        }
        let states: Vec<_> = (0..8).collect();
        bitmap.clear();
        assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true),
                   reference(&predecessors, &states));
        assert_eq!(bitmap.len(), 2, "fixture must actually use dense accumulation");
        // Bits above the last valid state still represent literal dangling IDs.
        predecessors[0].push(127);
        assert_eq!(predecessor_frontier_union(&predecessors, &states, &mut bitmap, true),
                   reference(&predecessors, &states));
        let mut disabled = Vec::new();
        predecessor_frontier_union(&predecessors, &states, &mut disabled, false);
        assert_eq!(disabled.capacity(), 0);
        let huge = vec![PredecessorSet::new(); 65_537];
        let mut oversized = Vec::new();
        assert_eq!(predecessor_frontier_union(&huge, &[0, 1, 2, 3], &mut oversized, true), Some(StateSubset::new()));
        assert_eq!(oversized.capacity(), 0);
    }

    #[test]
    fn bitmap_depth_queries_preserve_cache_failure_and_exact_visit_accounting() {
        let mut graphs = Vec::new();
        for n in [1, 8, 65, 257] {
            let mut graph = vec![PredecessorSet::new(); n];
            for (state, row) in graph.iter_mut().enumerate() {
                for offset in 0..n.min(40) {
                    row.push(((state + offset) % n) as u32);
                    if offset % 3 == 0 { row.push(((state + offset) % n) as u32); }
                }
            }
            graphs.push(graph.clone());
            graph[0].push(u32::MAX);
            graphs.push(graph);
        }
        graphs.push(vec![PredecessorSet::new(); 8]);
        for graph in graphs {
            for limit in [1, 4, 100_000, usize::MAX] {
                let mut caches = [FxHashMap::default(), FxHashMap::default()];
                let mut budgets = [default_unit_inline_budget(), default_unit_inline_budget()];
                for b in &mut budgets { b.max_ms = u128::MAX; b.max_stack_effect_visits = limit; }
                for (origin, depth) in [(0, 0), (0, 1), (0, 3), (0, 3), (1, 2), (0, 8), (u32::MAX, 0), (u32::MAX, 1)] {
                    let left = states_at_depth_with_bitmap(&graph, origin, depth, &mut caches[0], &budgets[0], false).cloned();
                    let right = states_at_depth_with_bitmap(&graph, origin, depth, &mut caches[1], &budgets[1], true).cloned();
                    assert_eq!(left, right, "origin={origin} depth={depth} limit={limit}");
                    assert_eq!(caches[0], caches[1]);
                    assert_eq!(budgets[0].stack_effect_visits(), budgets[1].stack_effect_visits());
                    assert_eq!(budgets[0].is_aborted(), budgets[1].is_aborted());
                    assert_eq!(budgets[0].report().reason, budgets[1].report().reason);
                }
            }
        }
    }

    #[test]
    fn completed_work_certificate_is_strict_monotone_and_overflow_safe() {
        let mut budget = default_unit_inline_budget();
        budget.max_stack_effect_visits = 10;
        budget.stack_effect_visits.store(3, std::sync::atomic::Ordering::Relaxed);
        let work = CompletedStackEffectWork::new(&budget).unwrap();
        assert!(!work.exceeded());
        work.record_completed(7);
        assert!(!work.exceeded(), "work exactly at the cap must continue");
        work.record_completed(1);
        assert!(work.exceeded());
        work.record_completed(usize::MAX);
        assert!(work.exceeded(), "addition must not wrap below the cap");

        budget.max_stack_effect_visits = usize::MAX - 1;
        let work = CompletedStackEffectWork::new(&budget).unwrap();
        work.record_completed(usize::MAX);
        assert!(work.exceeded());
        budget.max_stack_effect_visits = usize::MAX;
        assert!(CompletedStackEffectWork::new(&budget).is_none());
    }

    #[test]
    fn completed_work_production_policy_keeps_legacy_and_lalr_scheduling() {
        assert!(completed_work_abort_eligible(GlrTableConstruction::ExperimentalCoreMerged));
        assert!(!completed_work_abort_eligible(GlrTableConstruction::LegacyRowBisim));
        assert!(!completed_work_abort_eligible(GlrTableConstruction::Lalr));
    }

    #[test]
    fn completed_work_certificate_counts_locally_exhausted_states_concurrently() {
        let mut budget = default_unit_inline_budget();
        budget.max_stack_effect_visits = 63;
        let work = CompletedStackEffectWork::new(&budget).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            (0..16).into_par_iter().for_each(|_| {
                let child = budget.child_with_stack_effect_visit_limit(3);
                for _ in 0..3 { assert!(child.record_stack_effect_visit()); }
                assert!(!child.record_stack_effect_visit());
                work.record_completed(child.stack_effect_visits());
            });
        });
        assert!(work.exceeded());
        // The certificate never aborts or spends the parent's live budget.
        // The read-only phase must finish before folding/rollback is selected.
        assert!(!budget.is_aborted());
        assert_eq!(budget.stack_effect_visits(), 0);
    }

    #[test]
    fn completed_work_abort_preserves_successful_and_rolled_back_tables() {
        fn table(groups: usize, construction: GlrTableConstruction) -> GLRTable {
            let n = groups * 5;
            let mut action = vec![ActionRow::default(); n];
            let mut goto = vec![GotoRow::default(); n];
            for group in 0..groups {
                let base = group * 5;
                action[base + 1].insert(0, Action::Shift((base + 2) as u32, false));
                action[base + 2].insert(0, Action::Split {
                    shift: Some(((base + 4) as u32, false)),
                    reduces: vec![(10, 1)], accept: false,
                });
                action[base + 3].insert(0, Action::Shift((base + 4) as u32, false));
                goto[base + 1].insert(10, ((base + 3) as u32, true));
            }
            let mut table = GLRTable {
                action, goto, num_states: n as u32, num_terminals: 1,
                num_rules: 0, rules: Vec::new(), nonterminal_display_names: Vec::new(),
                construction, admission_policy: AdmissionPolicy::ExactSimulation,
                advance: Vec::new(), unconditional_advance: Vec::new(),
                forwarded_shifts: FxHashSet::default(),
                control_terminals: Default::default(), skip_terminals: Default::default(),
                guarded_shift_index: Vec::new(), direct_regular_wide_frontiers: Vec::new(),
            };
            table.rebuild_advance_rows_from_actions();
            table
        }

        let mut aborts = 0;
        let mut successes = 0;
        let mut changed = 0;
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            for construction in [GlrTableConstruction::LegacyRowBisim,
                                 GlrTableConstruction::ExperimentalCoreMerged] {
                for groups in [1, 8, 64] {
                    for limit in [1, 8, 64, 100_000, usize::MAX] {
                        let input = table(groups, construction);
                        let protected = BitSet::new(1);
                        let mut results = Vec::new();
                        for stop in [false, true] {
                            let mut result = input.clone();
                            let mut budget = default_unit_inline_budget();
                            budget.max_ms = u128::MAX;
                            budget.max_stack_effect_visits = limit;
                            let mut undo = UnitInlineUndo::new(&result);
                            if limit == 1 {
                                // Model a prior iteration that already changed
                                // an original row: certified abort must undo it,
                                // not merely leave the current phase untouched.
                                undo.record_cell(&result, 0, 0);
                                result.action[0].insert(0, Action::Shift(1, false));
                                result.advance[0].set(0);
                                budget.stack_effect_visits.store(
                                    1, std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            pool.install(|| {
                                result.collapse_sr_unit_reductions_with_completed_work_abort(
                                    &budget, &mut undo, &protected, stop,
                                );
                            });
                            if budget.is_aborted() { undo.rollback(&mut result); }
                            results.push((result, budget.is_aborted()));
                        }
                        assert_eq!(results[0].1, results[1].1,
                                   "abort policy threads={threads} groups={groups} limit={limit}");
                        assert_eq!(bincode::serialize(&results[0].0).unwrap(),
                                   bincode::serialize(&results[1].0).unwrap(),
                                   "complete table threads={threads} groups={groups} limit={limit}");
                        if results[0].1 {
                            aborts += 1;
                            assert_eq!(results[0].0.action, input.action);
                            assert_eq!(results[0].0.goto, input.goto);
                        } else {
                            successes += 1;
                            changed += usize::from(results[0].0.action != input.action);
                        }
                    }
                }
            }
        }
        assert!(aborts > 0 && successes > 0 && changed > 0,
                "must cover real aborts and successful nontrivial optimization");
    }

    #[test]
    fn control_predecessor_propagation_matches_synchronous_reference() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        fn random_pushes(seed: &mut u64, nstates: u32) -> Vec<u32> {
            let len = (next(seed) % 3 + 1) as usize;
            (0..len)
                .map(|_| (next(seed) % u64::from(nstates)) as u32)
                .collect()
        }

        fn random_action(seed: &mut u64, nstates: u32) -> Action {
            match next(seed) % 8 {
                0 => Action::Shift(
                    (next(seed) % u64::from(nstates)) as u32,
                    next(seed) & 1 != 0,
                ),
                1 => Action::ReplaceShifts(
                    (0..(next(seed) % 3 + 1))
                        .map(|_| (next(seed) % u64::from(nstates)) as u32)
                        .collect(),
                ),
                2 => Action::StackShifts(
                    (0..(next(seed) % 3 + 1))
                        .map(|_| StackShift {
                            pop: (next(seed) % 4) as u32,
                            pushes: random_pushes(seed, nstates),
                        })
                        .collect(),
                ),
                3 => Action::GuardedStackShifts(
                    (0..(next(seed) % 3 + 1))
                        .map(|_| GuardedStackShift {
                            guards: Vec::new(),
                            pop: (next(seed) % 4) as u32,
                            pushes: random_pushes(seed, nstates),
                        })
                        .collect(),
                ),
                4 => Action::Split {
                    shift: Some((
                        (next(seed) % u64::from(nstates)) as u32,
                        next(seed) & 1 != 0,
                    )),
                    reduces: Vec::new(),
                    accept: false,
                },
                5 => Action::Skip,
                6 => Action::Reduce((next(seed) % 4) as u32, (next(seed) % 4) as u32),
                _ => Action::Accept,
            }
        }

        let mut seed = 0xD1CE_F00D_5EED_BAADu64;
        for case in 0..256 {
            let nstates = (next(&mut seed) % 9 + 2) as usize;
            let mut action = vec![ActionRow::default(); nstates];
            let mut goto = vec![GotoRow::default(); nstates];

            for state in 0..nstates {
                for terminal in 0..4 {
                    if !next(&mut seed).is_multiple_of(3) {
                        action[state].insert(
                            terminal,
                            random_action(&mut seed, nstates as u32),
                        );
                    }
                }
                for nonterminal in 0..4 {
                    if next(&mut seed) & 1 != 0 {
                        goto[state].insert(
                            nonterminal,
                            (
                                (next(&mut seed) % nstates as u64) as u32,
                                next(&mut seed) & 1 != 0,
                            ),
                        );
                    }
                }
            }

            let table = GLRTable {
                action,
                goto,
                num_states: nstates as u32,
                num_terminals: 4,
                num_rules: 0,
                rules: Vec::new(),
                nonterminal_display_names: Vec::new(),
                construction: GlrTableConstruction::LegacyRowBisim,
                admission_policy: AdmissionPolicy::RowPresenceExact,
                advance: Vec::new(),
                unconditional_advance: Vec::new(),
                forwarded_shifts: FxHashSet::default(),
                control_terminals: Default::default(),
                skip_terminals: Default::default(),
                guarded_shift_index: Vec::new(),
                direct_regular_wide_frontiers: Vec::new(),
            };

            let expected = build_control_elimination_predecessors_synchronous_reference(&table)
                .unwrap_or_else(|error| panic!("reference failed for case {case}: {error}"));
            let actual = build_control_elimination_predecessors(&table)
                .unwrap_or_else(|error| panic!("optimized failed for case {case}: {error}"));
            assert_eq!(actual, expected, "case {case}");

            let mut origins = (0..nstates as u32)
                .filter(|_| next(&mut seed) & 3 == 0)
                .collect::<Vec<_>>();
            if origins.is_empty() {
                origins.push((next(&mut seed) % nstates as u64) as u32);
            }
            origins.sort_unstable();
            origins.dedup();
            let demand = build_control_elimination_predecessors_demand(&table, &origins)
                .unwrap_or_else(|error| panic!("demand solver failed for case {case}: {error}"));
            let mut required = vec![false; nstates];
            let mut queue = VecDeque::from(origins.clone());
            while let Some(state) = queue.pop_front() {
                if std::mem::replace(&mut required[state as usize], true) {
                    continue;
                }
                queue.extend(expected[state as usize].iter().copied());
            }
            for state in 0..nstates {
                if required[state] || !demand[state].is_empty() {
                    assert_eq!(
                        demand[state], expected[state],
                        "materialized demand row differs for case {case}, state {state}, origins {origins:?}",
                    );
                }
            }
        }
    }

    fn guarded_effect(pop: u32, guard_count: u32) -> GuardedStackShift {
        GuardedStackShift {
            guards: (0..guard_count)
                .map(|guard_pop| StackShiftGuard {
                    pop: guard_pop,
                    states: vec![0],
                })
                .collect(),
            pop: pop.max(guard_count.saturating_sub(1)),
            pushes: vec![1],
        }
    }

    fn table_with_stack_shifts(
        shifts: Vec<StackShift>,
        goto_rows: &[(u32, &[(NonterminalID, (u32, bool))])],
    ) -> GLRTable {
        let num_states = 8;
        let mut action = vec![ActionRow::default(); num_states];
        action[0].insert(0, Action::StackShifts(shifts));

        let mut goto = vec![GotoRow::default(); num_states];
        for &(state, row) in goto_rows {
            for &(nt, target) in row {
                goto[state as usize].insert(nt, target);
            }
        }

        GLRTable {
            action,
            goto,
            num_states: num_states as u32,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        }
    }

    fn stack_shifts_at_start(table: &GLRTable) -> Vec<StackShift> {
        match table.action(0, 0).expect("expected action at state 0 terminal 0") {
            Action::StackShifts(shifts) => shifts.clone(),
            action => panic!("expected stack shifts, got {action:?}"),
        }
    }

    #[test]
    fn contextual_state_sharing_compacts_ambiguous_top_with_distinct_stack_contexts() {
        // Two reachable parser paths have different caller states (1 and 2)
        // but structurally corresponding tops (3 and 4). Sharing 3~4 should
        // turn the GSS frontier from two top LR IDs into one while retaining
        // the caller distinction one frame below as an exact guard.
        let mut action = vec![ActionRow::default(); 6];
        action[0].insert(10, Action::Shift(1, false));
        action[0].insert(11, Action::Shift(2, false));
        action[1].insert(20, Action::Shift(3, false));
        action[2].insert(21, Action::Shift(4, false));
        action[3].insert(0, Action::Shift(5, true));
        action[4].insert(0, Action::Shift(5, true));
        action[5].insert(1, Action::Shift(5, true));
        let mut table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 6],
            num_states: 6,
            num_terminals: 22,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::ExactSimulation,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();
        let baseline = table.clone();
        let mapping = table.share_context_distinguishable_states_exact(&[vec![3, 4]]);

        assert_eq!(mapping[3], mapping[4]);
        assert!(table.num_states < baseline.num_states);

        let acc = TerminalsDisallowed::new();
        let baseline_before = ParserGSS::from_stacks(&[
            (vec![0, 1, 3], acc.clone()),
            (vec![0, 2, 4], acc.clone()),
        ]);
        let shared_before = ParserGSS::from_stacks(&[
            (
                vec![mapping[0], mapping[1], mapping[3]],
                acc.clone(),
            ),
            (vec![mapping[0], mapping[2], mapping[4]], acc),
        ]);
        assert_eq!(baseline_before.peek_values().len(), 2);
        assert_eq!(shared_before.peek_values().len(), 1);

        let baseline_after = advance_stacks(&baseline, &baseline_before, 0);
        let shared_after = advance_stacks(&table, &shared_before, 0);
        let mut expected = baseline_after
            .to_stacks(32)
            .unwrap()
            .into_iter()
            .map(|(stack, acc)| {
                (
                    stack
                        .into_iter()
                        .map(|state| mapping[state as usize])
                        .collect::<Vec<_>>(),
                    acc,
                )
            })
            .collect::<Vec<_>>();
        let mut actual = shared_after.to_stacks(32).unwrap();
        expected.sort_by(|left, right| left.0.cmp(&right.0));
        actual.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(actual, expected);
    }

    #[test]
    fn state_local_stack_effect_exhaustion_does_not_abort_parent_budget() {
        let parent = default_unit_inline_budget();
        let child = parent.child_with_stack_effect_visit_limit(2);

        assert!(child.record_stack_effect_visit());
        assert!(child.record_stack_effect_visit());
        assert!(!child.record_stack_effect_visit());
        assert_eq!(
            child.abort_code.load(std::sync::atomic::Ordering::Relaxed),
            ABORT_STACK_EFFECT_VISITS,
        );

        parent.fold_in_isolated_state(&child);

        assert_eq!(parent.stack_effect_visits(), 3);
        assert!(!parent.is_aborted());
    }

    #[test]
    fn state_local_work_still_counts_towards_global_stack_effect_limit() {
        let mut parent = default_unit_inline_budget();
        parent.max_stack_effect_visits = 4;

        let first = parent.child_with_stack_effect_visit_limit(2);
        assert!(first.record_stack_effect_visit());
        assert!(first.record_stack_effect_visit());
        assert!(!first.record_stack_effect_visit());
        parent.fold_in_isolated_state(&first);
        assert!(!parent.is_aborted());

        let second = parent.child_with_stack_effect_visit_limit(2);
        assert!(second.record_stack_effect_visit());
        assert!(second.record_stack_effect_visit());
        parent.fold_in_isolated_state(&second);

        assert!(parent.is_aborted());
        assert_eq!(parent.report().reason, Some("stack_effect_visits"));
    }

    #[test]
    fn core_merged_stack_effect_lowering_accepts_only_unguarded_effects() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set(MAX_GUARDED_STACK_EFFECTS_ENV, "1000000");
        let mut table = table_with_stack_shifts(Vec::new(), &[]);
        table.construction = GlrTableConstruction::ExperimentalCoreMerged;

        let unguarded = vec![GuardedStackShift {
            guards: Vec::new(),
            pop: 2,
            pushes: vec![2],
        }];
        assert!(matches!(
            stack_effect_action(&table, unguarded),
            Some(Action::StackShifts(shifts)) if shifts.len() == 1
        ));

        assert!(stack_effect_action(&table, vec![guarded_effect(1, 1)]).is_none());
    }

    #[test]
    fn legacy_stack_effect_lowering_still_accepts_guarded_effects() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set(MAX_GUARDED_STACK_EFFECTS_ENV, "1000000");
        let table = table_with_stack_shifts(Vec::new(), &[]);

        assert!(matches!(
            stack_effect_action(&table, vec![guarded_effect(1, 1)]),
            Some(Action::GuardedStackShifts(shifts)) if shifts.len() == 1
        ));
    }

    #[test]
    fn row_fingerprint_is_independent_of_sparse_row_insertion_order() {
        let mut action_left = ActionRow::default();
        action_left.insert(3, Action::Reduce(7, 1));
        action_left.insert(1, Action::Shift(4, false));
        action_left.insert(9, Action::Accept);
        let mut action_right = ActionRow::default();
        action_right.insert(9, Action::Accept);
        action_right.insert(1, Action::Shift(4, false));
        action_right.insert(3, Action::Reduce(7, 1));

        let mut goto_left = GotoRow::default();
        goto_left.insert(5, (2, false));
        goto_left.insert(1, (6, true));
        let mut goto_right = GotoRow::default();
        goto_right.insert(1, (6, true));
        goto_right.insert(5, (2, false));

        assert!(rows_equal(
            &action_left,
            &goto_left,
            None,
            &action_right,
            &goto_right,
            None,
        ));
        assert_eq!(
            row_fingerprint(&action_left, &goto_left, None),
            row_fingerprint(&action_right, &goto_right, None),
        );
    }

    #[test]
    fn incremental_row_merge_matches_full_quotient_on_small_generated_tables() {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        fn action(seed: &mut u64, nstates: u32) -> Action {
            match next(seed) % 5 {
                0 => Action::Shift((next(seed) % nstates as u64) as u32, next(seed) & 1 != 0),
                1 => Action::Reduce((next(seed) % 3) as u32, (next(seed) % 3) as u32),
                2 => Action::Split {
                    shift: (next(seed) & 1 != 0)
                        .then(|| ((next(seed) % nstates as u64) as u32, next(seed) & 1 != 0)),
                    reduces: vec![((next(seed) % 3) as u32, (next(seed) % 3) as u32)],
                    accept: next(seed) & 1 != 0,
                },
                3 => Action::StackShifts(vec![
                    StackShift {
                        pop: (next(seed) % 3) as u32,
                        pushes: vec![(next(seed) % nstates as u64) as u32],
                    },
                    StackShift {
                        pop: (next(seed) % 3) as u32,
                        pushes: vec![(next(seed) % nstates as u64) as u32],
                    },
                ]),
                _ => Action::Accept,
            }
        }

        let mut seed = 0xA3B1_C2D3_E4F5_6789u64;
        for _case in 0..128 {
            let nstates = 6usize;
            let mut action_rows = vec![ActionRow::default(); nstates];
            let mut goto_rows = vec![GotoRow::default(); nstates];
            for state in 0..nstates {
                for terminal in 0..3 {
                    if !next(&mut seed).is_multiple_of(4) {
                        action_rows[state].insert(terminal, action(&mut seed, nstates as u32));
                    }
                }
                for nonterminal in 0..3 {
                    if next(&mut seed) & 1 != 0 {
                        goto_rows[state].insert(
                            nonterminal,
                            ((next(&mut seed) % nstates as u64) as u32, next(&mut seed) & 1 != 0),
                        );
                    }
                }
            }

            let mut base = GLRTable {
                action: action_rows,
                goto: goto_rows,
                num_states: nstates as u32,
                num_terminals: 3,
                num_rules: 0,
                rules: Vec::new(),
                nonterminal_display_names: Vec::new(),
                construction: GlrTableConstruction::LegacyRowBisim,
                admission_policy: AdmissionPolicy::RowPresenceExact,
                advance: Vec::new(),
                unconditional_advance: Vec::new(),
                forwarded_shifts: FxHashSet::default(),
                control_terminals: Default::default(),
                skip_terminals: Default::default(),
                guarded_shift_index: Vec::new(),
                direct_regular_wide_frontiers: Vec::new(),
            };
            base.merge_identical_rows();
            if base.num_states == 0 {
                continue;
            }

            let dirty = (next(&mut seed) % base.num_states as u64) as u32;
            let terminal = (next(&mut seed) % 3) as u32;
            base.action[dirty as usize].insert(terminal, action(&mut seed, base.num_states));

            let mut expected = base.clone();
            expected.merge_identical_rows();
            let mut incremental = base;
            incremental.merge_identical_rows_from_dirty(&[dirty]);

            assert_eq!(incremental.num_states, expected.num_states);
            assert_eq!(incremental.action, expected.action);
            assert_eq!(incremental.goto, expected.goto);
            assert_eq!(incremental.advance, expected.advance);
            assert_eq!(incremental.forwarded_shifts, expected.forwarded_shifts);
        }
    }

    #[test]
    fn incremental_row_merge_matches_full_quotient_for_noncanonical_stack_shifts() {
        let mut action = vec![ActionRow::default(); 4];
        action[0].insert(
            0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 2,
                    pushes: vec![2],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![3],
                },
            ]),
        );
        // Initially distinct from state 0, so the starting table is already
        // fully exact-minimized before the local mutation below.
        action[1].insert(
            0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![3],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![3],
                },
            ]),
        );
        action[2].insert(0, Action::Shift(2, false));
        action[3].insert(0, Action::Accept);

        let table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 4],
            num_states: 4,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };

        let mut base = table.clone();
        base.merge_identical_rows();
        assert_eq!(base.num_states, 4);

        // Stack-effect equality remaps and normalizes alternatives. The two
        // rows are exact equals under the identity map, while their direct
        // Action hashes differ because their alternative order differs.
        base.action[1].insert(
            0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![3],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![2],
                },
            ]),
        );
        let identity = (0..base.num_states).collect::<Vec<_>>();
        assert!(rows_equal_after_remap(
            &base.action[0],
            &base.goto[0],
            None,
            &base.action[1],
            &base.goto[1],
            None,
            &identity,
        ));
        assert_ne!(
            row_fingerprint(&base.action[0], &base.goto[0], None),
            row_fingerprint(&base.action[1], &base.goto[1], None),
        );

        let mut expected = base.clone();
        expected.merge_identical_rows();
        assert_eq!(expected.num_states, 3);

        let mut incremental = base;
        assert!(incremental.merge_identical_rows_from_dirty(&[1]));
        assert_eq!(incremental.num_states, expected.num_states);
        assert_eq!(incremental.action, expected.action);
        assert_eq!(incremental.goto, expected.goto);
        assert_eq!(incremental.advance, expected.advance);
        assert_eq!(incremental.forwarded_shifts, expected.forwarded_shifts);
    }

    #[test]
    fn unit_reduction_inlining_budget_abort_keeps_original_table() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvVarGuard::set(UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS_ENV, "1");

        let mut action = vec![ActionRow::default(); 5];
        action[2].insert(
            0,
            Action::Split {
                shift: Some((4, false)),
                reduces: vec![(10, 1)],
                accept: false,
            },
        );
        action[3].insert(0, Action::Shift(4, false));

        let mut goto = vec![GotoRow::default(); 5];
        goto[1].insert(10, (3, true));

        let mut table = GLRTable {
            action,
            goto,
            num_states: 5,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let original_action = format!("{:?}", table.action);
        let original_goto = format!("{:?}", table.goto);
        let original_num_states = table.num_states;

        let report = table.collapse_sr_unit_reductions_with_compatible_gotos();

        assert!(report.aborted);
        assert_eq!(report.reason, Some("stack_effect_visits"));
        assert_eq!(table.num_states, original_num_states);
        assert_eq!(format!("{:?}", table.action), original_action);
        assert_eq!(format!("{:?}", table.goto), original_goto);
    }

    #[test]
    fn canonicalizes_stack_shift_predecessor_to_goto_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            }]
        );
    }

    #[test]
    fn leaves_protected_stack_shift_terminal_unchanged() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );
        let mut protected = BitSet::new(table.num_terminals as usize);
        protected.set(0);

        table.canonicalize_stack_shift_predecessors_except(&protected);

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn leaves_stack_shift_predecessors_unchanged_when_canonicalization_is_disabled() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors_with_enabled(
            false,
            &BitSet::new(table.num_terminals as usize),
        );

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn does_not_canonicalize_stack_shift_predecessors_when_shared_goto_target_differs() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (22, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn does_not_canonicalize_empty_goto_row_to_nonempty_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[(1, &[(10, (20, true))])],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn canonicalizes_buried_middle_stack_shift_predecessor_to_goto_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![StackShift {
                pop: 1,
                pushes: vec![9, 1, 3, 4],
            }]
        );
    }

    #[test]
    fn does_not_canonicalize_top_pushed_state_even_when_goto_rows_are_compatible() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 1],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 2],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 1],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 2],
                },
            ]
        );
    }

    #[test]
    fn reduce_frame_allows_origin_dependent_multiple_goto_targets() {
        let mut table = table_with_stack_shifts(Vec::new(), &[
            (1, &[(10, (3, false))]),
            (2, &[(10, (4, false))]),
        ]);
        table.num_states = 6;
        table.action.resize(6, ActionRow::default());
        table.goto.resize(6, GotoRow::default());

        let mut predecessors = vec![PredecessorSet::new(); 6];
        predecessors[5] = smallvec![1, 2];
        let budget = default_unit_inline_budget();

        let result = apply_reduce_to_frame(
            &table,
            &predecessors,
            5,
            StackEffectFrame {
                pop: 0,
                pushes: Vec::new(),
                guards: Vec::new(),
            },
            10,
            1,
            &mut FxHashMap::default(),
            &budget,
        );

        let Some(ReduceFrameResult::Frames { frames, origin_dependent }) = result else {
            panic!("expected frames");
        };
        assert!(origin_dependent);
        assert_eq!(
            frames,
            vec![
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![3],
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![1],
                    }],
                },
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![4],
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![2],
                    }],
                },
            ]
        );
    }

    #[test]
    fn reduce_frame_allows_origin_dependent_single_goto_target() {
        let mut table = table_with_stack_shifts(Vec::new(), &[
            (1, &[(10, (3, false))]),
            (2, &[(10, (3, false))]),
        ]);
        table.num_states = 6;
        table.action.resize(6, ActionRow::default());
        table.goto.resize(6, GotoRow::default());

        let mut predecessors = vec![PredecessorSet::new(); 6];
        predecessors[5] = smallvec![1, 2];
        let budget = default_unit_inline_budget();

        let result = apply_reduce_to_frame(
            &table,
            &predecessors,
            5,
            StackEffectFrame {
                pop: 0,
                pushes: Vec::new(),
                guards: Vec::new(),
            },
            10,
            1,
            &mut FxHashMap::default(),
            &budget,
        );

        let Some(ReduceFrameResult::Frames { frames, origin_dependent }) = result else {
            panic!("expected frames");
        };
        assert!(origin_dependent);
        assert_eq!(
            frames,
            vec![
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![3],
                    guards: Vec::new(),
                }
            ]
        );
    }

    #[test]
    fn action_target_remap_preserves_variant_and_normalizes_only_published_alternatives() {
        let mapping = vec![3, 0, 2, 1, 0, 5];
        let shifts = Action::StackShifts(vec![
            StackShift { pop: 1, pushes: vec![0, 3] },
            StackShift { pop: 1, pushes: vec![0, 1] },
            StackShift { pop: 1, pushes: vec![0, 4] },
        ]);
        let partition_shifts = Action::StackShifts(vec![
            StackShift { pop: 1, pushes: vec![3, 1] },
            StackShift { pop: 1, pushes: vec![3, 0] },
            StackShift { pop: 1, pushes: vec![3, 0] },
        ]);
        assert_eq!(remap_action_to_partition(&shifts, &mapping), partition_shifts);
        let cases = [
            (Action::Shift(1, false), Action::Shift(0, false)),
            (Action::Reduce(7, 2), Action::Reduce(7, 2)),
            (
                Action::Split {
                    shift: Some((3, true)),
                    reduces: vec![(4, 1), (5, 2)],
                    accept: false,
                },
                Action::Split {
                    shift: Some((1, true)),
                    reduces: vec![(4, 1), (5, 2)],
                    accept: false,
                },
            ),
            (
                shifts,
                Action::StackShifts(vec![
                    StackShift { pop: 1, pushes: vec![3, 0] },
                    StackShift { pop: 1, pushes: vec![3, 1] },
                ]),
            ),
            (
                Action::GuardedStackShifts(vec![GuardedStackShift {
                    guards: vec![StackShiftGuard { pop: 1, states: vec![0, 3, 1, 4] }],
                    pop: 2,
                    pushes: vec![1, 4],
                }]),
                Action::GuardedStackShifts(vec![GuardedStackShift {
                    guards: vec![StackShiftGuard { pop: 1, states: vec![0, 1, 3] }],
                    pop: 2,
                    pushes: vec![0, 0],
                }]),
            ),
            (
                Action::ReplaceShifts(vec![0, 1, 4].into()),
                Action::ReplaceShifts(vec![0, 3].into()),
            ),
            (Action::Accept, Action::Accept),
            (Action::Skip, Action::Skip),
        ];
        for (mut action, expected) in cases {
            assert_eq!(remap_action_targets(&action, &mapping), expected);
            remap_action_targets_in_place(&mut action, &mapping);
            assert_eq!(action, expected);
        }
    }

    #[test]
    fn trivial_origin_dependent_one_push_shortcut_matches_generic_refusal() {
        let mut action = vec![ActionRow::default(); 6];
        action[3].insert(0, Action::Shift(5, true));
        action[4].insert(0, Action::Shift(5, true));
        let mut goto = vec![GotoRow::default(); 6];
        goto[1].insert(10, (3, false));
        goto[2].insert(10, (4, false));
        let table = GLRTable {
            action,
            goto,
            num_states: 6,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 6];
        predecessors[0] = smallvec![1, 2];
        let reduce = Action::Reduce(10, 1);
        let mut shortcut_reads = Vec::new();
        assert!(is_trivial_origin_dependent_one_push(
            &table,
            &predecessors,
            0,
            0,
            &reduce,
            &mut shortcut_reads,
        ));
        assert_eq!(shortcut_reads, vec![((0, 0), (3, 0)), ((0, 0), (4, 0))]);

        let budget = default_unit_inline_budget();
        let generic = try_inline_action_to_stack_shifts_detailed(
            &table,
            &predecessors,
            0,
            0,
            &reduce,
            &mut FxHashMap::default(),
            &mut Vec::new(),
            &budget,
        );
        assert!(generic.action.is_none());
        assert!(matches!(
            generic.outcome,
            StackInlineOutcomeKind::OriginDependentOnePush
        ));
    }

    #[test]
    fn trivial_origin_dependent_one_push_shortcut_rejects_nonreplace_continuation() {
        let mut action = vec![ActionRow::default(); 4];
        action[3].insert(0, Action::Shift(3, false));
        let mut goto = vec![GotoRow::default(); 4];
        goto[1].insert(10, (3, false));
        let table = GLRTable {
            action,
            goto,
            num_states: 4,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 4];
        predecessors[0].push(1);
        assert!(!is_trivial_origin_dependent_one_push(
            &table,
            &predecessors,
            0,
            0,
            &Action::Reduce(10, 1),
            &mut Vec::new(),
        ));
    }

    #[test]
    fn inline_action_to_stack_shifts_keeps_multishift_replacement_reduce_chain() {
        let mut action = vec![ActionRow::default(); 5];
        action[2].insert(
            0,
            Action::Split {
                shift: Some((4, false)),
                reduces: vec![(10, 1)],
                accept: false,
            },
        );
        action[3].insert(0, Action::Shift(4, false));

        let mut goto = vec![GotoRow::default(); 5];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 5,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 5];
        predecessors[2].push(1);
        let budget = default_unit_inline_budget();

        let action = table.action(2, 0).expect("expected split action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut Vec::new(),
            &budget,
        );

        let Some(Action::StackShifts(shifts)) = result else {
            panic!("expected multi-stack-shift action, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                StackShift {
                    pop: 0,
                    pushes: vec![4],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![3, 4],
                },
            ]
        );
    }

    #[test]
    fn inline_action_to_stack_shifts_handles_replace_shift_and_replace_goto() {
        let mut action = vec![ActionRow::default(); 6];
        action[2].insert(
            0,
            Action::Split {
                shift: Some((4, true)),
                reduces: vec![(10, 1)],
                accept: false,
            },
        );
        action[3].insert(0, Action::Shift(5, true));

        let mut goto = vec![GotoRow::default(); 6];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 6,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 6];
        predecessors[2].push(1);
        let budget = default_unit_inline_budget();

        let action = table.action(2, 0).expect("expected split action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut Vec::new(),
            &budget,
        );

        let Some(Action::StackShifts(shifts)) = result else {
            panic!("expected replacement stack shifts, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![4],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![5],
                },
            ]
        );
    }

    #[test]
    fn inline_action_to_stack_shifts_guards_divergent_replace_gotos_by_predecessor() {
        let mut action = vec![ActionRow::default(); 9];
        action[2].insert(0, Action::Reduce(10, 1));
        action[3].insert(0, Action::Shift(7, false));
        action[4].insert(0, Action::Shift(8, false));

        let mut goto = vec![GotoRow::default(); 9];
        goto[1].insert(10, (3, true));
        goto[6].insert(10, (4, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 9,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 9];
        predecessors[2].extend_from_slice(&[1, 6]);
        let budget = default_unit_inline_budget();

        let action = table.action(2, 0).expect("expected reduce action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut Vec::new(),
            &budget,
        );

        let Some(Action::GuardedStackShifts(shifts)) = result else {
            panic!("expected guarded replacement stack shifts, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![1],
                    }],
                    pop: 2,
                    pushes: vec![3, 7],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![6],
                    }],
                    pop: 2,
                    pushes: vec![4, 8],
                },
            ]
        );
    }

    #[test]
    fn unit_reduce_destination_without_predecessors_is_not_inlineable() {
        let table = GLRTable {
            action: vec![ActionRow::default(); 2],
            goto: vec![GotoRow::default(); 2],
            num_states: 2,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::ExperimentalCoreMerged,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let predecessors = vec![PredecessorSet::new(); 2];

        assert_eq!(unit_reduce_destination(&table, &predecessors, 1, 10), None);
    }

    #[test]
    fn compatible_goto_unit_destination_still_refuses_replace_goto() {
        let action = vec![ActionRow::default(); 4];
        let mut goto = vec![GotoRow::default(); 4];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 4,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        let mut predecessors = vec![PredecessorSet::new(); 4];
        predecessors[2].push(1);

        assert_eq!(unit_reduce_destination(&table, &predecessors, 2, 10), None);
    }

    #[test]
    fn suffix_quotient_collapses_same_pop_stack_shift_fanout() {
        let token0 = 0;
        let token1 = 1;
        let mut action = vec![ActionRow::default(); 8];
        action[0].insert(
            token0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 2],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![3, 4],
                },
            ]),
        );
        action[2].insert(
            token1,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![5],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![6],
                },
            ]),
        );
        action[4].insert(
            token1,
            Action::StackShifts(vec![StackShift {
                pop: 2,
                pushes: vec![7],
            }]),
        );

        let mut table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 8],
            num_states: 8,
            num_terminals: 2,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();

        table.quotient_recognizer_stack_suffixes();

        assert!(matches!(table.action(0, token0), Some(Action::Shift(_, true))));
        assert!(
            table.ambiguous_actions().is_empty(),
            "{:#?}",
            table.ambiguous_actions()
        );
    }


    #[test]
    fn suffix_quotient_builds_synthetic_rows_through_reductions() {
        let produce = 0;
        let consume = 1;
        let mut action = vec![ActionRow::default(); 9];
        action[0].insert(
            produce,
            Action::StackShifts(vec![
                StackShift { pop: 0, pushes: vec![1] },
                StackShift { pop: 0, pushes: vec![2] },
            ]),
        );
        action[1].insert(consume, Action::Reduce(10, 1));
        action[2].insert(consume, Action::Reduce(11, 1));
        action[3].insert(consume, Action::Shift(7, false));
        action[4].insert(consume, Action::Shift(8, false));

        let mut goto = vec![GotoRow::default(); 9];
        goto[0].insert(10, (3, false));
        goto[0].insert(11, (4, false));

        let mut table = GLRTable {
            action,
            goto,
            num_states: 9,
            num_terminals: 2,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();

        table.quotient_recognizer_stack_suffixes();

        let synthetic = match table.action(0, produce) {
            Some(Action::StackShifts(producer_shifts)) => {
                assert_eq!(producer_shifts.len(), 1);
                assert_eq!(producer_shifts[0].pop, 0);
                assert_eq!(producer_shifts[0].pushes.len(), 1);
                producer_shifts[0].pushes[0]
            }
            Some(Action::Shift(target, replace)) => {
                assert!(!replace);
                *target
            }
            other => panic!("expected producer to be rewritten to one target action, got {other:?}"),
        };
        let Some(Action::GuardedStackShifts(shifts)) = table.action(synthetic, consume) else {
            panic!("expected synthetic row to compile reductions into guarded stack effects");
        };
        assert!(!shifts.is_empty());
        assert!(shifts.iter().all(|shift| shift.pop == 1));
        assert!(shifts.iter().all(|shift| shift.guards.len() == 1));
        assert!(shifts.iter().all(|shift| shift.guards[0].pop == 1));
        assert!(shifts.iter().all(|shift| shift.guards[0].states == vec![0]));
    }

    #[test]
    fn suffix_quotient_preserves_guarded_stack_shift_guards() {
        let token = 0;
        let guard = StackShiftGuard {
            pop: 1,
            states: vec![9],
        };
        let mut action = vec![ActionRow::default(); 12];
        action[0].insert(
            token,
            Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![guard.clone()],
                    pop: 1,
                    pushes: vec![1, 2],
                },
                GuardedStackShift {
                    guards: vec![guard.clone()],
                    pop: 1,
                    pushes: vec![3, 4],
                },
            ]),
        );

        let mut table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 12],
            num_states: 12,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();

        table.quotient_recognizer_stack_suffixes();

        let Some(Action::GuardedStackShifts(shifts)) = table.action(0, token) else {
            panic!("expected one guarded stack-shift action");
        };
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0].guards.len(), 1);
        assert_eq!(shifts[0].guards[0].pop, guard.pop);
        assert!(!shifts[0].guards[0].states.is_empty());
        assert_eq!(shifts[0].pop, 1);
        assert_eq!(shifts[0].pushes.len(), 1);
        assert!(
            table.ambiguous_actions().is_empty(),
            "{:#?}",
            table.ambiguous_actions()
        );
    }

    #[test]
    fn suffix_quotient_rolls_back_when_nested_construction_hits_depth_bound() {
        let outer_suffixes = vec![vec![10, 1], vec![10, 2]];

        let mut table = GLRTable {
            action: vec![ActionRow::default(); 11],
            goto: vec![GotoRow::default(); 11],
            num_states: 11,
            num_terminals: 0,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.goto[1].insert(0, (3, false));
        table.goto[2].insert(0, (4, false));
        table.rebuild_advance_rows_from_actions();

        let original_num_states = table.num_states;
        let original_action = table.action.clone();
        let original_goto = table.goto.clone();
        let original_advance = table.advance.clone();

        let mut quotient = SuffixQuotient {
            suffix_to_state: FxHashMap::default(),
            failed_suffixes: FxHashSet::default(),
            max_states: 4096,
            max_alts: 8,
            max_width: 8,
            max_depth: 1,
            build_depth: 0,
            created_states: 0,
        };

        assert_eq!(
            quotient.ensure_suffix_state(&mut table, outer_suffixes.clone()),
            Err(())
        );
        assert_eq!(table.num_states, original_num_states);
        assert_eq!(table.action, original_action);
        assert_eq!(table.goto, original_goto);
        assert_eq!(table.advance, original_advance);
        assert_eq!(quotient.created_states, 0);
        assert_eq!(quotient.build_depth, 0);
        assert!(quotient.failed_suffixes.contains(&outer_suffixes));
        assert!(quotient
            .suffix_to_state
            .values()
            .all(|&state| state < original_num_states));
    }

    #[test]
    fn suffix_quotient_rolls_back_nested_created_states_on_outer_failure() {
        let outer_suffixes = vec![vec![10, 1], vec![10, 2]];

        let mut table = GLRTable {
            action: vec![ActionRow::default(); 11],
            goto: vec![GotoRow::default(); 11],
            num_states: 11,
            num_terminals: 0,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            construction: GlrTableConstruction::LegacyRowBisim,
            admission_policy: AdmissionPolicy::RowPresenceExact,
            advance: Vec::new(),
            unconditional_advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            control_terminals: Default::default(),
            skip_terminals: Default::default(),
            guarded_shift_index: Vec::new(),
            direct_regular_wide_frontiers: Vec::new(),
        };
        table.goto[1].insert(0, (3, false));
        table.goto[2].insert(0, (4, false));
        table.goto[1].insert(1, (5, false));
        table.goto[2].insert(1, (6, false));
        table.rebuild_advance_rows_from_actions();

        let original_num_states = table.num_states;
        let original_action_len = table.action.len();
        let original_goto_len = table.goto.len();
        let original_advance_len = table.advance.len();

        let mut quotient = SuffixQuotient {
            suffix_to_state: FxHashMap::default(),
            failed_suffixes: FxHashSet::default(),
            max_states: 2,
            max_alts: 8,
            max_width: 8,
            max_depth: 128,
            build_depth: 0,
            created_states: 0,
        };

        assert_eq!(
            quotient.ensure_suffix_state(&mut table, outer_suffixes.clone()),
            Err(())
        );
        assert_eq!(table.num_states, original_num_states);
        assert_eq!(table.action.len(), original_action_len);
        assert_eq!(table.goto.len(), original_goto_len);
        assert_eq!(table.advance.len(), original_advance_len);
        assert_eq!(quotient.created_states, 0);
        assert!(quotient.failed_suffixes.contains(&outer_suffixes));
        assert!(
            quotient
                .suffix_to_state
                .values()
                .all(|&state| state < original_num_states)
        );
    }
}

fn try_inline_unit_reductions_for_cell_inner(
    table: &mut GLRTable,
    predecessors: &[PredecessorSet],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<StateSubset>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    visiting: &mut BTreeSet<(u32, TerminalID)>,
    owner: UnitInlineCell,
    dependencies: &mut UnitInlineDependencies,
    budget: &UnitInlineBudget,
) -> Result<Option<CellUpdate>, ()> {
    if !budget.record_stack_effect_visit() {
        return Err(());
    }
    if !visiting.insert((state, tid)) {
        return Ok(None);
    }

    let mut pending = PendingAction::default();
    let mut reduces: Vec<(NonterminalID, u32)> = Vec::new();

    match action {
        Action::Shift(target, replace) => pending.push_shift(*target, *replace),
        Action::StackShifts(_) | Action::ReplaceShifts(_) => return Ok(None),
        Action::GuardedStackShifts(_) | Action::Skip => return Ok(None),
        Action::Reduce(nt, len) => reduces.push((*nt, *len)),
        Action::Split {
            shift,
            reduces: action_reduces,
            accept,
        } => {
            if let Some((target, replace)) = shift {
                pending.push_shift(*target, *replace);
            }
            reduces.extend(action_reduces.iter().copied());
            if *accept {
                pending.push_accept();
            }
        }
        Action::Accept => pending.push_accept(),
    }

    let mut changed = false;
    for (lhs, pop_len) in reduces {
        if pop_len != 1 {
            pending.push_reduce(lhs, pop_len);
            continue;
        }

        let Some(reduce_dst) = unit_reduce_destination(table, predecessors, state, lhs) else {
            pending.push_reduce(lhs, pop_len);
            continue;
        };

        dependencies.record_read(owner, (reduce_dst, tid));
        match table.action[reduce_dst as usize].get(&tid).cloned() {
            None => {
                pending.push_reduce(lhs, pop_len);
            }
            Some(inline_action) => {
                let resolved_inline = match try_inline_unit_reductions_for_cell_inner(
                    table,
                    predecessors,
                    reduce_dst,
                    tid,
                    &inline_action,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    visiting,
                    owner,
                    dependencies,
                    budget,
                )? {
                    Some(CellUpdate::Set(action)) => Some(action),
                    Some(CellUpdate::Remove) => None,
                    None => Some(inline_action),
                };

                let Some(resolved_inline) = resolved_inline else {
                    changed = true;
                    continue;
                };

                // The runtime interprets a raw replace-flagged Shift as
                // APPEND when its (state,terminal) is forwarded. Copying the
                // same action into a different cell with different forwarding
                // metadata changes the stack effect. Retain the reduction
                // instead of silently changing append to replace (or reverse).
                let raw_replace_shift = match &resolved_inline {
                    Action::Shift(_, replace) => *replace,
                    Action::Split { shift: Some((_, replace)), .. } => *replace,
                    _ => false,
                };
                // Copying an APPEND across a unit reduction can leave the
                // wrong state underneath the pushed state. Original:
                // [P,R] --reduce+goto D--> [P,D] --append E--> [P,D,E].
                // Copied action: [P,R] --append E--> [P,R,E]. Goto-row
                // equality of D and R does not prove these stacks equivalent.
                // Only copy a plain REPLACE shift without forwarded semantics.
                let forwarded_source = table.forwarded_shifts.contains(&(reduce_dst, tid));
                let forwarded_destination = table.forwarded_shifts.contains(&(state, tid));
                if !raw_replace_shift || forwarded_source || forwarded_destination {
                    // Only Shift or shift-bearing Split cases are governed by
                    // this conservative extra rule. Other action kinds retain
                    // their preexisting reduction/inlining behavior.
                    let has_shift = matches!(
                        &resolved_inline,
                        Action::Shift(..) | Action::Split { shift: Some(_), .. }
                    );
                    if has_shift {
                        pending.push_reduce(lhs, pop_len);
                        continue;
                    }
                }

                merge_action_into_pending(
                    &mut pending,
                    &resolved_inline,
                    table,
                    constituent_sets,
                    subset_to_state,
                    failed_subsets,
                    budget,
                )?;
                changed = true;
            }
        }
    }

    let result = if !changed {
        Ok(None)
    } else {
        Ok(match pending.maybe_finish() {
            Some(action) => Some(CellUpdate::Set(action)),
            None => Some(CellUpdate::Remove),
        })
    };
    visiting.remove(&(state, tid));
    result
}

fn remap_action_row_targets_in_place(action_row: &mut ActionRow, mapping: &[u32]) {
    action_row.for_each_value_mut(|action| remap_action_targets_in_place(action, mapping));
}

/// Remap action targets without cloning reductions, shift lists, or guarded
/// alternatives. This is used while compacting a quotient; it preserves the
/// exact normalization performed by `remap_action_targets` for actions whose
/// target IDs can collide after a merge.
fn remap_action_targets_in_place(action: &mut Action, mapping: &[u32]) {
    remap_action_state_references(action, mapping, true);
}

/// Partition refinement deliberately preserves the order and multiplicity of
/// stack-shift alternatives. Published quotient rows normalize them instead.
fn remap_action_state_references(
    action: &mut Action,
    mapping: &[u32],
    normalize_stack_alternatives: bool,
) {
    match action {
        Action::Shift(target, _) => {
            *target = mapping[*target as usize];
        }
        Action::ReplaceShifts(targets) => {
            let mut remapped = targets
                .iter()
                .map(|&target| mapping[target as usize])
                .collect::<Vec<_>>();
            remapped.sort_unstable();
            remapped.dedup();
            *targets = remapped.into();
        }
        Action::StackShifts(shifts) => {
            for shift in shifts.iter_mut() {
                for target in &mut shift.pushes {
                    *target = mapping[*target as usize];
                }
            }
            if normalize_stack_alternatives {
                normalize_stack_shifts(shifts);
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts.iter_mut() {
                for guard in shift.guards.iter_mut() {
                    for state in &mut guard.states {
                        *state = mapping[*state as usize];
                    }
                    guard.states.sort_unstable();
                    guard.states.dedup();
                }
                for target in &mut shift.pushes {
                    *target = mapping[*target as usize];
                }
            }
        }
        Action::Reduce(_, _) | Action::Accept | Action::Skip => {}
        Action::Split { shift, .. } => {
            if let Some((target, _)) = shift {
                *target = mapping[*target as usize];
            }
        }
    }
}

fn remap_goto_row_targets_in_place(goto_row: &mut GotoRow, mapping: &[u32]) {
    goto_row.for_each_value_mut(|(target, _)| {
        *target = mapping[*target as usize];
    });
}

fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {
    let mut action = action.clone();
    remap_action_targets_in_place(&mut action, mapping);
    action
}

#[cfg(test)]
fn remap_action_to_partition(action: &Action, partition: &[u32]) -> Action {
    let mut action = action.clone();
    remap_action_state_references(&mut action, partition, false);
    action
}

fn core_classes(core_keys: &[Vec<Item>]) -> Vec<u32> {
    // Use the first state in each LR(0)-core class as its label. These labels
    // are sparse, but they let refinement touch only non-singleton classes;
    // an overwhelmingly LR(0)-unique grammar such as BFCL catalog-512 should
    // not build and hash row signatures for every singleton state.
    let mut class_of = Vec::with_capacity(core_keys.len());
    let mut key_to_representative: FxHashMap<Vec<Item>, u32> = FxHashMap::default();
    for (state, key) in core_keys.iter().enumerate() {
        let representative = *key_to_representative
            .entry(key.clone())
            .or_insert(state as u32);
        class_of.push(representative);
    }
    class_of
}

fn nontrivial_partition_members(partition: &[u32]) -> FxHashMap<u32, Vec<usize>> {
    let mut members = FxHashMap::<u32, Vec<usize>>::default();
    for (state, &representative) in partition.iter().enumerate() {
        if representative == state as u32 {
            continue;
        }
        members
            .entry(representative)
            .or_insert_with(|| vec![representative as usize])
            .push(state);
    }
    members
}

fn refine_same_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {
    let profile = std::env::var("GLRMASK_PROFILE_GLR_CORE_MERGE_DETAIL").is_ok_and(|value| value == "1");
    let mut partition = core_classes(core_keys);
    let observations = RowObservations::new(table);
    let mut iteration = 0usize;
    loop {
        let members = nontrivial_partition_members(&partition);
        if members.is_empty() {
            return partition;
        }
        iteration += 1;
        let started = profile.then(std::time::Instant::now);
        let mut active = members.values().flat_map(|members| members.iter().copied()).collect::<Vec<_>>();
        active.sort_unstable();
        let fingerprint = |&state: &usize| {
            (state as u32, unordered_entry_hash(&(
                partition[state], observations.fingerprint(state as u32, &partition, false),
            )))
        };
        let fingerprints = if active.len() >= 256 {
            active.par_iter().map(fingerprint).collect::<Vec<_>>()
        } else {
            active.iter().map(fingerprint).collect::<Vec<_>>()
        };
        let mut buckets = FxHashMap::<u64, SmallVec<[u32; 1]>>::default();
        let mut next = partition.clone();
        for (state, hash) in fingerprints {
            next[state as usize] = state;
            buckets.entry(hash).or_default().push(state);
        }
        let mut pairs = Vec::new();
        for bucket in buckets.into_values() {
            exact_row_bucket(bucket, |left, right| {
                partition[left as usize] == partition[right as usize]
                    && observations.equal(left, right, &partition, false)
            }, &mut pairs);
        }
        for (winner, loser) in pairs {
            next[loser as usize] = winner;
        }
        let changed = next != partition;
        if let Some(started) = started {
            eprintln!("[glrmask/profile][same_core_refine] iteration={iteration} states={} active_states={} changed={changed} elapsed_ms={:.3}",
                partition.len(), active.len(), started.elapsed().as_secs_f64() * 1000.0);
        }
        if !changed {
            return partition;
        }
        partition = next;
    }
}

pub(super) fn merge_same_core_lr1_states(mut table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {
    let partition = refine_same_core_partition(&table, core_keys);
    let n = table.num_states as usize;
    let keep = partition.iter().enumerate().map(|(state, &rep)| state as u32 == rep).collect::<Vec<_>>();
    if keep.iter().all(|&keep| keep) {
        return table;
    }
    let mut compact = vec![u32::MAX; n];
    let mut next = 0u32;
    for (state, &keep) in keep.iter().enumerate() {
        if keep {
            compact[state] = next;
            next += 1;
        }
    }
    let mapping = partition.iter().map(|&rep| compact[rep as usize]).collect::<Vec<_>>();
    table.publish_state_mapping(&mapping, &keep);
    table
}
