use super::*;
use crate::ds::bitset::BitSet;
use rustc_hash::FxHasher;
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
const MAX_GUARDED_STACK_EFFECTS_ENV: &str = "GLRMASK_MAX_GUARDED_STACK_EFFECTS";
const UNIT_INLINE_WORK_MAX_WALL_MS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_MS";
const UNIT_INLINE_WORK_MAX_ITERATIONS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_ITERATIONS";
const UNIT_INLINE_WORK_MAX_CELLS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_CELLS";
const UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES_ENV: &str =
    "GLRMASK_UNIT_REDUCTION_INLINE_MAX_SYNTHETIC_STATES";
const UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS_ENV: &str = "GLRMASK_UNIT_REDUCTION_INLINE_MAX_STACK_VISITS";
const DEFAULT_UNIT_INLINE_WORK_MAX_WALL_MS: u128 = 5_000;
const DEFAULT_UNIT_INLINE_WORK_MAX_ITERATIONS: usize = 64;
const DEFAULT_UNIT_INLINE_WORK_MAX_CELLS: usize = 2_000_000;
const DEFAULT_UNIT_INLINE_WORK_MAX_SYNTHETIC_STATES: usize = 4_096;
const DEFAULT_UNIT_INLINE_WORK_MAX_STACK_EFFECT_VISITS: usize = 100_000;

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
}

struct UnitInlineBudget {
    started_at: std::time::Instant,
    max_ms: u128,
    max_iterations: usize,
    max_cells: usize,
    max_synthetic_states: usize,
    max_stack_effect_visits: usize,
    iterations: usize,
    cells: usize,
    synthetic_states: usize,
    stack_effect_visits: usize,
    abort_reason: Option<&'static str>,
}

impl UnitInlineBudget {
    fn from_env() -> Self {
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
            iterations: 0,
            cells: 0,
            synthetic_states: 0,
            stack_effect_visits: 0,
            abort_reason: None,
        }
    }

    fn report(&self) -> UnitReductionInliningReport {
        UnitReductionInliningReport {
            aborted: self.abort_reason.is_some(),
            reason: self.abort_reason,
            iterations: self.iterations,
            cells: self.cells,
            synthetic_states: self.synthetic_states,
            stack_effect_visits: self.stack_effect_visits,
            elapsed_ms: self.started_at.elapsed().as_secs_f64() * 1000.0,
        }
    }

    fn abort(&mut self, reason: &'static str) {
        if self.abort_reason.is_none() {
            self.abort_reason = Some(reason);
        }
    }

    fn check_elapsed(&mut self) -> bool {
        if self.abort_reason.is_some() {
            return false;
        }
        if self.started_at.elapsed().as_millis() > self.max_ms {
            self.abort("elapsed_ms");
            return false;
        }
        true
    }

    fn record_iteration(&mut self) -> bool {
        self.iterations += 1;
        if self.iterations > self.max_iterations {
            self.abort("iterations");
            return false;
        }
        self.check_elapsed()
    }

    fn record_cell(&mut self) -> bool {
        self.cells += 1;
        if self.cells > self.max_cells {
            self.abort("cells");
            return false;
        }
        self.check_elapsed()
    }

    fn record_synthetic_state(&mut self) -> bool {
        self.synthetic_states += 1;
        if self.synthetic_states > self.max_synthetic_states {
            self.abort("synthetic_states");
            return false;
        }
        self.check_elapsed()
    }

    fn record_stack_effect_visit(&mut self) -> bool {
        self.stack_effect_visits += 1;
        if self.stack_effect_visits > self.max_stack_effect_visits {
            self.abort("stack_effect_visits");
            return false;
        }
        if self.stack_effect_visits & 0x3ff == 0 {
            return self.check_elapsed();
        }
        self.abort_reason.is_none()
    }
}

impl GLRTable {
    pub(super) fn canonicalize_stack_shift_predecessors(&mut self) {
        self.canonicalize_stack_shift_predecessors_with_enabled(
            stack_shift_predecessor_canonicalization_enabled(),
        );
    }

    fn canonicalize_stack_shift_predecessors_with_enabled(&mut self, enabled: bool) {
        if !enabled {
            return;
        }

        for state in 0..self.num_states as usize {
            let terminals: Vec<TerminalID> = self.action[state].keys().collect();
            for terminal in terminals {
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

    /// Merge states with identical (action, goto) rows.
    /// Iterates until no more merges are possible, since remapping targets
    /// can reveal new equivalences.
    pub(super) fn merge_identical_rows(&mut self) {
        let profile_detail = std::env::var("GLRMASK_PROFILE_GLR_ROW_MERGE_DETAIL")
            .map(|value| value == "1")
            .unwrap_or(false);
        let mut iteration = 0usize;
        loop {
            iteration += 1;
            let states_before = self.num_states;
            let scan_started_at = profile_detail.then(std::time::Instant::now);
            // Almost every row fingerprint is unique. Keep the first
            // representative inline and allocate a collision list only when a
            // fingerprint actually contains distinct rows. This preserves the
            // old first-representative search order exactly.
            let mut sig_to_first_rep: FxHashMap<u64, u32> = FxHashMap::default();
            let mut collision_reps: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
            let mut remap: Vec<u32> = (0..self.num_states).collect();
            let mut changed = false;
            let mut fingerprint_collisions = 0usize;
            let mut equality_checks = 0usize;
            let mut matched_rows = 0usize;

            let has_advance_rows = self.advance.len() == self.num_states as usize;
            for state in 0..self.num_states as usize {
                let advance_row = has_advance_rows.then(|| &self.advance[state]);
                let fingerprint = row_fingerprint(&self.action[state], &self.goto[state], advance_row);
                let Some(&first_rep) = sig_to_first_rep.get(&fingerprint) else {
                    sig_to_first_rep.insert(fingerprint, state as u32);
                    continue;
                };
                fingerprint_collisions += 1;
                let row_matches = |rep: u32| {
                    rows_equal(
                        &self.action[state],
                        &self.goto[state],
                        advance_row,
                        &self.action[rep as usize],
                        &self.goto[rep as usize],
                        has_advance_rows.then(|| &self.advance[rep as usize]),
                    )
                };
                equality_checks += 1;
                if row_matches(first_rep) {
                    remap[state] = first_rep;
                    changed = true;
                    matched_rows += 1;
                    continue;
                }

                let reps = collision_reps.entry(fingerprint).or_default();
                let mut matching_rep = None;
                for &rep in reps.iter() {
                    equality_checks += 1;
                    if row_matches(rep) {
                        matching_rep = Some(rep);
                        break;
                    }
                }
                if let Some(rep) = matching_rep {
                    remap[state] = rep;
                    changed = true;
                    matched_rows += 1;
                } else {
                    reps.push(state as u32);
                }
            }

            if !changed {
                if let Some(scan_started_at) = scan_started_at {
                    eprintln!(
                        "[glrmask/profile][row_merge] iteration={} states_before={} states_after={} changed=false fingerprints={} fingerprint_collisions={} equality_checks={} matched_rows={} scan_ms={:.3} remap_ms=0.000",
                        iteration,
                        states_before,
                        states_before,
                        sig_to_first_rep.len(),
                        fingerprint_collisions,
                        equality_checks,
                        matched_rows,
                        scan_started_at.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                break;
            }

            let remap_started_at = profile_detail.then(std::time::Instant::now);
            // Build old_to_new: compose remap (merge) with sequential renumbering
            let mut new_id = 0u32;
            let mut rep_to_new: FxHashMap<u32, u32> = FxHashMap::default();
            let mut kept: Vec<u32> = Vec::new();
            for state in 0..self.num_states as usize {
                if remap[state] == state as u32 {
                    rep_to_new.insert(state as u32, new_id);
                    kept.push(state as u32);
                    new_id += 1;
                }
            }
            let mapping: Vec<u32> = (0..self.num_states as usize)
                .map(|s| rep_to_new[&remap[s]])
                .collect();

            // Move surviving rows forward instead of cloning every action,
            // goto, and advance row on every refinement round.  The remap is
            // still applied to exactly the same representatives in the same
            // state order.
            let old_action = std::mem::take(&mut self.action);
            let old_goto = std::mem::take(&mut self.goto);
            let mut old_advance = has_advance_rows.then(|| {
                std::mem::take(&mut self.advance)
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>()
            });
            let mut new_action = Vec::with_capacity(kept.len());
            let mut new_goto = Vec::with_capacity(kept.len());
            let mut new_advance = Vec::with_capacity(kept.len());
            for (state, (mut action_row, mut goto_row)) in
                old_action.into_iter().zip(old_goto).enumerate()
            {
                if remap[state] != state as u32 {
                    continue;
                }
                remap_action_row_targets_in_place(&mut action_row, &mapping);
                remap_goto_row_targets_in_place(&mut goto_row, &mapping);
                new_action.push(action_row);
                new_goto.push(goto_row);
                if let Some(old_advance) = &mut old_advance {
                    new_advance.push(
                        old_advance[state]
                            .take()
                            .expect("kept state has an advance row"),
                    );
                }
            }

            self.action = new_action;
            self.goto = new_goto;
            if has_advance_rows {
                self.advance = new_advance;
            }
            self.forwarded_shifts = self.forwarded_shifts
                .iter()
                .map(|&(state, terminal)| (mapping[state as usize], terminal))
                .collect();
            self.num_states = kept.len() as u32;
            if let (Some(scan_started_at), Some(remap_started_at)) =
                (scan_started_at, remap_started_at)
            {
                eprintln!(
                    "[glrmask/profile][row_merge] iteration={} states_before={} states_after={} changed=true fingerprints={} fingerprint_collisions={} equality_checks={} matched_rows={} scan_ms={:.3} remap_ms={:.3}",
                    iteration,
                    states_before,
                    self.num_states,
                    sig_to_first_rep.len(),
                    fingerprint_collisions,
                    equality_checks,
                    matched_rows,
                    remap_started_at.duration_since(scan_started_at).as_secs_f64() * 1000.0,
                    remap_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }
    }


    pub(super) fn prune_unreachable_states(&mut self) {
        if self.num_states == 0 {
            return;
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
            return;
        }

        let mut mapping = vec![0u32; self.num_states as usize];
        let mut kept = Vec::new();
        for (state, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                mapping[state] = kept.len() as u32;
                kept.push(state as u32);
            }
        }

        self.action = kept
            .iter()
            .map(|&state| {
                self.action[state as usize]
                    .iter()
                    .map(|(terminal, action)| (terminal, remap_action_targets(action, &mapping)))
                    .collect()
            })
            .collect();
        self.goto = kept
            .iter()
            .map(|&state| {
                self.goto[state as usize]
                    .iter()
                    .map(|(&nonterminal, &(target, replace))| {
                        (nonterminal, (mapping[target as usize], replace))
                    })
                    .collect()
            })
            .collect();
        if self.advance.len() == reachable.len() {
            self.advance = kept
                .iter()
                .map(|&state| self.advance[state as usize].clone())
                .collect();
        }
        self.forwarded_shifts = self.forwarded_shifts
            .iter()
            .filter_map(|&(state, terminal)| {
                reachable[state as usize].then_some((mapping[state as usize], terminal))
            })
            .collect();
        self.num_states = kept.len() as u32;
    }

    /// Collapse unit reductions by inlining their destination actions.
    ///
    /// When inlining produces multiple shift destinations, create a synthetic
    /// merged state whose row is the union of its constituents' rows. This
    /// keeps the parser representation unchanged: every action cell still has
    /// at most one shift slot, but that shift target may be a merged state.
    pub(super) fn collapse_sr_unit_reductions_with_compatible_gotos(
        &mut self,
    ) -> UnitReductionInliningReport {
        // Keep the original only as an abort rollback. Moving it out of
        // `self` avoids cloning the complete table twice on the usual
        // successful path; `work` is the sole mutable copy.
        let original = std::mem::replace(
            self,
            Self {
                action: Vec::new(),
                goto: Vec::new(),
                num_states: 0,
                num_terminals: 0,
                num_rules: 0,
                rules: Vec::new(),
                nonterminal_display_names: Vec::new(),
                construction: GlrTableConstruction::default(),
                admission_policy: AdmissionPolicy::default(),
                advance: Vec::new(),
                forwarded_shifts: FxHashSet::default(),
                guarded_shift_index: Vec::new(),
            },
        );
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let clone_started_at = profile_enabled.then(std::time::Instant::now);
        let mut work = original.clone();
        let clone_ms = clone_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        let inner_started_at = profile_enabled.then(std::time::Instant::now);
        let mut budget = UnitInlineBudget::from_env();
        work.collapse_sr_unit_reductions_with_compatible_gotos_inner(&mut budget);
        let report = budget.report();
        let inner_ms = inner_started_at
            .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0);
        if report.aborted {
            *self = original;
        } else {
            *self = work;
        }
        if let (Some(clone_ms), Some(inner_ms)) = (clone_ms, inner_ms) {
            eprintln!(
                "[glrmask/profile][unit_reduction_inlining] outcome={} reason={} iterations={} cells={} synthetic_states={} stack_effect_visits={} clone_ms={:.3} inner_ms={:.3} elapsed_ms={:.3} max_ms={} max_iterations={} max_cells={} max_synthetic_states={} max_stack_effect_visits={}",
                if report.aborted { "aborted" } else { "committed" },
                report.reason.unwrap_or("none"),
                report.iterations,
                report.cells,
                report.synthetic_states,
                report.stack_effect_visits,
                clone_ms,
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
        budget: &mut UnitInlineBudget,
    ) {
        let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some();
        let original_num_states = self.num_states;
        let mut constituent_sets: Vec<BTreeSet<u32>> = (0..self.num_states)
            .map(|state| BTreeSet::from([state]))
            .collect();
        let mut subset_to_state: FxHashMap<Vec<u32>, u32> = (0..self.num_states)
            .map(|state| (vec![state], state))
            .collect();
        let mut failed_subsets: FxHashSet<Vec<u32>> = FxHashSet::default();
        let mut dirty_original_states: BTreeSet<u32> = BTreeSet::new();

        loop {
            if !budget.record_iteration() {
                break;
            }
            let iteration = budget.iterations;
            let cells_before = budget.cells;
            let visits_before = budget.stack_effect_visits;
            let refresh_started_at = profile_enabled.then(std::time::Instant::now);
            if !dirty_original_states.is_empty() {
                refresh_merged_states_depending_on(
                    self,
                    original_num_states,
                    &mut constituent_sets,
                    &mut subset_to_state,
                    &mut failed_subsets,
                    &dirty_original_states,
                    budget,
                );
                if budget.abort_reason.is_some() {
                    break;
                }
                dirty_original_states.clear();
            }

            let refresh_ms = refresh_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let predecessors_started_at = profile_enabled.then(std::time::Instant::now);
            let predecessors = build_runtime_state_predecessors(self, original_num_states, &constituent_sets);
            let predecessors_ms = predecessors_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            // The scan below computes updates against a stable snapshot of the
            // current original rows for this iteration. Delayed synthetic
            // states may be appended, but existing rows are not mutated until
            // after the scan, so stack-effect reductions use a shared depth
            // cache while their rows remain stable.
            let mut states_at_depth_cache: FxHashMap<(u32, u32), Option<BTreeSet<u32>>> =
                FxHashMap::default();
            let nstates = original_num_states as usize;
            let mut pending_updates: Vec<(usize, TerminalID, CellUpdate)> = Vec::new();
            let scan_started_at = profile_enabled.then(std::time::Instant::now);

            for state in 0..nstates {
                // Inlining may synthesize states through `&mut self`, so take a
                // compact snapshot of the row before the analysis. This still
                // avoids the old keys-only allocation followed by a second hash
                // lookup for every cell.
                let actions: Vec<(TerminalID, Action)> = self.action[state]
                    .iter()
                    .map(|(tid, action)| (tid, action.clone()))
                    .collect();
                for (tid, action) in actions {
                    if !budget.record_cell() {
                        break;
                    }

                    let Ok(update) = try_inline_unit_reductions_for_cell(
                        self,
                        &predecessors,
                        state as u32,
                        tid,
                        &action,
                        &mut constituent_sets,
                        &mut states_at_depth_cache,
                        &mut subset_to_state,
                        &mut failed_subsets,
                        budget,
                    ) else {
                        continue;
                    };
                    if budget.abort_reason.is_some() {
                        break;
                    }

                    match update {
                        Some(CellUpdate::Set(new_action)) if new_action != action => {
                            pending_updates.push((state, tid, CellUpdate::Set(new_action)));
                        }
                        Some(CellUpdate::Remove) => {
                            pending_updates.push((state, tid, CellUpdate::Remove));
                        }
                        _ => {}
                    }
                }
                if budget.abort_reason.is_some() {
                    break;
                }
            }

            if pending_updates.is_empty() || budget.abort_reason.is_some() {
                if profile_enabled {
                    eprintln!(
                        "[glrmask/profile][unit_reduction_inlining_iteration] iteration={} refresh_ms={:.3} predecessors_ms={:.3} scan_ms={:.3} apply_ms=0.000 scanned_cells={} stack_effect_visits={} pending_updates={} terminal={}",
                        iteration,
                        refresh_ms,
                        predecessors_ms,
                        scan_started_at
                            .expect("profile start exists when profiling is enabled")
                            .elapsed()
                            .as_secs_f64()
                            * 1000.0,
                        budget.cells - cells_before,
                        budget.stack_effect_visits - visits_before,
                        pending_updates.len(),
                        if budget.abort_reason.is_some() { "abort" } else { "stable" },
                    );
                }
                break;
            }

            let update_count = pending_updates.len();
            let scan_ms = scan_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let apply_started_at = profile_enabled.then(std::time::Instant::now);
            for (state, tid, update) in pending_updates {
                match update {
                    CellUpdate::Set(new_action) => {
                        self.action[state].insert(tid, new_action);
                    }
                    CellUpdate::Remove => {
                        self.action[state].remove(&tid);
                        if self.advance.len() == self.num_states as usize
                            && let Some(bit) = self.terminal_bit(tid)
                        {
                            self.advance[state].clear(bit);
                        }
                    }
                }
                dirty_original_states.insert(state as u32);
            }
            if profile_enabled {
                eprintln!(
                    "[glrmask/profile][unit_reduction_inlining_iteration] iteration={} refresh_ms={:.3} predecessors_ms={:.3} scan_ms={:.3} apply_ms={:.3} scanned_cells={} stack_effect_visits={} pending_updates={} terminal=continue",
                    iteration,
                    refresh_ms,
                    predecessors_ms,
                    scan_ms,
                    apply_started_at
                        .expect("profile start exists when profiling is enabled")
                        .elapsed()
                        .as_secs_f64()
                        * 1000.0,
                    budget.cells - cells_before,
                    budget.stack_effect_visits - visits_before,
                    update_count,
                );
            }
        }
    }

    /// Merge states that are equivalent for recognition purposes.
    ///
    /// Unlike `merge_identical_rows` which requires exact action/goto match,
    /// this treats two Reduce actions as equivalent when they have the same
    /// `(lhs, rhs_len)`, since the parser only uses those two fields.
    /// It also merges goto columns for nonterminals that become equivalent.
    /// Iterates until stable.
    pub(super) fn merge_recognizer_equivalent(&mut self) {
        loop {
            let prev_states = self.num_states;

            // Step 1: With Reduce(nt, len) representation, reduces are already
            // canonicalized by (lhs, rhs_len). Just merge identical rows.

            // Step 2: Merge states with identical rows.
            self.merge_identical_rows();

            // Step 3: Merge goto columns for nonterminals whose goto vectors
            // are identical across all states (i.e., they always land in the
            // same state, or are both absent).
            let nstates = self.num_states as usize;
            let mut all_nts: BTreeSet<NonterminalID> = BTreeSet::new();
            let mut columns_by_nt: FxHashMap<NonterminalID, Vec<(u32, (u32, bool))>> =
                FxHashMap::default();
            for (state, goto_row) in self.goto.iter().enumerate() {
                for (&nt, &target) in goto_row {
                    all_nts.insert(nt);
                    columns_by_nt
                        .entry(nt)
                        .or_default()
                        .push((state as u32, target));
                }
            }

            // Build sparse goto signatures for each nonterminal and group by them.
            let mut column_to_canon: FxHashMap<Vec<(u32, (u32, bool))>, NonterminalID> =
                FxHashMap::default();
            let mut nt_remap: FxHashMap<NonterminalID, NonterminalID> = FxHashMap::default();
            for &nt in &all_nts {
                let col = columns_by_nt.remove(&nt).unwrap_or_default();
                if let Some(&canon) = column_to_canon.get(&col) {
                    nt_remap.insert(nt, canon);
                } else {
                    column_to_canon.insert(col, nt);
                }
            }

            if !nt_remap.is_empty() {
                // Rewrite goto entries: merge columns.
                for state in 0..nstates {
                    let old = std::mem::take(&mut self.goto[state]);
                    let mut new_goto = GotoRow::default();
                    for (&nt, &target) in old.iter() {
                        let canon_nt = nt_remap.get(&nt).copied().unwrap_or(nt);
                        // All remapped NTs should have the same target; just insert.
                        new_goto.insert(canon_nt, target);
                    }
                    self.goto[state] = new_goto;
                }

                // Rewrite nonterminal IDs in action entries (Reduce and Split reduces).
                for state in 0..nstates {
                    let old = std::mem::take(&mut self.action[state]);
                    let new_action: ActionRow = old
                        .iter()
                        .map(|(tid, action)| {
                            let remapped = match action {
                                Action::Reduce(nt, len) => {
                                    let canon = nt_remap.get(&nt).copied().unwrap_or(*nt);
                                    Action::Reduce(canon, *len)
                                }
                                Action::StackShifts(shifts) => Action::StackShifts(shifts.clone()),
                                Action::GuardedStackShifts(shifts) => Action::GuardedStackShifts(shifts.clone()),
                                Action::Split { shift, reduces, accept } => {
                                    let reduces = reduces
                                        .into_iter()
                                        .map(|(nt, len)| {
                                            let canon = nt_remap.get(nt).copied().unwrap_or(*nt);
                                            (canon, *len)
                                        })
                                        .collect();
                                    Action::Split { shift: *shift, reduces, accept: *accept }
                                }
                                other => other.clone(),
                            };
                            (tid, remapped)
                        })
                        .collect();
                    self.action[state] = new_action;
                }

                // Rewrite rule LHS to use canonical NTs.
                for rule in &mut self.rules {
                    if let Some(&canon) = nt_remap.get(&rule.lhs) {
                        rule.lhs = canon;
                    }
                }

                // Rewrite rule RHS nonterminals to use canonical NTs.
                for rule in &mut self.rules {
                    for sym in &mut rule.rhs {
                        if let Symbol::Nonterminal(nt) = sym {
                            if let Some(&canon) = nt_remap.get(nt) {
                                *nt = canon;
                            }
                        }
                    }
                }

                // Merge identical rows again after NT merging.
                self.merge_identical_rows();
            }

            // Step 4: Local split collapsing.
            // For each remaining Split action, check if all reduces land in the
            // same goto target from every predecessor state.  If so, the split
            // is invisible to a recognizer and we can collapse it.
            //
            // Two sub-passes:
            //  4a (original) — immediate goto-target equality from all static predecessors.
            //  4b (new)      — speculative reduce-chain convergence: simulate
            //      each alternative reduce for up to MAX_SPEC_DEPTH steps,
            //      collecting the set of (top-state) the chain reaches.
            //      If all alternatives converge to the same set, collapse.
            let nstates2 = self.num_states as usize;

            // Build predecessor map: for each state, which states can be
            // "goto_from" after a rhs_len=K pop.
            // For rhs_len=1: predecessor is any state X such that
            //   goto[X][*] == this_state  OR  shift in action[X][*] -> this_state
            let mut predecessors: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); nstates2];
            for x in 0..nstates2 {
                for (_, action) in &self.action[x] {
                    if let Some(target) = action.shift_target() {
                        predecessors[target as usize].insert(x as u32);
                    }
                }
                for (_, &(target, _)) in &self.goto[x] {
                    predecessors[target as usize].insert(x as u32);
                }
            }

            let mut collapsed_any = false;
            let mut collapses: Vec<(usize, TerminalID, (NonterminalID, u32))> = Vec::new();
            for state in 0..nstates2 {
                for (tid, action) in &self.action[state] {
                    if let Action::Split { shift, reduces, accept } = action {
                        // Only handle pure-reduce splits (no shift, no accept).
                        if shift.is_some() || *accept {
                            continue;
                        }
                        if reduces.is_empty() {
                            continue;
                        }
                        // Check: do all reduces have the same rhs_len?
                        let (_, rhs_len) = reduces[0];
                        if reduces.iter().any(|&(_, l)| l != rhs_len) {
                            continue;
                        }
                        // For rhs_len=K, find all states that are K levels
                        // up in the stack (predecessors^K).
                        let mut candidate_froms: BTreeSet<u32> = BTreeSet::new();
                        candidate_froms.insert(state as u32);
                        for _ in 0..rhs_len {
                            let mut next = BTreeSet::new();
                            for &s in &candidate_froms {
                                if let Some(preds) = predecessors.get(s as usize) {
                                    next.extend(preds);
                                }
                            }
                            candidate_froms = next;
                        }
                        if candidate_froms.is_empty() {
                            continue;
                        }
                        // Check if all reduces lead to the same goto target
                        // from every predecessor.
                        let lhss: Vec<NonterminalID> = reduces
                            .iter()
                            .map(|&(nt, _)| nt)
                            .collect();
                        let mut all_same = true;
                        'pred_loop: for &pred in &candidate_froms {
                            let first_target = self.goto[pred as usize].get(&lhss[0]).map(|&(t, _)| t);
                            for &lhs in &lhss[1..] {
                                let target = self.goto[pred as usize].get(&lhs).map(|&(t, _)| t);
                                if target != first_target {
                                    all_same = false;
                                    break 'pred_loop;
                                }
                            }
                        }
                        if all_same {
                            collapses.push((state, tid, reduces[0]));
                        }
                    }
                }
            }

            for (state, tid, reduce_info) in collapses {
                self.action[state].insert(tid, Action::Reduce(reduce_info.0, reduce_info.1));
                collapsed_any = true;
            }

            // Step 4b: Deep split collapsing via stack-relative chain following.
            //
            // For pure R/R splits not collapsed in 4a, simulate the full reduce
            // chain for each alternative.  Track predecessor depth relative to
            // the ORIGINAL split state S (not intermediate chain states).
            //
            // The stack at the split: …→ preds^K(S) →…→ S (top)
            //
            // After alternative reduce Ri (pop=rhs_len(Ri)):
            //   - Expose state at depth rhs_len(Ri) below S
            //   - goto from that state with lhs(Ri) → push T1
            //   - If T1 has another reduce on the same terminal, follow it:
            //     pop rhs_len from T1's position, which goes further below S
            //   - Continue until we reach a non-reduce action
            //
            // If all alternatives' chains converge to the same final state
            // (same goto target from preds^(total_depth) of S), collapse.
            //
            // Two sub-passes:
            //  4b-i: filter out split-state predecessors (handles circular deps)
            //  4b-ii: deep chain following for remaining unconverged splits
            let mut spec_collapses: Vec<(usize, TerminalID, (NonterminalID, u32))> = Vec::new();

            // Build set of (state, terminal) pairs that have pure R/R splits
            let pure_rr_splits: BTreeSet<(usize, TerminalID)> = {
                let mut set = BTreeSet::new();
                for s in 0..nstates2 {
                    for (t, a) in &self.action[s] {
                        if let Action::Split { shift, reduces: _, accept } = a {
                            if shift.is_none() && !*accept {
                                set.insert((s, t));
                            }
                        }
                    }
                }
                set
            };

            for state in 0..nstates2 {
                for (tid, action) in &self.action[state] {
                    let Action::Split { shift, reduces, accept } = action else { continue };
                    if shift.is_some() || *accept { continue }
                    if reduces.is_empty() { continue }

                    let (_, rhs_len) = reduces[0];
                    if reduces.iter().any(|&(_, l)| l != rhs_len) {
                        continue;
                    }
                    let reduces = reduces.clone();

                    // Compute candidate_froms (predecessors^K of the split state)
                    let mut candidate_froms: BTreeSet<u32> = BTreeSet::new();
                    candidate_froms.insert(state as u32);
                    for _ in 0..rhs_len {
                        let mut next = BTreeSet::new();
                        for &s in &candidate_froms {
                            if let Some(preds) = predecessors.get(s as usize) {
                                next.extend(preds);
                            }
                        }
                        candidate_froms = next;
                    }
                    if candidate_froms.is_empty() { continue }

                    // 4b-i: Filter out predecessors that are themselves split states
                    let filtered: BTreeSet<u32> = candidate_froms.iter()
                        .filter(|&&p| !pure_rr_splits.contains(&(p as usize, tid)))
                        .copied()
                        .collect();

                    if filtered.is_empty() {
                        spec_collapses.push((state, tid, reduces[0]));
                        continue;
                    }

                    // Simple check: do all reduces converge from filtered preds?
                    let lhss: Vec<NonterminalID> = reduces
                        .iter()
                        .map(|&(nt, _)| nt)
                        .collect();
                    let mut simple_converge = true;
                    'pred_simple: for &pred in &filtered {
                        let first_target = self.goto[pred as usize].get(&lhss[0]).map(|&(t, _)| t);
                        for &lhs in &lhss[1..] {
                            if self.goto[pred as usize].get(&lhs).map(|&(t, _)| t) != first_target {
                                simple_converge = false;
                                break 'pred_simple;
                            }
                        }
                    }
                    if simple_converge {
                        spec_collapses.push((state, tid, reduces[0]));
                        continue;
                    }


                    // 4b-ii: Deep chain following.
                    // For each alternative, simulate the reduce chain and track
                    // the total depth popped from the original split state S.
                    //
                    // Stack model: After initial reduce Ri (pop=K) from S:
                    //   base_depth = K (below S)
                    //   goto_from = preds^K(S)
                    //   push T1 = goto[goto_from][lhs(Ri)]
                    //   T1 sits at depth K-1 (one above goto_from)
                    //
                    // After follow-up reduce Rj (pop=M) from T1:
                    //   We pop M items from T1's position. T1 is at K-1.
                    //   Popping 1 removes T1 itself (back to K).
                    //   Popping M total goes to depth K + M - 1.
                    //   base_depth = K + M - 1
                    //   goto_from = preds^(K+M-1)(S)
                    //   push T2, sits at K + M - 2
                    //
                    // In general: after n reduces with pop values K1,K2,...,Kn,
                    //   base_depth = K1 + K2 + ... + Kn - (n-1)
                    //   = sum(Ki) - n + 1
                    //
                    // The chain terminates when the action at the pushed state
                    // is not a Reduce on terminal T.
                    //
                    // All alternatives converge if they reach the same
                    // (base_depth, final_lhs) and goto[preds^base_depth][lhs]
                    // agrees for all preds.
                    const MAX_CHAIN: usize = 32;

                    // Follow one alternative's chain.  Returns (base_depth, final_lhs)
                    // or None if the chain diverges or is too deep.
                    let follow = |first_nt: NonterminalID, _first_len: u32| -> Option<(usize, NonterminalID)> {
                        let mut depth = rhs_len as usize; // after initial reduce

                        // Compute goto targets from preds^depth(state) with lhs
                        let preds_at_depth = |d: usize| -> BTreeSet<u32> {
                            let mut s = BTreeSet::new();
                            s.insert(state as u32);
                            for _ in 0..d {
                                let mut next = BTreeSet::new();
                                for &st in &s {
                                    if let Some(ps) = predecessors.get(st as usize) {
                                        next.extend(ps);
                                    }
                                }
                                s = next;
                            }
                            s
                        };

                        let mut current_lhs = first_nt;
                        for _ in 0..MAX_CHAIN {
                            let preds = preds_at_depth(depth);
                            if preds.is_empty() { return None }

                            // Get goto targets
                            let mut goto_targets: BTreeSet<u32> = BTreeSet::new();
                            for &p in &preds {
                                if let Some(&(gt, _)) = self.goto[p as usize].get(&current_lhs) {
                                    goto_targets.insert(gt);
                                }
                            }
                            if goto_targets.is_empty() { return None }

                            // Check action at goto targets on terminal tid
                            let mut next_reduce: Option<(NonterminalID, u32)> = None;
                            let mut all_reduce = true;
                            for &gt in &goto_targets {
                                match self.action.get(gt as usize).and_then(|r| r.get(&tid)) {
                                    Some(Action::Reduce(nt, len)) => {
                                        let info = (*nt, *len);
                                        match next_reduce {
                                            None => next_reduce = Some(info),
                                            Some(nr) if nr == info => {}
                                            _ => { all_reduce = false; break }
                                        }
                                    }
                                    _ => {
                                        // Chain terminates
                                        return Some((depth, current_lhs));
                                    }
                                }
                            }
                            if !all_reduce { return None }

                            // Follow the next reduce
                            let (next_nt, next_len) = next_reduce.unwrap();
                            depth = depth + next_len as usize - 1;
                            current_lhs = next_nt;
                        }
                        None // Too deep
                    };

                    // Follow all alternatives
                    let mut first_result: Option<(usize, NonterminalID)> = None;
                    let mut chain_converge = true;
                    for &(nt, len) in &reduces {
                        match follow(nt, len) {
                            Some(result) => {
                                match first_result {
                                    None => first_result = Some(result),
                                    Some(prev) if prev == result => {}
                                    _ => { chain_converge = false; break }
                                }
                            }
                            None => { chain_converge = false; break }
                        }
                    }

                    if !chain_converge { continue }
                    let Some((final_depth, final_lhs)) = first_result else { continue };

                    // All alternatives converge to (final_depth, final_lhs).
                    // Check: from preds^final_depth(state), do all gotos agree?
                    let mut final_preds = BTreeSet::new();
                    final_preds.insert(state as u32);
                    for _ in 0..final_depth {
                        let mut next = BTreeSet::new();
                        for &s in &final_preds {
                            if let Some(ps) = predecessors.get(s as usize) {
                                next.extend(ps);
                            }
                        }
                        final_preds = next;
                    }

                    let mut goto_target_val: Option<Option<u32>> = None;
                    let mut targets_agree = true;
                    for &pred in &final_preds {
                        let target = self.goto[pred as usize].get(&final_lhs).map(|&(t, _)| t);
                        match goto_target_val {
                            None => goto_target_val = Some(target),
                            Some(prev) if prev == target => {}
                            _ => { targets_agree = false; break }
                        }
                    }

                    if targets_agree {
                        spec_collapses.push((state, tid, reduces[0]));
                    }
                }
            }

            for (state, tid, reduce_info) in spec_collapses {
                self.action[state].insert(tid, Action::Reduce(reduce_info.0, reduce_info.1));
                collapsed_any = true;
            }

            if collapsed_any {
                self.merge_identical_rows();
            }

            if self.num_states == prev_states {
                break;
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
        if !recognizer_suffix_quotient_enabled() {
            return;
        }

        let mut quotient = SuffixQuotient::new();
        let original_states = self.num_states as usize;
        let mut changed = false;

        for state in 0..original_states {
            let terminals: Vec<TerminalID> = self.action[state].keys().collect();
            for terminal in terminals {
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
        let built = (|| {
            let action = self.build_suffix_action_row(table, &suffixes)?;
            let goto = self.build_suffix_goto_row(table, &suffixes)?;
            Ok::<_, ()>((action, goto))
        })();

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
    for effect in effects.iter_mut() {
        for guard in &mut effect.guards {
            guard.states.sort_unstable();
            guard.states.dedup();
        }
        effect.guards.retain(|guard| !guard.states.is_empty());
        effect.guards.sort_by_key(|guard| guard.pop);
        effect.guards.dedup();
    }
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

fn unguarded_suffix_effect(
    suffix: &[u32],
    pop: u32,
    pushes: &[u32],
) -> Result<GuardedStackShift, ()> {
    suffix_effect(suffix, Vec::new(), pop, pushes)
}

fn guarded_suffix_effect(
    suffix: &[u32],
    shift: &GuardedStackShift,
) -> Result<Option<GuardedStackShift>, ()> {
    let suffix_len = suffix.len() as u32;
    let mut translated_guards = Vec::new();

    for guard in &shift.guards {
        if guard.pop < suffix_len {
            let index = suffix.len() - 1 - guard.pop as usize;
            if guard.states.binary_search(&suffix[index]).is_err() {
                return Ok(None);
            }
        } else {
            translated_guards.push(StackShiftGuard {
                pop: 1 + (guard.pop - suffix_len),
                states: guard.states.clone(),
            });
        }
    }

    let effect = suffix_effect(suffix, translated_guards, shift.pop, &shift.pushes)?;
    Ok(Some(effect))
}

fn suffix_effect(
    suffix: &[u32],
    mut guards: Vec<StackShiftGuard>,
    pop: u32,
    pushes: &[u32],
) -> Result<GuardedStackShift, ()> {
    if suffix.is_empty() {
        return Err(());
    }
    let suffix_len = suffix.len() as u32;
    let (macro_pop, macro_pushes) = if pop <= suffix_len {
        let keep = suffix.len() - pop as usize;
        let mut out = suffix[..keep].to_vec();
        out.extend_from_slice(pushes);
        (1, out)
    } else {
        (1 + (pop - suffix_len), pushes.to_vec())
    };

    guards.sort_by_key(|guard| guard.pop);
    guards.dedup();
    debug_assert!(guards.iter().all(|guard| guard.pop <= macro_pop));

    Ok(GuardedStackShift {
        guards,
        pop: macro_pop,
        pushes: macro_pushes,
    })
}

// Compile one concrete action row of a synthetic suffix state into an
// equivalent macro stack effect.
//
// Invariant.  A synthetic suffix state Q_S denotes the finite set S of concrete
// LR stack suffixes.  If the lower stack is alpha, a concrete member suffix s in
// S denotes alpha·s, while the synthetic stack denotes alpha·Q_S.  A macro
// effect produced here is correct when applying it to alpha·Q_S yields exactly
// the same lower-stack result as applying the original LR action sequence to
// alpha·s, for every alpha satisfying its guards.
//
// Shifts and existing stack effects are translated by replacing the synthetic
// state with the concrete suffix, applying the concrete pop/push operation, and
// then re-encoding the result as one macro pop plus pushes.  Reductions are the
// only subtle case: the reduce pop may expose a state inside the represented
// suffix, or it may cross below Q_S into alpha.  In the first case the exposed
// predecessor is known statically.  In the second case the predecessor is not
// known until runtime, so we enumerate table goto sources for the reduced
// nonterminal and add a guard at the exposed depth.  The guarded alternatives
// are disjoint by predecessor state, and states without the goto simply produce
// no branch, matching ordinary LR reduce failure.  After goto, we recursively
// continue on the same terminal until a consuming shift/stack-effect/accept is
// reached, exactly as the GLR interpreter does for reductions.
fn collect_suffix_effects_from_frame(
    table: &GLRTable,
    terminal: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    effects: &mut Vec<GuardedStackShift>,
    accepts: &mut usize,
) -> Result<(), ()> {
    let key = StackEffectVisitKey {
        state,
        tid: terminal,
        action_tag: stack_effect_action_tag(action),
        frame: frame.clone(),
    };
    if !visiting.insert(key.clone()) {
        return Err(());
    }

    let result = (|| {
        match action {
            Action::Shift(target, replace) => {
                let mut frame = frame;
                let effective_replace = *replace && !table.forwarded_shifts.contains(&(state, terminal));
                push_transition_to_frame(&mut frame, *target, effective_replace);
                effects.push(frame_to_guarded_shift(frame));
                Ok(())
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    let mut frame = frame.clone();
                    pop_frame(&mut frame, shift.pop);
                    frame.pushes.extend_from_slice(&shift.pushes);
                    effects.push(frame_to_guarded_shift(frame));
                }
                Ok(())
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    if let Some(frame) = compose_guarded_shift_with_frame(frame.clone(), shift).ok_or(())? {
                        effects.push(frame_to_guarded_shift(frame));
                    }
                }
                Ok(())
            }
            Action::Reduce(nt, len) => {
                for frame in reduce_suffix_frame(table, frame, *nt, *len)? {
                    let Some(&next_state) = frame.pushes.last() else {
                        continue;
                    };
                    let Some(next_action) = table.action[next_state as usize].get(&terminal) else {
                        continue;
                    };
                    collect_suffix_effects_from_frame(
                        table,
                        terminal,
                        next_state,
                        next_action,
                        frame,
                        visiting,
                        effects,
                        accepts,
                    )?;
                }
                Ok(())
            }
            Action::Split { shift, reduces, accept } => {
                if *accept {
                    *accepts += 1;
                }
                if let Some((target, replace)) = shift {
                    let shift_action = Action::Shift(*target, *replace);
                    collect_suffix_effects_from_frame(
                        table,
                        terminal,
                        state,
                        &shift_action,
                        frame.clone(),
                        visiting,
                        effects,
                        accepts,
                    )?;
                }
                for &(nt, len) in reduces {
                    let reduce_action = Action::Reduce(nt, len);
                    collect_suffix_effects_from_frame(
                        table,
                        terminal,
                        state,
                        &reduce_action,
                        frame.clone(),
                        visiting,
                        effects,
                        accepts,
                    )?;
                }
                Ok(())
            }
            Action::Accept => {
                *accepts += 1;
                Ok(())
            }
        }
    })();

    visiting.remove(&key);
    result
}

fn reduce_suffix_frame(
    table: &GLRTable,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
) -> Result<Vec<StackEffectFrame>, ()> {
    pop_frame(&mut frame, len);

    let goto_froms: Vec<(u32, Option<Vec<u32>>)> = if let Some(&state) = frame.pushes.last() {
        vec![(state, None)]
    } else {
        table
            .goto
            .iter()
            .enumerate()
            .filter(|(_, row)| row.contains_key(&nt))
            .map(|(state, _)| (state as u32, Some(vec![state as u32])))
            .collect()
    };

    if goto_froms.is_empty() {
        return Ok(Vec::new());
    }

    let guard_pop = frame.pop;
    let mut out = Vec::new();
    for (goto_from, guard_states) in goto_froms {
        let Some((target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            continue;
        };
        let mut next = frame.clone();
        if let Some(states) = guard_states
            && !add_guard_to_frame(&mut next, guard_pop, states)
        {
            continue;
        }
        push_transition_to_frame(&mut next, target, replace);
        out.push(next);
    }
    out.sort();
    out.dedup();
    Ok(out)
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

fn row_fingerprint(
    action_row: &ActionRow,
    goto_row: &GotoRow,
    advance_row: Option<&BitSet>,
) -> u64 {
    let (action_sum, action_xor) = unordered_row_hash(
        action_row.iter().map(|(terminal, action)| (terminal, action)),
        action_row.len(),
    );
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

fn push_reachable_state(state: u32, reachable: &mut [bool], stack: &mut Vec<u32>) {
    let Some(slot) = reachable.get_mut(state as usize) else {
        return;
    };
    if !*slot {
        *slot = true;
        stack.push(state);
    }
}

fn push_action_targets(action: &Action, reachable: &mut [bool], stack: &mut Vec<u32>) {
    match action {
        Action::Shift(target, _) => push_reachable_state(*target, reachable, stack),
        Action::StackShifts(shifts) => {
            for shift in shifts {
                for &state in &shift.pushes {
                    push_reachable_state(state, reachable, stack);
                }
            }
        }
        Action::GuardedStackShifts(shifts) => {
            for shift in shifts {
                for guard in &shift.guards {
                    for &state in &guard.states {
                        push_reachable_state(state, reachable, stack);
                    }
                }
                for &state in &shift.pushes {
                    push_reachable_state(state, reachable, stack);
                }
            }
        }
        Action::Reduce(_, _) | Action::Accept => {}
        Action::Split { shift, .. } => {
            if let Some((target, _)) = shift {
                push_reachable_state(*target, reachable, stack);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CellUpdate {
    Set(Action),
    Remove,
}

fn build_runtime_state_predecessors(
    table: &GLRTable,
    original_num_states: u32,
    constituent_sets: &[BTreeSet<u32>],
) -> Vec<BTreeSet<u32>> {
    let mut predecessors = vec![BTreeSet::new(); table.num_states as usize];

    for src in 0..table.num_states as usize {
        for action in table.action[src].values() {
            match action {
                Action::Shift(dst, false) => {
                    predecessors[*dst as usize].extend(constituent_sets[src].iter().copied());
                }
                Action::Split { shift: Some((dst, false)), .. } => {
                    predecessors[*dst as usize].extend(constituent_sets[src].iter().copied());
                }
                _ => {}
            }
        }
        for &(dst, replace) in table.goto[src].values() {
            if !replace {
                predecessors[dst as usize].extend(constituent_sets[src].iter().copied());
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for src in 0..original_num_states as usize {
            let src_preds = predecessors[src].clone();
            for action in table.action[src].values() {
                match action {
                    Action::Shift(dst, true) => {
                        let before = predecessors[*dst as usize].len();
                        predecessors[*dst as usize].extend(src_preds.iter().copied());
                        changed |= predecessors[*dst as usize].len() != before;
                    }
                    Action::Split { shift: Some((dst, true)), .. } => {
                        let before = predecessors[*dst as usize].len();
                        predecessors[*dst as usize].extend(src_preds.iter().copied());
                        changed |= predecessors[*dst as usize].len() != before;
                    }
                    _ => {}
                }
            }
            for &(dst, replace) in table.goto[src].values() {
                if replace {
                    let before = predecessors[dst as usize].len();
                    predecessors[dst as usize].extend(src_preds.iter().copied());
                    changed |= predecessors[dst as usize].len() != before;
                }
            }
        }
    }

    predecessors
}

fn subset_key(subset: &BTreeSet<u32>) -> Vec<u32> {
    subset.iter().copied().collect()
}

fn union_state_subsets(
    states: impl IntoIterator<Item = u32>,
    constituent_sets: &[BTreeSet<u32>],
) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for state in states {
        out.extend(constituent_sets[state as usize].iter().copied());
    }
    out
}

fn merge_shift_into_pending(
    pending: &mut PendingAction,
    target: u32,
    replace: bool,
    table: &mut GLRTable,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &mut UnitInlineBudget,
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
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &mut UnitInlineBudget,
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
        Action::StackShifts(_) => Err(()),
        Action::GuardedStackShifts(_) => Err(()),
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
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &mut UnitInlineBudget,
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
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &mut UnitInlineBudget,
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
        let mut target_subset = BTreeSet::new();
        let mut saw_target = false;

        for &state in subset {
            if let Some(&(target, is_replace)) = table.goto[state as usize].get(&nt) {
                saw_target = true;
                match replace {
                    None => replace = Some(is_replace),
                    Some(existing) if existing == is_replace => {}
                    Some(_) => return Err(()),
                }
                target_subset.extend(constituent_sets[target as usize].iter().copied());
            }
        }

        if !saw_target {
            continue;
        }

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

fn union_advance_rows(table: &GLRTable, subset: &BTreeSet<u32>) -> BitSet {
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
    subset: &BTreeSet<u32>,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &mut UnitInlineBudget,
) -> Result<u32, ()> {
    debug_assert!(!subset.is_empty());
    if subset.len() == 1 {
        return Ok(*subset.iter().next().unwrap());
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
            subset_to_state.remove(&key);
            failed_subsets.insert(key);
            table.action.pop();
            table.goto.pop();
            if had_advance_rows {
                table.advance.pop();
            }
            table.num_states -= 1;
            constituent_sets.pop();
            Err(())
        }
    }
}

fn refresh_merged_states_depending_on(
    table: &mut GLRTable,
    original_num_states: u32,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    dirty_original_states: &BTreeSet<u32>,
    budget: &mut UnitInlineBudget,
) {
    let mut state = original_num_states as usize;
    while state < table.num_states as usize {
        if !budget.check_elapsed() {
            break;
        }
        // Merged-state rows depend only on their flattened original
        // constituents, so only subsets intersecting changed original rows
        // need to be rebuilt.
        if constituent_sets[state].is_disjoint(dirty_original_states) {
            state += 1;
            continue;
        }

        let subset = constituent_sets[state].clone();
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

        if let Ok((action_row, goto_row)) = rebuilt {
            table.action[state] = action_row;
            table.goto[state] = goto_row;
            if table.advance.len() == table.num_states as usize {
                table.advance[state] = union_advance_rows(table, &subset);
            }
        }

        state += 1;
    }
}

fn unit_reduce_destination(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    lhs: NonterminalID,
) -> Option<u32> {
    let preds = &predecessors[state as usize];
    assert!(!preds.is_empty());

    let relevant_preds: Vec<u32> = preds
        .iter()
        .copied()
        .filter(|&pred| table.goto[pred as usize].contains_key(&lhs))
        .collect();
    if relevant_preds.is_empty() {
        return None;
    }

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

enum ReduceFrameResult {
    Dead,
    Frames {
        frames: Vec<StackEffectFrame>,
        origin_dependent: bool,
    },
}

fn states_at_depth<'a>(
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    depth: u32,
    cache: &'a mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
    budget: &mut UnitInlineBudget,
) -> Option<&'a BTreeSet<u32>> {
    let cache_key = (origin_state, depth);
    if !cache.contains_key(&cache_key) {
        let mut states = BTreeSet::from([origin_state]);
        for _ in 0..depth {
            if !budget.record_stack_effect_visit() {
                return None;
            }
            let mut next = BTreeSet::new();
            for state in states {
                next.extend(predecessors.get(state as usize)?.iter().copied());
            }
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

fn pop_frame(frame: &mut StackEffectFrame, pop: u32) {
    if pop as usize <= frame.pushes.len() {
        let keep = frame.pushes.len() - pop as usize;
        frame.pushes.truncate(keep);
    } else {
        frame.pop += pop - frame.pushes.len() as u32;
        frame.pushes.clear();
    }
}

fn push_transition_to_frame(frame: &mut StackEffectFrame, target: u32, replace: bool) {
    if replace {
        if let Some(top) = frame.pushes.last_mut() {
            *top = target;
        } else {
            frame.pop += 1;
            frame.pushes.push(target);
        }
    } else {
        frame.pushes.push(target);
    }
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
    }
}

fn apply_reduce_to_frame(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
    budget: &mut UnitInlineBudget,
) -> Option<ReduceFrameResult> {
    pop_frame(&mut frame, len);

    let mut origin_dependent = false;
    let direct_goto_from;
    let goto_froms = if let Some(&state) = frame.pushes.last() {
        direct_goto_from = BTreeSet::from([state]);
        &direct_goto_from
    } else {
        origin_dependent = true;
        states_at_depth(
            predecessors,
            origin_state,
            frame.pop,
            states_at_depth_cache,
            budget,
        )?
    };

    if goto_froms.len() == 1 {
        let goto_from = goto_froms
            .first()
            .copied()
            .expect("a singleton set has one element");
        let Some((target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            return Some(ReduceFrameResult::Dead);
        };
        push_transition_to_frame(&mut frame, target, replace);
        return Some(ReduceFrameResult::Frames {
            frames: vec![frame],
            origin_dependent,
        });
    }

    let guard_pop = frame.pop;
    let mut by_target: BTreeMap<(u32, bool), BTreeSet<u32>> = BTreeMap::new();
    let mut missing = 0usize;
    for &goto_from in goto_froms {
        let Some((next_target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            missing += 1;
            continue;
        };
        by_target
            .entry((next_target, replace))
            .or_default()
            .insert(goto_from);
    }

    if missing > 0 && by_target.is_empty() {
        return Some(ReduceFrameResult::Dead);
    }

    let needs_guard = missing > 0 || by_target.len() > 1;
    let mut frames = Vec::new();
    for ((target, replace), froms) in by_target {
        let mut next_frame = frame.clone();
        if needs_guard && !add_guard_to_frame(&mut next_frame, guard_pop, froms.into_iter()) {
            continue;
        }
        push_transition_to_frame(&mut next_frame, target, replace);
        frames.push(next_frame);
    }

    if frames.is_empty() {
        Some(ReduceFrameResult::Dead)
    } else {
        frames.sort();
        frames.dedup();
        Some(ReduceFrameResult::Frames {
            frames,
            origin_dependent,
        })
    }
}

fn compose_guarded_shift_with_frame(
    mut frame: StackEffectFrame,
    shift: &GuardedStackShift,
) -> Option<Option<StackEffectFrame>> {
    let pushed_len = frame.pushes.len() as u32;

    for guard in &shift.guards {
        if guard.states.is_empty() {
            return Some(None);
        }

        if guard.pop < pushed_len {
            let idx = (pushed_len - 1 - guard.pop) as usize;
            let known_state = frame.pushes[idx];
            if guard.states.binary_search(&known_state).is_err() {
                return Some(None);
            }
        } else {
            let translated_pop = frame.pop + (guard.pop - pushed_len);
            if !add_guard_to_frame(&mut frame, translated_pop, guard.states.iter().copied()) {
                return Some(None);
            }
        }
    }

    if shift.pop < shift.guards.iter().map(|guard| guard.pop).max().unwrap_or(0) {
        return None;
    }

    pop_frame(&mut frame, shift.pop);
    frame.pushes.extend_from_slice(&shift.pushes);
    Some(Some(frame))
}

fn stack_effects_for_action(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    tid: TerminalID,
    state: u32,
    action: &Action,
    frame: StackEffectFrame,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    budget: &mut UnitInlineBudget,
) -> Option<StackEffectResult> {
    if !budget.record_stack_effect_visit() {
        return None;
    }
    let key = StackEffectVisitKey {
        state,
        tid,
        action_tag: stack_effect_action_tag(action),
        frame: frame.clone(),
    };
    if !visiting.insert(key.clone()) {
        return None;
    }

    let result = (|| {
        let mut out = Vec::new();
        let mut origin_dependent = false;
        match action {
            Action::Shift(target, replace) => {
                let mut frame = frame;
                let effective_replace = *replace && !table.forwarded_shifts.contains(&(state, tid));
                push_transition_to_frame(&mut frame, *target, effective_replace);
                out.push(frame_to_guarded_shift(frame));
            }
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    let mut frame = frame.clone();
                    pop_frame(&mut frame, shift.pop);
                    frame.pushes.extend_from_slice(&shift.pushes);
                    out.push(frame_to_guarded_shift(frame));
                }
            }
            Action::GuardedStackShifts(shifts) => {
                for shift in shifts {
                    match compose_guarded_shift_with_frame(frame.clone(), shift)? {
                        None => {}
                        Some(frame) => out.push(frame_to_guarded_shift(frame)),
                    }
                }
            }
            Action::Reduce(nt, len) => {
                let frames = match apply_reduce_to_frame(
                    table,
                    predecessors,
                    origin_state,
                    frame,
                    *nt,
                    *len,
                    states_at_depth_cache,
                    budget,
                )? {
                    ReduceFrameResult::Dead => {
                        return Some(StackEffectResult {
                            effects: Vec::new(),
                            origin_dependent,
                        })
                    }
                    ReduceFrameResult::Frames {
                        frames,
                        origin_dependent: reduce_origin_dependent,
                    } => {
                        origin_dependent |= reduce_origin_dependent;
                        frames
                    }
                };
                for frame in frames {
                    let Some(&next_state) = frame.pushes.last() else {
                        continue;
                    };
                    let Some(next) = table.action[next_state as usize].get(&tid) else {
                        continue;
                    };
                    let next_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        next_state,
                        next,
                        frame,
                        states_at_depth_cache,
                        visiting,
                        budget,
                    )?;
                    origin_dependent |= next_result.origin_dependent;
                    out.extend(next_result.effects);
                }
            }
            Action::Split { shift, reduces, accept } => {
                if *accept {
                    return None;
                }
                if let Some((target, replace)) = shift {
                    let shift_action = Action::Shift(*target, *replace);
                    let shift_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        state,
                        &shift_action,
                        frame.clone(),
                        states_at_depth_cache,
                        visiting,
                        budget,
                    )?;
                    origin_dependent |= shift_result.origin_dependent;
                    out.extend(shift_result.effects);
                }
                for &(nt, len) in reduces {
                    let reduce_action = Action::Reduce(nt, len);
                    let reduce_result = stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        state,
                        &reduce_action,
                        frame.clone(),
                        states_at_depth_cache,
                        visiting,
                        budget,
                    )?;
                    origin_dependent |= reduce_result.origin_dependent;
                    out.extend(reduce_result.effects);
                }
            }
            Action::Accept => return None,
        }

        out.sort();
        out.dedup();
        Some(StackEffectResult {
            effects: out,
            origin_dependent,
        })
    })();

    visiting.remove(&key);
    result
}

fn normalize_guarded_effects(effects: &mut Vec<GuardedStackShift>) {
    for effect in effects.iter_mut() {
        for guard in effect.guards.iter_mut() {
            guard.states.sort_unstable();
            guard.states.dedup();
        }
        effect.guards.retain(|guard| !guard.states.is_empty());
        effect.guards.sort_by_key(|guard| guard.pop);
        effect.guards.dedup();
    }
    effects.retain(|effect| !effect.pushes.is_empty());
    effects.sort();
    effects.dedup();
}

fn stack_effect_action(table: &GLRTable, mut effects: Vec<GuardedStackShift>) -> Option<Action> {
    normalize_guarded_effects(&mut effects);
    if effects.is_empty() {
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

fn try_inline_action_to_stack_shifts(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
    budget: &mut UnitInlineBudget,
) -> Option<Action> {
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
        return None;
    }

    let result = stack_effects_for_action(
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
        budget,
    )?;
    let effects = result.effects;
    if result.origin_dependent
        && matches!(action, Action::Reduce(_, 1))
        && !effects.is_empty()
        && effects.iter().all(|effect| effect.pushes.len() == 1)
    {
        return None;
    }
    if effects.is_empty() {
        return None;
    }
    stack_effect_action(table, effects)
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        }
    }

    fn stack_shifts_at_start(table: &GLRTable) -> Vec<StackShift> {
        match table.action(0, 0).expect("expected action at state 0 terminal 0") {
            Action::StackShifts(shifts) => shifts.clone(),
            action => panic!("expected stack shifts, got {action:?}"),
        }
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
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

        table.canonicalize_stack_shift_predecessors_with_enabled(false);

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

        let mut predecessors = vec![BTreeSet::new(); 6];
        predecessors[5] = BTreeSet::from([1, 2]);
        let mut budget = UnitInlineBudget::from_env();

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
            &mut budget,
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

        let mut predecessors = vec![BTreeSet::new(); 6];
        predecessors[5] = BTreeSet::from([1, 2]);
        let mut budget = UnitInlineBudget::from_env();

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
            &mut budget,
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        let mut predecessors = vec![BTreeSet::new(); 5];
        predecessors[2].insert(1);
        let mut budget = UnitInlineBudget::from_env();

        let action = table.action(2, 0).expect("expected split action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut budget,
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        let mut predecessors = vec![BTreeSet::new(); 6];
        predecessors[2].insert(1);
        let mut budget = UnitInlineBudget::from_env();

        let action = table.action(2, 0).expect("expected split action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut budget,
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        let mut predecessors = vec![BTreeSet::new(); 9];
        predecessors[2].extend([1, 6]);
        let mut budget = UnitInlineBudget::from_env();

        let action = table.action(2, 0).expect("expected reduce action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut budget,
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        let mut predecessors = vec![BTreeSet::new(); 4];
        predecessors[2].insert(1);

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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
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
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
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

fn try_inline_unit_reductions_for_cell(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    states_at_depth_cache: &mut FxHashMap<(u32, u32), Option<BTreeSet<u32>>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    budget: &mut UnitInlineBudget,
) -> Result<Option<CellUpdate>, ()> {
    if let Some(action) = try_inline_action_to_stack_shifts(
        table,
        predecessors,
        state,
        tid,
        action,
        states_at_depth_cache,
        budget,
    ) {
        return Ok(Some(CellUpdate::Set(action)));
    }

    match action {
        Action::Split {
            shift: Some(_),
            accept: false,
            ..
        }
        | Action::Shift(_, _) => {}
        _ => return Ok(None),
    }

    let mut visiting = BTreeSet::new();
    try_inline_unit_reductions_for_cell_inner(
        table,
        predecessors,
        state,
        tid,
        action,
        constituent_sets,
        subset_to_state,
        failed_subsets,
        &mut visiting,
        budget,
    )
}

fn try_inline_unit_reductions_for_cell_inner(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
    visiting: &mut BTreeSet<(u32, TerminalID)>,
    budget: &mut UnitInlineBudget,
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
        Action::StackShifts(_) => return Ok(None),
        Action::GuardedStackShifts(_) => return Ok(None),
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
    action_row.for_each_value_mut(|action| {
        *action = remap_action_targets(action, mapping);
    });
}

fn remap_goto_row_targets_in_place(goto_row: &mut GotoRow, mapping: &[u32]) {
    goto_row.for_each_value_mut(|(target, _)| {
        *target = mapping[*target as usize];
    });
}

fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {
    match action {
        Action::Shift(target, replace) => Action::Shift(mapping[*target as usize], *replace),
        Action::StackShifts(shifts) => {
            let mut remapped = shifts
                .iter()
                .map(|shift| StackShift {
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| mapping[state as usize]).collect(),
                })
                .collect();
            normalize_stack_shifts(&mut remapped);
            Action::StackShifts(remapped)
        }
        Action::GuardedStackShifts(shifts) => Action::GuardedStackShifts(
            shifts
                .iter()
                .map(|shift| GuardedStackShift {
                    guards: shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states: Vec<u32> = guard
                                .states
                                .iter()
                                .map(|&state| mapping[state as usize])
                                .collect();
                            states.sort_unstable();
                            states.dedup();
                            StackShiftGuard {
                                pop: guard.pop,
                                states,
                            }
                        })
                        .collect(),
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| mapping[state as usize]).collect(),
                })
                .collect(),
        ),
        Action::Reduce(nt, len) => Action::Reduce(*nt, *len),
        Action::Split {
            shift,
            reduces,
            accept,
        } => Action::Split {
            shift: shift.map(|(target, replace)| (mapping[target as usize], replace)),
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => Action::Accept,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ActionSig {
    Shift(u32, bool),
    StackShifts(Vec<(u32, Vec<u32>)>),
    GuardedStackShifts(Vec<(Vec<(u32, Vec<u32>)>, u32, Vec<u32>)>),
    Reduce(NonterminalID, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(NonterminalID, u32)>,
        accept: bool,
    },
    Accept,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowSignature {
    core_class: u32,
    action: Vec<(TerminalID, ActionSig)>,
    goto: Vec<(NonterminalID, (u32, bool))>,
    advance: Option<BitSet>,
}

fn remap_action_to_partition(action: &Action, partition: &[u32]) -> ActionSig {
    match action {
        Action::Shift(target, replace) => ActionSig::Shift(partition[*target as usize], *replace),
        Action::StackShifts(shifts) => ActionSig::StackShifts(
            shifts
                .iter()
                .map(|shift| {
                    (
                        shift.pop,
                        shift.pushes.iter().map(|&state| partition[state as usize]).collect(),
                    )
                })
                .collect(),
        ),
        Action::GuardedStackShifts(shifts) => ActionSig::GuardedStackShifts(
            shifts
                .iter()
                .map(|shift| {
                    let guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states: Vec<u32> = guard
                                .states
                                .iter()
                                .map(|&state| partition[state as usize])
                                .collect();
                            states.sort_unstable();
                            states.dedup();
                            (guard.pop, states)
                        })
                        .collect();
                    let pushes = shift
                        .pushes
                        .iter()
                        .map(|&state| partition[state as usize])
                        .collect();
                    (guards, shift.pop, pushes)
                })
                .collect(),
        ),
        Action::Reduce(nt, len) => ActionSig::Reduce(*nt, *len),
        Action::Split {
            shift,
            reduces,
            accept,
        } => ActionSig::Split {
            shift: shift.map(|(target, replace)| (partition[target as usize], replace)),
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => ActionSig::Accept,
    }
}

fn core_classes(core_keys: &[Vec<Item>]) -> Vec<u32> {
    let mut class_of = vec![0; core_keys.len()];
    let mut key_to_class: FxHashMap<Vec<Item>, u32> = FxHashMap::default();
    let mut next = 0u32;

    for (state, key) in core_keys.iter().enumerate() {
        let class = *key_to_class.entry(key.clone()).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        class_of[state] = class;
    }

    class_of
}

fn refine_same_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {
    let profile_detail = std::env::var("GLRMASK_PROFILE_GLR_CORE_MERGE_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false);
    let core_classes_started_at = profile_detail.then(std::time::Instant::now);
    let nstates = table.num_states as usize;
    let has_advance_rows = table.advance.len() == nstates;
    let core_class_of = core_classes(core_keys);
    let core_classes_ms = core_classes_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let mut partition = core_class_of.clone();
    let mut iteration = 0usize;

    loop {
        iteration += 1;
        let iteration_started_at = profile_detail.then(std::time::Instant::now);
        let mut sig_to_part: FxHashMap<RowSignature, u32> = FxHashMap::default();
        let mut next_partition = vec![0u32; nstates];
        let mut next_id = 0u32;
        let mut action_entries = 0usize;
        let mut goto_entries = 0usize;

        for state in 0..nstates {
            let action = table.action[state]
                .iter()
                .map(|(terminal, action)| {
                    (terminal, remap_action_to_partition(action, &partition))
                })
                .collect();
            action_entries += table.action[state].len();
            let goto = table.goto[state]
                .iter()
                .map(|(&nt, &(target, replace))| (nt, (partition[target as usize], replace)))
                .collect();
            goto_entries += table.goto[state].len();
            let signature = RowSignature {
                core_class: core_class_of[state],
                action,
                goto,
                advance: has_advance_rows.then(|| table.advance[state].clone()),
            };

            let class = *sig_to_part.entry(signature).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            next_partition[state] = class;
        }

        let changed = next_partition != partition;
        if let Some(iteration_started_at) = iteration_started_at {
            eprintln!(
                "[glrmask/profile][same_core_refine] iteration={} states={} core_classes_ms={:.3} unique_partitions={} action_entries={} goto_entries={} changed={} elapsed_ms={:.3}",
                iteration,
                nstates,
                core_classes_ms,
                next_id,
                action_entries,
                goto_entries,
                changed,
                iteration_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        // Refinement only splits classes. Once every state has its own class,
        // no later pass can change the partition; because classes are assigned
        // in source-state order, this is also the identity partition.
        if next_id as usize == nstates {
            return next_partition;
        }
        if !changed {
            return partition;
        }
        partition = next_partition;
    }
}

pub(super) fn merge_same_core_lr1_states(table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {
    let partition = refine_same_core_partition(&table, core_keys);
    let nstates = table.num_states as usize;
    let has_merge = partition
        .iter()
        .enumerate()
        .any(|(state, &group)| group != state as u32);
    if !has_merge {
        return table;
    }
    let ngroups = partition.iter().copied().max().map(|x| x + 1).unwrap_or(0) as usize;

    let mut representatives = vec![u32::MAX; ngroups];
    for state in 0..nstates {
        let group = partition[state] as usize;
        if representatives[group] == u32::MAX {
            representatives[group] = state as u32;
        }
    }

    let action = representatives
        .iter()
        .map(|&rep| {
            table.action[rep as usize]
                .iter()
                .map(|(terminal, action)| (terminal, remap_action_targets(action, &partition)))
                .collect()
        })
        .collect();
    let goto = representatives
        .iter()
        .map(|&rep| {
            table.goto[rep as usize]
                .iter()
                .map(|(&nt, &(target, replace))| (nt, (partition[target as usize], replace)))
                .collect()
        })
        .collect();
    let advance = if table.advance.len() == nstates {
        representatives
            .iter()
            .map(|&rep| table.advance[rep as usize].clone())
            .collect()
    } else {
        Vec::new()
    };

    // Remap forwarded_shifts to use merged state IDs
    let forwarded_shifts: FxHashSet<(u32, TerminalID)> = table.forwarded_shifts
        .iter()
        .map(|&(state, terminal)| (partition[state as usize], terminal))
        .collect();

    GLRTable {
        action,
        goto,
        num_states: ngroups as u32,
        num_terminals: table.num_terminals,
        num_rules: table.num_rules,
        rules: table.rules,
        nonterminal_display_names: table.nonterminal_display_names,
        construction: table.construction,
        admission_policy: table.admission_policy,
        advance,
        forwarded_shifts,
        guarded_shift_index: Vec::new(),
    }
}
