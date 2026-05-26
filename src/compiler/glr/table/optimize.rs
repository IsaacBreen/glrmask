use super::*;
use crate::ds::bitset::BitSet;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

const DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION_ENV: &str =
    "GLRMASK_DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION";
const DISABLE_RECOGNITION_QUOTIENT_ENV: &str = "GLRMASK_DISABLE_RECOGNITION_QUOTIENT";
const RECOGNITION_QUOTIENT_MAX_STATES_ENV: &str = "GLRMASK_RECOGNITION_QUOTIENT_MAX_STATES";
const RECOGNITION_QUOTIENT_MAX_ITERS_ENV: &str = "GLRMASK_RECOGNITION_QUOTIENT_MAX_ITERS";

fn stack_shift_predecessor_canonicalization_enabled() -> bool {
    !env_flag_enabled(DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION_ENV, false)
}

fn recognition_quotient_enabled() -> bool {
    !env_flag_enabled(DISABLE_RECOGNITION_QUOTIENT_ENV, false)
}

fn recognition_quotient_max_states() -> usize {
    env_usize(RECOGNITION_QUOTIENT_MAX_STATES_ENV, 100_000)
}

fn recognition_quotient_max_iters() -> usize {
    env_usize(RECOGNITION_QUOTIENT_MAX_ITERS_ENV, 128)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StackEffectActionKey {
    Shift(u32, bool),
    StackShifts(Vec<StackShift>),
    GuardedStackShifts(Vec<GuardedStackShift>),
    Reduce(NonterminalID, u32),
    Split,
    Accept,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackEffectKey {
    origin_state: u32,
    state: u32,
    tid: TerminalID,
    action: StackEffectActionKey,
    frame: StackEffectFrame,
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
        loop {
            let mut sig_to_reps: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
            let mut remap: Vec<u32> = (0..self.num_states).collect();
            let mut changed = false;

            let has_advance_rows = self.advance.len() == self.num_states as usize;
            for state in 0..self.num_states as usize {
                let advance_row = has_advance_rows.then(|| &self.advance[state]);
                let fingerprint = row_fingerprint(&self.action[state], &self.goto[state], advance_row);
                let reps = sig_to_reps.entry(fingerprint).or_default();
                if let Some(&rep) = reps.iter().find(|&&rep| {
                    rows_equal(
                        &self.action[state],
                        &self.goto[state],
                        advance_row,
                        &self.action[rep as usize],
                        &self.goto[rep as usize],
                        has_advance_rows.then(|| &self.advance[rep as usize]),
                    )
                }) {
                    remap[state] = rep;
                    changed = true;
                } else {
                    reps.push(state as u32);
                }
            }

            if !changed {
                break;
            }

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

            // Extract representative rows and remap all state references
            let new_action: Vec<_> = kept
                .iter()
                .map(|&s| {
                    self.action[s as usize]
                        .iter()
                        .map(|(tid, action)| (tid, remap_action_targets(action, &mapping)))
                        .collect()
                })
                .collect();
            let new_goto: Vec<_> = kept
                .iter()
                .map(|&s| {
                    self.goto[s as usize]
                        .iter()
                        .map(|(&nt, &(target, replace))| (nt, (mapping[target as usize], replace)))
                        .collect()
                })
                .collect();

            let new_advance = has_advance_rows.then(|| {
                kept.iter()
                    .map(|&s| self.advance[s as usize].clone())
                    .collect::<Vec<_>>()
            });

            self.action = new_action;
            self.goto = new_goto;
            if let Some(new_advance) = new_advance {
                self.advance = new_advance;
            }
            self.forwarded_shifts = self.forwarded_shifts
                .iter()
                .map(|&(state, terminal)| (mapping[state as usize], terminal))
                .collect();
            self.num_states = kept.len() as u32;
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
    pub(super) fn collapse_sr_unit_reductions_with_compatible_gotos(&mut self) {
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
            if !dirty_original_states.is_empty() {
                refresh_merged_states_depending_on(
                    self,
                    original_num_states,
                    &mut constituent_sets,
                    &mut subset_to_state,
                    &mut failed_subsets,
                    &dirty_original_states,
                );
                dirty_original_states.clear();
            }

            let predecessors = build_runtime_state_predecessors(self, original_num_states, &constituent_sets);
            // The scan below computes updates against a stable snapshot of the
            // current original rows for this iteration. Delayed synthetic
            // states may be appended, but existing rows are not mutated until
            // after the scan, so stack-effect results are memoizable.
            let mut stack_effect_memo: FxHashMap<StackEffectKey, Option<StackEffectResult>> =
                FxHashMap::default();
            let nstates = original_num_states as usize;
            let mut pending_updates: Vec<(usize, TerminalID, CellUpdate)> = Vec::new();

            for state in 0..nstates {
                let tids: Vec<TerminalID> = self.action[state].keys().collect();
                for tid in tids {
                    let Some(action) = self.action[state].get(&tid).cloned() else {
                        continue;
                    };

                    let Ok(update) = try_inline_unit_reductions_for_cell(
                        self,
                        &predecessors,
                        state as u32,
                        tid,
                        &action,
                        &mut constituent_sets,
                        &mut stack_effect_memo,
                        &mut subset_to_state,
                        &mut failed_subsets,
                    ) else {
                        continue;
                    };

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
            }

            if pending_updates.is_empty() {
                break;
            }

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

    /// Quotient LR table states by recognition semantics and normalize action
    /// alternatives through the quotient.
    ///
    /// This is intentionally after unit-reduction stack-effect lowering: the
    /// visible ambiguity may be a `StackShifts`/`GuardedStackShifts` fanout
    /// rather than an explicit reduce/reduce split. The equivalence is a fixed
    /// point over table rows, where state references are observed only through
    /// their current quotient class and nonterminal identities are observed only
    /// through their goto columns. The admission/advance row is part of the
    /// state signature, so masks and token admission remain exact.
    pub(super) fn eliminate_recognition_equivalent_ambiguity(&mut self) {
        if !recognition_quotient_enabled() || self.num_states <= 1 {
            self.normalize_action_alternatives_identity();
            return;
        }

        if self.num_states as usize > recognition_quotient_max_states() {
            // Size guard: still perform local action normalization/deduplication,
            // but avoid the O(states × nonterminals) quotient refinement on very
            // large generated tables unless explicitly allowed by the caller.
            self.normalize_action_alternatives_identity();
            return;
        }

        self.normalize_action_alternatives_identity();
        let quotient = recognition_quotient(self);
        self.apply_recognition_quotient(&quotient);
        self.normalize_action_alternatives_identity();
        self.merge_identical_rows();
    }

    pub(super) fn action_cell_count(&self) -> usize {
        self.action.iter().map(ActionRow::len).sum()
    }

    fn normalize_action_alternatives_identity(&mut self) {
        let state_identity: Vec<u32> = (0..self.num_states).collect();
        let max_nt = table_nonterminal_capacity(self);
        let nt_identity: Vec<u32> = (0..max_nt as u32).collect();
        let nt_reps: Vec<NonterminalID> = (0..max_nt as NonterminalID).collect();

        for state in 0..self.num_states as usize {
            let entries: Vec<(TerminalID, Action)> = self.action[state]
                .iter()
                .filter_map(|(terminal, action)| {
                    normalize_action_to_quotient(action, &state_identity, &nt_identity, &nt_reps)
                        .map(|action| (terminal, action))
                })
                .collect();
            self.action[state] = entries.into_iter().collect();
        }
    }

    fn apply_recognition_quotient(&mut self, quotient: &RecognizerQuotient) {
        let nclasses = quotient.state_representatives.len();
        if nclasses == 0 {
            return;
        }

        let had_advance_rows = self.advance.len() == self.num_states as usize;
        let mut action = Vec::with_capacity(nclasses);
        let mut goto = Vec::with_capacity(nclasses);
        let mut advance = if had_advance_rows {
            Vec::with_capacity(nclasses)
        } else {
            Vec::new()
        };

        for &rep in &quotient.state_representatives {
            let action_row: ActionRow = self.action[rep as usize]
                .iter()
                .filter_map(|(terminal, action)| {
                    normalize_action_to_quotient(
                        action,
                        &quotient.state_class_of,
                        &quotient.nt_class_of,
                        &quotient.nt_representatives,
                    )
                    .map(|action| (terminal, action))
                })
                .collect();
            action.push(action_row);

            let mut goto_row = GotoRow::default();
            for (&nt, &(target, replace)) in self.goto[rep as usize].iter() {
                let mapped_nt = quotient.canonical_nonterminal(nt);
                let mapped_target = quotient.state_class_of[target as usize];
                match goto_row.get(&mapped_nt).copied() {
                    Some(existing) if existing == (mapped_target, replace) => {}
                    Some(_) => {
                        // The partition refinement treats such rows as distinct;
                        // reaching this arm means an out-of-range NT slipped in.
                        // Preserve the first mapping rather than inventing a new
                        // behavior.
                    }
                    None => {
                        goto_row.insert(mapped_nt, (mapped_target, replace));
                    }
                }
            }
            goto.push(goto_row);

            if had_advance_rows {
                advance.push(self.advance[rep as usize].clone());
            }
        }

        self.action = action;
        self.goto = goto;
        self.advance = advance;
        self.forwarded_shifts = self
            .forwarded_shifts
            .iter()
            .filter_map(|&(state, terminal)| {
                quotient
                    .state_class_of
                    .get(state as usize)
                    .copied()
                    .map(|state| (state, terminal))
            })
            .collect();
        self.num_states = nclasses as u32;

        for rule in &mut self.rules {
            rule.lhs = quotient.canonical_nonterminal(rule.lhs);
            for symbol in &mut rule.rhs {
                if let Symbol::Nonterminal(nt) = symbol {
                    *nt = quotient.canonical_nonterminal(*nt);
                }
            }
        }
    }
}

fn row_fingerprint(
    action_row: &ActionRow,
    goto_row: &GotoRow,
    advance_row: Option<&BitSet>,
) -> u64 {
    let mut hasher = FxHasher::default();

    let mut action = action_row.iter().collect::<Vec<_>>();
    action.sort_unstable_by_key(|(terminal, _)| *terminal);
    action.len().hash(&mut hasher);
    for (terminal, action) in action {
        terminal.hash(&mut hasher);
        action.hash(&mut hasher);
    }

    let mut goto = goto_row.iter().collect::<Vec<_>>();
    goto.sort_unstable_by_key(|(nonterminal, _)| **nonterminal);
    goto.len().hash(&mut hasher);
    for (nonterminal, target) in goto {
        nonterminal.hash(&mut hasher);
        target.hash(&mut hasher);
    }

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


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RecognizerActionSig {
    Shift(u32, bool),
    StackShifts(Vec<(u32, Vec<u32>)>),
    GuardedStackShifts(Vec<(Vec<(u32, Vec<u32>)>, u32, Vec<u32>)>),
    Reduce(u32, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(u32, u32)>,
        accept: bool,
    },
    Accept,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RecognizerStateSig {
    advance: Option<BitSet>,
    action: Vec<(TerminalID, RecognizerActionSig)>,
    goto: Vec<(u32, u32, bool)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RecognizerNtSig {
    column: Vec<(u32, u32, bool)>,
}

#[derive(Debug, Clone)]
struct RecognizerQuotient {
    state_class_of: Vec<u32>,
    state_representatives: Vec<u32>,
    nt_class_of: Vec<u32>,
    nt_representatives: Vec<NonterminalID>,
}

impl RecognizerQuotient {
    fn canonical_nonterminal(&self, nt: NonterminalID) -> NonterminalID {
        self.nt_class_of
            .get(nt as usize)
            .and_then(|&class| self.nt_representatives.get(class as usize).copied())
            .unwrap_or(nt)
    }
}

fn table_nonterminal_capacity(table: &GLRTable) -> usize {
    let mut max_nt = table.nonterminal_display_names.len();
    for rule in &table.rules {
        max_nt = max_nt.max(rule.lhs as usize + 1);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(nt) = symbol {
                max_nt = max_nt.max(*nt as usize + 1);
            }
        }
    }
    for row in &table.goto {
        for &nt in row.keys() {
            max_nt = max_nt.max(nt as usize + 1);
        }
    }
    for row in &table.action {
        for (_, action) in row.iter() {
            max_nonterminal_in_action(action, &mut max_nt);
        }
    }
    max_nt
}

fn max_nonterminal_in_action(action: &Action, max_nt: &mut usize) {
    match action {
        Action::Reduce(nt, _) => *max_nt = (*max_nt).max(*nt as usize + 1),
        Action::Split { reduces, .. } => {
            for &(nt, _) in reduces {
                *max_nt = (*max_nt).max(nt as usize + 1);
            }
        }
        _ => {}
    }
}

fn recognition_quotient(table: &GLRTable) -> RecognizerQuotient {
    let nstates = table.num_states as usize;
    let nnts = table_nonterminal_capacity(table);
    let mut state_class_of = vec![0u32; nstates];
    let mut nt_class_of = vec![0u32; nnts];
    let mut converged = false;

    for _ in 0..recognition_quotient_max_iters() {
        let (next_nt_class_of, _) = refine_recognizer_nt_partition(table, &state_class_of, nnts);
        let (next_state_class_of, _) = refine_recognizer_state_partition(
            table,
            &state_class_of,
            &next_nt_class_of,
        );

        if next_state_class_of == state_class_of && next_nt_class_of == nt_class_of {
            converged = true;
            break;
        }

        state_class_of = next_state_class_of;
        nt_class_of = next_nt_class_of;
    }

    if !converged {
        return identity_recognition_quotient(table, nnts);
    }

    // One final NT refinement with the fixed state partition gives canonical
    // representatives that correspond exactly to the table we are about to emit.
    let (nt_class_of, nt_representatives) = refine_recognizer_nt_partition(table, &state_class_of, nnts);
    let state_representatives = representatives_for_partition(&state_class_of);

    RecognizerQuotient {
        state_class_of,
        state_representatives,
        nt_class_of,
        nt_representatives,
    }
}

fn identity_recognition_quotient(table: &GLRTable, nnts: usize) -> RecognizerQuotient {
    RecognizerQuotient {
        state_class_of: (0..table.num_states).collect(),
        state_representatives: (0..table.num_states).collect(),
        nt_class_of: (0..nnts as u32).collect(),
        nt_representatives: (0..nnts as NonterminalID).collect(),
    }
}

fn refine_recognizer_nt_partition(
    table: &GLRTable,
    state_class_of: &[u32],
    nnts: usize,
) -> (Vec<u32>, Vec<NonterminalID>) {
    let mut signatures = Vec::with_capacity(nnts);
    for nt in 0..nnts as NonterminalID {
        let mut column = Vec::new();
        for (state, row) in table.goto.iter().enumerate() {
            if let Some(&(target, replace)) = row.get(&nt) {
                column.push((state as u32, state_class_of[target as usize], replace));
            }
        }
        signatures.push(RecognizerNtSig { column });
    }
    partition_with_representatives(signatures)
}

fn refine_recognizer_state_partition(
    table: &GLRTable,
    state_class_of: &[u32],
    nt_class_of: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let has_advance_rows = table.advance.len() == table.num_states as usize;
    let mut signatures = Vec::with_capacity(table.num_states as usize);

    for state in 0..table.num_states as usize {
        let mut action = table.action[state]
            .iter()
            .map(|(terminal, action)| {
                (
                    terminal,
                    recognizer_action_sig(action, state_class_of, nt_class_of),
                )
            })
            .collect::<Vec<_>>();
        action.sort_unstable_by_key(|(terminal, _)| *terminal);

        let mut goto = table.goto[state]
            .iter()
            .map(|(&nt, &(target, replace))| {
                (
                    nt_class_of.get(nt as usize).copied().unwrap_or(nt),
                    state_class_of[target as usize],
                    replace,
                )
            })
            .collect::<Vec<_>>();
        goto.sort_unstable();
        goto.dedup();

        signatures.push(RecognizerStateSig {
            advance: has_advance_rows.then(|| table.advance[state].clone()),
            action,
            goto,
        });
    }

    partition_with_representatives(signatures)
}

fn partition_with_representatives<T, R>(signatures: Vec<T>) -> (Vec<u32>, Vec<R>)
where
    T: Eq + Hash,
    R: From<u32>,
{
    let mut sig_to_class: FxHashMap<T, u32> = FxHashMap::default();
    let mut class_of = Vec::with_capacity(signatures.len());
    let mut representatives = Vec::new();

    for (idx, signature) in signatures.into_iter().enumerate() {
        let class = match sig_to_class.get(&signature).copied() {
            Some(class) => class,
            None => {
                let class = sig_to_class.len() as u32;
                sig_to_class.insert(signature, class);
                representatives.push(R::from(idx as u32));
                class
            }
        };
        class_of.push(class);
    }

    (class_of, representatives)
}

fn representatives_for_partition(partition: &[u32]) -> Vec<u32> {
    let nclasses = partition.iter().copied().max().map(|class| class + 1).unwrap_or(0) as usize;
    let mut reps = vec![u32::MAX; nclasses];
    for (state, &class) in partition.iter().enumerate() {
        let slot = &mut reps[class as usize];
        if *slot == u32::MAX {
            *slot = state as u32;
        }
    }
    reps
}

fn recognizer_action_sig(
    action: &Action,
    state_class_of: &[u32],
    nt_class_of: &[u32],
) -> RecognizerActionSig {
    match action {
        Action::Shift(target, replace) => {
            RecognizerActionSig::Shift(state_class_of[*target as usize], *replace)
        }
        Action::StackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| {
                    (
                        shift.pop,
                        shift.pushes.iter().map(|&state| state_class_of[state as usize]).collect(),
                    )
                })
                .collect::<Vec<_>>();
            shifts.sort_unstable();
            shifts.dedup();
            RecognizerActionSig::StackShifts(shifts)
        }
        Action::GuardedStackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| {
                    let mut guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states = guard
                                .states
                                .iter()
                                .map(|&state| state_class_of[state as usize])
                                .collect::<Vec<_>>();
                            states.sort_unstable();
                            states.dedup();
                            (guard.pop, states)
                        })
                        .collect::<Vec<_>>();
                    guards.sort_unstable();
                    guards.dedup();
                    let pushes = shift
                        .pushes
                        .iter()
                        .map(|&state| state_class_of[state as usize])
                        .collect::<Vec<_>>();
                    (guards, shift.pop, pushes)
                })
                .collect::<Vec<_>>();
            shifts.sort_unstable();
            shifts.dedup();
            RecognizerActionSig::GuardedStackShifts(shifts)
        }
        Action::Reduce(nt, len) => {
            RecognizerActionSig::Reduce(nt_class_of.get(*nt as usize).copied().unwrap_or(*nt), *len)
        }
        Action::Split { shift, reduces, accept } => {
            let shift = shift.map(|(target, replace)| (state_class_of[target as usize], replace));
            let mut reduces = reduces
                .iter()
                .map(|&(nt, len)| (nt_class_of.get(nt as usize).copied().unwrap_or(nt), len))
                .collect::<Vec<_>>();
            reduces.sort_unstable();
            reduces.dedup();
            RecognizerActionSig::Split { shift, reduces, accept: *accept }
        }
        Action::Accept => RecognizerActionSig::Accept,
    }
}

fn normalize_action_to_quotient(
    action: &Action,
    state_class_of: &[u32],
    nt_class_of: &[u32],
    nt_representatives: &[NonterminalID],
) -> Option<Action> {
    match action {
        Action::Shift(target, replace) => Some(Action::Shift(state_class_of[*target as usize], *replace)),
        Action::StackShifts(shifts) => {
            let shifts = shifts
                .iter()
                .map(|shift| StackShift {
                    pop: shift.pop,
                    pushes: shift
                        .pushes
                        .iter()
                        .map(|&state| state_class_of[state as usize])
                        .collect(),
                })
                .collect();
            stack_shift_action(shifts)
        }
        Action::GuardedStackShifts(shifts) => {
            let mut shifts = shifts
                .iter()
                .map(|shift| GuardedStackShift {
                    guards: shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states = guard
                                .states
                                .iter()
                                .map(|&state| state_class_of[state as usize])
                                .collect::<Vec<_>>();
                            states.sort_unstable();
                            states.dedup();
                            StackShiftGuard { pop: guard.pop, states }
                        })
                        .collect(),
                    pop: shift.pop,
                    pushes: shift
                        .pushes
                        .iter()
                        .map(|&state| state_class_of[state as usize])
                        .collect(),
                })
                .collect::<Vec<_>>();
            guarded_stack_shift_action(&mut shifts)
        }
        Action::Reduce(nt, len) => Some(Action::Reduce(
            canonical_nt(*nt, nt_class_of, nt_representatives),
            *len,
        )),
        Action::Split { shift, reduces, accept } => {
            let shift = shift.map(|(target, replace)| (state_class_of[target as usize], replace));
            let reduces = reduces
                .iter()
                .map(|&(nt, len)| (canonical_nt(nt, nt_class_of, nt_representatives), len))
                .collect();
            normalize_split_action(shift, reduces, *accept)
        }
        Action::Accept => Some(Action::Accept),
    }
}

fn canonical_nt(
    nt: NonterminalID,
    nt_class_of: &[u32],
    nt_representatives: &[NonterminalID],
) -> NonterminalID {
    nt_class_of
        .get(nt as usize)
        .and_then(|&class| nt_representatives.get(class as usize).copied())
        .unwrap_or(nt)
}

fn normalize_split_action(
    shift: Option<(u32, bool)>,
    reduces: Vec<(NonterminalID, u32)>,
    accept: bool,
) -> Option<Action> {
    PendingAction { shift, reduces, accept }.maybe_finish()
}

fn guarded_stack_shift_action(shifts: &mut Vec<GuardedStackShift>) -> Option<Action> {
    normalize_guarded_effects(shifts);
    if shifts.is_empty() {
        return None;
    }

    if shifts.iter().all(|shift| shift.guards.is_empty()) {
        let shifts = shifts
            .drain(..)
            .map(|shift| StackShift {
                pop: shift.pop,
                pushes: shift.pushes,
            })
            .collect();
        return stack_shift_action(shifts);
    }

    Some(Action::GuardedStackShifts(std::mem::take(shifts)))
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
) -> Result<ActionRow, ()> {
    let mut terminals = BTreeSet::new();
    for &state in subset {
        for tid in table.action[state as usize].keys() {
            terminals.insert(tid);
        }
    }

    let mut row = ActionRow::default();
    for tid in terminals {
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
) -> Result<GotoRow, ()> {
    let mut nts = BTreeSet::new();
    for &state in subset {
        for &nt in table.goto[state as usize].keys() {
            nts.insert(nt);
        }
    }

    let mut row = GotoRow::default();
    for nt in nts {
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
        )?;
        let goto_row = build_merged_goto_row(
            table,
            subset,
            constituent_sets,
            subset_to_state,
            failed_subsets,
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
) {
    let mut state = original_num_states as usize;
    while state < table.num_states as usize {
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
            )?;
            let goto_row = build_merged_goto_row(
                table,
                &subset,
                constituent_sets,
                subset_to_state,
                failed_subsets,
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

fn states_at_depth(
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    depth: u32,
) -> Option<BTreeSet<u32>> {
    let mut states = BTreeSet::from([origin_state]);
    for _ in 0..depth {
        let mut next = BTreeSet::new();
        for state in states {
            next.extend(predecessors.get(state as usize)?.iter().copied());
        }
        if next.is_empty() {
            return None;
        }
        states = next;
    }
    Some(states)
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

fn stack_effect_action_key(action: &Action) -> StackEffectActionKey {
    match action {
        Action::Shift(target, replace) => StackEffectActionKey::Shift(*target, *replace),
        Action::StackShifts(shifts) => StackEffectActionKey::StackShifts(shifts.clone()),
        Action::GuardedStackShifts(shifts) => {
            StackEffectActionKey::GuardedStackShifts(shifts.clone())
        }
        Action::Reduce(nt, len) => StackEffectActionKey::Reduce(*nt, *len),
        Action::Split { .. } => StackEffectActionKey::Split,
        Action::Accept => StackEffectActionKey::Accept,
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
) -> Option<ReduceFrameResult> {
    pop_frame(&mut frame, len);

    let mut origin_dependent = false;
    let goto_froms = if let Some(&state) = frame.pushes.last() {
        BTreeSet::from([state])
    } else {
        origin_dependent = true;
        states_at_depth(predecessors, origin_state, frame.pop)?
    };

    let guard_pop = frame.pop;
    let mut by_target: BTreeMap<(u32, bool), BTreeSet<u32>> = BTreeMap::new();
    let mut missing = 0usize;
    for goto_from in goto_froms {
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
    visiting: &mut FxHashSet<StackEffectVisitKey>,
    memo: &mut FxHashMap<StackEffectKey, Option<StackEffectResult>>,
) -> Option<StackEffectResult> {
    let memo_key = StackEffectKey {
        origin_state,
        state,
        tid,
        action: stack_effect_action_key(action),
        frame: frame.clone(),
    };
    if let Some(cached) = memo.get(&memo_key) {
        return cached.clone();
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
                let frames = match apply_reduce_to_frame(table, predecessors, origin_state, frame, *nt, *len)? {
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
                        visiting,
                        memo,
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
                        visiting,
                        memo,
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
                        visiting,
                        memo,
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
    memo.insert(memo_key, result.clone());
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
    Some(Action::GuardedStackShifts(effects))
}

fn try_inline_action_to_stack_shifts(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    stack_effect_memo: &mut FxHashMap<StackEffectKey, Option<StackEffectResult>>,
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

    let effects = stack_effects_for_action(
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
        &mut FxHashSet::default(),
        stack_effect_memo,
    )?
    .effects;
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
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
        }
    }

    fn stack_shifts_at_start(table: &GLRTable) -> Vec<StackShift> {
        match table.action(0, 0).expect("expected action at state 0 terminal 0") {
            Action::StackShifts(shifts) => shifts.clone(),
            action => panic!("expected stack shifts, got {action:?}"),
        }
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
    fn recognition_quotient_collapses_equivalent_stack_shift_fanout() {
        let token = 0;
        let next = 1;
        let mut table = crate::compiler::glr::table::testing::build_test_table(
            7,
            2,
            &[
                &[(
                    token,
                    Action::StackShifts(vec![
                        StackShift { pop: 1, pushes: vec![1, 3] },
                        StackShift { pop: 1, pushes: vec![2, 4] },
                    ]),
                )],
                &[],
                &[],
                &[(next, Action::Shift(5, false))],
                &[(next, Action::Shift(6, false))],
                &[(next, Action::Accept)],
                &[(next, Action::Accept)],
            ],
            &[&[], &[], &[], &[], &[], &[], &[]],
        );

        assert!(table.has_ambiguity());
        table.eliminate_recognition_equivalent_ambiguity();

        assert!(!table.has_ambiguity(), "quotient should deduplicate equivalent stack effects");
        match table.action(0, token).expect("state 0 token action") {
            Action::StackShifts(shifts) => {
                assert_eq!(shifts.len(), 1);
                assert_eq!(shifts[0].pop, 1);
                assert_eq!(shifts[0].pushes.len(), 2);
            }
            action => panic!("expected one normalized stack-shift effect, got {action:?}"),
        }
    }

    #[test]
    fn recognition_quotient_is_guard_aware() {
        let token = 0;
        let next = 1;
        let mut table = crate::compiler::glr::table::testing::build_test_table(
            7,
            2,
            &[
                &[(
                    token,
                    Action::GuardedStackShifts(vec![
                        GuardedStackShift {
                            guards: vec![StackShiftGuard { pop: 1, states: vec![1] }],
                            pop: 1,
                            pushes: vec![3],
                        },
                        GuardedStackShift {
                            guards: vec![StackShiftGuard { pop: 1, states: vec![2] }],
                            pop: 1,
                            pushes: vec![4],
                        },
                    ]),
                )],
                &[],
                &[],
                &[(next, Action::Shift(5, false))],
                &[(next, Action::Shift(6, false))],
                &[(next, Action::Accept)],
                &[(next, Action::Accept)],
            ],
            &[&[], &[], &[], &[], &[], &[], &[]],
        );

        table.eliminate_recognition_equivalent_ambiguity();

        assert!(!table.has_ambiguity(), "guarded alternatives that differ only by quotient-equivalent states should merge");
        match table.action(0, token).expect("state 0 token action") {
            Action::GuardedStackShifts(shifts) => {
                assert_eq!(shifts.len(), 1);
                assert_eq!(shifts[0].guards.len(), 1);
                assert_eq!(shifts[0].guards[0].states.len(), 1);
            }
            action => panic!("expected one normalized guarded stack-shift effect, got {action:?}"),
        }
    }

    #[test]
    fn recognition_quotient_collapses_split_reduces_with_equivalent_goto_columns() {
        let token = 0;
        let nt_a = 10;
        let nt_b = 11;
        let mut table = crate::compiler::glr::table::testing::build_test_table(
            4,
            1,
            &[
                &[(
                    token,
                    Action::Split {
                        shift: None,
                        reduces: vec![(nt_a, 1), (nt_b, 1)],
                        accept: false,
                    },
                )],
                &[],
                &[(token, Action::Accept)],
                &[(token, Action::Accept)],
            ],
            &[&[], &[(nt_a, (2, false)), (nt_b, (3, false))], &[], &[]],
        );

        assert!(table.has_ambiguity());
        table.eliminate_recognition_equivalent_ambiguity();

        assert!(!table.has_ambiguity());
        assert!(matches!(table.action(0, token), Some(Action::Reduce(_, 1))));
    }

    #[test]
    fn recognition_quotient_is_idempotent_on_normalized_tables() {
        let token = 0;
        let mut table = crate::compiler::glr::table::testing::build_test_table(
            3,
            1,
            &[
                &[(token, Action::Shift(1, false))],
                &[(token, Action::Shift(2, true))],
                &[(token, Action::Accept)],
            ],
            &[&[], &[], &[]],
        );

        table.eliminate_recognition_equivalent_ambiguity();
        let states = table.num_states;
        let actions = table.action.clone();
        let gotos = table.goto.clone();
        table.eliminate_recognition_equivalent_ambiguity();

        assert_eq!(table.num_states, states);
        assert_eq!(table.action, actions);
        assert_eq!(table.goto, gotos);
    }
  }

fn try_inline_unit_reductions_for_cell(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    stack_effect_memo: &mut FxHashMap<StackEffectKey, Option<StackEffectResult>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<Option<CellUpdate>, ()> {
    if let Some(action) = try_inline_action_to_stack_shifts(
        table,
        predecessors,
        state,
        tid,
        action,
        stack_effect_memo,
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
) -> Result<Option<CellUpdate>, ()> {
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

fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {
    match action {
        Action::Shift(target, replace) => Action::Shift(mapping[*target as usize], *replace),
        Action::StackShifts(shifts) => {
            let remapped = shifts
                .iter()
                .map(|shift| StackShift {
                    pop: shift.pop,
                    pushes: shift.pushes.iter().map(|&state| mapping[state as usize]).collect(),
                })
                .collect();
            stack_shift_action(remapped).expect("remapping a non-empty stack-shift action should stay non-empty")
        }
        Action::GuardedStackShifts(shifts) => {
            let mut remapped = shifts
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
                .collect();
            guarded_stack_shift_action(&mut remapped)
                .expect("remapping a non-empty guarded stack-shift action should stay non-empty")
        }
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
    let nstates = table.num_states as usize;
    let has_advance_rows = table.advance.len() == nstates;
    let core_class_of = core_classes(core_keys);
    let mut partition = core_class_of.clone();

    loop {
        let mut sig_to_part: FxHashMap<RowSignature, u32> = FxHashMap::default();
        let mut next_partition = vec![0u32; nstates];
        let mut next_id = 0u32;

        for state in 0..nstates {
            let action = table.action[state]
                .iter()
                .map(|(terminal, action)| {
                    (terminal, remap_action_to_partition(action, &partition))
                })
                .collect();
            let goto = table.goto[state]
                .iter()
                .map(|(&nt, &(target, replace))| (nt, (partition[target as usize], replace)))
                .collect();
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

        if next_partition == partition {
            return partition;
        }
        partition = next_partition;
    }
}

pub(super) fn merge_same_core_lr1_states(table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {
    let partition = refine_same_core_partition(&table, core_keys);
    let nstates = table.num_states as usize;
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
        advance,
        forwarded_shifts,
    }
}
