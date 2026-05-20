use super::*;

const DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION_ENV: &str =
    "GLRMASK_DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION";
const ENABLE_DELAYED_STACK_SHIFT_STATES_ENV: &str = "GLRMASK_ENABLE_DELAYED_STACK_SHIFT_STATES";

fn stack_shift_predecessor_canonicalization_enabled() -> bool {
    !env_flag_enabled(DISABLE_STACK_SHIFT_PREDECESSOR_CANONICALIZATION_ENV, false)
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
struct TableRowKey {
    action: Vec<(TerminalID, Action)>,
    goto: Vec<(NonterminalID, (u32, bool))>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StackEffectKey {
    origin_state: u32,
    state: u32,
    tid: TerminalID,
    action: Action,
    frame: StackEffectFrame,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DelayedStackShiftKey {
    origin_state: u32,
    depth: u32,
    effects: Vec<GuardedStackShift>,
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
            let mut sig_to_rep: FxHashMap<TableRowKey, u32> = FxHashMap::default();
            let mut remap: Vec<u32> = (0..self.num_states).collect();
            let mut changed = false;

            for state in 0..self.num_states as usize {
                let row_key = row_key(&self.action[state], &self.goto[state]);
                let rep = *sig_to_rep.entry(row_key).or_insert(state as u32);
                if rep != state as u32 {
                    remap[state] = rep;
                    changed = true;
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

            self.action = new_action;
            self.goto = new_goto;
            self.forwarded_shifts = self.forwarded_shifts
                .iter()
                .map(|&(state, terminal)| (mapping[state as usize], terminal))
                .collect();
            self.num_states = kept.len() as u32;
        }
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
        let mut delayed_stack_shift_states: FxHashMap<TableRowKey, u32> = FxHashMap::default();
        let mut delayed_stack_shift_effect_states: FxHashMap<DelayedStackShiftKey, Option<u32>> =
            FxHashMap::default();
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
            let mut stack_effect_memo: FxHashMap<StackEffectKey, Option<Vec<GuardedStackShift>>> =
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
                        &mut delayed_stack_shift_states,
                        &mut delayed_stack_shift_effect_states,
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
}

fn row_key(
    action_row: &ActionRow,
    goto_row: &GotoRow,
) -> TableRowKey {
    let mut action = action_row
        .iter()
        .map(|(terminal, action)| (terminal, action.clone()))
        .collect::<Vec<_>>();
    action.sort_unstable_by_key(|(terminal, _)| *terminal);

    let mut goto = goto_row
        .iter()
        .map(|(&nonterminal, &target)| (nonterminal, target))
        .collect::<Vec<_>>();
    goto.sort_unstable_by_key(|(nonterminal, _)| *nonterminal);

    TableRowKey {
        action,
        goto,
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

    let state = table.num_states;
    table.num_states += 1;
    table.action.push(ActionRow::default());
    table.goto.push(GotoRow::default());
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
    Frames(Vec<StackEffectFrame>),
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

fn apply_reduce_to_frame(
    table: &GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    mut frame: StackEffectFrame,
    nt: NonterminalID,
    len: u32,
) -> Option<ReduceFrameResult> {
    pop_frame(&mut frame, len);

    let goto_froms = if let Some(&state) = frame.pushes.last() {
        BTreeSet::from([state])
    } else {
        states_at_depth(predecessors, origin_state, frame.pop)?
    };

    let guard_pop = frame.pop;
    let mut target: Option<u32> = None;
    let mut by_replace: BTreeMap<bool, BTreeSet<u32>> = BTreeMap::new();
    let mut missing = 0usize;
    for goto_from in goto_froms {
        let Some((next_target, replace)) = table.goto[goto_from as usize].get(&nt).copied() else {
            missing += 1;
            continue;
        };
        match target {
            None => target = Some(next_target),
            Some(existing) if existing == next_target => {}
            Some(_) => return None,
        }
        by_replace.entry(replace).or_default().insert(goto_from);
    }

    if missing > 0 && by_replace.is_empty() {
        return Some(ReduceFrameResult::Dead);
    }

    let target = target?;
    let needs_guard = missing > 0 || by_replace.len() > 1;
    let mut frames = Vec::new();
    for (replace, froms) in by_replace {
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
        Some(ReduceFrameResult::Frames(frames))
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
    visiting: &mut BTreeSet<(u32, TerminalID, u8, u32, Vec<u32>, Vec<StackShiftGuard>)>,
    memo: &mut FxHashMap<StackEffectKey, Option<Vec<GuardedStackShift>>>,
) -> Option<Vec<GuardedStackShift>> {
    let memo_key = StackEffectKey {
        origin_state,
        state,
        tid,
        action: action.clone(),
        frame: frame.clone(),
    };
    if let Some(cached) = memo.get(&memo_key) {
        return cached.clone();
    }

    let action_tag = match action {
        Action::Shift(..) => 0,
        Action::StackShifts(_) => 1,
        Action::GuardedStackShifts(_) => 2,
        Action::Reduce(..) => 3,
        Action::Split { .. } => 4,
        Action::Accept => 5,
    };
    let key = (state, tid, action_tag, frame.pop, frame.pushes.clone(), frame.guards.clone());
    if !visiting.insert(key.clone()) {
        return None;
    }

    let result = (|| {
        let mut out = Vec::new();
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
                    ReduceFrameResult::Dead => return Some(Vec::new()),
                    ReduceFrameResult::Frames(frames) => frames,
                };
                for frame in frames {
                    let Some(&next_state) = frame.pushes.last() else {
                        continue;
                    };
                    let Some(next) = table.action[next_state as usize].get(&tid) else {
                        continue;
                    };
                    out.extend(stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        next_state,
                        next,
                        frame,
                        visiting,
                        memo,
                    )?);
                }
            }
            Action::Split { shift, reduces, accept } => {
                if *accept {
                    return None;
                }
                if let Some((target, replace)) = shift {
                    let shift_action = Action::Shift(*target, *replace);
                    out.extend(stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        state,
                        &shift_action,
                        frame.clone(),
                        visiting,
                        memo,
                    )?);
                }
                for &(nt, len) in reduces {
                    let reduce_action = Action::Reduce(nt, len);
                    out.extend(stack_effects_for_action(
                        table,
                        predecessors,
                        origin_state,
                        tid,
                        state,
                        &reduce_action,
                        frame.clone(),
                        visiting,
                        memo,
                    )?);
                }
            }
            Action::Accept => return None,
        }

        out.sort();
        out.dedup();
        Some(out)
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

fn effects_can_be_delayed(effects: &[GuardedStackShift]) -> bool {
    effects.len() > 1
        && effects.iter().all(|effect| effect.pop > 0 && !effect.pushes.is_empty())
        && effects.iter().any(|effect| effect.guards.is_empty())
}

fn try_inline_action_to_stack_shifts(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    delayed_stack_shift_states: &mut FxHashMap<TableRowKey, u32>,
    delayed_stack_shift_effect_states: &mut FxHashMap<DelayedStackShiftKey, Option<u32>>,
    stack_effect_memo: &mut FxHashMap<StackEffectKey, Option<Vec<GuardedStackShift>>>,
) -> Option<Action> {
    let Action::Split {
        reduces,
        accept: false,
        ..
    } = action
    else {
        return None;
    };
    if reduces.is_empty() {
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
        &mut BTreeSet::new(),
        stack_effect_memo,
    )?;
    if effects.is_empty() {
        return None;
    }
    if effects_can_be_delayed(&effects)
        && env_flag_enabled(ENABLE_DELAYED_STACK_SHIFT_STATES_ENV, false)
    {
        if effects.iter().any(|effect| effect.pop == 0) {
            return None;
        }
        if let Some(delay_state) = try_create_delayed_stack_shift_state(
            table,
            predecessors,
            state,
              &effects,
              constituent_sets,
              delayed_stack_shift_states,
              delayed_stack_shift_effect_states,
              stack_effect_memo,
              0,
          ) {
            return Some(Action::Shift(delay_state, true));
        }
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

        for rep_idx in 0..idx {
            if shifts[idx].pop != shifts[rep_idx].pop
                || shifts[idx].pushes.len() != shifts[rep_idx].pushes.len()
                || shifts[idx].pushes[1..] != shifts[rep_idx].pushes[1..]
            {
                continue;
            }

            // The predecessor is buried below an identical pushed suffix. Once
            // buried, it can only be observed by a later reduction querying its
            // goto row, so prefer a predecessor whose goto row is a compatible
            // superset and let the otherwise identical stack paths merge.
            let pred = shifts[idx].pushes[0];
            let rep = shifts[rep_idx].pushes[0];
            if goto_row_is_target_compatible_subset(table, pred, rep) {
                shifts[idx].pushes[0] = rep;
                break;
            }
            if goto_row_is_target_compatible_subset(table, rep, pred) {
                shifts[rep_idx].pushes[0] = pred;
                break;
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
}

fn try_create_delayed_stack_shift_state(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    origin_state: u32,
    effects: &[GuardedStackShift],
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    delayed_stack_shift_states: &mut FxHashMap<TableRowKey, u32>,
    delayed_stack_shift_effect_states: &mut FxHashMap<DelayedStackShiftKey, Option<u32>>,
    stack_effect_memo: &mut FxHashMap<StackEffectKey, Option<Vec<GuardedStackShift>>>,
    depth: u32,
) -> Option<u32> {
    if depth >= 8 || effects.len() <= 1 || effects.iter().any(|effect| effect.pop == 0 || effect.pushes.is_empty()) {
        return None;
    }

    let mut normalized_effects = effects.to_vec();
    normalize_guarded_effects(&mut normalized_effects);
    if normalized_effects.len() <= 1
        || normalized_effects
            .iter()
            .any(|effect| effect.pop == 0 || effect.pushes.is_empty())
    {
        return None;
    }
    let effect_key = DelayedStackShiftKey {
        origin_state,
        depth,
        effects: normalized_effects,
    };
    // This cache is per stable collapse iteration. A delayed-state row is a pure
    // expansion of its normalized guarded effects plus the origin/predecessor
    // context, so identical requests can reuse the already materialized state.
    if let Some(cached) = delayed_stack_shift_effect_states.get(&effect_key) {
        return *cached;
    }
    let effects = effect_key.effects.as_slice();

    let mut terminals = BTreeSet::new();
    for effect in effects {
        let top = *effect.pushes.last()?;
        for terminal in table.action.get(top as usize)?.keys() {
            terminals.insert(terminal);
        }
    }

    let mut row = ActionRow::default();
    for terminal in terminals {
        let mut composed = Vec::new();
        for effect in effects {
            let top = *effect.pushes.last()?;
            let Some(action) = table.action[top as usize].get(&terminal).cloned() else {
                continue;
            };
            composed.extend(stack_effects_for_action(
                table,
                predecessors,
                origin_state,
                terminal,
                top,
                &action,
                StackEffectFrame {
                    pop: effect.pop,
                    pushes: effect.pushes.clone(),
                    guards: effect.guards.clone(),
                },
                &mut BTreeSet::new(),
                stack_effect_memo,
            )?);
        }
        normalize_guarded_effects(&mut composed);
        let action = if effects_can_be_delayed(&composed) {
            if let Some(next_state) = try_create_delayed_stack_shift_state(
                table,
                predecessors,
                origin_state,
                  &composed,
                  constituent_sets,
                  delayed_stack_shift_states,
                  delayed_stack_shift_effect_states,
                  stack_effect_memo,
                  depth + 1,
              ) {
                Action::Shift(next_state, true)
            } else {
                stack_effect_action(table, composed)?
            }
        } else {
            stack_effect_action(table, composed)?
        };
        row.insert(terminal, action);
    }

    if row.is_empty() {
        delayed_stack_shift_effect_states.insert(effect_key, None);
        return None;
    }

    let empty_goto = GotoRow::default();
    let key = row_key(&row, &empty_goto);
    // Delayed stack-shift states always have empty goto rows, so identical
    // rows are equivalent to the later identical-row merge and can be reused.
    if let Some(&state) = delayed_stack_shift_states.get(&key) {
        delayed_stack_shift_effect_states.insert(effect_key, Some(state));
        return Some(state);
    }

    let state = table.num_states;
    table.num_states += 1;
    table.action.push(row);
      table.goto.push(GotoRow::default());
      constituent_sets.push(BTreeSet::from([state]));
      delayed_stack_shift_states.insert(key, state);
      delayed_stack_shift_effect_states.insert(effect_key, Some(state));
      Some(state)
  }

fn try_inline_unit_reductions_for_cell(
    table: &mut GLRTable,
    predecessors: &[BTreeSet<u32>],
    state: u32,
    tid: TerminalID,
    action: &Action,
    constituent_sets: &mut Vec<BTreeSet<u32>>,
    delayed_stack_shift_states: &mut FxHashMap<TableRowKey, u32>,
    delayed_stack_shift_effect_states: &mut FxHashMap<DelayedStackShiftKey, Option<u32>>,
    stack_effect_memo: &mut FxHashMap<StackEffectKey, Option<Vec<GuardedStackShift>>>,
    subset_to_state: &mut FxHashMap<Vec<u32>, u32>,
    failed_subsets: &mut FxHashSet<Vec<u32>>,
) -> Result<Option<CellUpdate>, ()> {
    if let Some(action) = try_inline_action_to_stack_shifts(
        table,
        predecessors,
        state,
        tid,
          action,
          constituent_sets,
          delayed_stack_shift_states,
          delayed_stack_shift_effect_states,
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
        forwarded_shifts,
    }
}
