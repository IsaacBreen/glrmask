use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::OnceLock;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::analysis::{EOF, AnalyzedGrammar};
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};

fn env_flag_enabled(key: &str) -> bool {
    std::env::var(key).map_or(false, |v| v == "1")
}

/// Check once whether replace shifts are enabled via env vars.
fn replace_shifts_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !env_flag_enabled("GLRMASK_DISABLE_REPLACE")
            && !env_flag_enabled("GLRMASK_DISABLE_REPLACE_SHIFT")
    })
}

/// Check once whether replace gotos are enabled via env vars.
fn replace_gotos_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !env_flag_enabled("GLRMASK_DISABLE_REPLACE")
            && !env_flag_enabled("GLRMASK_DISABLE_REPLACE_GOTO")
    })
}

std::thread_local! {
    pub(crate) static LOCAL_FORWARD_REPLACE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

fn local_forward_replace_enabled() -> bool {
    LOCAL_FORWARD_REPLACE_OVERRIDE.with(|c| {
        if let Some(v) = c.get() {
            return v;
        }
        env_flag_enabled("GLRMASK_ENABLE_LOCAL_FORWARD_REPLACE")
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Shift(u32, bool),
    Reduce(NonterminalID, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(NonterminalID, u32)>,
        accept: bool,
    },
    Accept,
}

impl Action {
    /// The shift target, if any. Works for both Shift and Split.
    #[inline]
    pub fn shift_target(&self) -> Option<u32> {
        match self {
            Action::Shift(t, _) => Some(*t),
            Action::Split { shift: Some((t, _)), .. } => Some(*t),
            _ => None,
        }
    }

    /// Whether the shift is a replace (pop + push instead of just push).
    #[inline]
    pub fn shift_is_replace(&self) -> bool {
        match self {
            Action::Shift(_, r) => *r,
            Action::Split { shift: Some((_, r)), .. } => *r,
            _ => false,
        }
    }

    /// Iterate over reduce (lhs_nonterminal, reduce_length) pairs.
    #[inline]
    pub fn for_each_reduce(&self, mut f: impl FnMut(NonterminalID, u32)) {
        match self {
            Action::Reduce(nt, len) => f(*nt, *len),
            Action::Split { reduces, .. } => {
                for &(nt, len) in reduces {
                    f(nt, len);
                }
            }
            _ => {}
        }
    }

    /// Number of reduce alternatives.
    #[inline]
    pub fn reduce_count(&self) -> usize {
        match self {
            Action::Reduce(..) => 1,
            Action::Split { reduces, .. } => reduces.len(),
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    pub action: Vec<FxHashMap<TerminalID, Action>>,
    pub goto: Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
    /// Set of (state, terminal) pairs where the shift was created by the
    /// transfer mechanism.  The characterization should treat these as
    /// non-replace to avoid creating pop-0 reduces in the template NFA.
    pub forwarded_shifts: FxHashSet<(u32, TerminalID)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TableRowKey {
    action: Vec<(TerminalID, Action)>,
    goto: Vec<(NonterminalID, (u32, bool))>,
}

impl GLRTable {
    pub fn build(grammar: &AnalyzedGrammar) -> Self {
        let t0 = std::time::Instant::now();
        let (item_sets, transitions) = build_lr1_item_sets(grammar);
        let lr1_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let mut table = build_ielr_table(grammar, &item_sets, &transitions);
        let ielr_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let pre_merge_states = table.num_states;
        let t2 = std::time::Instant::now();
        table.merge_identical_rows();
        let merge_ms = t2.elapsed().as_secs_f64() * 1000.0;

        let pre_recog_states = table.num_states;
        let pre_recog_splits = {
            let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
                .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
                .unwrap_or(false);
            if debug_profile {
                table.action.iter().filter(|row| {
                    row.values().any(|a| matches!(a, Action::Split { .. }))
                }).count()
            } else { 0 }
        };
        let t3 = std::time::Instant::now();
        table.merge_recognizer_equivalent();
        let recog_ms = t3.elapsed().as_secs_f64() * 1000.0;

        let debug_profile = std::env::var("GLRMASK_DEBUG_PROFILE")
            .map(|v| { let n = v.trim().to_ascii_lowercase(); !matches!(n.as_str(), "" | "0" | "false" | "no" | "off") })
            .unwrap_or(false);
        if debug_profile {
            let max_items = item_sets.iter().map(|s| s.len()).max().unwrap_or(0);
            let total_items: usize = item_sets.iter().map(|s| s.len()).sum();
            let count_splits = |t: &GLRTable| -> usize {
                t.action.iter().filter(|row| {
                    row.values().any(|a| matches!(a, Action::Split { .. }))
                }).count()
            };
            eprintln!(
                "[glrmask/debug][glr_table] lr1_states={} lr1_ms={:.3} ielr_ms={:.3} pre_merge_states={} merge_ms={:.3} pre_recog_states={} pre_recog_splits={} recog_ms={:.3} final_states={} splits={} max_items_per_state={} total_items={}",
                item_sets.len(), lr1_ms, ielr_ms, pre_merge_states, merge_ms, pre_recog_states, pre_recog_splits, recog_ms, table.num_states, count_splits(&table), max_items, total_items,
            );
        }

        table
    }

    #[inline]
    pub fn action(&self, state: u32, terminal: TerminalID) -> Option<&Action> {
        self.action
            .get(state as usize)
            .and_then(|by_terminal| by_terminal.get(&terminal))
    }

    #[inline]
    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<(u32, bool)> {
        self.goto
            .get(state as usize)
            .and_then(|by_nt| by_nt.get(&nt).copied())
    }

    /// Merge states with identical (action, goto) rows.
    /// Iterates until no more merges are possible, since remapping targets
    /// can reveal new equivalences.
    fn merge_identical_rows(&mut self) {
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
                        .map(|(&tid, action)| (tid, remap_action_targets(action, &mapping)))
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

    /// Merge states that are equivalent for recognition purposes.
    ///
    /// Unlike `merge_identical_rows` which requires exact action/goto match,
    /// this treats two Reduce actions as equivalent when they have the same
    /// `(lhs, rhs_len)`, since the parser only uses those two fields.
    /// It also merges goto columns for nonterminals that become equivalent.
    /// Iterates until stable.
    fn merge_recognizer_equivalent(&mut self) {
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
            for goto_row in &self.goto {
                for &nt in goto_row.keys() {
                    all_nts.insert(nt);
                }
            }

            // Build goto column for each nonterminal.
            let mut nt_to_column: FxHashMap<NonterminalID, Vec<Option<(u32, bool)>>> =
                FxHashMap::default();
            for &nt in &all_nts {
                let col: Vec<Option<(u32, bool)>> = (0..nstates)
                    .map(|s| self.goto[s].get(&nt).copied())
                    .collect();
                nt_to_column.insert(nt, col);
            }

            // Group NTs by column.
            let mut column_to_canon: FxHashMap<Vec<Option<(u32, bool)>>, NonterminalID> =
                FxHashMap::default();
            let mut nt_remap: FxHashMap<NonterminalID, NonterminalID> = FxHashMap::default();
            for &nt in &all_nts {
                let col = &nt_to_column[&nt];
                let canon = *column_to_canon.entry(col.clone()).or_insert(nt);
                if canon != nt {
                    nt_remap.insert(nt, canon);
                }
            }

            if !nt_remap.is_empty() {
                // Rewrite goto entries: merge columns.
                for state in 0..nstates {
                    let old = std::mem::take(&mut self.goto[state]);
                    let mut new_goto: FxHashMap<NonterminalID, (u32, bool)> = FxHashMap::default();
                    for (nt, target) in old {
                        let canon_nt = nt_remap.get(&nt).copied().unwrap_or(nt);
                        // All remapped NTs should have the same target; just insert.
                        new_goto.insert(canon_nt, target);
                    }
                    self.goto[state] = new_goto;
                }

                // Rewrite nonterminal IDs in action entries (Reduce and Split reduces).
                for state in 0..nstates {
                    let old = std::mem::take(&mut self.action[state]);
                    let new_action: FxHashMap<TerminalID, Action> = old
                        .into_iter()
                        .map(|(tid, action)| {
                            let remapped = match action {
                                Action::Reduce(nt, len) => {
                                    let canon = nt_remap.get(&nt).copied().unwrap_or(nt);
                                    Action::Reduce(canon, len)
                                }
                                Action::Split { shift, reduces, accept } => {
                                    let reduces = reduces
                                        .into_iter()
                                        .map(|(nt, len)| {
                                            let canon = nt_remap.get(&nt).copied().unwrap_or(nt);
                                            (canon, len)
                                        })
                                        .collect();
                                    Action::Split { shift, reduces, accept }
                                }
                                other => other,
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
                for (&tid, action) in &self.action[state] {
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
                    for (&t, a) in &self.action[s] {
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
                for (&tid, action) in &self.action[state] {
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
    action_row: &FxHashMap<TerminalID, Action>,
    goto_row: &FxHashMap<NonterminalID, (u32, bool)>,
) -> TableRowKey {
    TableRowKey {
        action: action_row
            .iter()
            .map(|(&terminal, action)| (terminal, action.clone()))
            .collect(),
        goto: goto_row
            .iter()
            .map(|(&nonterminal, &target)| (nonterminal, target))
            .collect(),
    }
}

fn remap_action_targets(action: &Action, mapping: &[u32]) -> Action {
    match action {
        Action::Shift(target, replace) => Action::Shift(mapping[*target as usize], *replace),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Item {
    rule: u32,
    dot: u32,
    stack_depth: u32,
}

impl Item {
    fn new(rule: u32, dot: u32, stack_depth: u32) -> Self {
        Self { rule, dot, stack_depth }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

fn lr0_closure(items: &BTreeSet<Item>, rules: &[Rule]) -> BTreeSet<Item> {
    let mut result = items.clone();
    let mut queue: VecDeque<Item> = items.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            for (i, r) in rules.iter().enumerate() {
                if r.lhs == *nt {
                    let new_item = Item::new(i as u32, 0, r.rhs.len() as u32);
                    if result.insert(new_item) {
                        queue.push_back(new_item);
                    }
                }
            }
        }
    }
    result
}

fn lr0_goto_set(items: &BTreeSet<Item>, sym: &Symbol, rules: &[Rule]) -> BTreeSet<Item> {
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(Item::new(item.rule, item.dot + 1, item.stack_depth));
        }
    }
    lr0_closure(&kernel, rules)
}

fn build_item_sets<ItemT, NextSymbol, GotoSet>(
    initial: BTreeSet<ItemT>,
    next_symbol: NextSymbol,
    goto_set: GotoSet,
) -> (Vec<BTreeSet<ItemT>>, Vec<BTreeMap<Symbol, u32>>)
where
    ItemT: Copy + Ord + std::hash::Hash,
    NextSymbol: Fn(&ItemT) -> Option<Symbol>,
    GotoSet: Fn(&BTreeSet<ItemT>, &Symbol) -> BTreeSet<ItemT>,
{
    let mut item_sets = vec![initial.clone()];
    let mut transitions = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<ItemT>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        let symbols: BTreeSet<Symbol> = item_sets[state_id as usize]
            .iter()
            .filter_map(&next_symbol)
            .collect();

        for symbol in &symbols {
            let target_items = goto_set(&item_sets[state_id as usize], symbol);
            if target_items.is_empty() {
                continue;
            }

            let key: Vec<ItemT> = target_items.iter().copied().collect();
            let target_id = if let Some(&existing_id) = set_to_id.get(&key) {
                existing_id
            } else {
                let new_id = item_sets.len() as u32;
                set_to_id.insert(key, new_id);
                item_sets.push(target_items);
                transitions.push(BTreeMap::new());
                queue.push_back(new_id);
                new_id
            };

            transitions[state_id as usize].insert(symbol.clone(), target_id);
        }
    }

    (item_sets, transitions)
}

#[allow(dead_code)]
fn build_lr0_item_sets(grammar: &AnalyzedGrammar) -> (Vec<BTreeSet<Item>>, Vec<BTreeMap<Symbol, u32>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        s.insert(Item::new(0, 0, rules[0].rhs.len() as u32)); 
        lr0_closure(&s, rules)
    };

    build_item_sets(
        initial,
        |item| item.next_symbol(rules).cloned(),
        |items, sym| lr0_goto_set(items, sym, rules),
    )
}

#[derive(Debug, Default, Clone)]
struct PendingAction {
    shift: Option<(u32, bool)>,
    reduces: Vec<(NonterminalID, u32)>,
    accept: bool,
}

impl PendingAction {
    fn push_shift(&mut self, target: u32, is_replace: bool) {
        match self.shift {
            Some((existing, _)) => debug_assert_eq!(existing, target),
            None => self.shift = Some((target, is_replace)),
        }
    }

    fn push_reduce(&mut self, nt: NonterminalID, len: u32) {
        self.reduces.push((nt, len));
    }

    fn push_accept(&mut self) {
        self.accept = true;
    }

    fn finish(mut self) -> Action {
        self.reduces.sort_unstable();
        self.reduces.dedup();
        match (self.shift, self.reduces.len(), self.accept) {
            (Some((target, replace)), 0, false) => Action::Shift(target, replace),
            (None, 1, false) => Action::Reduce(self.reduces[0].0, self.reduces[0].1),
            (None, 0, true) => Action::Accept,
            (shift, _, accept) => Action::Split {
                shift,
                reduces: self.reduces,
                accept,
            },
        }
    }
}

fn initialize_pending_and_goto(
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> (
    Vec<BTreeMap<TerminalID, PendingAction>>,
    Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    FxHashSet<(u32, TerminalID)>,
) {
    let mut pending = std::iter::repeat_with(BTreeMap::<TerminalID, PendingAction>::new)
        .take(transitions.len())
        .collect::<Vec<_>>();
    let mut goto: Vec<FxHashMap<NonterminalID, (u32, bool)>> = (0..transitions.len()).map(|_| FxHashMap::default()).collect();
    let mut forwarded_shifts = FxHashSet::default();

    for (state_id, by_symbol) in transitions.iter().enumerate() {
        for (symbol, &(target, is_replace, is_forwarded)) in by_symbol {
            match symbol {
                Symbol::Terminal(terminal) => {
                    pending[state_id]
                        .entry(*terminal)
                        .or_default()
                        .push_shift(target, is_replace);
                    if is_forwarded {
                        forwarded_shifts.insert((state_id as u32, *terminal));
                    }
                }
                Symbol::Nonterminal(nonterminal) => {
                    goto[state_id].insert(*nonterminal, (target, is_replace));
                }
            }
        }
    }

    (pending, goto, forwarded_shifts)
}

fn finish_table(
    grammar: &AnalyzedGrammar,
    pending: Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    forwarded_shifts: FxHashSet<(u32, TerminalID)>,
) -> GLRTable {
    let action: Vec<FxHashMap<TerminalID, Action>> = pending
        .into_iter()
        .map(|by_terminal| {
            by_terminal
                .into_iter()
                .map(|(terminal, pending)| (terminal, pending.finish()))
                .collect()
        })
        .collect();
    let num_states = action.len() as u32;

    GLRTable {
        action,
        goto,
        num_states,
        num_terminals: grammar.num_terminals,
        num_rules: grammar.rules.len() as u32,
        rules: grammar.rules.clone(),
        forwarded_shifts,
    }
}

#[allow(dead_code)]
fn build_slr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
    // Convert old-style transitions to new format with replace=false, forwarded=false
    let transitions_with_replace: Vec<BTreeMap<Symbol, (u32, bool, bool)>> = transitions
        .iter()
        .map(|m| m.iter().map(|(s, &t)| (s.clone(), (t, false, false))).collect())
        .collect();
    let (mut pending, goto, forwarded_shifts) = initialize_pending_and_goto(&transitions_with_replace);

    for (state_id, items) in item_sets.iter().enumerate() {

        for item in items {
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            if item.rule == 0 {
                pending[state_id].entry(EOF).or_default().push_accept();
                continue;
            }

            for &lookahead in &grammar.follow[rule.lhs as usize] {
                pending[state_id]
                    .entry(lookahead)
                    .or_default()
                    .push_reduce(rule.lhs, item.stack_depth);
            }
        }
    }

    finish_table(grammar, pending, goto, forwarded_shifts)
}

// LR(1) item set construction.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LR1Item {
    rule: u32,
    dot: u32,
    lookahead: TerminalID,
    stack_depth: u32,
    /// When true, this item was "transferred" from a parent state to provide
    /// goto information.  Transferred items do NOT participate in closure,
    /// shift actions, or reduce actions — only goto transitions.
    transferred: bool,
}

impl LR1Item {
    fn new(rule: u32, dot: u32, lookahead: TerminalID, stack_depth: u32) -> Self {
        Self { rule, dot, lookahead, stack_depth, transferred: false }
    }

    fn next_symbol<'a>(&self, rules: &'a [Rule]) -> Option<&'a Symbol> {
        let rhs = &rules[self.rule as usize].rhs;
        rhs.get(self.dot as usize)
    }
}

/// Compute FIRST set for a sequence of symbols followed by a lookahead terminal.
fn first_of_sequence(
    symbols: &[Symbol],
    lookahead: TerminalID,
    first: &[BTreeSet<TerminalID>],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeSet<TerminalID> {
    let mut result = BTreeSet::new();
    let mut all_nullable = true;
    for sym in symbols {
        match sym {
            Symbol::Terminal(t) => {
                result.insert(*t);
                all_nullable = false;
                break;
            }
            Symbol::Nonterminal(nt) => {
                result.extend(&first[*nt as usize]);
                if !nullable.contains(nt) {
                    all_nullable = false;
                    break;
                }
            }
        }
    }
    if all_nullable {
        result.insert(lookahead);
    }
    result
}

fn lr1_closure(
    items: &BTreeSet<LR1Item>,
    grammar: &AnalyzedGrammar,
) -> BTreeSet<LR1Item> {
    let rules = &grammar.rules;
    let mut result = items.clone();
    let mut queue: VecDeque<LR1Item> = items.iter().copied().collect();

    while let Some(item) = queue.pop_front() {
        // Transferred items do not participate in closure.
        if item.transferred {
            continue;
        }
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            let rhs = &rules[item.rule as usize].rhs;
            let beta = &rhs[(item.dot as usize + 1)..];

            let lookaheads = first_of_sequence(beta, item.lookahead, &grammar.first, &grammar.nullable);

            for &i in &grammar.rules_by_lhs[*nt as usize] {
                let sd = grammar.rules[i as usize].rhs.len() as u32;
                for &la in &lookaheads {
                    let new_item = LR1Item::new(i, 0, la, sd);
                    if result.insert(new_item) {
                        queue.push_back(new_item);
                    }
                }
            }
        }
    }
    result
}

#[allow(dead_code)]
fn lr1_goto_set(
    items: &BTreeSet<LR1Item>,
    sym: &Symbol,
    grammar: &AnalyzedGrammar,
) -> BTreeSet<LR1Item> {
    let rules = &grammar.rules;
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead, item.stack_depth));
        }
    }
    lr1_closure(&kernel, grammar)
}

/// Compute transferred items for the local-forward replace optimisation.
///
/// For each dot-1 item `[A → X . rest, la]` in `kernel`, find "foo items" in
/// `source_items` — items whose symbol-after-dot is `Nonterminal(A)`.  These
/// are the items that generate gotos for `A` in the source state.  Transferring
/// them into the target kernel provides the same gotos at the target so the
/// transition can be marked replace.
///
/// Returns `None` if:
/// - any dot-1 item belongs to `rule == 0` (augmented start), or
/// - any dot-1 item is NOT completed (i.e., not a single-symbol production), or
/// - any dot-1 item's LHS nonterminal has NO foo items in the source.
///
/// Recursively follows single-symbol production chains: when a foo item is
/// itself a single-symbol production at dot=0, its LHS nonterminal also needs
/// foo items in the source.
///
/// Returns `Some(transferred)` with the set of transferred items otherwise.

/// Eagerly advance transferred items past completed nonterminals in the
/// same kernel.  For example, if the kernel has:
///   [LPAREN → '('. , sd=0]  (completed, non-transferred)
///   [f → .LPAREN e ')' , transferred]
/// then the transferred item can be directly advanced to [f → LPAREN.e ')']
/// because LPAREN is already completed.  This avoids creating a zero-pop
/// reduce + goto in the table.
fn eagerly_advance_transferred(
    kernel: &BTreeSet<LR1Item>,
    rules: &[Rule],
) -> BTreeSet<LR1Item> {
    // Collect completed nonterminals (LHS of completed non-transferred items
    // with sd == 0).
    let mut completed_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    for item in kernel {
        if item.transferred {
            continue;
        }
        let rule = &rules[item.rule as usize];
        if item.dot as usize == rule.rhs.len() && item.stack_depth == 0 {
            completed_nts.insert(rule.lhs);
        }
    }

    if completed_nts.is_empty() {
        return kernel.clone();
    }

    let mut advanced_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    let mut result = BTreeSet::new();
    for item in kernel {
        if item.transferred {
            if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                if completed_nts.contains(nt) {
                    // Advance the transferred item past the completed NT.
                    // Keep sd unchanged: the advancement replaces a
                    // reduce+goto pair that was a no-op in stack terms
                    // (zero-pop reduce followed by non-replace goto).
                    result.insert(LR1Item {
                        rule: item.rule,
                        dot: item.dot + 1,
                        lookahead: item.lookahead,
                        stack_depth: item.stack_depth,
                        transferred: false,
                    });
                    advanced_nts.insert(*nt);
                    continue;
                }
            }
        }
        result.insert(*item);
    }

    // Remove completed items whose nonterminal was consumed by advancement.
    // They would produce zero-pop reduces that are now unnecessary.
    let advanced_nts_copy = advanced_nts;
    result.retain(|item| {
        if item.transferred {
            return true;
        }
        let rule = &rules[item.rule as usize];
        if item.dot as usize == rule.rhs.len() && item.stack_depth == 0 {
            !advanced_nts_copy.contains(&rule.lhs)
        } else {
            true
        }
    });

    result
}

fn compute_transfer_items(
    kernel: &BTreeSet<LR1Item>,
    source_items: &BTreeSet<LR1Item>,
    rules: &[Rule],
) -> Option<Vec<LR1Item>> {
    let mut transferred = Vec::new();

    // Collect the LHS nonterminals of all dot-1 items.
    // Only completed dot-1 items (single-symbol productions) are eligible.
    let mut needed_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    for item in kernel.iter().filter(|it| it.dot == 1) {
        if item.rule == 0 {
            return None;
        }
        let rule = &rules[item.rule as usize];
        if (item.dot as usize) != rule.rhs.len() {
            return None;
        }
        needed_nts.insert(rule.lhs);
    }

    if needed_nts.is_empty() {
        return None;
    }

    // Iteratively find foo items, following the nonterminal chain.
    // The chain extends through ALL foo items' LHS nonterminals so that
    // gotos for every nonterminal in the reduce chain are available.
    let mut all_needed = needed_nts.clone();
    let mut found_nts: BTreeSet<NonterminalID> = BTreeSet::new();
    loop {
        let mut new_needed: BTreeSet<NonterminalID> = BTreeSet::new();
        for item in source_items {
            if item.transferred {
                continue;
            }
            if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                if all_needed.contains(nt) && !found_nts.contains(nt) {
                    transferred.push(LR1Item {
                        transferred: true,
                        ..*item
                    });
                    found_nts.insert(*nt);
                    // Add this foo item's LHS to the chain so that gotos
                    // for it are also generated in the target state.
                    if item.dot == 0 {
                        let foo_rule = &rules[item.rule as usize];
                        let chain_nt = foo_rule.lhs;
                        if !all_needed.contains(&chain_nt) {
                            new_needed.insert(chain_nt);
                        }
                    }
                }
            }
        }
        if new_needed.is_empty() {
            break;
        }
        all_needed.extend(&new_needed);
    }

    // ALL initially needed nonterminals must have at least one foo item.
    // Chain-extended nonterminals may not have foo items (e.g. the
    // augmented start nonterminal) which is fine.
    if !needed_nts.is_subset(&found_nts) {
        return None;
    }

    if transferred.is_empty() {
        return None;
    }

    Some(transferred)
}

fn build_lr1_item_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<BTreeSet<LR1Item>>, Vec<BTreeMap<Symbol, (u32, bool, bool)>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        let sd = rules[0].rhs.len() as u32;
        s.insert(LR1Item::new(0, 0, EOF, sd));
        lr1_closure(&s, grammar)
    };

    let mut item_sets = vec![initial.clone()];
    let mut transitions: Vec<BTreeMap<Symbol, (u32, bool, bool)>> = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<LR1Item>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    // Pre-compute safety checks for the transfer mechanism.
    // Transfer is unsafe for nullable grammars (epsilon productions create
    // split actions that the mechanism can't handle) and recursive grammars
    // (shared reduce states get corrupted).
    let transfer_safe = !grammar.nullable.iter().any(|&n| n != 0)
        && !grammar_has_recursion(rules);

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        let source_items = item_sets[state_id as usize].clone();

        // Build all goto kernels in a single pass over items.
        let mut kernels: BTreeMap<Symbol, BTreeSet<LR1Item>> = BTreeMap::new();
        for item in &source_items {
            // Transferred items only advance through nonterminal gotos.
            if item.transferred {
                if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
                    kernels
                        .entry(Symbol::Nonterminal(*nt))
                        .or_default()
                        .insert(LR1Item {
                            rule: item.rule,
                            dot: item.dot + 1,
                            lookahead: item.lookahead,
                            stack_depth: item.stack_depth,
                            transferred: false, // un-transfer on goto
                        });
                }
                continue;
            }
            if let Some(sym) = item.next_symbol(rules) {
                kernels
                    .entry(sym.clone())
                    .or_default()
                    .insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead, item.stack_depth));
            }
        }

        for (symbol, mut kernel) in kernels {
            let base_kernel = kernel.clone();
            // Check replace condition: is_replace iff no item in the kernel
            // has dot at position 1 (i.e., all items came from items that
            // already had dot > 0).
            let has_dot_1 = kernel.iter().any(|item| item.dot == 1);
            let mut is_replace = match &symbol {
                Symbol::Terminal(_) => !has_dot_1 && replace_shifts_enabled(),
                Symbol::Nonterminal(_) => !has_dot_1 && replace_gotos_enabled(),
            };
            let mut used_transfer = false;

            // Local-forward transfer: when has_dot_1 but transfer is enabled,
            // try to transfer foo items from the source state into the kernel
            // so the transition can still be marked as replace.
            if has_dot_1 && !is_replace && local_forward_replace_enabled() && transfer_safe {
                let enable_for_this = match &symbol {
                    Symbol::Terminal(_) => replace_shifts_enabled(),
                    Symbol::Nonterminal(_) => replace_gotos_enabled(),
                };
                if enable_for_this {
                    if let Some(transferred) =
                        compute_transfer_items(&kernel, &source_items, rules)
                    {
                        kernel.extend(transferred);
                        // De-duplicate: remove transferred items that also
                        // exist as non-transferred.
                        let to_remove: Vec<LR1Item> = kernel
                            .iter()
                            .filter(|it| it.transferred)
                            .filter(|it| {
                                kernel.contains(&LR1Item {
                                    transferred: false,
                                    ..**it
                                })
                            })
                            .copied()
                            .collect();
                        for item in to_remove {
                            kernel.remove(&item);
                        }
                        is_replace = true;
                        used_transfer = true;
                    }
                }
            }

            let mut kernel_has_transferred = used_transfer && kernel.iter().any(|it| it.transferred);

            // If replace, decrement stack_depth for non-transferred kernel
            // items — the replace absorbs one stack level for items that
            // went through the shift. Transferred items didn't go through
            // the shift; they were copied from the source state and await
            // nonterminal gotos, so their sd is already correct.
            let mut adjusted_kernel: BTreeSet<LR1Item> = if is_replace {
                kernel
                    .iter()
                    .map(|item| {
                        if item.transferred {
                            *item
                        } else {
                            LR1Item {
                                stack_depth: item.stack_depth.saturating_sub(1),
                                ..*item
                            }
                        }
                    })
                    .collect()
            } else {
                kernel
            };

            let mut target_items = lr1_closure(&adjusted_kernel, grammar);
            if used_transfer && matches!(symbol, Symbol::Terminal(_)) {
                let base_target_items = lr1_closure(&base_kernel, grammar);
                let base_has_zero_pop_completed = base_target_items.iter().any(|item| {
                    let rule = &rules[item.rule as usize];
                    (item.dot as usize) == rule.rhs.len() && item.stack_depth == 0
                });
                let transferred_has_zero_pop_completed = target_items.iter().any(|item| {
                    let rule = &rules[item.rule as usize];
                    (item.dot as usize) == rule.rhs.len() && item.stack_depth == 0
                });
                if transferred_has_zero_pop_completed && !base_has_zero_pop_completed {
                    kernel = base_kernel;
                    is_replace = false;
                    kernel_has_transferred = false;
                    adjusted_kernel = kernel;
                    target_items = lr1_closure(&adjusted_kernel, grammar);
                }
            }

            // Track whether this replace was created by the transfer mechanism.
            let is_forwarded = is_replace && kernel_has_transferred;

            if target_items.is_empty() {
                continue;
            }

            let key: Vec<LR1Item> = target_items.iter().copied().collect();
            let target_id = if let Some(&existing_id) = set_to_id.get(&key) {
                existing_id
            } else {
                let new_id = item_sets.len() as u32;
                set_to_id.insert(key, new_id);
                item_sets.push(target_items);
                transitions.push(BTreeMap::new());
                queue.push_back(new_id);
                new_id
            };

            transitions[state_id as usize].insert(symbol, (target_id, is_replace, is_forwarded));
        }
    }

    (item_sets, transitions)
}

fn current_unique_reduce_len(
    pending: &[BTreeMap<TerminalID, PendingAction>],
    state: u32,
    lookahead: TerminalID,
    nonterminal: NonterminalID,
) -> Option<u32> {
    let pending_action = pending.get(state as usize)?.get(&lookahead)?;
    let mut unique_len = None;
    for &(reduce_nt, reduce_len) in &pending_action.reduces {
        if reduce_nt != nonterminal {
            continue;
        }
        match unique_len {
            None => unique_len = Some(reduce_len),
            Some(existing) if existing == reduce_len => {}
            Some(_) => return None,
        }
    }
    unique_len
}

/// Check if any nonterminal can transitively reach itself through the
/// grammar's production rules. Returns true if any recursion exists.
fn grammar_has_recursion(rules: &[Rule]) -> bool {
    let max_nt = rules.iter().map(|r| r.lhs).max().unwrap_or(0) as usize + 1;
    // Build adjacency: lhs → set of nonterminals appearing in rhs.
    let mut adj = vec![vec![false; max_nt]; max_nt];
    for rule in rules {
        for sym in &rule.rhs {
            if let Symbol::Nonterminal(nt) = sym {
                adj[rule.lhs as usize][*nt as usize] = true;
            }
        }
    }
    // Transitive closure (Floyd-Warshall on small nonterminal set).
    for k in 0..max_nt {
        for i in 0..max_nt {
            for j in 0..max_nt {
                if adj[i][k] && adj[k][j] {
                    adj[i][j] = true;
                }
            }
        }
    }
    // Check diagonal: any nonterminal that can reach itself.
    (0..max_nt).any(|i| adj[i][i])
}

fn apply_local_forward_replace(
    pending: &mut Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: &mut Vec<FxHashMap<NonterminalID, (u32, bool)>>,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
    rules: &[Rule],
) {
    // Build incoming-transition count per state so we can check whether
    // a reduce-state is uniquely reachable.  If multiple transitions
    // lead into the reduce state, rewriting its reduce length would
    // corrupt other paths (e.g. recursive grammars).
    let mut in_count = vec![0u32; item_sets.len()];
    for t in transitions {
        for &(target, _, _) in t.values() {
            in_count[target as usize] += 1;
        }
    }

    loop {
        let mut changed = false;

        for source in 0..item_sets.len() {
            for (symbol, &(target, _, _)) in &transitions[source] {
                let currently_non_replace = match symbol {
                    Symbol::Terminal(terminal) => pending[source]
                        .get(terminal)
                        .and_then(|pending_action| pending_action.shift)
                        .is_some_and(|(_, is_replace)| !is_replace),
                    Symbol::Nonterminal(nonterminal) => goto[source]
                        .get(nonterminal)
                        .is_some_and(|&(_, is_replace)| !is_replace),
                };
                if !currently_non_replace {
                    continue;
                }

                let dot1_items: Vec<_> = item_sets[target as usize]
                    .iter()
                    .copied()
                    .filter(|item| item.dot == 1)
                    .collect();
                if dot1_items.is_empty() {
                    continue;
                }

                let mut forwarded: BTreeMap<(u32, NonterminalID), u32> = BTreeMap::new();
                let mut rewrites = Vec::new();
                let mut valid = true;

                for item in dot1_items {
                    if item.rule == 0 {
                        valid = false;
                        break;
                    }

                    let rule = &rules[item.rule as usize];
                    let reduce_nt = rule.lhs;
                    let Some(&(forward_target, _)) = goto[source].get(&reduce_nt) else {
                        valid = false;
                        break;
                    };

                    let mut reduce_state = target;
                    let mut chain_unique = in_count[target as usize] <= 1;
                    for next_symbol in &rule.rhs[item.dot as usize..] {
                        let Some(&(next_state, _, _)) = transitions[reduce_state as usize].get(next_symbol) else {
                            valid = false;
                            break;
                        };
                        reduce_state = next_state;
                        // If a state in the chain is reachable from
                        // multiple predecessors, we can't safely rewrite
                        // its reduce because other paths share it.
                        if in_count[reduce_state as usize] > 1 {
                            chain_unique = false;
                        }
                    }
                    if !valid {
                        break;
                    }
                    if !chain_unique {
                        valid = false;
                        break;
                    }

                    let Some(current_len) = current_unique_reduce_len(
                        pending,
                        reduce_state,
                        item.lookahead,
                        reduce_nt,
                    ) else {
                        valid = false;
                        break;
                    };
                    if current_len != 1 {
                        valid = false;
                        break;
                    }

                    match forwarded.get(&(reduce_state, reduce_nt)) {
                        Some(&existing_target) if existing_target != forward_target => {
                            valid = false;
                            break;
                        }
                        Some(_) => {}
                        None => {
                            if let Some(&(existing_target, existing_replace)) =
                                goto[reduce_state as usize].get(&reduce_nt)
                            {
                                if existing_target != forward_target || !existing_replace {
                                    valid = false;
                                    break;
                                }
                            }
                            forwarded.insert((reduce_state, reduce_nt), forward_target);
                        }
                    }

                    rewrites.push((reduce_state, item.lookahead, reduce_nt));
                }

                if !valid || rewrites.is_empty() {
                    continue;
                }

                changed = true;

                match symbol {
                    Symbol::Terminal(terminal) => {
                        if let Some(pending_action) = pending[source].get_mut(terminal) {
                            if let Some((_, is_replace)) = pending_action.shift.as_mut() {
                                *is_replace = true;
                            }
                        }
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        if let Some((_, is_replace)) = goto[source].get_mut(nonterminal) {
                            *is_replace = true;
                        }
                    }
                }

                for &(reduce_state, lookahead, reduce_nt) in &rewrites {
                    if let Some(pending_action) = pending[reduce_state as usize].get_mut(&lookahead) {
                        for (existing_nt, reduce_len) in pending_action.reduces.iter_mut() {
                            if *existing_nt == reduce_nt && *reduce_len == 1 {
                                *reduce_len = 0;
                            }
                        }
                    }
                }

                for ((reduce_state, reduce_nt), forward_target) in forwarded {
                    goto[reduce_state as usize].insert(reduce_nt, (forward_target, true));
                }
            }
        }

        if !changed {
            break;
        }
    }
}

/// Inline zero-pop reduces: when a state has Reduce(nt, 0) on some terminal,
/// follow the goto to the target state and copy the target's action for that
/// terminal into the current state. This eliminates the zero-pop reduce
/// entirely, replacing it with a direct shift or accept.
///
/// Iterates until no more inlining is possible (handles chains of zero-pop
/// reduces).
fn inline_zero_pop_reduces(
    pending: &mut Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: &mut Vec<FxHashMap<NonterminalID, (u32, bool)>>,
) {
    loop {
        let mut changed = false;

        for state in 0..pending.len() {
            // Collect (terminal, nt, target_state) triples for zero-pop reduces.
            let mut to_inline: Vec<(TerminalID, NonterminalID, u32)> = Vec::new();
            if let Some(by_terminal) = pending.get(state) {
                for (&terminal, pa) in by_terminal {
                    for &(reduce_nt, reduce_len) in &pa.reduces {
                        if reduce_len == 0 {
                            if let Some(&(target, _)) = goto[state].get(&reduce_nt) {
                                to_inline.push((terminal, reduce_nt, target));
                            }
                        }
                    }
                }
            }

            for (terminal, reduce_nt, target) in to_inline {
                if target as usize == state {
                    continue; // avoid self-reference
                }

                // Read action at target state for the same terminal.
                let target_pa = pending[target as usize].get(&terminal).cloned();

                if let Some(pa) = pending[state].get_mut(&terminal) {
                    // Remove the zero-pop reduce.
                    pa.reduces.retain(|&(nt, len)| !(nt == reduce_nt && len == 0));

                    // Inline target's actions.
                    if let Some(tpa) = target_pa {
                        if let Some((shift_target, _)) = tpa.shift {
                            pa.push_shift(shift_target, true);
                        }
                        for (nt, len) in &tpa.reduces {
                            pa.push_reduce(*nt, *len);
                        }
                        if tpa.accept {
                            pa.push_accept();
                        }

                        // Propagate gotos needed for any zero-pop reduces
                        // we just inlined from the target state.
                        for &(nt, len) in &tpa.reduces {
                            if len == 0 {
                                if let Some(&goto_entry) = goto[target as usize].get(&nt) {
                                    goto[state].entry(nt).or_insert(goto_entry);
                                }
                            }
                        }
                    }

                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }
}

fn build_lr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let (pending, goto, forwarded_shifts) = initialize_pending_and_goto(transitions);

    // Replace flags are now carried in the transitions map from
    // build_lr1_item_sets, so we don't need to recompute them here.

    let mut pending = pending;
    let mut goto = goto;
    for (state_id, items) in item_sets.iter().enumerate() {

        for item in items {
            // Transferred items do not generate reduces.
            if item.transferred {
                continue;
            }
            let rule = &grammar.rules[item.rule as usize];
            if item.dot as usize != rule.rhs.len() {
                continue;
            }

            if item.rule == 0 {
                pending[state_id].entry(item.lookahead).or_default().push_accept();
                continue;
            }

            pending[state_id]
                .entry(item.lookahead)
                .or_default()
                .push_reduce(rule.lhs, item.stack_depth);
        }
    }

    // Post-process: convert non-replace transitions to replace where
    // possible by following the dot-1 items through to their reduce
    // states and rewriting sd=1 reduces to sd=0 with forwarded gotos.
    // Only safe for grammars that are:
    // - not using the transferred-item mechanism
    // - non-recursive (no shared reduce states)
    // - non-nullable (no epsilon productions that create split actions)
    let has_forwarded = transitions.iter().any(|t| t.values().any(|&(_, _, f)| f));
    let has_recursion = grammar_has_recursion(&grammar.rules);
    let has_nullable = grammar.nullable.iter().any(|&n| n != 0);
    if local_forward_replace_enabled() && !has_forwarded && !has_recursion && !has_nullable {
        apply_local_forward_replace(
            &mut pending,
            &mut goto,
            item_sets,
            transitions,
            &grammar.rules,
        );
    }
    // For non-forwarded grammars (apply_local_forward_replace path),
    // inline the zero-pop reduces it created.
    if local_forward_replace_enabled() && !has_forwarded && !has_recursion && !has_nullable {
        inline_zero_pop_reduces(&mut pending, &mut goto);
    }

    finish_table(grammar, pending, goto, forwarded_shifts)
}

// IELR-style merge.

fn lr1_core_key(items: &BTreeSet<LR1Item>) -> Vec<Item> {
    let mut core = BTreeSet::new();
    for item in items {
        core.insert(Item::new(item.rule, item.dot, item.stack_depth));
    }
    core.into_iter().collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ActionSig {
    Shift(u32, bool),
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
                .map(|(&terminal, action)| {
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

fn merge_same_core_lr1_states(table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {
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
                .map(|(&terminal, action)| (terminal, remap_action_targets(action, &partition)))
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
        forwarded_shifts,
    }
}

fn build_ielr_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, (u32, bool, bool)>],
) -> GLRTable {
    let canonical = build_lr1_table(grammar, item_sets, transitions);
    let core_keys = item_sets.iter().map(lr1_core_key).collect::<Vec<_>>();
    merge_same_core_lr1_states(canonical, &core_keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::flat::GrammarDef;
    use crate::grammar::flat::tests::*;

    #[test]
    fn test_table_simple_ab() {
        
        let gdef = simple_ab_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.num_states >= 3);

        let a0 = table.action(0, 0);
        assert!(matches!(a0, Some(Action::Shift(_, _))));

        let shift_state = match a0 {
            Some(Action::Shift(s, _)) => *s,
            _ => panic!("expected shift"),
        };
        let a1 = table.action(shift_state, 1);
        assert!(matches!(a1, Some(Action::Shift(_, _))));
    }

    #[test]
    fn test_table_choice() {
        
        let gdef = choice_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.action(0, 0).is_some()); 
        assert!(table.action(0, 1).is_some()); 
    }

    #[test]
    fn test_table_accept() {
        
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![crate::grammar::flat::Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        let a0 = table.action(0, 0);
        let s1 = match a0 {
            Some(Action::Shift(s, _)) => *s,
            _ => panic!(),
        };
        let a1 = table.action(s1, EOF);
        assert!(matches!(
            a1,
            Some(Action::Reduce(_, _))
                | Some(Action::Accept)
                | Some(Action::Split { accept: true, .. })
        ));
    }

    #[test]
    fn test_table_two_nt() {
        
        let gdef = two_nt_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.action(0, 0).is_some());
    }

    #[test]
    fn test_table_ambiguous() {
        
        let gdef = GrammarDef {
            rules: vec![
                
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Nonterminal(0),
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(0),
                    ],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![
                crate::grammar::flat::Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                crate::grammar::flat::Terminal::Literal {
                    id: 1,
                    bytes: b"+".to_vec(),
                },
            ],
            ..Default::default()
        };
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);

        assert!(table.num_states > 0);

        let has_conflict = (0..table.num_states).any(|s| {
            matches!(table.action(s, 1), Some(Action::Split { shift: Some(_), .. }))
        });
        assert!(
            has_conflict,
            "Expected shift/reduce conflict for ambiguous grammar"
        );
    }

    #[test]
    fn test_pending_action_finish_normalizes_pure_cases() {
        let mut shift = PendingAction::default();
        shift.push_shift(7, false);
        assert_eq!(shift.finish(), Action::Shift(7, false));

        let mut reduce = PendingAction::default();
        reduce.push_reduce(1, 2);
        assert_eq!(reduce.finish(), Action::Reduce(1, 2));

        let mut accept = PendingAction::default();
        accept.push_accept();
        assert_eq!(accept.finish(), Action::Accept);
    }
}
