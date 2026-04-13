use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::analysis::{EOF, AnalyzedGrammar};
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Shift(u32),
    Reduce(u32),
    Split {
        shift: Option<u32>,
        reduces: Vec<u32>,
        accept: bool,
    },
    Accept,
}

impl Action {
    /// The shift target, if any. Works for both Shift and Split.
    #[inline]
    pub fn shift_target(&self) -> Option<u32> {
        match self {
            Action::Shift(t) => Some(*t),
            Action::Split { shift: Some(t), .. } => Some(*t),
            _ => None,
        }
    }

    /// Slice of reduce rule IDs. Empty for Shift/Accept.
    #[inline]
    pub fn reduce_rule_ids(&self) -> &[u32] {
        match self {
            Action::Reduce(id) => std::slice::from_ref(id),
            Action::Split { reduces, .. } => reduces.as_slice(),
            _ => &[],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLRTable {
    pub action: Vec<FxHashMap<TerminalID, Action>>,
    pub goto: Vec<FxHashMap<NonterminalID, u32>>,
    pub num_states: u32,
    pub num_terminals: u32,
    pub num_rules: u32,
    pub rules: Vec<Rule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TableRowKey {
    action: Vec<(TerminalID, Action)>,
    goto: Vec<(NonterminalID, u32)>,
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
    pub fn goto_target(&self, state: u32, nt: NonterminalID) -> Option<u32> {
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
                        .map(|(&nt, &target)| (nt, mapping[target as usize]))
                        .collect()
                })
                .collect();

            self.action = new_action;
            self.goto = new_goto;
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

            // Step 1: Canonicalize rule IDs by (lhs, rhs_len).
            let mut lhs_rhs_to_canon: FxHashMap<(NonterminalID, usize), u32> =
                FxHashMap::default();
            let rule_canon: Vec<u32> = self
                .rules
                .iter()
                .enumerate()
                .map(|(id, rule)| {
                    let key = (rule.lhs, rule.rhs.len());
                    *lhs_rhs_to_canon.entry(key).or_insert(id as u32)
                })
                .collect();

            // Rewrite all action entries with canonical rule IDs.
            for state in 0..self.num_states as usize {
                let old = std::mem::take(&mut self.action[state]);
                let mut new_action = FxHashMap::default();
                for (tid, action) in old {
                    new_action.insert(tid, canonicalize_action_rules(&action, &rule_canon));
                }
                self.action[state] = new_action;
            }

            // Step 2: Merge states with now-identical rows.
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
            let mut nt_to_column: FxHashMap<NonterminalID, Vec<Option<u32>>> =
                FxHashMap::default();
            for &nt in &all_nts {
                let col: Vec<Option<u32>> = (0..nstates)
                    .map(|s| self.goto[s].get(&nt).copied())
                    .collect();
                nt_to_column.insert(nt, col);
            }

            // Group NTs by column.
            let mut column_to_canon: FxHashMap<Vec<Option<u32>>, NonterminalID> =
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
                    let mut new_goto: FxHashMap<NonterminalID, u32> = FxHashMap::default();
                    for (nt, target) in old {
                        let canon_nt = nt_remap.get(&nt).copied().unwrap_or(nt);
                        // All remapped NTs should have the same target; just insert.
                        new_goto.insert(canon_nt, target);
                    }
                    self.goto[state] = new_goto;
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
                for (_, &target) in &self.goto[x] {
                    predecessors[target as usize].insert(x as u32);
                }
            }

            let mut collapsed_any = false;
            let mut collapses: Vec<(usize, TerminalID, u32)> = Vec::new();
            for state in 0..nstates2 {
                for (&tid, action) in &self.action[state] {
                    if let Action::Split { shift, reduces, accept } = action {
                        // Only handle pure-reduce splits (no shift, no accept).
                        if shift.is_some() || *accept {
                            continue;
                        }
                        // Check: do all reduces have the same rhs_len?
                        let rhs_len = self.rules[reduces[0] as usize].rhs.len();
                        if reduces.iter().any(|&r| self.rules[r as usize].rhs.len() != rhs_len) {
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
                            .map(|&r| self.rules[r as usize].lhs)
                            .collect();
                        let mut all_same = true;
                        'pred_loop: for &pred in &candidate_froms {
                            let first_target = self.goto[pred as usize].get(&lhss[0]).copied();
                            for &lhs in &lhss[1..] {
                                let target = self.goto[pred as usize].get(&lhs).copied();
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

            for (state, tid, rule_id) in collapses {
                self.action[state].insert(tid, Action::Reduce(rule_id));
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
            let mut spec_collapses: Vec<(usize, TerminalID, u32)> = Vec::new();

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

                    let rhs_len = self.rules[reduces[0] as usize].rhs.len();
                    if reduces.iter().any(|&r| self.rules[r as usize].rhs.len() != rhs_len) {
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
                        .map(|&r| self.rules[r as usize].lhs)
                        .collect();
                    let mut simple_converge = true;
                    'pred_simple: for &pred in &filtered {
                        let first_target = self.goto[pred as usize].get(&lhss[0]).copied();
                        for &lhs in &lhss[1..] {
                            if self.goto[pred as usize].get(&lhs).copied() != first_target {
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
                    let follow = |first_rule: u32| -> Option<(usize, NonterminalID)> {
                        let mut depth = rhs_len; // after initial reduce
                        let mut rule_id = first_rule;
                        let lhs = self.rules[rule_id as usize].lhs;

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

                        let mut current_lhs = lhs;
                        for _ in 0..MAX_CHAIN {
                            let preds = preds_at_depth(depth);
                            if preds.is_empty() { return None }

                            // Get goto targets
                            let mut goto_targets: BTreeSet<u32> = BTreeSet::new();
                            for &p in &preds {
                                if let Some(&gt) = self.goto[p as usize].get(&current_lhs) {
                                    goto_targets.insert(gt);
                                }
                            }
                            if goto_targets.is_empty() { return None }

                            // Check action at goto targets on terminal tid
                            let mut next_rule: Option<u32> = None;
                            let mut all_reduce = true;
                            for &gt in &goto_targets {
                                match self.action.get(gt as usize).and_then(|r| r.get(&tid)) {
                                    Some(Action::Reduce(r)) => {
                                        match next_rule {
                                            None => next_rule = Some(*r),
                                            Some(nr) if nr == *r => {}
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
                            rule_id = next_rule.unwrap();
                            let next_rhs_len = self.rules[rule_id as usize].rhs.len();
                            // depth after this reduce: current depth was where we pushed
                            // the goto target (depth-1 for the pushed state).
                            // Popping next_rhs_len from there:
                            // new_base = (depth-1) + next_rhs_len
                            //   = depth + next_rhs_len - 1
                            depth = depth + next_rhs_len - 1;
                            current_lhs = self.rules[rule_id as usize].lhs;
                        }
                        None // Too deep
                    };

                    // Follow all alternatives
                    let mut first_result: Option<(usize, NonterminalID)> = None;
                    let mut chain_converge = true;
                    for &rule_id in &reduces {
                        match follow(rule_id) {
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

                    let mut goto_target: Option<Option<u32>> = None;
                    let mut targets_agree = true;
                    for &pred in &final_preds {
                        let target = self.goto[pred as usize].get(&final_lhs).copied();
                        match goto_target {
                            None => goto_target = Some(target),
                            Some(prev) if prev == target => {}
                            _ => { targets_agree = false; break }
                        }
                    }

                    if targets_agree {
                        spec_collapses.push((state, tid, reduces[0]));
                    }
                }
            }

            for (state, tid, rule_id) in spec_collapses {
                self.action[state].insert(tid, Action::Reduce(rule_id));
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

fn canonicalize_action_rules(action: &Action, rule_canon: &[u32]) -> Action {
    match action {
        Action::Shift(t) => Action::Shift(*t),
        Action::Reduce(rule) => Action::Reduce(rule_canon[*rule as usize]),
        Action::Split {
            shift,
            reduces,
            accept,
        } => {
            let mut canon_reduces: Vec<u32> =
                reduces.iter().map(|r| rule_canon[*r as usize]).collect();
            canon_reduces.sort_unstable();
            canon_reduces.dedup();
            if canon_reduces.len() == 1 && shift.is_none() && !accept {
                Action::Reduce(canon_reduces[0])
            } else {
                Action::Split {
                    shift: *shift,
                    reduces: canon_reduces,
                    accept: *accept,
                }
            }
        }
        Action::Accept => Action::Accept,
    }
}

fn row_key(
    action_row: &FxHashMap<TerminalID, Action>,
    goto_row: &FxHashMap<NonterminalID, u32>,
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
        Action::Shift(target) => Action::Shift(mapping[*target as usize]),
        Action::Reduce(rule) => Action::Reduce(*rule),
        Action::Split {
            shift,
            reduces,
            accept,
        } => Action::Split {
            shift: shift.map(|target| mapping[target as usize]),
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
}

impl Item {
    fn new(rule: u32, dot: u32) -> Self {
        Self { rule, dot }
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
                    let new_item = Item::new(i as u32, 0);
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
            kernel.insert(Item::new(item.rule, item.dot + 1));
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
        s.insert(Item::new(0, 0)); 
        lr0_closure(&s, rules)
    };

    build_item_sets(
        initial,
        |item| item.next_symbol(rules).cloned(),
        |items, sym| lr0_goto_set(items, sym, rules),
    )
}

#[derive(Default)]
struct PendingAction {
    shift: Option<u32>,
    reduces: Vec<u32>,
    accept: bool,
}

impl PendingAction {
    fn push_shift(&mut self, target: u32) {
        match self.shift {
            Some(existing) => debug_assert_eq!(existing, target),
            None => self.shift = Some(target),
        }
    }

    fn push_reduce(&mut self, rule_id: u32) {
        self.reduces.push(rule_id);
    }

    fn push_accept(&mut self) {
        self.accept = true;
    }

    fn finish(mut self) -> Action {
        self.reduces.sort_unstable();
        self.reduces.dedup();
        match (self.shift, self.reduces.len(), self.accept) {
            (Some(target), 0, false) => Action::Shift(target),
            (None, 1, false) => Action::Reduce(self.reduces[0]),
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
    transitions: &[BTreeMap<Symbol, u32>],
) -> (
    Vec<BTreeMap<TerminalID, PendingAction>>,
    Vec<FxHashMap<NonterminalID, u32>>,
) {
    let mut pending = std::iter::repeat_with(BTreeMap::<TerminalID, PendingAction>::new)
        .take(transitions.len())
        .collect::<Vec<_>>();
    let mut goto: Vec<FxHashMap<NonterminalID, u32>> = (0..transitions.len()).map(|_| FxHashMap::default()).collect();

    for (state_id, by_symbol) in transitions.iter().enumerate() {
        for (symbol, &target) in by_symbol {
            match symbol {
                Symbol::Terminal(terminal) => {
                    pending[state_id]
                        .entry(*terminal)
                        .or_default()
                        .push_shift(target);
                }
                Symbol::Nonterminal(nonterminal) => {
                    goto[state_id].insert(*nonterminal, target);
                }
            }
        }
    }

    (pending, goto)
}

fn finish_table(
    grammar: &AnalyzedGrammar,
    pending: Vec<BTreeMap<TerminalID, PendingAction>>,
    goto: Vec<FxHashMap<NonterminalID, u32>>,
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
    }
}

#[allow(dead_code)]
fn build_slr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
    let (mut pending, goto) = initialize_pending_and_goto(transitions);

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
                    .push_reduce(item.rule);
            }
        }
    }

    finish_table(grammar, pending, goto)
}

// LR(1) item set construction.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct LR1Item {
    rule: u32,
    dot: u32,
    lookahead: TerminalID,
}

impl LR1Item {
    fn new(rule: u32, dot: u32, lookahead: TerminalID) -> Self {
        Self { rule, dot, lookahead }
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
        if let Some(Symbol::Nonterminal(nt)) = item.next_symbol(rules) {
            let rhs = &rules[item.rule as usize].rhs;
            let beta = &rhs[(item.dot as usize + 1)..];

            let lookaheads = first_of_sequence(beta, item.lookahead, &grammar.first, &grammar.nullable);

            for &i in &grammar.rules_by_lhs[*nt as usize] {
                for &la in &lookaheads {
                    let new_item = LR1Item::new(i, 0, la);
                    if result.insert(new_item) {
                        queue.push_back(new_item);
                    }
                }
            }
        }
    }
    result
}

fn lr1_goto_set(
    items: &BTreeSet<LR1Item>,
    sym: &Symbol,
    grammar: &AnalyzedGrammar,
) -> BTreeSet<LR1Item> {
    let rules = &grammar.rules;
    let mut kernel = BTreeSet::new();
    for item in items {
        if item.next_symbol(rules) == Some(sym) {
            kernel.insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead));
        }
    }
    lr1_closure(&kernel, grammar)
}

fn build_lr1_item_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<BTreeSet<LR1Item>>, Vec<BTreeMap<Symbol, u32>>) {
    let rules = &grammar.rules;

    let initial = {
        let mut s = BTreeSet::new();
        s.insert(LR1Item::new(0, 0, EOF));
        lr1_closure(&s, grammar)
    };

    let mut item_sets = vec![initial.clone()];
    let mut transitions = vec![BTreeMap::new()];
    let mut set_to_id: FxHashMap<Vec<LR1Item>, u32> = FxHashMap::default();
    set_to_id.insert(initial.iter().copied().collect(), 0);

    let mut queue = VecDeque::from([0u32]);
    while let Some(state_id) = queue.pop_front() {
        // Build all goto kernels in a single pass over items.
        let mut kernels: BTreeMap<Symbol, BTreeSet<LR1Item>> = BTreeMap::new();
        for item in &item_sets[state_id as usize] {
            if let Some(sym) = item.next_symbol(rules) {
                kernels
                    .entry(sym.clone())
                    .or_default()
                    .insert(LR1Item::new(item.rule, item.dot + 1, item.lookahead));
            }
        }

        for (symbol, kernel) in &kernels {
            let target_items = lr1_closure(kernel, grammar);
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

            transitions[state_id as usize].insert(symbol.clone(), target_id);
        }
    }

    (item_sets, transitions)
}

fn build_lr1_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, u32>],
) -> GLRTable {
    let (mut pending, goto) = initialize_pending_and_goto(transitions);

    for (state_id, items) in item_sets.iter().enumerate() {

        for item in items {
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
                .push_reduce(item.rule);
        }
    }

    finish_table(grammar, pending, goto)
}

// IELR-style merge.

fn lr1_core_key(items: &BTreeSet<LR1Item>) -> Vec<Item> {
    let mut core = BTreeSet::new();
    for item in items {
        core.insert(Item::new(item.rule, item.dot));
    }
    core.into_iter().collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ActionSig {
    Shift(u32),
    Reduce(u32),
    Split {
        shift: Option<u32>,
        reduces: Vec<u32>,
        accept: bool,
    },
    Accept,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowSignature {
    core_class: u32,
    action: Vec<(TerminalID, ActionSig)>,
    goto: Vec<(NonterminalID, u32)>,
}

fn remap_action_to_partition(action: &Action, partition: &[u32]) -> ActionSig {
    match action {
        Action::Shift(target) => ActionSig::Shift(partition[*target as usize]),
        Action::Reduce(rule) => ActionSig::Reduce(*rule),
        Action::Split {
            shift,
            reduces,
            accept,
        } => ActionSig::Split {
            shift: shift.map(|target| partition[target as usize]),
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
                .map(|(&nt, &target)| (nt, partition[target as usize]))
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
                .map(|(&nt, &target)| (nt, partition[target as usize]))
                .collect()
        })
        .collect();

    GLRTable {
        action,
        goto,
        num_states: ngroups as u32,
        num_terminals: table.num_terminals,
        num_rules: table.num_rules,
        rules: table.rules,
    }
}

fn build_ielr_table(
    grammar: &AnalyzedGrammar,
    item_sets: &[BTreeSet<LR1Item>],
    transitions: &[BTreeMap<Symbol, u32>],
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
        assert!(matches!(a0, Some(Action::Shift(_))));

        let shift_state = match a0 {
            Some(Action::Shift(s)) => *s,
            _ => panic!("expected shift"),
        };
        let a1 = table.action(shift_state, 1);
        assert!(matches!(a1, Some(Action::Shift(_))));
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
            Some(Action::Shift(s)) => *s,
            _ => panic!(),
        };
        let a1 = table.action(s1, EOF);
        assert!(matches!(a1, Some(Action::Reduce(_))));
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
        shift.push_shift(7);
        assert_eq!(shift.finish(), Action::Shift(7));

        let mut reduce = PendingAction::default();
        reduce.push_reduce(11);
        assert_eq!(reduce.finish(), Action::Reduce(11));

        let mut accept = PendingAction::default();
        accept.push_accept();
        assert_eq!(accept.finish(), Action::Accept);
    }
}
