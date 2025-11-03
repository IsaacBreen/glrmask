use crate::glr::grammar::regex_name;
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{
    compute_all_characterizations, compute_below_bottom_characterization, BelowBottomCharacterization,
};
use crate::weighted_automata::{
    DWA, NWA as WaNWA, NWABody as WaNWABody, NWAStates as WaNWAStates, StateID as WaStateID, Weight as WaWeight,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use std::ops::BitOrAssign;
use std::time::Instant;

/// Error while building an AugmentedNwa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AugmentedNwaBuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedNwaBody {
    pub nwa: WaNWABody,
    pub nt_nodes: BTreeMap<NonTerminalID, WaStateID>,
    pub end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedNwa {
    pub states: WaNWAStates,
    pub body: AugmentedNwaBody,
}

fn encode_symbol(id: ParserStateID) -> Result<u16, AugmentedNwaBuildError> {
    u16::try_from(id.0).map_err(|_| AugmentedNwaBuildError::ParserStateIdOutOfRange { state_id: id })
}

pub fn build_augmented_nwa_for_terminal(
    parser: &GLRParser,
    terminal_id: TerminalID,
) -> Result<AugmentedNwa, AugmentedNwaBuildError> {
    let bb = compute_below_bottom_characterization(parser, terminal_id);
    build_augmented_nwa_from_characterization(parser, &bb)
}

/// Identity NWA for ignore terminals: passes through any stack unchanged.
pub fn build_augmented_nwa_for_ignore_terminal() -> AugmentedNwa {
    let mut states = WaNWAStates::default();
    let start_state = states.add_state();
    let mut end_map = BTreeMap::new();
    end_map.insert(start_state, BTreeSet::from([vec![]]));

    AugmentedNwa {
        states,
        body: AugmentedNwaBody {
            nwa: WaNWABody { start_states: BTreeSet::from([start_state]) },
            nt_nodes: BTreeMap::new(),
            end_map,
        },
    }
}

pub fn build_augmented_nwas(
    parser: &GLRParser,
) -> Result<BTreeMap<TerminalID, AugmentedNwa>, AugmentedNwaBuildError> {
    let all = compute_all_characterizations(parser);

    crate::debug!(5, "\n--- Terminal Characterizations ---");
    for (terminal_id, bb) in &all {
        let terminal = parser.terminal_map.get_by_right(terminal_id).cloned().unwrap_or(regex_name("UNKNOWN"));
        crate::debug!(5, "Terminal {} ({}) Characterization:\n{}", terminal_id.0, terminal, bb);
    }
    crate::debug!(5, "--- End Terminal Characterizations ---\n");

    let mut out = BTreeMap::new();
    for (term, bb) in all {
        let aug = build_augmented_nwa_from_characterization(parser, &bb)?;
        out.insert(term, aug);
    }
    Ok(out)
}

/// Core builder: turns a BelowBottomCharacterization into an AugmentedNwa.
///
/// Construction rules:
/// - Create one initial start node, one unique `end_state`, and one node per nonterminal.
/// - Initial shifts: start --(initial_state)--> end_state; record [initial_state, shift_state].
/// - Initial reduces (initial_state, len, nt): first step on initial_state, then len default steps to nt.
/// - Per-nonterminal reduces:
///   - Reveal-and-rereduces (revealed_state, len, reduce_nt): first on revealed_state, then len default steps to reduce_nt.
///   - Reveal-goto-shift escapes: from nt node, on revealed_state go to end_state; record [revealed_state, goto_state, shift_state].
pub fn build_augmented_nwa_from_characterization(
    parser: &GLRParser,
    bb: &BelowBottomCharacterization,
) -> Result<AugmentedNwa, AugmentedNwaBuildError> {
    let mut states = WaNWAStates::default();
    let start = states.add_state();
    let end_state = states.add_state();
    let mut body = AugmentedNwaBody {
        nwa: WaNWABody { start_states: BTreeSet::from([start]) },
        nt_nodes: BTreeMap::new(),
        end_map: BTreeMap::new(),
    };

    for &nt in parser.non_terminal_map.right_values() {
        let id = states.add_state();
        body.nt_nodes.insert(nt, id);
    }

    let w_all = WaWeight::all();

    for &(initial_state, shift_state) in &bb.initial_shifts {
        let ch = encode_symbol(initial_state)?;
        states.add_transition(start, ch, end_state, w_all.clone());
        body.end_map.entry(end_state).or_default().insert(vec![initial_state, shift_state]);
    }

    for &(initial_state, len, nt) in &bb.initial_reduces {
        let ch = encode_symbol(initial_state)?;
        let target_nt = *body.nt_nodes.get(&nt).expect("nonterminal node must exist");
        let mut from = start;

        let next_state = if len == 0 { target_nt } else { states.add_state() };
        states.add_transition(from, ch, next_state, w_all.clone());
        from = next_state;

        for i in 0..len {
            let to = if i == len - 1 { target_nt } else { states.add_state() };
            states.add_default_transition(from, to, w_all.clone());
            from = to;
        }
    }

    for (nt, rc) in &bb.reduce_characterizations {
        let src_nt = *body.nt_nodes.get(nt).expect("reduce_characterizations nt must exist");

        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            let ch = encode_symbol(revealed_state)?;
            let dst_nt = *body.nt_nodes.get(&reduce_nt).expect("reduce target nonterminal must exist");
            let mut from = src_nt;

            let next_state = if len == 0 { dst_nt } else { states.add_state() };
            states.add_transition(from, ch, next_state, w_all.clone());
            from = next_state;

            for i in 0..len {
                let to = if i == len - 1 { dst_nt } else { states.add_state() };
                states.add_default_transition(from, to, w_all.clone());
                from = to;
            }
        }

        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let ch = encode_symbol(revealed_state)?;
            states.add_transition(src_nt, ch, end_state, w_all.clone());
            body.end_map.entry(end_state).or_default().insert(vec![revealed_state, goto_state, shift_state]);
        }
    }

    Ok(AugmentedNwa { states, body })
}

impl AugmentedNwaBody {
    pub fn remap_states(&mut self, mapping: &[WaStateID]) {
        self.nwa.start_states = self.nwa.start_states.iter().map(|&s| mapping[s]).collect();
        for v in self.nt_nodes.values_mut() {
            *v = mapping[*v];
        }
        self.end_map = std::mem::take(&mut self.end_map).into_iter().map(|(k, v)| (mapping[k], v)).collect();
    }

    pub fn process_stack(
        &self,
        states: &WaNWAStates,
        stack: &[ParserStateID],
    ) -> Result<Vec<(WaStateID, WaStateID, WaWeight)>, AugmentedNwaBuildError> {
        let encoded: Vec<u16> =
            stack.iter().map(|&s| encode_symbol(s)).collect::<Result<_, _>>()?;
        Ok(states.process_stack_u16_from_starts(&self.nwa.start_states, &encoded))
    }

    /// Precompute: for each state s reachable from our starts (ignoring labels),
    /// the union of all stacks associated with end_map states reachable from s (also ignoring labels).
    /// This removes the need for a per-stop BFS during combination.
    /// Complexity: single backward fixpoint over the body subgraph per use.
    pub fn combine_right_into_on_shared(
        // shared arena
        states: &mut WaNWAStates,
        left: &mut AugmentedNwaBody,
        right: &AugmentedNwaBody,
        weight: &WaWeight,
        right_end_reach: Option<&BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>>>,
    ) -> Result<(), AugmentedNwaBuildError> {
        let now = Instant::now();
        let left_end_snapshot = left.end_map.clone();
        let mut new_end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();

        let mut total_process_stack_time = std::time::Duration::new(0, 0);
        let mut total_reachable_time = std::time::Duration::new(0, 0);
        let mut total_end_map_build_time = std::time::Duration::new(0, 0);

        for (left_end_state, stacks) in left_end_snapshot {
            for left_stack in stacks {
                let encoded: Vec<u16> =
                    left_stack.iter().rev().map(|&s| encode_symbol(s)).collect::<Result<_, _>>()?;

                let process_now = Instant::now();
                let stops = states.process_stack_u16_from_starts(&right.nwa.start_states, &encoded);
                total_process_stack_time += process_now.elapsed();

                for (pos, right_stop_state, path_weight) in stops {
                    let combined_weight = &path_weight & weight;

                    let end_map_build_now = Instant::now();
                    let r_sets = if !right.end_map.is_empty() {
                        if let Some(cache) = right_end_reach {
                            cache.get(&right_stop_state).cloned().unwrap_or_default()
                        } else {
                            let mut tmp = BTreeSet::new();
                            let reachable_now = Instant::now();
                            let reachable = states.reachable_states_ignoring_labels(right_stop_state);
                            for r_state in reachable {
                                if let Some(r_stacks) = right.end_map.get(&r_state) {
                                    tmp.extend(r_stacks.clone());
                                }
                            }
                            total_reachable_time += reachable_now.elapsed();
                            tmp
                        }
                    } else {
                        BTreeSet::new()
                    };
                    states.add_epsilon_transition(left_end_state, right_stop_state, combined_weight);

                    for r_stack in &r_sets {
                        let keep_len = left_stack.len().saturating_sub(pos);
                        let mut combined: Vec<ParserStateID> = left_stack[..keep_len].to_vec();
                        combined.extend(r_stack.iter().cloned());
                        new_end_map.entry(right_stop_state).or_default().insert(combined);
                    }
                    total_end_map_build_time += end_map_build_now.elapsed();
                }
            }
        }
        left.end_map = new_end_map;
        println!(
            "    combine_right_into_on_shared took: {:?}, process_stack: {:?}, reachable: {:?}, end_map_build: {:?}",
            now.elapsed(),
            total_process_stack_time,
            total_reachable_time,
            total_end_map_build_time
        );
        Ok(())
    }

    pub fn union_with_on_shared(
        _states: &mut WaNWAStates,
        left: &mut AugmentedNwaBody,
        right: &AugmentedNwaBody,
    ) {
        for (other_end_state, stacks) in &right.end_map {
            left.end_map.entry(*other_end_state).or_default().extend(stacks.clone());
        }
        for &start in &right.nwa.start_states {
            left.nwa.start_states.insert(start);
        }
    }

    /// Compute, for this body's subgraph, the union of reachable end stacks (ignoring labels)
    /// for every state reachable from our starts. Uses a backward fixpoint over reverse edges.
    pub fn compute_end_reach_map(
        &self,
        states: &WaNWAStates,
    ) -> BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> {
        use std::collections::VecDeque;
        let now = Instant::now();

        // 1) Collect the subgraph nodes reachable from our starts ignoring labels.
        let mut sub_nodes: BTreeSet<WaStateID> = BTreeSet::new();
        for &s in &self.nwa.start_states {
            let reach = states.reachable_states_ignoring_labels(s);
            sub_nodes.extend(reach);
        }

        // 2) Build reverse adjacency restricted to subgraph.
        let mut preds: BTreeMap<WaStateID, Vec<WaStateID>> = BTreeMap::new();
        for &u in &sub_nodes {
            // epsilons
            for (v, _) in &states[u].epsilon_transitions {
                if sub_nodes.contains(v) {
                    preds.entry(*v).or_default().push(u);
                }
            }
            // default (vector)
            if let Some(def) = states[u].transitions.get_default() {
                for (v, _) in def {
                    if sub_nodes.contains(v) {
                        preds.entry(*v).or_default().push(u);
                    }
                }
            }
            // exceptions per char
            for vec in states[u].transitions.exceptions.values() {
                for (v, _) in vec {
                    if sub_nodes.contains(v) {
                        preds.entry(*v).or_default().push(u);
                    }
                }
            }
        }

        // 3) Initialize results with end_map entries present in subgraph; worklist seeded.
        let mut result: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();
        let mut q: VecDeque<WaStateID> = VecDeque::new();
        for (k, stacks) in &self.end_map {
            if sub_nodes.contains(k) && !stacks.is_empty() {
                result.insert(*k, stacks.clone());
                q.push_back(*k);
            }
        }

        // 4) Backward propagation fixpoint over reverse edges.
        while let Some(v) = q.pop_front() {
            let vset = result.get(&v).cloned().unwrap_or_default();
            if let Some(ps) = preds.get(&v) {
                for &p in ps {
                    let entry = result.entry(p).or_default();
                    let old_len = entry.len();
                    entry.extend(vset.iter().cloned());
                    if entry.len() > old_len {
                        q.push_back(p);
                    }
                }
            }
        }
        let elapsed = now.elapsed();
        if elapsed.as_millis() > 5 {
            println!(
                "AugmentedNwaBody::compute_end_reach_map: subgraph_nodes={}, end_map_keys={}, took: {:?}",
                sub_nodes.len(),
                self.end_map.len(),
                elapsed
            );
        }
        result
    }

    /// Build an epsilon-free, k-depth determinized "front" from the (possibly many) start states.
    /// - Level counts label-consuming steps; epsilon-steps do not count.
    /// - Within the front (levels < k), transitions are deterministic per character and per default.
    /// - No epsilon transitions are created in the front.
    /// - On the boundary (level+1 == k), labeled/default edges target the original graph
    ///   (after epsilon-closure), possibly with multiple targets (we do not determinize beyond k).
    /// - Appends new states into the shared arena and replaces start_states with a single summary start.
    pub fn simplify_front_k_inplace(
        &mut self,
        states: &mut WaNWAStates,
        depth: usize,
    ) {
        use std::collections::{HashMap, VecDeque};
        if depth == 0 || self.nwa.start_states.is_empty() {
            return;
        }
        let has_epsilons = states.0.iter().any(|s| !s.epsilon_transitions.is_empty());

        // Helper: map -> sorted key vector, dropping empty weights.
        let to_key = |comp: &BTreeMap<WaStateID, WaWeight>| -> Vec<(WaStateID, WaWeight)> {
            comp.iter().filter(|(_, w)| !w.is_empty()).map(|(k, w)| (*k, w.clone())).collect()
        };

        // Initial composition: starts with weight ALL, then epsilon-closure.
        let mut start_raw: BTreeMap<WaStateID, WaWeight> = BTreeMap::new();
        for &s in &self.nwa.start_states {
            start_raw.insert(s, WaWeight::all());
        }
        let start_map = if has_epsilons {
            states.epsilon_closure(start_raw)
        } else {
            start_raw
        };
        let start_key = to_key(&start_map);
        if start_key.is_empty() {
            // Degenerate: nothing reachable; leave as-is.
            return;
        }

        // Known compositions -> new summary state id.
        let mut comp2id: HashMap<Vec<(WaStateID, WaWeight)>, WaStateID> = HashMap::new();
        let mut work: VecDeque<(Vec<(WaStateID, WaWeight)>, usize)> = VecDeque::new();

        let new_start = if let Some(&id) = comp2id.get(&start_key) {
            id
        } else {
            let id = states.add_state();
            comp2id.insert(start_key.clone(), id);
            work.push_back((start_key, 0));
            id
        };
        // Replace body's starts with the single merged start.
        self.nwa.start_states.clear();
        self.nwa.start_states.insert(new_start);

        while let Some((comp, lvl)) = work.pop_front() {
            let sid = *comp2id.get(&comp).unwrap();

            // Aggregate final weight for this summary state.
            let mut is_final = false;
            let mut agg_final = WaWeight::zeros();
            let mut critical_points: BTreeSet<u16> = BTreeSet::new();
            for (nid, pw) in &comp {
                if let Some(fw) = &states[*nid].final_weight {
                    is_final = true;
                    agg_final |= &(pw & fw);
                }
                for &ch in states[*nid].transitions.exceptions.keys() {
                    critical_points.insert(ch);
                }
            }
            if is_final {
                states.set_final_weight(sid, agg_final);
            }

            // Default transition (no character).
            let mut def_next_raw: BTreeMap<WaStateID, WaWeight> = BTreeMap::new();
            let mut def_weight_agg = WaWeight::zeros();
            for (nid, pw) in &comp {
                if let Some(defs) = states[*nid].transitions.get_default() {
                    for (to, tw) in defs {
                        let w = pw & tw;
                        if !w.is_empty() {
                            def_next_raw.entry(*to).or_default().bitor_assign(&w);
                            def_weight_agg |= &w;
                        }
                    }
                }
            }
            if !def_next_raw.is_empty() {
                if lvl + 1 < depth {
                    let def_closure = if has_epsilons { states.epsilon_closure(def_next_raw) } else { def_next_raw };
                    let def_key = to_key(&def_closure);
                    if !def_key.is_empty() {
                        let tid = if let Some(&id) = comp2id.get(&def_key) {
                            id
                        } else {
                            let id = states.add_state();
                            comp2id.insert(def_key.clone(), id);
                            work.push_back((def_key, lvl + 1));
                            id
                        };
                        states.add_default_transition(sid, tid, def_weight_agg);
                    }
                } else {
                    // Boundary: transition into original graph (after epsilon-closure).
                    let def_closure = if has_epsilons { states.epsilon_closure(def_next_raw) } else { def_next_raw };
                    for (to, w) in def_closure {
                        states.add_default_transition(sid, to, w);
                    }
                }
            }

            // Exception transitions per character in the union of underlying exceptions.
            for ch in critical_points {
                let mut next_raw: BTreeMap<WaStateID, WaWeight> = BTreeMap::new();
                let mut ch_weight_agg = WaWeight::zeros();
                for (nid, pw) in &comp {
                    if let Some(vec) = states[*nid].transitions.get(ch) {
                        for (to, tw) in vec {
                            let w = pw & tw;
                            if !w.is_empty() {
                                next_raw.entry(*to).or_default().bitor_assign(&w);
                                ch_weight_agg |= &w;
                            }
                        }
                    }
                }
                if next_raw.is_empty() {
                    continue;
                }
                if lvl + 1 < depth {
                    let closure = if has_epsilons { states.epsilon_closure(next_raw) } else { next_raw };
                    let key = to_key(&closure);
                    if !key.is_empty() {
                        let tid = if let Some(&id) = comp2id.get(&key) {
                            id
                        } else {
                            let id = states.add_state();
                            comp2id.insert(key.clone(), id);
                            work.push_back((key, lvl + 1));
                            id
                        };
                        states.add_transition(sid, ch, tid, ch_weight_agg);
                    }
                } else {
                    // Boundary: add to original targets (after epsilon-closure).
                    let closure = if has_epsilons { states.epsilon_closure(next_raw) } else { next_raw };
                    for (to, w) in closure {
                        states.add_transition(sid, ch, to, w);
                    }
                }
            }
        }
    }
}

impl AugmentedNwa {
    /// Implementation note: rebase the right's states/body into `self` first, then combine on shared states.
    pub fn combine_right_into(
        &mut self,
        right: &AugmentedNwa,
        weight: &WaWeight,
    ) -> Result<(), AugmentedNwaBuildError> {
        let mapping = self.states.append_copy_from(&right.states);
        let mut mapped_right_body = right.body.clone();
        mapped_right_body.remap_states(&mapping);
        // Precompute reachable end stacks for the right body to avoid per-stop BFS.
        let right_end_reach = mapped_right_body.compute_end_reach_map(&self.states);
        AugmentedNwaBody::combine_right_into_on_shared(
            &mut self.states,
            &mut self.body,
            &mapped_right_body,
            weight,
            Some(&right_end_reach),
        )
    }

    /// Union of two augmented NWAs into self.
    pub fn union_with(&mut self, other: &AugmentedNwa) {
        let mapping = self.states.append_copy_from(&other.states);
        let mut other_body = other.body.clone();
        other_body.remap_states(&mapping);
        AugmentedNwaBody::union_with_on_shared(&mut self.states, &mut self.body, &other_body);
    }


    /// Determinize to DWA using combined NWA separation.
    pub fn determinize(&self) -> DWA {
        WaNWA::determinize_components(&self.states, &self.body.nwa)
    }
}


impl Display for AugmentedNwa {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Augmented NWA")?;
        writeln!(f, "  Nonterminal Nodes:")?;
        for (nt, state) in &self.body.nt_nodes {
            writeln!(f, "    - NT {}: State {}", nt.0, state)?;
        }
        if !self.body.end_map.is_empty() {
            writeln!(f, "  End Map:")?;
            for (state, stacks) in &self.body.end_map {
                writeln!(f, "    - State {}:", state)?;
                for stack in stacks {
                    let stack_str: Vec<String> = stack.iter().map(|s| s.0.to_string()).collect();
                    writeln!(f, "      - [{}]", stack_str.join(", "))?;
                }
            }
        }
        writeln!(f, "  Underlying NWA (starts: {:?}):", self.body.nwa.start_states)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "    State {}:", id)?;
            if let Some(w) = &state.final_weight {
                writeln!(f, "      final_weight: {}", w)?;
            }
            for (to, weight) in &state.epsilon_transitions {
                writeln!(f, "      ε -> {} (weight: {})", to, weight)?;
            }
            if let Some(default) = &state.transitions.default {
                for (to, weight) in default {
                    writeln!(f, "      * -> {} (weight: {})", to, weight)?;
                }
            }
            for (on, transitions) in &state.transitions.exceptions {
                for (to, weight) in transitions {
                    let char_repr = if let Some(c) = char::from_u32(*on as u32) {
                        if c.is_ascii_graphic() || c == ' ' {
                            format!("'{}'", c)
                        } else {
                            format!("{}", *on)
                        }
                    } else {
                        format!("{}", *on)
                    };
                    writeln!(f, "      {} -> {} (weight: {})", char_repr, to, weight)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_symbol_fails_on_large_state_ids() {
        #[cfg(target_pointer_width = "64")]
        {
            let big: ParserStateID = ParserStateID(u32::MAX as usize + 10usize);
            let err = super::encode_symbol(big).unwrap_err();
            match err {
                AugmentedNwaBuildError::ParserStateIdOutOfRange { state_id } => assert_eq!(state_id, big),
            }
        }
    }

    fn build_simple_aug_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let end = states.add_state();
        states.add_transition(start, 100, end, WaWeight::all());

        let mut end_map = BTreeMap::new();
        end_map.insert(end, BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]));

        AugmentedNwa {
            states,
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_states: BTreeSet::from([start]) },
                nt_nodes: BTreeMap::new(),
                end_map,
            },
        }
    }

    #[test]
    fn test_combine_with_ignore_on_left() {
        let mut lhs = build_augmented_nwa_for_ignore_terminal();
        let mut rhs = build_simple_aug_nwa();
        let end_state = rhs.body.end_map.keys().cloned().next().unwrap();
        rhs.states.set_final_weight(end_state, WaWeight::all());
        let weight = WaWeight::all();

        crate::debug!(5, "Left NWA (ignore):\n{}", lhs);
        crate::debug!(5, "Right NWA (simple):\n{}", rhs);

        lhs.combine_right_into(&rhs, &weight).unwrap();

        let mut states = WaNWAStates::default();
        let s0 = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        states.add_epsilon_transition(s0, s1, WaWeight::all());
        states.add_transition(s1, 100, s2, WaWeight::all());
        states.set_final_weight(s2, WaWeight::all());

        let expected_end_map =
            BTreeMap::from([(s2, BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]))]);

        let expected_aug_nwa = AugmentedNwa {
            states,
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_states: BTreeSet::from([s0]) },
                nt_nodes: BTreeMap::new(),
                end_map: expected_end_map,
            },
        };

        crate::debug!(5, "Expected NWA:\n{}", expected_aug_nwa);
        crate::debug!(5, "Resulting NWA:\n{}", lhs);

        assert_eq!(lhs, expected_aug_nwa);
    }

    #[test]
    fn test_combine_with_ignore_on_right() {
        let mut lhs = build_simple_aug_nwa();
        let mut rhs = build_augmented_nwa_for_ignore_terminal();
        rhs.states.set_final_weight(*rhs.body.nwa.start_states.iter().next().unwrap(), WaWeight::all());
        let weight = WaWeight::all();

        lhs.combine_right_into(&rhs, &weight).unwrap();

        let mut states = WaNWAStates::default();
        let s0 = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        states.add_transition(s0, 100, s1, WaWeight::all());
        states.add_epsilon_transition(s1, s2, WaWeight::all());
        states.set_final_weight(s2, WaWeight::all());

        let expected_end_map =
            BTreeMap::from([(s2, BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]))]);

        let expected_aug_nwa = AugmentedNwa {
            states,
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_states: BTreeSet::from([s0]) },
                nt_nodes: BTreeMap::new(),
                end_map: expected_end_map,
            },
        };

        assert_eq!(lhs, expected_aug_nwa);
    }

    fn build_terminal1_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start_state = states.add_state();
        let end_state = states.add_state();
        let nt0_state = states.add_state();
        let nt1_state = states.add_state();
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(start_state, 0, end_state, WaWeight::all());
        states.add_transition(start_state, 1, nt1_state, WaWeight::all());
        states.add_transition(start_state, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([(NonTerminalID(0), nt0_state), (NonTerminalID(1), nt1_state)]);
        let end_map = BTreeMap::from([(end_state, BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]))]);

        AugmentedNwa {
            states,
            body: AugmentedNwaBody { nwa: WaNWABody { start_states: BTreeSet::from([start_state]) }, nt_nodes, end_map },
        }
    }

    fn build_terminal2_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start_state = states.add_state();
        let end_state = states.add_state();
        let nt0_state = states.add_state();
        let nt1_state = states.add_state();
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(start_state, 1, nt1_state, WaWeight::all());
        states.add_transition(start_state, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([(NonTerminalID(0), nt0_state), (NonTerminalID(1), nt1_state)]);

        AugmentedNwa {
            states,
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_states: BTreeSet::from([start_state]) },
                nt_nodes,
                end_map: BTreeMap::new(),
            },
        }
    }

    fn build_terminal0_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start_state = states.add_state();
        let end_state = states.add_state();
        let nt0_state = states.add_state();
        let nt1_state = states.add_state();
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(start_state, 1, nt1_state, WaWeight::all());
        states.add_transition(start_state, 2, end_state, WaWeight::all());
        states.add_transition(start_state, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([(NonTerminalID(0), nt0_state), (NonTerminalID(1), nt1_state)]);
        let end_map = BTreeMap::from([(end_state, BTreeSet::from([vec![ParserStateID(2), ParserStateID(0)]]))]);

        AugmentedNwa {
            states,
            body: AugmentedNwaBody { nwa: WaNWABody { start_states: BTreeSet::from([start_state]) }, nt_nodes, end_map },
        }
    }

    #[test]
    fn test_right_to_left_combination() {
        let mut states = WaNWAStates::default();
        let initial_state = states.add_state();
        states.set_final_weight(initial_state, WaWeight::all());
        let mut current_aug_nwa = AugmentedNwa {
            states: states.clone(),
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_states: BTreeSet::from([initial_state]) },
                nt_nodes: BTreeMap::new(),
                end_map: BTreeMap::from([(initial_state, BTreeSet::from([vec![]]))]),
            },
        };

        let terminal_nwas_with_id =
            vec![(0, build_terminal0_nwa()), (2, build_terminal2_nwa()), (1, build_terminal1_nwa())];
        let weight = WaWeight::all();

        for (i, (term_id, term_nwa)) in terminal_nwas_with_id.iter().rev().enumerate() {
            crate::debug!(5, "\n--- Combination Step {} (Term {} on LEFT) ---", i, term_id);
            crate::debug!(5, "LEFT NWA (Term {}):\n{}", term_id, term_nwa);
            crate::debug!(5, "RIGHT NWA (Current):\n{}", current_aug_nwa);

            let mut new_current = term_nwa.clone();
            new_current.combine_right_into(&current_aug_nwa, &weight).unwrap();
            current_aug_nwa = new_current;

            crate::debug!(5, "Resulting Combined NWA:\n{}", current_aug_nwa);
        }

        assert_eq!(current_aug_nwa.states.len(), 13);
        assert!(current_aug_nwa.body.end_map.is_empty());
    }

    // to create an epsilon path to a final state.
    fn build_nwa_from_prompt_left() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start_state = states.add_state();
        let end_state = states.add_state();
        let nt0_state = states.add_state();
        let nt1_state = states.add_state();

        states.add_transition(start_state, 1, nt1_state, WaWeight::all());
        states.add_transition(start_state, 2, end_state, WaWeight::all());
        states.add_transition(start_state, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([(NonTerminalID(0), nt0_state), (NonTerminalID(1), nt1_state)]);
        let end_map = BTreeMap::from([(end_state, BTreeSet::from([vec![ParserStateID(2), ParserStateID(0)]]))]);

        AugmentedNwa {
            states,
            body: AugmentedNwaBody { nwa: WaNWABody { start_states: BTreeSet::from([start_state]) }, nt_nodes, end_map },
        }
    }

    // Helper to build the "LEFT" NWA from the prompt, which acts as `right` (the right operand)
    // in the `combine_right_into` call.
    fn build_nwa_from_prompt_right() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start_state = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        let s3 = states.add_state();
        let s4 = states.add_state();
        let end_state = states.add_state();

        let w3 = WaWeight::from_item(3);
        states.add_epsilon_transition(start_state, s1, w3.clone());
        states.add_transition(s1, 0, s2, WaWeight::all());
        states.add_transition(s1, 1, s4, WaWeight::all());
        states.add_transition(s1, 3, s3, WaWeight::all());
        states.add_epsilon_transition(s2, end_state, w3.clone());
        states.set_final_weight(end_state, WaWeight::all());

        let end_map = BTreeMap::from([(end_state, BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]))]);

        AugmentedNwa {
            states,
            body: AugmentedNwaBody { nwa: WaNWABody { start_states: BTreeSet::from([start_state]) }, nt_nodes: BTreeMap::new(), end_map },
        }
    }

    #[test]
    fn test_combination_from_prompt_example() {
        let mut self_nwa = build_nwa_from_prompt_left();
        let right_nwa = build_nwa_from_prompt_right();
        let weight = WaWeight::all();
        crate::debug!(5, "RIGHT NWA:\n{}", right_nwa);
        crate::debug!(5, "LEFT NWA:\n{}", self_nwa);

        self_nwa.combine_right_into(&right_nwa, &weight).unwrap();

        crate::debug!(5, "Resulting NWA after combination:\n{}", self_nwa);

        // Build expected result
        let mut states = WaNWAStates::default();
        let start_state = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        let s3 = states.add_state();
        // Copied states
        let s4 = states.add_state();
        let s5 = states.add_state();
        let s6 = states.add_state();
        let s7 = states.add_state();
        let s8 = states.add_state();
        let s9 = states.add_state();

        // Original part
        states.add_transition(start_state, 1, s3, WaWeight::all());
        states.add_transition(start_state, 2, s1, WaWeight::all());
        states.add_transition(start_state, 3, s2, WaWeight::all());

        // Connection
        let w3 = WaWeight::from_item(3);
        states.add_epsilon_transition(s1, s9, w3.clone());

        // Copied part
        states.add_epsilon_transition(s4, s5, w3.clone());
        states.add_transition(s5, 0, s6, WaWeight::all());
        states.add_transition(s5, 1, s8, WaWeight::all());
        states.add_transition(s5, 3, s7, WaWeight::all());
        states.add_epsilon_transition(s6, s9, w3.clone());
        states.set_final_weight(s9, WaWeight::all());

        let expected_nt_nodes = BTreeMap::from([(NonTerminalID(0), s2), (NonTerminalID(1), s3)]);
        let expected_end_map =
            BTreeMap::from([(s9, BTreeSet::from([vec![ParserStateID(2), ParserStateID(0), ParserStateID(1)]]))]);

        let expected_aug_nwa = AugmentedNwa {
            states,
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_states: BTreeSet::from([start_state]) },
                nt_nodes: expected_nt_nodes,
                end_map: expected_end_map,
            },
        };

        crate::debug!(5, "Expected NWA:\n{}", expected_aug_nwa);

        assert_eq!(self_nwa, expected_aug_nwa);
    }
}
