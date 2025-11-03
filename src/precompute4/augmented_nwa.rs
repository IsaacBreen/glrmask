use crate::glr::grammar::regex_name;
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{
    compute_all_characterizations, compute_below_bottom_characterization, BelowBottomCharacterization,
};
use crate::weighted_automata::{
    DWA, NWA as WaNWA, NWABody as WaNWABody, NWAStates as WaNWAStates, StateID as WaStateID, Weight as WaWeight,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque, HashMap};
use std::fmt::{Display, Formatter};
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

/// Internal helper: iterate neighbors ignoring labels (ε, default, exceptions).
fn neighbors_ignoring_labels(states: &WaNWAStates, u: WaStateID, out: &mut Vec<WaStateID>) {
    // ε edges
    for (v, _) in &states[u].epsilon_transitions {
        out.push(*v);
    }
    // default edges
    if let Some(def) = states[u].transitions.get_default() {
        for (v, _) in def {
            out.push(*v);
        }
    }
    // exception edges
    for vec in states[u].transitions.exceptions.values() {
        for (v, _) in vec {
            out.push(*v);
        }
    }
}

impl AugmentedNwaBody {
    /// Compute, for the subgraph reachable from `self.nwa.start_states` ignoring labels,
    /// the set of end-map keys each state can reach (ignoring labels).
    ///
    /// Returns a map: state -> {end_map_key_states}
    pub fn precompute_end_reach(
        &self,
        states: &WaNWAStates,
    ) -> BTreeMap<WaStateID, BTreeSet<WaStateID>> {
        let mut subgraph: BTreeSet<WaStateID> = BTreeSet::new();
        let mut preds: BTreeMap<WaStateID, Vec<WaStateID>> = BTreeMap::new();
        let mut q: VecDeque<WaStateID> = VecDeque::new();

        // 1) Collect subgraph reachable from starts ignoring labels; build predecessor map.
        for &s in &self.nwa.start_states {
            if subgraph.insert(s) {
                q.push_back(s);
            }
        }
        let mut neigh = Vec::new();
        while let Some(u) = q.pop_front() {
            neigh.clear();
            neighbors_ignoring_labels(states, u, &mut neigh);
            for v in neigh.iter().copied() {
                preds.entry(v).or_default().push(u);
                if subgraph.insert(v) {
                    q.push_back(v);
                }
            }
        }

        // 2) Reverse BFS from each end_map key to accumulate "can reach that key".
        let mut result: BTreeMap<WaStateID, BTreeSet<WaStateID>> = BTreeMap::new();
        let mut work: VecDeque<WaStateID> = VecDeque::new();
        for &end_key in self.end_map.keys() {
            if !subgraph.contains(&end_key) {
                continue;
            }
            let mut seen: BTreeSet<WaStateID> = BTreeSet::new();
            seen.insert(end_key);
            work.push_back(end_key);
            while let Some(v) = work.pop_front() {
                result.entry(v).or_default().insert(end_key);
                if let Some(ps) = preds.get(&v) {
                    for &p in ps {
                        if subgraph.contains(&p) && seen.insert(p) {
                            work.push_back(p);
                        }
                    }
                }
            }
        }
        result
    }
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

    pub fn combine_right_into_on_shared(
        states: &mut WaNWAStates,
        left: &mut AugmentedNwaBody,
        right: &AugmentedNwaBody,
        weight: &WaWeight,
    ) -> Result<(), AugmentedNwaBuildError> {
        let now = Instant::now();
        let left_end_snapshot = left.end_map.clone();
        let mut new_end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();

        let mut total_process_stack_time = std::time::Duration::new(0, 0);
        let mut total_reachable_time = std::time::Duration::new(0, 0);
        let mut total_end_map_build_time = std::time::Duration::new(0, 0);

        // Precompute reachable end-map states per right-state (ignoring labels) once.
        let precompute_now = Instant::now();
        let end_targets = right.precompute_end_reach(states);
        let precompute_elapsed = precompute_now.elapsed();

        for (left_end_state, stacks) in left_end_snapshot {
            for left_stack in stacks {
                let encoded: Vec<u16> =
                    left_stack.iter().rev().map(|&s| encode_symbol(s)).collect::<Result<_, _>>()?;

                let process_now = Instant::now();
                let stops = states.process_stack_u16_from_starts(&right.nwa.start_states, &encoded);
                total_process_stack_time += process_now.elapsed();

                for (pos, right_stop_state, path_weight) in stops {
                    let combined_weight = &path_weight & weight;
                    states.add_epsilon_transition(left_end_state, right_stop_state, combined_weight);

                    if right.end_map.values().all(|stacks| stacks.is_empty()) {
                        continue;
                    }

                    // Use precomputed targets instead of BFS over the (huge) shared arena.
                    let reachable_now = Instant::now();
                    let targets = end_targets.get(&right_stop_state).cloned().unwrap_or_default();
                    total_reachable_time += reachable_now.elapsed();

                    let end_map_build_now = Instant::now();
                    for r_state in targets {
                        if let Some(r_stacks) = right.end_map.get(&r_state) {
                            for r_stack in r_stacks {
                                let keep_len = left_stack.len().saturating_sub(pos);
                                let mut combined: Vec<ParserStateID> = left_stack[..keep_len].to_vec();
                                combined.extend(r_stack.iter().cloned());
                                new_end_map.entry(r_state).or_default().insert(combined);
                            }
                        }
                    }
                    total_end_map_build_time += end_map_build_now.elapsed();
                }
            }
        }
        left.end_map = new_end_map;
        println!(
            "    combine_right_into_on_shared took: {:?}, precompute_end_reach: {:?}, process_stack: {:?}, reachable: {:?}, end_map_build: {:?}",
            now.elapsed(),
            precompute_elapsed,
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
}

impl AugmentedNwaBody {
    /// Local epsilon-closure helper (no debug). Initial map is state -> path-weight.
    fn epsilon_closure_map(
        states: &WaNWAStates,
        initial_states: BTreeMap<WaStateID, WaWeight>,
    ) -> BTreeMap<WaStateID, WaWeight> {
        let mut closure = initial_states.clone();
        let mut worklist: VecDeque<WaStateID> = closure.keys().cloned().collect();
        while let Some(u_id) = worklist.pop_front() {
            let u_weight = closure.get(&u_id).unwrap().clone();
            if u_weight.is_empty() {
                continue;
            }
            if states[u_id].epsilon_transitions.is_empty() {
                continue;
            }
            for (v_id, trans_weight) in &states[u_id].epsilon_transitions {
                let new_v_weight = &u_weight & trans_weight;
                if new_v_weight.is_empty() {
                    continue;
                }
                let current_v_weight = closure.entry(*v_id).or_insert_with(WaWeight::zeros);
                let old_len = current_v_weight.len();
                *current_v_weight |= &new_v_weight;
                if current_v_weight.len() > old_len {
                    worklist.push_back(*v_id);
                }
            }
        }
        closure
    }

    /// Merge start states and build an epsilon-free front-end up to depth `k`. At the boundary,
    /// reconnect to the original states via epsilon edges to preserve exact semantics.
    pub fn simplify_front_k_epsilon_free(
        &mut self,
        states: &mut WaNWAStates,
        k: usize,
    ) {
        if self.nwa.start_states.is_empty() {
            return;
        }
        // Build initial composition = epsilon-closure of merged starts with weight=ALL.
        let mut init_raw: BTreeMap<WaStateID, WaWeight> = BTreeMap::new();
        for &s in &self.nwa.start_states {
            init_raw.insert(s, WaWeight::all());
        }
        let start_comp = Self::epsilon_closure_map(states, init_raw);
        // Key conversion: keep only non-empty weights and sorted order.
        let to_key = |comp: &BTreeMap<WaStateID, WaWeight>| -> Vec<(WaStateID, WaWeight)> {
            comp.iter().filter(|(_, w)| !w.is_empty()).map(|(k, v)| (*k, v.clone())).collect()
        };
        let start_key = to_key(&start_comp);
        if start_key.is_empty() {
            // Nothing reachable; keep as-is.
            return;
        }

        // Mappings: comp_key -> new_state_id, and comp_key -> BTreeMap form.
        let mut comp_to_id: HashMap<Vec<(WaStateID, WaWeight)>, WaStateID> = HashMap::new();
        let mut comp_to_map: HashMap<Vec<(WaStateID, WaWeight)>, BTreeMap<WaStateID, WaWeight>> = HashMap::new();
        let mut work: VecDeque<(Vec<(WaStateID, WaWeight)>, usize)> = VecDeque::new();

        let new_start_id = states.add_state();
        comp_to_id.insert(start_key.clone(), new_start_id);
        comp_to_map.insert(start_key.clone(), start_comp);
        work.push_back((start_key, 0));

        // Helper to compute final_weight aggregate for a composition.
        let mut aggregate_final = |comp_map: &BTreeMap<WaStateID, WaWeight>| -> Option<WaWeight> {
            let mut agg: Option<WaWeight> = None;
            for (sid, pw) in comp_map {
                if let Some(fw) = &states[*sid].final_weight {
                    let w = pw & fw;
                    if !w.is_empty() {
                        if let Some(ref mut a) = agg { *a |= &w; } else { agg = Some(w); }
                    }
                }
            }
            agg
        };

        while let Some((comp_key, depth)) = work.pop_front() {
            let cur_id = *comp_to_id.get(&comp_key).unwrap();
            let comp_map = comp_to_map.get(&comp_key).unwrap();

            // Final weight for this front-end state
            if let Some(final_w) = aggregate_final(comp_map) {
                states[cur_id].final_weight = Some(final_w);
            }

            if depth == k {
                // Boundary: reconnect to original states by epsilon edges with their path weights.
                for (sid, pw) in comp_map {
                    states[cur_id].epsilon_transitions.push((*sid, pw.clone()));
                }
                continue;
            }

            // Collect critical points (exception alphabet) from underlying states.
            let mut critical_points: BTreeSet<u16> = BTreeSet::new();
            for (sid, _) in comp_map {
                for &ch in states[*sid].transitions.exceptions.keys() {
                    critical_points.insert(ch);
                }
            }

            // Default transition aggregate
            let mut default_next_raw: BTreeMap<WaStateID, WaWeight> = BTreeMap::new();
            let mut default_weight_agg = WaWeight::zeros();
            for (sid, pw) in comp_map {
                if let Some(def) = states[*sid].transitions.get_default() {
                    for (to, tw) in def {
                        let w = pw & tw;
                        if !w.is_empty() {
                            default_next_raw.entry(*to).or_insert_with(WaWeight::zeros).bitor_assign(&w);
                            default_weight_agg |= &w;
                        }
                    }
                }
            }
            if !default_next_raw.is_empty() {
                let next_map = Self::epsilon_closure_map(states, default_next_raw);
                let next_key = to_key(&next_map);
                if !next_key.is_empty() {
                    let tgt_id = *comp_to_id.entry(next_key.clone()).or_insert_with(|| {
                        let nid = states.add_state();
                        comp_to_map.insert(next_key.clone(), next_map);
                        work.push_back((next_key.clone(), depth + 1));
                        nid
                    });
                    states[cur_id].transitions.default.get_or_insert_with(Vec::new).push((tgt_id, default_weight_agg));
                }
            }

            // Exception transitions
            for ch in critical_points {
                let mut next_raw: BTreeMap<WaStateID, WaWeight> = BTreeMap::new();
                let mut ex_weight_agg = WaWeight::zeros();
                for (sid, pw) in comp_map {
                    if let Some(vec) = states[*sid].transitions.exceptions.get(&ch) {
                        for (to, tw) in vec {
                            let w = pw & tw;
                            if !w.is_empty() {
                                next_raw.entry(*to).or_insert_with(WaWeight::zeros).bitor_assign(&w);
                                ex_weight_agg |= &w;
                            }
                        }
                    }
                }
                if next_raw.is_empty() {
                    continue;
                }
                let next_map = Self::epsilon_closure_map(states, next_raw);
                let next_key = to_key(&next_map);
                if next_key.is_empty() {
                    continue;
                }
                let tgt_id = *comp_to_id.entry(next_key.clone()).or_insert_with(|| {
                    let nid = states.add_state();
                    comp_to_map.insert(next_key.clone(), next_map);
                    work.push_back((next_key.clone(), depth + 1));
                    nid
                });
                states[cur_id].transitions.exceptions.entry(ch).or_default().push((tgt_id, ex_weight_agg));
            }
        }

        // Replace with single merged start.
        self.nwa.start_states = BTreeSet::from([new_start_id]);
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
        AugmentedNwaBody::combine_right_into_on_shared(
            &mut self.states,
            &mut self.body,
            &mapped_right_body,
            weight,
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

