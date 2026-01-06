use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::constraint::StateIDBV;
use crate::glr::parser::GLRParser;
use crate::glr::table::{iter_rows, StateID as ParserStateID};
use crate::precompute4::utils::{decode_symbol_i16, DEFAULT_TRANSITION_SYMBOL};
use crate::precompute4::weighted_automata::{NWA, StateID, Weight};

/// For any state with a final weight, subtract that weight from all outgoing transitions.
/// This prunes paths that continue after a word has already been accepted with a given weight.
pub(crate) fn prune_continuations_from_final_states(nwa: &mut NWA) -> bool {
    let mut changed = false;
    for i in 0..nwa.states.len() {
        if let Some(final_weight) = nwa.states[i].final_weight.clone() {
            if final_weight.is_empty() {
                continue;
            }
            let state = &mut nwa.states[i];

            // Epsilon transitions
            for (_, w) in &mut state.epsilons {
                let old_w = w.clone();
                *w -= &final_weight;
                if *w != old_w {
                    changed = true;
                }
            }

            // Labeled transitions
            for targets in state.transitions.values_mut() {
                for (_, w) in targets {
                    let old_w = w.clone();
                    *w -= &final_weight;
                    if *w != old_w {
                        changed = true;
                    }
                }
            }
        }
    }
    changed
}

/// If a default transition for A -> B exists with weight W, subtract W from the weights of all
/// non-default transitions A -> B (and remove if the resulting weight is empty).
pub(crate) fn minimize_default_transitions(nwa: &mut NWA) -> bool {
    let mut changed = false;
    for i in 0..nwa.states.len() {
        let state = &mut nwa.states[i];

        let mut default_weights: BTreeMap<StateID, Weight> = BTreeMap::new();
        if let Some(default_targets) = state.transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            for (target, weight) in default_targets {
                if !weight.is_empty() {
                    *default_weights.entry(*target).or_insert_with(Weight::zeros) |= weight;
                }
            }
        }
        if default_weights.is_empty() {
            continue;
        }

        for (label, targets) in state.transitions.iter_mut() {
            if *label == DEFAULT_TRANSITION_SYMBOL {
                continue;
            }
            for (target, weight) in targets.iter_mut() {
                if let Some(default_weight) = default_weights.get(target) {
                    let old_weight = weight.clone();
                    *weight -= default_weight;
                    if *weight != old_weight {
                        changed = true;
                    }
                }
            }
        }

        for targets in state.transitions.values_mut() {
            let old_len = targets.len();
            targets.retain(|(_, w)| !w.is_empty());
            if targets.len() != old_len {
                changed = true;
            }
        }
        let old_len = state.transitions.len();
        state.transitions.retain(|_, targets| !targets.is_empty());
        if state.transitions.len() != old_len {
            changed = true;
        }
    }
    changed
}

fn build_label_follower_map(parser: &GLRParser) -> BTreeMap<ParserStateID, StateIDBV> {
    let mut follower_map: BTreeMap<ParserStateID, StateIDBV> = BTreeMap::new();
    let mut add_follower = |from_sid: ParserStateID, to_sid: ParserStateID| {
        follower_map.entry(from_sid).or_default().insert(to_sid.0);
    };

    for (from_sid, row) in iter_rows(&parser.table) {
        for &to_sid in row
            .get_shifts_and_reduces_map()
            .values()
            .filter_map(|action| match action {
                crate::glr::table::Stage7ShiftsAndReducesLookaheadValue::Shift(sid) => Some(sid),
                crate::glr::table::Stage7ShiftsAndReducesLookaheadValue::Split { shift, .. } => shift.as_ref(),
                _ => None,
            })
        {
            add_follower(*from_sid, to_sid);
        }
        for goto in row.gotos.values() {
            if let Some(to_sid) = goto.state_id {
                add_follower(*from_sid, to_sid);
            }
        }
    }

    let default_sid = ParserStateID(DEFAULT_TRANSITION_SYMBOL as usize);
    for sid in 0..parser.table.len() {
        let state_id = ParserStateID(sid);
        add_follower(default_sid, state_id);
        add_follower(state_id, default_sid);
    }

    follower_map
}

/// Propagate label weights along the NWA and prune transitions whose labels are never reachable.
pub(crate) fn propagate_and_prune_labels(parser: &GLRParser, nwa: &mut NWA) {
    crate::debug!(5, "Starting label propagation and pruning...");
    let now = std::time::Instant::now();

    let follower_map = build_label_follower_map(parser);

    let mut state_info: Vec<BTreeMap<ParserStateID, Weight>> = vec![BTreeMap::new(); nwa.states.len()];
    let mut worklist: VecDeque<StateID> = VecDeque::new();
    let mut in_worklist: BTreeSet<StateID> = BTreeSet::new();
    let mut initial_states: BTreeSet<StateID> = BTreeSet::new();

    // Initialize from all start states
    for &start_node in &nwa.body.start_states {
        if start_node >= nwa.states.len() { continue; }
        let start_state = &nwa.states[start_node];
        
        // Propagate to immediate neighbors of start states
        for (_, targets) in &start_state.transitions {
            for (target_state, w) in targets {
                initial_states.insert(*target_state);
                let s_init = *target_state;
                
                // Look at outgoing transitions of the neighbor to determine initial valid labels
                for (label, _) in &nwa.states[s_init].transitions {
                    if let Ok((is_pos, p_id)) = decode_symbol_i16(*label) {
                        if is_pos {
                            let entry = state_info[s_init].entry(p_id).or_insert_with(Weight::zeros);
                            *entry |= w;
                        }
                    }
                }
                if !state_info[s_init].is_empty() && in_worklist.insert(s_init) {
                    worklist.push_back(s_init);
                }
            }
        }
    }

    while let Some(u) = worklist.pop_front() {
        in_worklist.remove(&u);
        let info_at_u = state_info[u].clone();
        if info_at_u.is_empty() {
            continue;
        }

        for (l, targets) in &nwa.states[u].transitions {
            for (v, w_uv) in targets {
                let v = *v;
                let mut changed = false;

                let process_propagation = |pw: &Weight, followers: &StateIDBV, state_info: &mut Vec<BTreeMap<ParserStateID, Weight>>| -> bool {
                    let mut any_change = false;
                    for follower_id_val in followers.iter_up_to(usize::MAX) {
                        let follower_id = ParserStateID(follower_id_val);
                        let entry = state_info[v].entry(follower_id).or_insert_with(Weight::zeros);
                        let old_len = entry.len();
                        *entry |= pw;
                        if entry.len() != old_len {
                            any_change = true;
                        }
                    }
                    any_change
                };

                match decode_symbol_i16(*l) {
                    Ok((is_pos, p_id)) => {
                        if is_pos {
                            if let Some(w_p) = info_at_u.get(&p_id) {
                                let pw = w_p & w_uv;
                                if pw.is_empty() {
                                    continue;
                                }
                                if let Some(followers) = follower_map.get(&p_id) {
                                    if process_propagation(&pw, followers, &mut state_info) {
                                        changed = true;
                                    }
                                }
                            }
                        } else {
                            panic!("Unexpected negative label during label propagation: {}", l);
                        }
                    }
                    Err(_) => panic!("Unexpected unknown non-default label during label propagation: {}", l),
                }

                if changed && in_worklist.insert(v) {
                    worklist.push_back(v);
                }
            }
        }
    }
    crate::debug!(5, "Label propagation fixpoint took: {:?}", now.elapsed());

    let now_prune = std::time::Instant::now();
    let mut changed_count = 0;
    let start_states_set: BTreeSet<StateID> = nwa.body.start_states.iter().cloned().collect();

    for u in 0..nwa.states.len() {
        if initial_states.contains(&u) || start_states_set.contains(&u) {
            continue;
        }

        let info_at_u = &state_info[u];
        let state = &mut nwa.states[u];

        for (l, targets) in state.transitions.iter_mut() {
            let valid_incoming_weight = match decode_symbol_i16(*l) {
                Ok((is_pos, p_id)) => {
                    if is_pos {
                        info_at_u.get(&p_id).cloned().unwrap_or_else(Weight::zeros)
                    } else {
                        panic!("Unexpected negative label during pruning: {}", l);
                    }
                }
                Err(_) => panic!("Unexpected unknown non-default label during pruning: {}", l),
            };

            if valid_incoming_weight.is_empty() {
                for (_, w_uv) in targets.iter_mut() {
                    if !w_uv.is_empty() {
                        changed_count += 1;
                        *w_uv = Weight::zeros();
                    }
                }
            } else {
                for (_, w_uv) in targets.iter_mut() {
                    let old_w = w_uv.clone();
                    *w_uv &= &valid_incoming_weight;
                    if *w_uv != old_w {
                        changed_count += 1;
                    }
                }
            }
        }
    }

    crate::debug!(6, "state_info after propagation:");
    for (i, info) in state_info.iter().enumerate() {
        crate::debug!(6, "  State {}: {:?}", i, info);
    }

    for state in &mut nwa.states.0 {
        for targets in state.transitions.values_mut() {
            targets.retain(|(_, w)| !w.is_empty());
        }
        state.transitions.retain(|_, v| !v.is_empty());
    }
    crate::debug!(5, "Pruning pass changed {} weights and took: {:?}", changed_count, now_prune.elapsed());
}
