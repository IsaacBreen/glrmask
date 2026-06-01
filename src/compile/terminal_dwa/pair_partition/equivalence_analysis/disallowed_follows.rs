use std::collections::BTreeMap;

use crate::automata::unweighted::dfa::{DFA, Label};
use crate::sets::bitset::BitSet;

pub(crate) fn normalize_disallowed_follows(
    num_groups: usize,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Vec<BitSet> {
    let mut normalized = vec![BitSet::new(num_groups); num_groups];
    for gid in 0..num_groups {
        if let Some(bits) = disallowed_follows.get(&(gid as u32)) {
            let mut out = BitSet::new(num_groups);
            for bit in bits.iter() {
                if bit < num_groups {
                    out.set(bit);
                }
            }
            normalized[gid] = out;
        }
    }
    normalized
}

pub(crate) fn build_disallowed_follow_dfa(disallowed_follows: &[BitSet]) -> DFA {
    let num_groups = disallowed_follows.len();
    if num_groups == 0 {
        return DFA::new();
    }

    let mut dfa = DFA::new();
    let start = dfa.start_state;
    let accept = dfa.add_state();
    dfa.set_accepting(accept, true);

    let mut previous_terminal_states = Vec::with_capacity(num_groups);
    for _ in 0..num_groups {
        previous_terminal_states.push(dfa.add_state());
    }

    for prev_gid in 0..num_groups {
        let prev_state = previous_terminal_states[prev_gid];
        dfa.add_transition(start, prev_gid as Label, prev_state);

        for next_gid in 0..num_groups {
            let target = if disallowed_follows[prev_gid].contains(next_gid) {
                accept
            } else {
                previous_terminal_states[next_gid]
            };
            dfa.add_transition(prev_state, next_gid as Label, target);
        }
    }

    for gid in 0..num_groups {
        dfa.add_transition(accept, gid as Label, accept);
    }

    dfa
}