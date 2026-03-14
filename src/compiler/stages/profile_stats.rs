use std::collections::BTreeSet;
use std::fmt;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::weighted_u32::dwa::DWA as WeightedDwa;
use crate::automata::weighted_u32::nwa::NWA as WeightedNwa;

#[derive(Debug, Clone, Default)]
pub(crate) struct WeightedNwaStats {
    pub states: usize,
    pub start_states: usize,
    pub final_states: usize,
    pub epsilon_edges: usize,
    pub labeled_edges: usize,
    pub transitions: usize,
}

impl fmt::Display for WeightedNwaStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "nwa_states={}", self.states)
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct WeightedDwaStats {
    pub states: usize,
    pub final_states: usize,
    pub transitions: usize,
    pub state_pairs: usize,
    pub transitions_to_final: usize,
    pub transitions_to_nonfinal: usize,
    pub pairs_to_final: usize,
    pub pairs_to_nonfinal: usize,
}

impl fmt::Display for WeightedDwaStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dwa_states={} dwa_transitions={} state_pairs={} trans_to_leaf={} trans_to_nonleaf={} pairs_to_leaf={} pairs_to_nonleaf={}",
            self.states, self.transitions, self.state_pairs,
            self.transitions_to_final, self.transitions_to_nonfinal,
            self.pairs_to_final, self.pairs_to_nonfinal,
        )
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct UnweightedDfaStats {
    pub states: usize,
    pub final_states: usize,
    pub transitions: usize,
}

pub(crate) fn collect_weighted_nwa_stats(nwa: &WeightedNwa) -> WeightedNwaStats {
    let mut stats = WeightedNwaStats {
        states: nwa.states.len(),
        start_states: nwa.start_states.len(),
        ..WeightedNwaStats::default()
    };

    for state in &nwa.states {
        if state
            .final_weight
            .as_ref()
            .map(|weight| !weight.is_empty())
            .unwrap_or(false)
        {
            stats.final_states += 1;
        }
        stats.epsilon_edges += state.epsilons.len();
        stats.labeled_edges += state.transitions.values().map(Vec::len).sum::<usize>();
    }
    stats.transitions = stats.epsilon_edges + stats.labeled_edges;
    stats
}

pub(crate) fn collect_weighted_dwa_stats(dwa: &WeightedDwa) -> WeightedDwaStats {
    let mut stats = WeightedDwaStats {
        states: dwa.states.len(),
        ..WeightedDwaStats::default()
    };

    let mut all_pairs: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut pairs_to_final: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut pairs_to_nonfinal: BTreeSet<(u32, u32)> = BTreeSet::new();

    for (src_idx, state) in dwa.states.iter().enumerate() {
        let src = src_idx as u32;
        if state
            .final_weight
            .as_ref()
            .map(|weight| !weight.is_empty())
            .unwrap_or(false)
        {
            stats.final_states += 1;
        }
        stats.transitions += state.transitions.len();
        for &(target, _) in state.transitions.values() {
            all_pairs.insert((src, target));
            let target_state = &dwa.states[target as usize];
            let target_has_final = target_state
                .final_weight
                .as_ref()
                .map(|w| !w.is_empty())
                .unwrap_or(false);
            let target_is_leaf = target_has_final && target_state.transitions.is_empty();
            if target_is_leaf {
                stats.transitions_to_final += 1;
                pairs_to_final.insert((src, target));
            } else {
                stats.transitions_to_nonfinal += 1;
                pairs_to_nonfinal.insert((src, target));
            }
        }
    }
    stats.state_pairs = all_pairs.len();
    stats.pairs_to_final = pairs_to_final.len();
    stats.pairs_to_nonfinal = pairs_to_nonfinal.len();
    stats
}

pub(crate) fn collect_unweighted_dfa_stats(dfa: &UnweightedDfa) -> UnweightedDfaStats {
    let mut stats = UnweightedDfaStats {
        states: dfa.states.len(),
        ..UnweightedDfaStats::default()
    };

    for state in &dfa.states {
        if state.is_accepting {
            stats.final_states += 1;
        }
        stats.transitions += state.transitions.len();
    }
    stats
}
