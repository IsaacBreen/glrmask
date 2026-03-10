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

#[derive(Debug, Clone, Default)]
pub(crate) struct WeightedDwaStats {
    pub states: usize,
    pub final_states: usize,
    pub transitions: usize,
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

    for state in &dwa.states {
        if state
            .final_weight
            .as_ref()
            .map(|weight| !weight.is_empty())
            .unwrap_or(false)
        {
            stats.final_states += 1;
        }
        stats.transitions += state.transitions.len();
    }
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
