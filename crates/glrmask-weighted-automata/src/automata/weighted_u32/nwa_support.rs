//! Exact path-support restriction for acyclic weighted NFAs, including epsilon edges.
//!
//! This is the epsilon-capable counterpart of the ordinary acyclic minimizer's
//! backward weight push. It makes no token-determinism assumption and performs
//! no state merging. All weights remain in their original correlated domain.

use std::time::Instant;

use super::nwa::NWA;
use crate::ds::weight::{ScopedWeightOpCache, Weight};

#[derive(Debug, Default)]
pub struct NwaSupportProfile {
    pub states: usize,
    pub edges_before: usize,
    pub edges_after: usize,
    pub backward_ms: f64,
    pub forward_ms: f64,
}

/// Restrict edges to coordinates occurring on accepting suffixes. Optionally
/// also restrict them to coordinates reachable from the actual start states.
///
/// B(u) = final(u) union union_(u,w,v) [w intersection B(v)]. Replacing w by
/// w intersection B(v) preserves the weight of EVERY accepting path, because
/// that path's complete weight is already contained in its suffix support.
/// Epsilon edges obey exactly the same recurrence. A subsequent forward pass
/// similarly intersects edges/finals with their source's reachable support.
/// It preserves the whole language from the given starts, not residual
/// languages outside those reachable domains. Both passes only delete weight
/// membership, so they cannot introduce an accepted coordinate or word.
///
/// Invalid/cyclic inputs return Err BEFORE any mutation. A successful result
/// is exact, with no time-budget truncation or approximation.
pub fn restrict_acyclic_nwa_support(
    nwa: &mut NWA, restrict_forward: bool,
) -> Result<NwaSupportProfile, String> {
    let n = nwa.states().len();
    let starts = nwa.start_states().to_vec();
    if starts.iter().any(|&s| s as usize >= n) {
        return Err("NWA support restriction has an invalid start state".into());
    }
    let mut indegree = vec![0usize; n];
    for state in nwa.states() {
        for (target, _) in state.epsilons.iter().chain(state.transitions.values().flatten()) {
            let Some(degree) = indegree.get_mut(*target as usize) else {
                return Err("NWA support restriction has an invalid target".into());
            };
            *degree += 1;
        }
    }
    let mut topo = (0..n).filter(|&s| indegree[s] == 0).collect::<Vec<_>>();
    let mut head = 0;
    while head < topo.len() {
        let source = topo[head]; head += 1;
        let state = &nwa.states()[source];
        for (target, _) in state.epsilons.iter().chain(state.transitions.values().flatten()) {
            let target = *target as usize;
            indegree[target] -= 1;
            if indegree[target] == 0 { topo.push(target); }
        }
    }
    if topo.len() != n { return Err("NWA support restriction requires an acyclic graph".into()); }
    let mut profile = NwaSupportProfile { states: n, edges_before: nwa.num_transitions(), ..Default::default() };
    let mut ops = ScopedWeightOpCache::default();
    let mut backward = vec![Weight::empty(); n];
    let started = Instant::now();
    {
        let states = nwa.states_mut();
        for &source in topo.iter().rev() {
            let state = &mut states[source];
            let mut parts = Vec::new();
            if let Some(final_weight) = &state.final_weight {
                if !final_weight.is_empty() { parts.push(final_weight.clone()); }
            }
            for (target, weight) in state.epsilons.iter_mut().chain(state.transitions.values_mut().flatten()) {
                *weight = ops.intersection(weight, &backward[*target as usize]);
                if !weight.is_empty() { parts.push(weight.clone()); }
            }
            state.epsilons.retain(|(_, w)| !w.is_empty());
            for targets in state.transitions.values_mut() { targets.retain(|(_, w)| !w.is_empty()); }
            state.transitions.retain(|_, targets| !targets.is_empty());
            backward[source] = ops.union_all(parts.iter());
        }
    }
    profile.backward_ms = started.elapsed().as_secs_f64() * 1000.0;
    if restrict_forward {
        let started = Instant::now();
        let mut forward = vec![Weight::empty(); n];
        for start in starts { forward[start as usize] = Weight::all(); }
        let states = nwa.states_mut();
        for source in topo {
            let state = &mut states[source];
            let incoming = forward[source].clone();
            if incoming.is_empty() {
                state.transitions.clear(); state.epsilons.clear(); state.final_weight = None;
                continue;
            }
            if let Some(final_weight) = &mut state.final_weight {
                *final_weight = ops.intersection(final_weight, &incoming);
            }
            for (target, weight) in state.epsilons.iter_mut().chain(state.transitions.values_mut().flatten()) {
                *weight = ops.intersection(weight, &incoming);
                let target = *target as usize;
                forward[target] = ops.union(&forward[target], weight);
            }
            state.epsilons.retain(|(_, w)| !w.is_empty());
            for targets in state.transitions.values_mut() { targets.retain(|(_, w)| !w.is_empty()); }
            state.transitions.retain(|_, targets| !targets.is_empty());
        }
        profile.forward_ms = started.elapsed().as_secs_f64() * 1000.0;
    }
    profile.edges_after = nwa.num_transitions();
    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;
    use crate::weighted_u32::{determinize::determinize, equivalence::find_difference};

    fn token(id: u32) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([id..=id]))
    }

    #[test]
    fn nwa_support_preserves_generated_weighted_languages() {
        let weights = [Weight::empty(), Weight::all(), token(0), token(1), token(0).union(&token(1))];
        let mut rng = 113u64;
        let mut next = || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1); (rng >> 32) as usize };
        for case in 0..128 {
            let n = 4 + next() % 6;
            let mut original = NWA::new(1, 2);
            for _ in 0..n { original.add_state(); }
            original.set_start_states(if case % 3 == 0 { vec![0, 1] } else { vec![0] });
            for source in 0..n {
                if next() % 3 == 0 { original.set_final_weight(source as u32, weights[next() % weights.len()].clone()); }
                for target in source + 1..n {
                    let kind = next() % 7;
                    let weight = weights[next() % weights.len()].clone();
                    if kind == 0 { original.add_epsilon(source as u32, target as u32, weight); }
                    else if kind <= 3 { original.add_transition(source as u32, kind as i32, target as u32, weight); }
                }
            }
            let expected = determinize(&original).unwrap();
            for forward in [false, true] {
                let mut candidate = original.clone();
                restrict_acyclic_nwa_support(&mut candidate, forward).unwrap();
                assert_eq!(find_difference(&expected, &determinize(&candidate).unwrap()).unwrap(), None,
                    "case {case}, forward {forward}");
            }
        }
    }

    #[test]
    fn nwa_support_removes_contradictory_paths_and_declines_cycles() {
        let mut nwa = NWA::new(1, 2); for _ in 0..3 { nwa.add_state(); }
        nwa.set_start_states(vec![0]); nwa.add_epsilon(0, 1, token(0));
        nwa.add_transition(1, 7, 2, token(1)); nwa.set_final_weight(2, Weight::all());
        let profile = restrict_acyclic_nwa_support(&mut nwa, true).unwrap();
        assert_eq!(profile.edges_after, 0);
        let mut cycle = NWA::new(1, 1); cycle.add_state(); cycle.set_start_states(vec![0]);
        cycle.add_epsilon(0, 0, Weight::all());
        assert!(restrict_acyclic_nwa_support(&mut cycle, true).is_err());
        assert_eq!(cycle.states()[0].epsilons.len(), 1, "decline must not mutate graph");
    }
}
