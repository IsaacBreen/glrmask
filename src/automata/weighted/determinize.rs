//! Acyclic NWA → CompDwa determinization.
//!
//! Converts a [`Nwa`] (assumed acyclic) into a [`CompDwa`] via weighted
//! subset construction.  The algorithm proceeds in two phases:
//!
//! 1. **Unweighted exploration** – discover the DWA state space using
//!    unweighted ε-closures and subset construction.
//! 2. **Weight computation** – for each discovered DWA state compute
//!    weighted ε-closures and assign transition / final weights.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use rustc_hash::FxHashMap;

use super::dwa::{CompDwa, CompDwaState};
use super::nwa::{Label, Nwa};
use super::weight::Weight;
use crate::GlrMaskError;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Determinize an acyclic NWA into a compilation-time DWA.
///
/// Returns an error if the NWA contains cycles.
pub fn determinize_acyclic(nwa: &Nwa) -> Result<CompDwa, GlrMaskError> {
    let n = nwa.states.len();
    if n == 0 {
        return Ok(CompDwa::new(nwa.num_tsids, nwa.max_token));
    }

    // 1. Topological sort.
    let topo = topo_sort(nwa)?;

    // 2. Unweighted ε-closures.
    let eps_uw = unweighted_epsilon_closures(nwa, &topo);

    // 3. Unweighted subset construction – discover DWA structure.
    let (subsets, uw_transitions) = unweighted_subset_construction(nwa, &eps_uw);

    // 4. Weighted ε-closures (maps NWA state → [(nwa_state, weight)]).
    let eps_w = weighted_epsilon_closures(nwa, &topo);

    // 5. Build CompDwa with weights.
    build_comp_dwa(nwa, &subsets, &uw_transitions, &eps_w)
}

// ---------------------------------------------------------------------------
// Topological sort  (Kahn's algorithm)
// ---------------------------------------------------------------------------

fn topo_sort(nwa: &Nwa) -> Result<Vec<u32>, GlrMaskError> {
    let n = nwa.states.len();
    let mut indegree = vec![0u32; n];

    for st in &nwa.states {
        for (t, _) in &st.epsilons {
            indegree[*t as usize] += 1;
        }
        for targets in st.transitions.values() {
            for (t, _) in targets {
                indegree[*t as usize] += 1;
            }
        }
    }

    let mut queue: VecDeque<u32> = indegree
        .iter()
        .enumerate()
        .filter(|&(_, d)| *d == 0)
        .map(|(i, _)| i as u32)
        .collect();

    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        let st = &nwa.states[u as usize];
        for (v, _) in &st.epsilons {
            let d = &mut indegree[*v as usize];
            *d -= 1;
            if *d == 0 {
                queue.push_back(*v);
            }
        }
        for targets in st.transitions.values() {
            for (v, _) in targets {
                let d = &mut indegree[*v as usize];
                *d -= 1;
                if *d == 0 {
                    queue.push_back(*v);
                }
            }
        }
    }

    if order.len() != n {
        return Err(GlrMaskError::Compilation(
            "NWA contains a cycle; only acyclic NWAs are supported".into(),
        ));
    }
    Ok(order)
}

// ---------------------------------------------------------------------------
// Unweighted ε-closures
// ---------------------------------------------------------------------------

/// For each NWA state, compute the set of states reachable via ε-transitions.
fn unweighted_epsilon_closures(nwa: &Nwa, topo: &[u32]) -> Vec<BTreeSet<u32>> {
    let n = nwa.states.len();
    let mut closures: Vec<BTreeSet<u32>> = (0..n)
        .map(|i| {
            let mut s = BTreeSet::new();
            s.insert(i as u32);
            s
        })
        .collect();

    // Process in reverse topo order: when we process u, all targets of
    // u's ε-transitions already have complete closures.
    for &u in topo.iter().rev() {
        let targets: Vec<u32> = nwa.states[u as usize]
            .epsilons
            .iter()
            .map(|(t, _)| *t)
            .collect();
        for t in targets {
            let ext: Vec<u32> = closures[t as usize].iter().copied().collect();
            closures[u as usize].extend(ext);
        }
    }

    closures
}

// ---------------------------------------------------------------------------
// Unweighted subset construction
// ---------------------------------------------------------------------------

/// Explore the DWA state space without weights.
///
/// Returns:
/// - `subsets[dwa_id]` = set of NWA states forming that DWA state.
/// - `transitions[dwa_id]` = vec of (label, target_dwa_id).
fn unweighted_subset_construction(
    nwa: &Nwa,
    eps_uw: &[BTreeSet<u32>],
) -> (Vec<BTreeSet<u32>>, Vec<Vec<(Label, u32)>>) {
    let mut subsets: Vec<BTreeSet<u32>> = Vec::new();
    let mut transitions: Vec<Vec<(Label, u32)>> = Vec::new();
    let mut seen: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut queue: VecDeque<u32> = VecDeque::new();

    // Build start subset = ε-closure of all start states.
    let mut start_set = BTreeSet::new();
    for &s in &nwa.start_states {
        start_set.extend(eps_uw[s as usize].iter().copied());
    }

    let _start_id =
        intern_subset(&start_set, &mut subsets, &mut transitions, &mut seen, &mut queue);

    while let Some(sid) = queue.pop_front() {
        let subset = subsets[sid as usize].clone();

        // Gather all labels reachable from this subset.
        let mut by_label: BTreeMap<Label, BTreeSet<u32>> = BTreeMap::new();
        for &u in &subset {
            for (label, targets) in &nwa.states[u as usize].transitions {
                let entry = by_label.entry(*label).or_default();
                for (v, _) in targets {
                    entry.extend(eps_uw[*v as usize].iter().copied());
                }
            }
        }

        let mut trans = Vec::new();
        for (label, target_set) in by_label {
            if target_set.is_empty() {
                continue;
            }
            let tid = intern_subset(
                &target_set,
                &mut subsets,
                &mut transitions,
                &mut seen,
                &mut queue,
            );
            trans.push((label, tid));
        }
        transitions[sid as usize] = trans;
    }

    (subsets, transitions)
}

/// Intern a subset: if already seen return its id, otherwise register it.
fn intern_subset(
    subset: &BTreeSet<u32>,
    subsets: &mut Vec<BTreeSet<u32>>,
    transitions: &mut Vec<Vec<(Label, u32)>>,
    seen: &mut FxHashMap<Vec<u32>, u32>,
    queue: &mut VecDeque<u32>,
) -> u32 {
    // Use Vec<u32> as key for cheaper hashing than BTreeSet.
    let key: Vec<u32> = subset.iter().copied().collect();
    if let Some(&id) = seen.get(&key) {
        return id;
    }
    let id = subsets.len() as u32;
    seen.insert(key, id);
    subsets.push(subset.clone());
    transitions.push(Vec::new());
    queue.push_back(id);
    id
}

// ---------------------------------------------------------------------------
// Weighted ε-closures
// ---------------------------------------------------------------------------

/// For each NWA state `u`, compute:
///   closure[u] = { (v, w) | v reachable from u via ε, w = ∩ of edge-weights }
///
/// Multiple paths to the same state v are combined with ∪.
fn weighted_epsilon_closures(nwa: &Nwa, topo: &[u32]) -> Vec<BTreeMap<u32, Weight>> {
    let n = nwa.states.len();
    let nt = nwa.num_tsids;
    let max_pos = nwa.max_position();

    let mut closures: Vec<BTreeMap<u32, Weight>> = (0..n)
        .map(|i| {
            let mut m = BTreeMap::new();
            m.insert(i as u32, Weight::all(max_pos, nt));
            m
        })
        .collect();

    for &u in topo.iter().rev() {
        // Snapshot ε-targets to avoid borrow issues.
        let eps: Vec<(u32, Weight)> = nwa.states[u as usize].epsilons.clone();
        for (t, w_eps) in &eps {
            // For each (v, w_v) in closure[t], add (v, w_eps ∩ w_v) to closure[u].
            let t_entries: Vec<(u32, Weight)> = closures[*t as usize]
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .collect();
            for (v, w_v) in t_entries {
                let combined = w_eps.intersection(&w_v);
                if combined.is_empty() {
                    continue;
                }
                closures[u as usize]
                    .entry(v)
                    .and_modify(|existing| *existing = existing.union(&combined))
                    .or_insert(combined);
            }
        }
    }

    closures
}

// ---------------------------------------------------------------------------
// Build CompDwa with weights
// ---------------------------------------------------------------------------

fn build_comp_dwa(
    nwa: &Nwa,
    subsets: &[BTreeSet<u32>],
    uw_transitions: &[Vec<(Label, u32)>],
    eps_w: &[BTreeMap<u32, Weight>],
) -> Result<CompDwa, GlrMaskError> {
    let nt = nwa.num_tsids;
    let max_tok = nwa.max_token;

    let num_dwa = subsets.len();
    let mut states: Vec<CompDwaState> = (0..num_dwa)
        .map(|_| CompDwaState::default())
        .collect();

    for (sid, subset) in subsets.iter().enumerate() {
        // Merge the weighted closures for all NWA states in this DWA state.
        let merged = merge_weighted_closures(subset, eps_w);

        // --- Final weight ---
        let mut final_w: Option<Weight> = None;
        for (nwa_st, w_acc) in &merged {
            if let Some(fw) = &nwa.states[*nwa_st as usize].final_weight {
                let c = w_acc.intersection(fw);
                if !c.is_empty() {
                    final_w = Some(match final_w {
                        Some(e) => e.union(&c),
                        None => c,
                    });
                }
            }
        }
        states[sid].final_weight = final_w;

        // --- Transition weights ---
        for &(label, target_dwa) in &uw_transitions[sid] {
            let mut tw: Option<Weight> = None;
            for (nwa_u, w_u) in &merged {
                if let Some(nwa_targets) = nwa.states[*nwa_u as usize].transitions.get(&label) {
                    for (nwa_v, w_trans) in nwa_targets {
                        // The transition weight contribution: w_u ∩ w_trans.
                        //
                        // We *could* further intersect with the target DWA
                        // state's weighted closure from nwa_v, but for
                        // standard NWA semantics the transition weight
                        // captures the source-side filtering; the target
                        // state's closure will be applied when that state
                        // is entered.
                        let _ = nwa_v; // target id used for routing, not weight calc
                        let c = w_u.intersection(w_trans);
                        if !c.is_empty() {
                            tw = Some(match tw {
                                Some(e) => e.union(&c),
                                None => c,
                            });
                        }
                    }
                }
            }
            if let Some(w) = tw {
                states[sid]
                    .transitions
                    .insert(label, (target_dwa, w));
            }
        }
    }

    Ok(CompDwa {
        states,
        start_state: 0,
        num_tsids: nt,
        max_token: max_tok,
    })
}

/// Merge weighted ε-closures for all NWA states in a subset.
fn merge_weighted_closures(
    subset: &BTreeSet<u32>,
    eps_w: &[BTreeMap<u32, Weight>],
) -> BTreeMap<u32, Weight> {
    let mut merged: BTreeMap<u32, Weight> = BTreeMap::new();
    for &u in subset {
        for (v, w) in &eps_w[u as usize] {
            merged
                .entry(*v)
                .and_modify(|e| *e = e.union(w))
                .or_insert_with(|| w.clone());
        }
    }
    // Drop empty weights (shouldn't happen but be safe).
    merged.retain(|_, w| !w.is_empty());
    merged
}

// ====================================================================
// Tests
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ds::RangeSet;

    #[test]
    fn test_determinize_trivial_accepting() {
        // Single-state accepting NWA → single-state accepting DWA.
        let mut nwa = Nwa::new(1, 5);
        let s = nwa.add_state();
        nwa.start_states.push(s);
        nwa.set_final_weight(s, Weight::all(5, 1));

        let dwa = determinize_acyclic(&nwa).unwrap();
        assert_eq!(dwa.num_states(), 1);
        assert!(dwa.states[0].final_weight.is_some());
    }

    #[test]
    fn test_determinize_linear() {
        // s0 --label 0--> s1 (accepting)
        let nt = 1u32;
        let max_tok = 5u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        nwa.start_states.push(s0);

        let w_all = Weight::all(nwa.max_position(), nt);
        nwa.add_transition(s0, 0, s1, w_all.clone());
        nwa.set_final_weight(s1, w_all);

        let dwa = determinize_acyclic(&nwa).unwrap();
        assert_eq!(dwa.num_states(), 2);
        assert!(dwa.states[0].final_weight.is_none());
        assert!(dwa.states[1].final_weight.is_some());

        // eval_word([0]) should be non-empty
        assert!(!dwa.eval_word(&[0]).is_empty());
        // eval_word([1]) should be empty (no transition for label 1)
        assert!(dwa.eval_word(&[1]).is_empty());
    }

    #[test]
    fn test_determinize_nondeterminism() {
        // Two transitions on the same label with disjoint weights.
        // s0 --0,w1--> s1 (accepting)
        // s0 --0,w2--> s2 (accepting)
        let nt = 1u32;
        let max_tok = 5u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.start_states.push(s0);

        let w1 = Weight::from_positions(&RangeSet::from_range(0, 2), nt);
        let w2 = Weight::from_positions(&RangeSet::from_range(3, 5), nt);
        nwa.add_transition(s0, 0, s1, w1);
        nwa.add_transition(s0, 0, s2, w2);
        nwa.set_final_weight(s1, Weight::all(nwa.max_position(), nt));
        nwa.set_final_weight(s2, Weight::all(nwa.max_position(), nt));

        let dwa = determinize_acyclic(&nwa).unwrap();
        let result = dwa.eval_word(&[0]);
        assert!(!result.is_empty());
        // Should see positions from both w1 and w2 (0..=5 → 6 positions).
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_determinize_epsilon() {
        // s0 --ε--> s1 --label 0--> s2 (accepting)
        let nt = 1u32;
        let max_tok = 5u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.start_states.push(s0);

        let w_all = Weight::all(nwa.max_position(), nt);
        nwa.add_epsilon(s0, s1, w_all.clone());
        nwa.add_transition(s1, 0, s2, w_all.clone());
        nwa.set_final_weight(s2, w_all);

        let dwa = determinize_acyclic(&nwa).unwrap();
        assert!(!dwa.eval_word(&[0]).is_empty());
    }

    #[test]
    fn test_determinize_cycle_rejected() {
        let mut nwa = Nwa::new(1, 5);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        nwa.start_states.push(s0);
        let w = Weight::all(5, 1);
        nwa.add_epsilon(s0, s1, w.clone());
        nwa.add_epsilon(s1, s0, w);

        assert!(determinize_acyclic(&nwa).is_err());
    }

    #[test]
    fn test_determinize_empty_nwa() {
        let nwa = Nwa::new(1, 5);
        let dwa = determinize_acyclic(&nwa).unwrap();
        // CompDwa::new creates a single dead start state.
        assert_eq!(dwa.num_states(), 1);
        assert!(dwa.states[0].final_weight.is_none());
    }

    #[test]
    fn test_determinize_no_start_states() {
        // NWA with states but no start states → start subset = ∅ → 1 dead DWA state.
        let mut nwa = Nwa::new(1, 5);
        let s0 = nwa.add_state();
        nwa.set_final_weight(s0, Weight::all(5, 1));
        // No start_states pushed.
        let dwa = determinize_acyclic(&nwa).unwrap();
        assert_eq!(dwa.num_states(), 1);
        assert!(dwa.states[0].final_weight.is_none());
    }

    #[test]
    fn test_determinize_chain_with_epsilon() {
        // s0 --0,w_all--> s1 --ε,w_all--> s2 --1,w_all--> s3 (accepting)
        let nt = 1u32;
        let max_tok = 5u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        let s3 = nwa.add_state();
        nwa.start_states.push(s0);

        let w_all = Weight::all(nwa.max_position(), nt);
        nwa.add_transition(s0, 0, s1, w_all.clone());
        nwa.add_epsilon(s1, s2, w_all.clone());
        nwa.add_transition(s2, 1, s3, w_all.clone());
        nwa.set_final_weight(s3, w_all);

        let dwa = determinize_acyclic(&nwa).unwrap();
        // Word [0, 1] should reach the accepting state.
        assert!(!dwa.eval_word(&[0, 1]).is_empty());
        // Word [0] alone should NOT be accepting.
        assert!(dwa.eval_word(&[0]).is_empty());
        // Word [1] alone should NOT have a transition from start.
        assert!(dwa.eval_word(&[1]).is_empty());
    }

    #[test]
    fn test_determinize_weight_filtering() {
        // s0 --0,w_small--> s1 (accepting with w_all)
        // Only positions in w_small should survive.
        let nt = 1u32;
        let max_tok = 10u32;
        let mut nwa = Nwa::new(nt, max_tok);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        nwa.start_states.push(s0);

        let w_small = Weight::from_positions(&RangeSet::from_range(2, 5), nt);
        let w_all = Weight::all(nwa.max_position(), nt);
        nwa.add_transition(s0, 0, s1, w_small);
        nwa.set_final_weight(s1, w_all);

        let dwa = determinize_acyclic(&nwa).unwrap();
        let result = dwa.eval_word(&[0]);
        assert!(!result.is_empty());
        // Only positions 2..=5 survive the intersection.
        assert_eq!(result.len(), 4);
    }
}
