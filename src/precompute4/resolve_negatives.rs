use crate::precompute4::weighted_automata::{DWA, NWA, NWAStateID, NWAStates, Weight};
use crate::profiler::PROGRESS_BAR_ENABLED;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, VecDeque};

/// Resolve negative codes in a DWA by a single, high-performance, semantics-preserving NWA rewrite:
/// - Convert DWA -> NWA (preserves default transitions and weights).
/// - For every negative-labeled edge A --neg(p,w_neg)--> B:
///     1) Compute ε-closure of B (weighted), and for every t in closure:
///        if t has a transition on p (or default), say t --p,w_p--> C,
///        add ε-edge A --ε, (w_neg ∧ w(B⇒t) ∧ w_p)--> C.  (Cancellation of neg(p) with p across ε)
///     2) We will later split B (per incoming neg-edge) to a canonical "negative-only" copy B⁻ (no default, no positive-labeled edges, no final),
///        and retarget A --neg(p)--> B⁻. We share one B⁻ per original state B.
/// - After inserting all cancellation ε-edges, compute the least fixpoint of "final weights" propagated
///   along ε-edges and negative-labeled edges backward (union over paths with ∧ along edges).
///   This collapses all multi-pass determinization effects into one pass.
/// - Split/retarget negative edges using the updated final closure and ε-closure shape:
///     For A --neg(p)--> B, if B is final under the computed closure, or if B's ε-closure provides any
///     positive/default transitions, retarget to B⁻; otherwise keep it.
/// - Finally determinize back to DWA and simplify.
///
/// This construction is equivalent to the previous iterative algorithm with interleaved determinization passes,
/// but runs in near-linear time in the size of the NWA subgraph involved in negative-code resolution.
pub fn resolve_negative_codes_in_dwa(dwa: &mut DWA) {
    let pb = if PROGRESS_BAR_ENABLED {
        let p = ProgressBar::new(4);
        p.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [Resolving negative codes: {elapsed_precise}] [{wide_bar:.cyan/blue}] step {pos}/{len} ({msg})")
                .expect("progress-bar"),
        );
        Some(p)
    } else {
        None
    };

    // 0) Convert to NWA for easier local rewrites
    if let Some(p) = &pb { p.set_message("DWA -> NWA"); p.set_position(1); }
    let mut nwa = NWA::from_dwa(dwa);

    // 1) Scan negative edges once, compute ε-cancellations via ε-closure, cache per-target results.
    if let Some(p) = &pb { p.set_message("Scan negatives, build ε-cancellations"); p.set_position(2); }
    let (neg_edges, eps_cancellations, b_has_pos_or_default_in_eps) = compute_cancellations_and_shape(&nwa.states);

    // Add all created ε-cancellation edges in one batch.
    for (from, to, w) in eps_cancellations {
        if !w.is_empty() {
            nwa.states.add_epsilon(from, to, w);
        }
    }

    // 2) Compute final-weight fixpoint along ε-edges and negative-labeled edges (no labeled positives/defaults).
    if let Some(p) = &pb { p.set_message("Compute final-weight fixpoint (ε + neg)"); p.set_position(3); }
    let final_fix = compute_final_fixpoint_eps_and_neg(&nwa.states, &neg_edges);

    // Install updated finals
    for sid in 0..nwa.states.len() {
        let f = &final_fix[sid];
        if f.is_empty() {
            nwa.states.0[sid].final_weight = None;
        } else {
            nwa.states.0[sid].final_weight = Some(f.clone());
        }
    }

    // 3) Split/retarget: for each negative edge A --neg(p)--> B,
    // if B is "problematic" (final under fixpoint OR has positive/default in its ε-closure),
    // retarget to a canonical negative-only copy B⁻ that strips positive/default/final.
    if let Some(p) = &pb { p.set_message("Split/retarget negatives to negative-only copies"); p.set_position(4); }
    retarget_negatives_to_negative_only(&mut nwa.states, &neg_edges, &final_fix, &b_has_pos_or_default_in_eps);

    // 4) Determinize back to DWA and simplify
    let mut result = nwa.determinize_to_dwa();
    result.simplify();
    *dwa = result;

    if let Some(p) = &pb {
        p.finish_with_message("Done");
    }
}

/// A negative-labeled edge (from, neg_label, to, weight).
#[derive(Clone)]
struct NegEdge {
    from: NWAStateID,
    neg_label: i16, // < 0
    to: NWAStateID,
    weight: Weight,
}

/// Compute all negative edges, ε-cancellation edges to add, and a per-target flag
/// "does ε-closure(target) expose any positive/default transitions".
fn compute_cancellations_and_shape(
    states: &NWAStates,
) -> (Vec<NegEdge>, Vec<(NWAStateID, NWAStateID, Weight)>, Vec<bool>) {
    let n = states.len();

    // 1) Collect all negative edges once.
    let mut neg_edges: Vec<NegEdge> = Vec::new();
    for from in 0..n {
        let st = &states.0[from];
        for (lbl, (to, w)) in &st.transitions {
            if *lbl < 0 {
                neg_edges.push(NegEdge {
                    from,
                    neg_label: *lbl,
                    to: *to,
                    weight: w.clone(),
                });
            }
        }
    }

    // 2) For each unique target B of a negative edge, compute ε-closure(B) and memoize.
    // Also decide whether ε-closure(B) exposes any positive/default transitions.
    let mut eps_closure_cache: HashMap<NWAStateID, Vec<(NWAStateID, Weight)>> = HashMap::new();
    let mut has_pos_or_default_in_eps: Vec<bool> = vec![false; n];

    let mut eps_cancellations: Vec<(NWAStateID, NWAStateID, Weight)> = Vec::new();

    for edge in &neg_edges {
        let b = edge.to;
        let p = edge.neg_label.wrapping_sub(i16::MIN); // positive counterpart

        // ε-closure of B (weighted), with memoization
        let closure = eps_closure_cache.entry(b).or_insert_with(|| compute_eps_closure(states, b));

        // Compute cancellation edges: For each t in ε-closure(B), if t has p (or default), add ε A -> target with the gated weight.
        let mut closure_exposes_pos_or_default = false;
        for (t, w_bt) in closure.iter() {
            // Check t's positive/default availability
            let t_state = &states[*t];
            let (maybe_to, maybe_w) = if let Some((to2, w2)) = t_state.transitions.get(&p) {
                (Some(*to2), Some(w2))
            } else if let Some((to_def, w_def)) = &t_state.default {
                (Some(*to_def), Some(w_def))
            } else {
                (None, None)
            };
            if maybe_to.is_some() {
                closure_exposes_pos_or_default = true;
            }

            // Cancellation epsilon addition
            if let (Some(to_target), Some(w_p)) = (maybe_to, maybe_w) {
                let w = (&edge.weight & w_bt) & w_p;
                if !w.is_empty() {
                    eps_cancellations.push((edge.from, to_target, w));
                }
            }
        }

        if closure_exposes_pos_or_default {
            has_pos_or_default_in_eps[b] = true;
        }
    }

    (neg_edges, eps_cancellations, has_pos_or_default_in_eps)
}

/// Compute weighted ε-closure from a start state:
/// Return Vec of (target, weight), containing at least (start, ALL).
fn compute_eps_closure(states: &NWAStates, start: NWAStateID) -> Vec<(NWAStateID, Weight)> {
    let mut map: HashMap<NWAStateID, Weight> = HashMap::new();
    let mut q: VecDeque<NWAStateID> = VecDeque::new();

    // Identity
    map.insert(start, Weight::all());
    q.push_back(start);

    while let Some(u) = q.pop_front() {
        let wu = map.get(&u).cloned().unwrap_or_else(Weight::zeros);
        if wu.is_empty() {
            continue;
        }
        for &(v, ref we) in &states[u].epsilons {
            let add = &wu & we;
            if add.is_empty() {
                continue;
            }
            match map.get_mut(&v) {
                Some(old) => {
                    let joined = &*old | &add;
                    if &joined != old {
                        *old = joined;
                        q.push_back(v);
                    }
                }
                None => {
                    map.insert(v, add);
                    q.push_back(v);
                }
            }
        }
    }

    let mut out: Vec<(NWAStateID, Weight)> = map.into_iter().collect();
    out.sort_by_key(|(sid, _)| *sid);
    out
}

/// Compute the least fixpoint of final weights propagated backward along:
/// - ε-edges (u --ε,w--> v contributes final[u] ⊇ final[v] ∧ w)
/// - negative edges (u --neg,w--> v contributes final[u] ⊇ final[v] ∧ w)
///
/// The graph is given by current NWA states plus the set of negative edges (which are still present as labeled transitions).
fn compute_final_fixpoint_eps_and_neg(states: &NWAStates, neg_edges: &[NegEdge]) -> Vec<Weight> {
    let n = states.len();
    let mut fut: Vec<Weight> = vec![Weight::zeros(); n];

    // Reverse adjacency across ε-edges and negative edges only.
    let mut rev: Vec<Vec<(NWAStateID, Weight)>> = vec![vec![]; n];

    // ε-edges
    for u in 0..n {
        for &(v, ref w) in &states[u].epsilons {
            if v < n {
                rev[v].push((u, w.clone()));
            }
        }
    }
    // Negative edges
    for e in neg_edges {
        if e.to < n {
            rev[e.to].push((e.from, e.weight.clone()));
        }
    }

    // Initialize with current finals
    let mut q: VecDeque<NWAStateID> = VecDeque::new();
    for s in 0..n {
        if let Some(ref fw) = states[s].final_weight {
            if !fw.is_empty() {
                fut[s] = fw.clone();
                q.push_back(s);
            }
        }
    }

    // Reverse-propagate until fixpoint
    while let Some(v) = q.pop_front() {
        let fv = fut[v].clone();
        if fv.is_empty() {
            continue;
        }
        for &(u, ref w_uv) in &rev[v] {
            let add = &fv & w_uv;
            if add.is_empty() {
                continue;
            }
            let old = &fut[u];
            // If add has any bits not already in fut[u], update.
            if (&add & old) != add {
                fut[u] |= &add;
                q.push_back(u);
            }
        }
    }

    fut
}

/// Retarget negative edges A --neg--> B to a canonical "negative-only" copy B⁻ when required:
/// Condition per original algorithm's eventual fixpoint:
///  - B is final under the computed final-fixpoint, or
///  - ε-closure(B) exposes any positive/default transitions.
///
/// The canonical B⁻ is built once per original B and re-used.
/// B⁻ has: no final, no default, and only negative-labeled transitions (ε-edges preserved).
fn retarget_negatives_to_negative_only(
    states: &mut NWAStates,
    neg_edges: &[NegEdge],
    final_fix: &[Weight],
    has_pos_or_default_in_eps: &[bool],
) {
    let n = states.len();

    // Map original-B -> B_neg_only (canonical copy)
    let mut neg_only_map: HashMap<NWAStateID, NWAStateID> = HashMap::new();

    // Helper to determine if original B needs splitting
    let mut needs_split_cache: HashMap<NWAStateID, bool> = HashMap::new();
    let mut get_or_create_neg_only_if_needed =
        |b: NWAStateID, states: &mut NWAStates| -> Option<NWAStateID> {
            let needs_split = {
                if let Some(&ans) = needs_split_cache.get(&b) {
                    ans
                } else {
                    let b_has_final = !final_fix[b].is_empty();
                    let b_has_pos_or_def = has_pos_or_default_in_eps.get(b).copied().unwrap_or(false);
                    let need = b_has_final || b_has_pos_or_def;
                    needs_split_cache.insert(b, need);
                    need
                }
            };

            if !needs_split {
                return None;
            }

            if let Some(&id) = neg_only_map.get(&b) {
                return Some(id);
            }

            // Otherwise, create a copy and strip to negative-only
            let new_id = states.copy_state(b);
            {
                let st = &mut states.0[new_id];
                st.final_weight = None;
                st.default = None;
                st.transitions.retain(|k, _| *k < 0);
                // keep epsilons as-is (per original algorithm)
            }
            neg_only_map.insert(b, new_id);
            Some(new_id)
        };

    // Finally retarget each negative edge
    for e in neg_edges {
        if e.from >= n {
            continue;
        }
        if e.neg_label >= 0 {
            continue;
        }

        if let Some(tgt_neg_only) = get_or_create_neg_only_if_needed(e.to, states) {
            // Update the (from, neg_label) transition target to tgt_neg_only, if still present
            if let Some((ref mut to_mut, _w)) = states.0[e.from].transitions.get_mut(&e.neg_label) {
                *to_mut = tgt_neg_only;
            }
        }
    }
}
