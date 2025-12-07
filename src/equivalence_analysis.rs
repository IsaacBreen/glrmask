use crate::finite_automata::Regex;
use crate::{equivalence_analysis_fast, equivalence_analysis_fast_new, equivalence_analysis_reference};
use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};

pub type EquivalenceResult = BTreeSet<Vec<usize>>;

pub fn find_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> EquivalenceResult {
    if std::env::var("SKIP_EQUIVALENCE_ANALYSIS_TEST").is_ok() {
        // Use old fast implementation (known to be correct)
        let fast =
            equivalence_analysis_fast::find_equivalence_classes(regex, strings, initial_states);
        return fast;
    }
    let instant = std::time::Instant::now();
    let reference =
        equivalence_analysis_reference::find_equivalence_classes(regex, strings, initial_states);
    crate::debug!(
        3,
        "Reference equivalence analysis took {:?}",
        instant.elapsed()
    );
    let instant = std::time::Instant::now();
    // Use old fast implementation for now
    let fast = equivalence_analysis_fast::find_equivalence_classes(regex, strings, initial_states);
    crate::debug!(3, "Fast equivalence analysis took {:?}", instant.elapsed());
    if reference != fast {
        fn trunc(v: &[usize], limit: usize) -> String {
            let take = v.iter().take(limit).cloned().collect::<Vec<_>>();
            if v.len() > limit {
                format!("{:?}... (len {})", take, v.len())
            } else {
                format!("{:?}", take)
            }
        }

        fn build_maps(groups: &EquivalenceResult) -> (HashMap<usize, usize>, HashMap<usize, &[usize]>) {
            let mut idx_to_rep = HashMap::new();
            let mut rep_to_group = HashMap::new();
            for g in groups {
                if let Some(&rep) = g.first() {
                    rep_to_group.insert(rep, g.as_slice());
                    for &idx in g {
                        idx_to_rep.insert(idx, rep);
                    }
                }
            }
            (idx_to_rep, rep_to_group)
        }

        let (ref_map, ref_groups) = build_maps(&reference);
        let (fast_map, fast_groups) = build_maps(&fast);

        eprintln!(
            "Equivalence mismatch: reference groups {} fast groups {}",
            reference.len(),
            fast.len()
        );

        let mut mismatches = Vec::new();
        for idx in 0..strings.len() {
            let r = ref_map.get(&idx);
            let f = fast_map.get(&idx);
            if r != f {
                mismatches.push((idx, r.copied(), f.copied()));
            }
            if mismatches.len() >= 10 {
                break;
            }
        }

        if mismatches.is_empty() {
            eprintln!("No per-index mismatch detected, sets differ in ordering only");
        } else {
            for (idx, r, f) in &mismatches {
                let ref_group = r.and_then(|rep| ref_groups.get(&rep)).copied().unwrap_or(&[]);
                let fast_group = f.and_then(|rep| fast_groups.get(&rep)).copied().unwrap_or(&[]);
                eprintln!(
                    "idx {} ref_rep {:?} fast_rep {:?} | ref_group {} fast_group {}",
                    *idx,
                    r,
                    f,
                    trunc(ref_group, 16),
                    trunc(fast_group, 16)
                );
            }
        }

        if std::env::var("EQ_DEBUG_STATE").is_ok() {
            let debug_idx = std::env::var("EQ_DEBUG_INDEX")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .or_else(|| mismatches.first().map(|(idx, _, _)| *idx))
                .unwrap_or(0);
            eprintln!("EQ_DEBUG_STATE idx {} len {}", debug_idx, strings.get(debug_idx).map(|s| s.len()).unwrap_or(0));
            let sig_ref: Vec<u64> = initial_states
                .iter()
                .map(|&st| equivalence_analysis_reference::compute_signature(regex, &strings[debug_idx], st))
                .collect();
            let sig_fast =
                equivalence_analysis_fast::compute_signature_debug(regex, &strings[debug_idx], initial_states);
            let mut printed = 0usize;
            let mut first_state: Option<(usize, u64, u64)> = None;
            for (state_idx, (&r_sig, &f_sig)) in sig_ref.iter().zip(sig_fast.iter()).enumerate() {
                if r_sig != f_sig {
                    let state = initial_states[state_idx];
                    eprintln!("state {} signature mismatch ref={} fast={}", state, r_sig, f_sig);
                    if first_state.is_none() {
                        first_state = Some((state_idx, r_sig, f_sig));
                    }
                    printed += 1;
                    if printed >= 16 {
                        break;
                    }
                }
            }
            if printed == 0 {
                eprintln!("No per-state signature mismatch found for idx {}", debug_idx);
            }

            if printed == 0 {
                for (idx, _, _) in mismatches.iter().take(10) {
                    if *idx == debug_idx {
                        continue;
                    }
                    let ref_sig: Vec<u64> = initial_states
                        .iter()
                        .map(|&st| equivalence_analysis_reference::compute_signature(regex, &strings[*idx], st))
                        .collect();
                    let fast_sig = equivalence_analysis_fast::compute_signature_debug(
                        regex,
                        &strings[*idx],
                        initial_states,
                    );
                    let mut mismatch = None;
                    for (state_idx, (&r, &f)) in ref_sig.iter().zip(fast_sig.iter()).enumerate() {
                        if r != f {
                            mismatch = Some((state_idx, r, f));
                            break;
                        }
                    }
                    if let Some((state_idx, r, f)) = mismatch {
                        let state = initial_states[state_idx];
                        eprintln!(
                            "Additional idx {} state {} mismatch ref={} fast={}",
                            idx, state, r, f
                        );
                        break;
                    }
                }
            }

            if std::env::var("EQ_DEBUG_SUMMARY").is_ok() {
                use std::hash::Hash;

                fn combined(sig: &[u64]) -> u64 {
                    use std::collections::hash_map::DefaultHasher;
                    let mut h = DefaultHasher::new();
                    for s in sig {
                        s.hash(&mut h);
                    }
                    h.finish()
                }

                for (idx, _, _) in mismatches.iter().take(5) {
                    let ref_sig: Vec<u64> = initial_states
                        .iter()
                        .map(|&st| equivalence_analysis_reference::compute_signature(regex, &strings[*idx], st))
                        .collect();
                    let fast_sig = equivalence_analysis_fast::compute_signature_debug(
                        regex,
                        &strings[*idx],
                        initial_states,
                    );
                    eprintln!(
                        "Summary idx {} ref_comb={} fast_comb={} ref_vec_len={}",
                        idx,
                        combined(&ref_sig),
                        combined(&fast_sig),
                        ref_sig.len()
                    );
                }

                if let Some((idx, r, f)) = mismatches.first() {
                    let ref_group = r.and_then(|rep| ref_groups.get(&rep)).copied().unwrap_or(&[]);
                    let fast_group = f.and_then(|rep| fast_groups.get(&rep)).copied().unwrap_or(&[]);
                    let mut seen: std::collections::BTreeMap<u64, Vec<(usize, u64)>> =
                        Default::default();

                    for id in ref_group.iter().take(12).chain(fast_group.iter().take(12)) {
                        let sig_vec = equivalence_analysis_fast::compute_signature_debug(
                            regex,
                            &strings[*id],
                            initial_states,
                        );
                        let sig = combined(&sig_vec);
                        let sig_fast_actual =
                            equivalence_analysis_fast::compute_signature_actual(
                                regex,
                                &strings[*id],
                                initial_states,
                            );
                        seen.entry(sig)
                            .or_default()
                            .push((*id, sig_fast_actual));
                    }

                    eprintln!("First mismatch idx {} group signature buckets: {:?}", idx, seen);
                }
            }

            if let Some((state_idx, r_sig, f_sig)) = first_state {
                fn trunc_edges(list: &[(usize, usize)], limit: usize) -> String {
                    let take: Vec<_> = list.iter().take(limit).cloned().collect();
                    if list.len() > limit {
                        format!("{:?}... (len {})", take, list.len())
                    } else {
                        format!("{:?}", take)
                    }
                }

                let state = initial_states[state_idx];
                let fast_edges = equivalence_analysis_fast::debug_pos0_edges(
                    regex,
                    &strings[debug_idx],
                    initial_states,
                )
                .get(state_idx)
                .cloned()
                .unwrap_or_default();

                let mut ref_edges: Vec<(usize, usize)> = regex
                    .execute_from_state_nonzero(&strings[debug_idx], state)
                    .matches
                    .iter()
                    .map(|m| (m.group_id, m.position))
                    .collect();
                ref_edges.sort_unstable_by_key(|e| e.0);

                eprintln!(
                    "state {} first mismatch ref_sig={} fast_sig={} | ref_edges {} fast_edges {}",
                    state,
                    r_sig,
                    f_sig,
                    trunc_edges(&ref_edges, 24),
                    trunc_edges(&fast_edges, 24)
                );

                if std::env::var("EQ_DEBUG_TRACE").is_ok() {
                    let bytes = &strings[debug_idx];
                    let mut cur = state;
                    for (i, &b) in bytes.iter().enumerate() {
                        let finals = &regex.dfa.states[cur].finalizers;
                        let next = regex.dfa.states[cur].transitions.get(b).copied();
                        eprintln!(
                            "step {} byte {} state {} finals {:?} next {:?}",
                            i,
                            b,
                            cur,
                            finals,
                            next
                        );
                        if let Some(n) = next {
                            cur = n;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        panic!("Mismatch between reference and fast equivalence analysis results");
    }
    reference
}
