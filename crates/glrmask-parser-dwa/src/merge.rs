use std::collections::BTreeMap;
use std::time::Instant;

use glrmask_artifact::mapped_artifact::MappedArtifact;
use glrmask_weight::Weight;
use glrmask_weighted_automata::automata::weighted_u32::determinize::determinize;
use glrmask_weighted_automata::automata::weighted_u32::dwa::DWA;
use glrmask_weighted_automata::automata::weighted_u32::nwa::NWA;

fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

fn immediate_dwa_accepting_edges(dwa: &DWA) -> Option<Vec<(i32, Weight)>> {
    if dwa.states().len() != 2 {
        return None;
    }
    let start_id = dwa.start_state() as usize;
    let final_id = 1usize.checked_sub(start_id)?;
    let start = &dwa.states()[start_id];
    let final_state = &dwa.states()[final_id];
    if start
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
        || !final_state.transitions.is_empty()
    {
        return None;
    }
    let final_weight = final_state.final_weight.as_ref()?;
    if final_weight.is_empty() || start.transitions.is_empty() {
        return None;
    }

    let mut accepting = Vec::with_capacity(start.transitions.len());
    for (&label, (target, edge_weight)) in &start.transitions {
        if label < 0 || *target as usize != final_id || edge_weight.is_empty() {
            return None;
        }
        let weight = edge_weight.intersection(final_weight);
        if weight.is_empty() {
            return None;
        }
        accepting.push((label, weight));
    }
    Some(accepting)
}

/// Reconcile parser-family ID spaces, then form their exact weighted union.
///
/// Parser labels include fallback semantics, so this deliberately skips the
/// generic weighted minimizer after determinization.
pub fn merge_mapped_parser_dwas(
    inputs: Vec<MappedArtifact<DWA>>,
    _num_tokenizer_states: usize,
    _max_token_id: u32,
) -> MappedArtifact<DWA> {
    assert!(
        !inputs.is_empty(),
        "merge_mapped_parser_dwas called with empty inputs"
    );
    if inputs.len() == 1 {
        return inputs.into_iter().next().unwrap();
    }

    let total_started_at = Instant::now();
    let reconcile_started_at = Instant::now();
    let reconciled = MappedArtifact::reconcile_vec(inputs);
    let reconcile_ms = reconcile_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwas, common_id_map) = reconciled.into_parts();

    let union_started_at = Instant::now();
    let mut union = NWA::new(
        common_id_map.num_tsids(),
        common_id_map.max_internal_token_id(),
    );
    let mut body = union.body();
    for dwa in dwas {
        body = union.union_in_place(&dwa.to_nwa(), &body);
    }
    union.set_start_states(body.start_states);
    let dwa = determinize(&union).expect("parser-family NWA union must determinize");
    let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_dwa_merge] mode=exact_reconciled_union reconcile_ms={:.3} union_ms={:.3} states={} transitions={} total_ms={:.3}",
            reconcile_ms,
            union_ms,
            dwa.num_states(),
            dwa.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    MappedArtifact::new(dwa, common_id_map)
}

/// Keep one depth-one parser family as an exact top-state acceptance overlay.
pub fn merge_mapped_parser_dwas_with_top_accept(
    inputs: Vec<MappedArtifact<DWA>>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> (MappedArtifact<DWA>, BTreeMap<i32, Weight>) {
    assert!(
        !inputs.is_empty(),
        "merge_mapped_parser_dwas_with_top_accept called with empty inputs"
    );
    if inputs.len() == 1 {
        return (inputs.into_iter().next().unwrap(), BTreeMap::new());
    }

    if inputs.len() == 2 {
        let total_started_at = Instant::now();
        let mut iter = inputs.into_iter();
        let left = iter.next().unwrap();
        let right = iter.next().unwrap();
        let left_is_immediate = immediate_dwa_accepting_edges(left.artifact()).is_some();
        let right_is_immediate = immediate_dwa_accepting_edges(right.artifact()).is_some();

        if left_is_immediate || right_is_immediate {
            let (primary, immediate) = if left_is_immediate {
                (right, left)
            } else {
                (left, right)
            };
            let (immediate_dwa, immediate_id_map) = immediate.into_parts();
            let top_accept: BTreeMap<i32, Weight> =
                immediate_dwa_accepting_edges(&immediate_dwa)
                    .expect("immediate parser shape was checked above")
                    .into_iter()
                    .collect();
            let reconciled = primary
                .pair_forced_common(MappedArtifact::new(top_accept, immediate_id_map));
            let ((primary_dwa, top_accept), common_id_map) = reconciled.into_parts();
            if compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][parser_dwa_merge] inputs=2 mode=top_accept_overlay overlay_labels={} states={} transitions={} total_ms={:.3}",
                    top_accept.len(),
                    primary_dwa.num_states(),
                    primary_dwa.num_transitions(),
                    total_started_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return (
                MappedArtifact::new(primary_dwa, common_id_map),
                top_accept,
            );
        }

        return (
            merge_mapped_parser_dwas(
                vec![left, right],
                num_tokenizer_states,
                max_token_id,
            ),
            BTreeMap::new(),
        );
    }

    (
        merge_mapped_parser_dwas(inputs, num_tokenizer_states, max_token_id),
        BTreeMap::new(),
    )
}
