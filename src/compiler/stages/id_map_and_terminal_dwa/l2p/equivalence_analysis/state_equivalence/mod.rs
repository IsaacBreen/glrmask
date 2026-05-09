use crate::compiler::stages::equiv_types::ManyToOneIdMap;

pub(crate) mod f_signature;
pub(crate) mod max_length;
pub(crate) mod pass;
pub(crate) mod pipeline;
pub(crate) mod vocab_trie_hash128;

pub(crate) use pass::StateEquivalenceScope;
pub(crate) use pipeline::{
    resolve_global_pipeline_config, resolve_l2p_pipeline_config, run_state_equivalence_pipeline,
};

pub(crate) fn identity_state_map(num_states: usize) -> ManyToOneIdMap {
    let original_to_internal: Vec<u32> = (0..num_states as u32).collect();
    ManyToOneIdMap::from_original_to_internal_with_representatives(
        original_to_internal.clone(),
        original_to_internal.len() as u32,
        original_to_internal,
    )
}

pub(crate) fn build_state_map_from_subset_representatives(
    states: &[usize],
    representative_states: &[usize],
    num_states: usize,
    initial_state_map: Option<&ManyToOneIdMap>,
) -> ManyToOneIdMap {
    debug_assert_eq!(states.len(), representative_states.len());

    let mut rep_to_internal = vec![u32::MAX; num_states];
    let mut next_internal = 0u32;

    for &rep in representative_states {
        if rep_to_internal[rep] == u32::MAX {
            rep_to_internal[rep] = next_internal;
            next_internal += 1;
        }
    }

    match initial_state_map {
        Some(initial_state_map) => {
            let mut composed_original_to_internal = vec![u32::MAX; num_states];
            let mut composed_internal_to_originals = vec![Vec::new(); next_internal as usize];
            let mut composed_reps = vec![u32::MAX; next_internal as usize];
            let mut init_rep_to_new_internal = vec![u32::MAX; num_states];

            for (&state, &rep) in states.iter().zip(representative_states.iter()) {
                init_rep_to_new_internal[state] = rep_to_internal[rep];
            }

            for (orig_state, &init_internal) in initial_state_map.original_to_internal.iter().enumerate() {
                if init_internal == u32::MAX
                    || (init_internal as usize) >= initial_state_map.representative_original_ids.len()
                {
                    continue;
                }
                let init_rep = initial_state_map.representative_original_ids[init_internal as usize] as usize;
                let new_internal = init_rep_to_new_internal[init_rep];
                if new_internal == u32::MAX {
                    continue;
                }
                composed_original_to_internal[orig_state] = new_internal;
                let bucket = &mut composed_internal_to_originals[new_internal as usize];
                if bucket.is_empty() {
                    composed_reps[new_internal as usize] = init_rep as u32;
                }
                bucket.push(orig_state as u32);
            }

            ManyToOneIdMap {
                original_to_internal: composed_original_to_internal,
                internal_to_originals: composed_internal_to_originals,
                representative_original_ids: composed_reps,
            }
        }
        None => {
            let mut original_to_internal = vec![u32::MAX; num_states];
            let mut internal_to_originals = vec![Vec::new(); next_internal as usize];
            let mut representative_original_ids = vec![u32::MAX; next_internal as usize];

            for (&state, &rep) in states.iter().zip(representative_states.iter()) {
                let internal = rep_to_internal[rep];
                original_to_internal[state] = internal;
                let bucket = &mut internal_to_originals[internal as usize];
                if bucket.is_empty() {
                    representative_original_ids[internal as usize] = rep as u32;
                }
                bucket.push(state as u32);
            }

            ManyToOneIdMap {
                original_to_internal,
                internal_to_originals,
                representative_original_ids,
            }
        }
    }
}