//! Global tokenizer-state quotient for Terminal-DWA construction.
//!
//! Both the Terminal DWA and the scan relation are sensitive to lexer state.
//! Before building local Terminal DWAs, the compiler can quotient tokenizer DFA
//! states by a max-token-length equivalence.  This file owns that global quotient
//! so it is not confused with either of the local direct/pair partitions.

use std::sync::Arc;
use std::time::Instant;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compile::id_space::ManyToOneIdMap;
use crate::compile::terminal_dwa::options;
use crate::compile::terminal_dwa::pair_partition::equivalence_analysis::state_equivalence::{
    resolve_global_pipeline_config,
    run_state_equivalence_pipeline,
    StateEquivalenceScope,
};
use crate::compile::terminal_dwa::types::compile_profile_enabled;
use crate::Vocab;

fn use_global_max_length(tokenizer: &Tokenizer) -> bool {
    match options::global_max_length_env_override() {
        Some(enabled) => enabled,
        None => tokenizer.num_states() > 50_000,
    }
}

pub(crate) fn build_global_max_length_state_map(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _flat_trans: &Arc<[u32]>,
) -> ManyToOneIdMap {
    let started_at = Instant::now();
    let num_states_u32 = tokenizer.num_states();
    let num_states = num_states_u32 as usize;
    let token_bytes: Vec<&[u8]> = vocab
        .entries
        .values()
        .map(|bytes| bytes.as_slice())
        .collect();
    let max_token_len = token_bytes
        .iter()
        .map(|bytes| bytes.len())
        .max()
        .unwrap_or(0);

    let config = resolve_global_pipeline_config(use_global_max_length(tokenizer));
    let (state_map, profile) = run_state_equivalence_pipeline(
        tokenizer,
        vocab,
        None,
        None,
        StateEquivalenceScope::Global,
        &config,
    );

    if compile_profile_enabled() {
        if profile.max_length_skipped {
            eprintln!(
                "[glrmask/profile][global_max_length] mode=identity skipped=true states={} reps={} tokens_included=0 max_token_len=0 ms={:.3}",
                num_states,
                state_map.representative_original_ids.len(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        } else {
            eprintln!(
                "[glrmask/profile][global_max_length] mode=stable skipped=false states={} reps={} tokens_included={} max_token_len={} ms={:.3}",
                num_states,
                state_map.representative_original_ids.len(),
                token_bytes.len(),
                max_token_len,
                profile.max_length_state_equiv_ms,
            );
        }
    }

    state_map
}
