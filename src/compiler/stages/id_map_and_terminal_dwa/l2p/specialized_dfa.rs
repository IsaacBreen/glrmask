//! Specialized (quotient) DFA construction for the L2+ terminal path.
//!
//! Given the full tokenizer DFA and a set of "active" (L2+) groups, this
//! module builds a *minimized* quotient DFA where states that are
//! indistinguishable with respect to the active groups are merged.  Both
//! the equivalence analysis and the NWA builder then operate on this
//! compact DFA, keeping their state spaces consistent.
//!
//! Group numbering is preserved: the quotient DFA has the same number of
//! group slots as the original, but only L2+ groups carry data.  This
//! avoids index remapping when terminal IDs are used as NWA labels.

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::ds::bitset::BitSet;

/// Result of building a specialized (quotient) tokenizer for L2+ groups.
pub(crate) struct SpecializedTokenizer {
    /// The quotient tokenizer (same group count as original, only L2+ data,
    /// minimized state count).
    pub tokenizer: Tokenizer,
    /// Maps each original DFA state → quotient state index.
    pub original_to_quotient: Vec<u32>,
}

/// Build a minimized quotient tokenizer that only tracks the given active
/// groups.  States that are indistinguishable under active-group finalizers
/// and futures are merged via iterative partition refinement.
///
/// Group numbering is preserved (not renumbered to 0..K).
pub(crate) fn build_specialized_tokenizer(
    tokenizer: &Tokenizer,
    active_groups: &[bool],
) -> SpecializedTokenizer {
    let dfa = &tokenizer.dfa;
    let states = dfa.states();
    let num_states = states.len();
    let num_groups = active_groups.len();

    // Compute per-state L2+-projected signature for initial partition.
    // Two vectors of active-group IDs: finalizers and futures.
    let mut state_l2p_finalizers: Vec<Vec<usize>> = Vec::with_capacity(num_states);
    let mut state_l2p_futures: Vec<Vec<usize>> = Vec::with_capacity(num_states);
    for (i, _state) in states.iter().enumerate() {
        let finals: Vec<usize> = dfa
            .finalizers(i as u32)
            .iter()
            .filter(|&gid| gid < num_groups && active_groups[gid])
            .collect();
        state_l2p_finalizers.push(finals);

        let futures: Vec<usize> = dfa
            .possible_future_group_ids(i as u32)
            .iter()
            .filter(|&gid| gid < num_groups && active_groups[gid])
            .collect();
        state_l2p_futures.push(futures);
    }

    // Build flat transition table: trans[state][byte] = target_state (u32::MAX = dead)
    let trans: Vec<[u32; 256]> = states
        .iter()
        .map(|state| {
            let mut table = [u32::MAX; 256];
            for (byte, &target) in state.transitions.iter() {
                table[byte as usize] = target;
            }
            table
        })
        .collect();

    // ---- Partition refinement ----
    // Initial partition: group by (L2+ finalizers, L2+ futures)
    let mut partition: Vec<u32> = vec![0; num_states];
    let mut num_blocks: u32;
    {
        let mut sig_to_block: BTreeMap<(&[usize], &[usize]), u32> = BTreeMap::new();
        let mut block_id = 0u32;
        for i in 0..num_states {
            let sig = (
                state_l2p_finalizers[i].as_slice(),
                state_l2p_futures[i].as_slice(),
            );
            let bid = *sig_to_block.entry(sig).or_insert_with(|| {
                let id = block_id;
                block_id += 1;
                id
            });
            partition[i] = bid;
        }
        num_blocks = block_id;
    }

    // Iterative refinement: split blocks by byte-transition targets
    let mut changed = true;
    while changed {
        changed = false;
        for byte in 0..256u16 {
            let b = byte as usize;
            // Group states within each current block by successor block
            let mut block_split: BTreeMap<u32, BTreeMap<u32, Vec<usize>>> = BTreeMap::new();
            for s in 0..num_states {
                let cur_block = partition[s];
                let succ = trans[s][b];
                let succ_block = if succ == u32::MAX {
                    u32::MAX
                } else {
                    partition[succ as usize]
                };
                block_split
                    .entry(cur_block)
                    .or_default()
                    .entry(succ_block)
                    .or_default()
                    .push(s);
            }
            for (_cur_block, succ_groups) in &block_split {
                if succ_groups.len() <= 1 {
                    continue;
                }
                let mut first = true;
                for (_succ_block, states_in_group) in succ_groups {
                    if first {
                        first = false;
                        continue;
                    }
                    let new_block = num_blocks;
                    num_blocks += 1;
                    for &s in states_in_group {
                        partition[s] = new_block;
                    }
                    changed = true;
                }
            }
        }
    }

    // Find canonical representative for each block
    let mut block_representative: BTreeMap<u32, usize> = BTreeMap::new();
    for s in 0..num_states {
        block_representative.entry(partition[s]).or_insert(s);
    }

    // Renumber blocks contiguously, ensuring start-state block gets ID 0
    let mut block_renumber: BTreeMap<u32, u32> = BTreeMap::new();
    let start_block = partition[tokenizer.start_state() as usize];
    block_renumber.insert(start_block, 0);
    let mut next_id = 1u32;
    for s in 0..num_states {
        let b = partition[s];
        if !block_renumber.contains_key(&b) {
            block_renumber.insert(b, next_id);
            next_id += 1;
        }
    }
    let num_quotient_states = next_id as usize;

    // Build original_to_quotient mapping
    let original_to_quotient: Vec<u32> = (0..num_states)
        .map(|s| block_renumber[&partition[s]])
        .collect();

    // Build the quotient DFA (same group count as original)
    let mut new_dfa = crate::automata::lexer::dfa::DFA::new(num_quotient_states);
    new_dfa.ensure_group_capacity(num_groups);

    for (&block_id, &repr) in &block_representative {
        let q_state = block_renumber[&block_id];

        // Transitions: remap targets through original_to_quotient
        let entries: Vec<(u8, u32)> = trans[repr]
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t != u32::MAX)
            .map(|(byte, &target)| (byte as u8, original_to_quotient[target as usize]))
            .collect();
        new_dfa.set_transitions_from_sorted_entries(q_state, entries);

        // Finalizers: only L2+ groups
        let mut finalizers = BitSet::new(num_groups);
        for &gid in &state_l2p_finalizers[repr] {
            finalizers.set(gid);
        }

        // Possible futures: only L2+ groups (from representative)
        let mut futures = BitSet::new(num_groups);
        for &gid in &state_l2p_futures[repr] {
            futures.set(gid);
        }

        new_dfa.overwrite_state_metadata(q_state, finalizers, futures);
    }

    let spec_tokenizer = Tokenizer {
        dfa: new_dfa,
        num_terminals: tokenizer.num_terminals,
        exprs: tokenizer.exprs.clone(),
    };

    SpecializedTokenizer {
        tokenizer: spec_tokenizer,
        original_to_quotient,
    }
}
