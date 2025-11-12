use crate::r#macro::is_debug_level_enabled;
use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{compute_all_characterizations, BelowBottomCharacterization};
use crate::precompute4::resolve_negatives::{apply_cancellations, apply_finality_fixpoint, remove_negative_transitions};
use crate::precompute4::utils;
use crate::precompute4::weighted_automata::{DWA, DWABody, DWAState, DWAStates, NWA, NWABuildError, NWAStates, NWABody, StateID, Weight};
use crate::constraint::LLMTokenBV;
use range_set_blaze::RangeSetBlaze;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

pub type Precomputed4 = DWA;
use crate::tokenizer::TokenizerStateID;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullDWABuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
    AutomatonBuild(NWABuildError),
}

impl From<NWABuildError> for FullDWABuildError {
    fn from(e: NWABuildError) -> Self {
        FullDWABuildError::AutomatonBuild(e)
    }
}

fn build_template_nwa_from_characterization(
    bb: &BelowBottomCharacterization,
) -> Result<NWA, FullDWABuildError> {
    let mut nwa = NWA::new();
    let w_all = Weight::all();

    // Create a node for each non-terminal, similar to the NWA construction.
    let mut nt_nodes: BTreeMap<NonTerminalID, StateID> = BTreeMap::new();
    for &nt in &bb.all_nts {
        let id = nwa.states.add_state();
        nt_nodes.insert(nt, id);
    }

    let start = nwa.body.start_state;

    // --- Initial Actions from Start State ---

    for &(initial_state, shift_state) in &bb.initial_shifts {
        let pos_initial = utils::encode_symbol_i16(initial_state)?;
        let neg_initial = utils::encode_negative_i16(initial_state)?;
        let neg_shift = utils::encode_negative_i16(shift_state)?;

        let s0 = nwa.states.add_state();
        let s1 = nwa.states.add_state();
        let s2 = nwa.states.add_state();
        let s3 = nwa.states.add_state();

        // start --(eps)--> s0 --(+initial)--> s1 --(-initial)--> s2 --(-shift)--> s3 (final)
        nwa.add_epsilon(start, s0, w_all.clone());
        nwa.add_transition(s0, pos_initial, s1, w_all.clone())?;
        nwa.add_transition(s1, neg_initial, s2, w_all.clone())?;
        nwa.add_transition(s2, neg_shift, s3, w_all.clone())?;
        nwa.states[s3].final_weight = Some(w_all.clone());
    }

    for &(initial_state, len, nt) in &bb.initial_reduces {
        let pos_initial = utils::encode_symbol_i16(initial_state)?;
        let target_nt_state = *nt_nodes.get(&nt).expect("nt_node must exist for initial_reduce");

        // Create a chain of default transitions for the pops.
        // start --(eps)--> s0 --(+initial)--> s1 --(default)*len--> target_nt_state
        let s0 = nwa.states.add_state();
        nwa.add_epsilon(start, s0, w_all.clone());
        let mut from = s0;
        let next_state = if len == 0 { target_nt_state } else { nwa.states.add_state() };
        nwa.add_transition(from, pos_initial, next_state, w_all.clone())?;
        from = next_state;

        for i in 0..len {
            let to = if i == len - 1 { target_nt_state } else { nwa.states.add_state() };
            nwa.add_default_transition(from, to, w_all.clone(), BTreeSet::new())?;
            from = to;
        }
    }

    // --- Actions from Non-Terminal States ---

    for (nt, rc) in &bb.reduce_characterizations {
        let src_nt_state = *nt_nodes.get(nt).expect("nt_node must exist for reduce_char");

        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;
            let dst_nt_state = *nt_nodes.get(&reduce_nt).expect("dst nt_node must exist");

            // src --(eps)--> s0 --(+revealed)--> s1 --(default)*len--> dst
            let s0 = nwa.states.add_state();
            nwa.add_epsilon(src_nt_state, s0, w_all.clone());
            let mut from = s0;
            let next_state = if len == 0 { dst_nt_state } else { nwa.states.add_state() };
            nwa.add_transition(from, pos_revealed, next_state, w_all.clone())?;
            from = next_state;

            for i in 0..len {
                let to = if i == len - 1 { dst_nt_state } else { nwa.states.add_state() };
                nwa.add_default_transition(from, to, w_all.clone(), BTreeSet::new())?;
                from = to;
            }
        }

        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let pos_revealed = utils::encode_symbol_i16(revealed_state)?;
            let neg_revealed = utils::encode_negative_i16(revealed_state)?;
            let neg_goto = utils::encode_negative_i16(goto_state)?;
            let neg_shift = utils::encode_negative_i16(shift_state)?;

            let s0 = nwa.states.add_state();
            let s1 = nwa.states.add_state();
            let s2 = nwa.states.add_state();
            let s3 = nwa.states.add_state();
            let s4 = nwa.states.add_state();

            // src --(eps)--> s0 --(+revealed)--> s1 --(-revealed)--> s2 --(-goto)--> s3 --(-shift)--> s4 (final)
            nwa.add_epsilon(src_nt_state, s0, w_all.clone());
            nwa.add_transition(s0, pos_revealed, s1, w_all.clone())?;
            nwa.add_transition(s1, neg_revealed, s2, w_all.clone())?;
            nwa.add_transition(s2, neg_goto, s3, w_all.clone())?;
            nwa.add_transition(s3, neg_shift, s4, w_all.clone())?;
            nwa.states[s4].final_weight = Some(w_all.clone());
        }
    }

    Ok(nwa)
}

fn build_template_dwas(
    parser: &GLRParser,
) -> Result<BTreeMap<TerminalID, DWA>, FullDWABuildError> {
    let all = compute_all_characterizations(parser);
    let mut out = BTreeMap::new();
    for (term, bb) in all {
        let nwa = build_template_nwa_from_characterization(&bb)?;
        let mut dwa = nwa.determinize_to_dwa();
        dwa.simplify();
        crate::debug!(5, "Built template DWA for terminal {:?}:", term);
        crate::debug!(5, "{}", dwa);
        out.insert(term, dwa);
    }
    Ok(out)
}

fn build_ignore_terminal_dwa() -> DWA {
    // Identity DWA: start is final, no transitions.
    let mut dwa = DWA::new();
    dwa.states[dwa.body.start_state].final_weight = Some(Weight::all());
    dwa
}

/// For any state with a final weight, subtract that weight from all outgoing transitions.
/// This prunes paths that continue after a word has already been accepted with a given weight.
fn prune_continuations_from_final_states(nwa: &mut NWA) -> bool {
    let mut changed = false;
    for i in 0..nwa.states.len() {
        if let Some(final_weight) = nwa.states[i].final_weight.clone() {
            if final_weight.is_empty() {
                continue;
            }

            let state = &mut nwa.states[i];

            // Epsilon transitions
            for (_, w) in &mut state.epsilons {
                let old_w = w.clone();
                *w -= &final_weight;
                if *w != old_w {
                    changed = true;
                }
            }

            // Labeled transitions
            for targets in state.transitions.values_mut() {
                for (_, w) in targets {
                    let old_w = w.clone();
                    *w -= &final_weight;
                    if *w != old_w {
                        changed = true;
                    }
                }
            }

            // Default transitions
            for def in &mut state.default {
                let old_w = def.weight.clone();
                def.weight -= &final_weight;
                if def.weight != old_w {
                    changed = true;
                }
            }
        }
    }
    changed
}

// Helper: collect final states of a DWA
fn collect_final_states(dwa: &DWA) -> BTreeSet<usize> {
    let mut finals = BTreeSet::new();
    for (i, st) in dwa.states.0.iter().enumerate() {
        if st.final_weight.is_some() {
            finals.insert(i);
        }
    }
    finals
}

// Helper: join_map for concatenation: map each left final to the right's start.
fn join_map_final_to_start(left: &DWA, right: &DWA) -> BTreeMap<usize, BTreeSet<usize>> {
    let left_final_states = collect_final_states(left);
    let right_start = right.body.start_state;
    let mut join_map: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for lf in left_final_states {
        join_map.insert(lf, BTreeSet::from([right_start]));
    }
    join_map
}

// Public API: precompute4 using NWA-first approach, determinize at the end.
pub fn precompute4(parser: &GLRParser, precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, trie1_god: &Trie1GodWrapper) -> DWA {
    let now_total = Instant::now();
    let now = Instant::now();
    crate::debug!(5, "Starting precompute4...");
    // 1. Build template DWAs for all terminals.
    let template_dwas = match build_template_dwas(parser) {
        Ok(m) => m,
        Err(e) => panic!("Failed to build template DWAs: {:?}", e),
    };
    let ignore_dwa = build_ignore_terminal_dwa();
    crate::debug!(4, "Built {} template DWAs in {:?}", template_dwas.len(), now.elapsed());
    if is_debug_level_enabled(5) {
        for (term, dwa) in template_dwas.iter().take(5) {
            crate::debug!(5, "Stats for template DWA for terminal {:?}:\n{}", term, dwa.stats());
        }
    }

    // 2. Set up shared NWA state arena.
    let states_arena = RefCell::new(NWAStates::default());

    // 3. Reverse the precompute1 trie.
    let trie1_roots: Vec<_> = precomputed1.values().cloned().collect();

    let all_nodes = Trie::all_nodes(trie1_god, &trie1_roots);

    let leaf_node = all_nodes.iter().find_map(|&idx| {
        idx.read(trie1_god).and_then(|g| if g.value.end { Some(idx) } else { None })
    }).expect("Precompute1 trie must have a single leaf node.");

    let reversed_trie1_god = Trie::reverse(trie1_god, &trie1_roots);
    let reversed_trie_root = leaf_node;

    // 4. Traverse the reversed trie with NWA bodies.
    let initial_nwa_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_state: start }
    };
    let initial_tokens = LLMTokenBV::max_ones();
    let initial_values: Vec<(Trie2Index, (NWABody, LLMTokenBV))> = vec![(reversed_trie_root, (initial_nwa_body, initial_tokens))];
    let traversal_data = Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root]).expect("Failed to compute traversal data for reversed trie1");
    let mut original_trie1_roots_map: BTreeMap<PrecomputeNode1Index, Vec<TokenizerStateID>> = BTreeMap::new();
    for (k, v) in precomputed1.iter() {
        original_trie1_roots_map.entry(v.clone()).or_default().push(*k);
    }

    let options = crate::datastructures::trie::PrettyPrintOptions::default()
        .omit_nodes()
        .omit_depth()
        ;
    crate::debug!(5, "Trie:\n{}", Trie::pretty_print_with_options(&trie1_god, &trie1_roots, &options));
    crate::debug!(5, "Reversed trie:\n{}", Trie::pretty_print_with_options(&reversed_trie1_god, &[reversed_trie_root], &options));

    let mut final_bodies: BTreeMap<TokenizerStateID, NWABody> = BTreeMap::new();

    let now = Instant::now();
    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_val: &(NWABody, LLMTokenBV), edge_terminal_opt, dest_map| {
            let (current_nwa_body, current_tokens) = current_val;
            let template_dwa: &DWA = if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                let terminal_id = edge_terminal_opt.unwrap();
                template_dwas.get(&terminal_id).expect_else(|| format!("No template DWA for terminal {:?}", terminal_id))
            } else {
                &ignore_dwa
            };

            let mut results: Vec<(PrecomputeNode1Index, (NWABody, LLMTokenBV))> = Vec::new();
            for (dest_idx, llm_token_bv) in dest_map.iter() {
                let next_tokens = current_tokens & llm_token_bv;
                if next_tokens.is_empty() {
                    continue;
                }

                let mut states = states_arena.borrow_mut();

                // Convert template DWA to NWA and copy it into the arena
                let template_nwa = NWA::from_dwa(template_dwa);
                crate::debug!(5, "Applying template NWA for terminal {:?} with epsilon gate weight {:?}...", edge_terminal_opt, llm_token_bv);
                let (template_start_in_arena, _) = states.copy_subgraph_from(&template_nwa.states, template_nwa.body.start_state);
                crate::debug!(5, "Template NWA copied into arena. Current arena size: {} states.", states.0.len());
                let left_body = NWABody { start_state: template_start_in_arena };

                // Concatenate: left then current (right) via epsilon with weight = llm_token_bv
                crate::debug!(5, "Starting NWA::concatenate_components: left_start={} right_start={}...", left_body.start_state, current_nwa_body.start_state);
                let eps_weight = Weight::from_rsb(llm_token_bv.inner.as_ref().clone());
                let composed_body = NWA::concatenate_components(&mut states, &left_body, current_nwa_body, &eps_weight);
                crate::debug!(5, "NWA::concatenate_components finished. New start state: {}.", composed_body.start_state);

                results.push((*dest_idx, (composed_body, next_tokens)));
            }
            results
        },
        // merge function: union them via epsilon
        |val1, val2| {
            let (body1, tokens1) = val1;
            let (body2, tokens2) = val2;
            let mut states = states_arena.borrow_mut();
            crate::debug!(5, "Starting NWA::union_components: body1_start={} body2_start={}...", body1.start_state, body2.start_state);
            *body1 = NWA::union_components(&mut states, body1, &body2);
            *tokens1 |= &tokens2;
            crate::debug!(5, "NWA::union_components finished. New start state: {}.", body1.start_state);
        },
        // process function: capture at original roots
        |_node_data, node_idx, val| {
            let (mut nwa_body, tokens) = val;
            if !tokens.is_empty() {
                // Simplify the NWA by determinizing and converting back.
                // This is an expensive but powerful simplification step.
                {
                    let mut states = states_arena.borrow_mut();

                    // Extract subgraph
                    let mut sub_nwa_states = NWAStates::default();
                    let (sub_start, _) = sub_nwa_states.copy_subgraph_from(&*states, nwa_body.start_state);
                    let mut sub_nwa = NWA {
                        states: sub_nwa_states,
                        body: NWABody { start_state: sub_start },
                    };
                    sub_nwa.simplify();

                    // Determinize, simplify, convert back
                    let mut temp_dwa = sub_nwa.determinize_to_dwa();
                    temp_dwa.simplify();
                    let new_nwa_from_dwa = NWA::from_dwa(&temp_dwa);

                    // Copy back into shared arena
                    let (new_start, _) = states.copy_subgraph_from(&new_nwa_from_dwa.states, new_nwa_from_dwa.body.start_state);
                    nwa_body.start_state = new_start;
                }

                if let Some(tokenizer_state_ids) = original_trie1_roots_map.get(&node_idx) {
                    for tokenizer_state_id in tokenizer_state_ids {
                        final_bodies.insert(*tokenizer_state_id, nwa_body.clone());
                    }
                }
                Some((nwa_body, tokens)) // continue traversal
            } else {
                None
            }
        },
    );
    crate::debug!(4, "Reversed trie traversal (special_map_grouped) took: {:?}", now.elapsed());

    // Combine all final NWA bodies into a single NWA
    let mut combined_nwa_states = states_arena.into_inner();
    let combined_start_state = combined_nwa_states.add_state();

    for (tok_id, body) in final_bodies {
        // Add a transition from the new combined start state to the start of the NWA for this tokenizer state.
        // The label is the tokenizer state ID.
        let label = tok_id.0 as i16;
        combined_nwa_states.add_transition(combined_start_state, label, body.start_state, Weight::all()).unwrap();
    }

    let mut combined_nwa = NWA {
        states: combined_nwa_states,
        body: NWABody { start_state: combined_start_state },
    };
    combined_nwa.simplify();
    crate::debug!(5, "Resolving negative codes in combined NWA: {}", combined_nwa);
    crate::debug!(4, "Combined NWA has {} states.", combined_nwa.states.len());
    crate::debug!(4, "Stats for combined NWA before negative resolution:\n{}", combined_nwa.stats());

    let now = Instant::now();
    crate::debug!(4, "Starting negative code resolution...");
    apply_cancellations(&mut combined_nwa);
    combined_nwa.simplify();
    apply_finality_fixpoint(&mut combined_nwa);
    combined_nwa.simplify();
    remove_negative_transitions(&mut combined_nwa);
    combined_nwa.simplify();
    crate::debug!(4, "Negative code resolution took: {:?}. NWA now has {} states.", now.elapsed(), combined_nwa.states.len());
    crate::debug!(4, "Stats for combined NWA after negative resolution:\n{}", combined_nwa.stats());

    let now = Instant::now();
    crate::debug!(4, "Pruning continuations from final states...");
    prune_continuations_from_final_states(&mut combined_nwa);
    crate::debug!(4, "Pruning and simplifying took: {:?}. NWA now has {} states.", now.elapsed(), combined_nwa.states.len());
    crate::debug!(4, "Stats for combined NWA after pruning:\n{}", combined_nwa.stats());

    let now = Instant::now();
    // Determinize the single combined NWA
    crate::debug!(4, "Determinizing final combined NWA...");
    let mut final_dwa = combined_nwa.determinize_to_dwa();
    final_dwa.simplify();
    crate::debug!(4, "Final determinize & simplify took: {:?}. Final DWA has {} states.", now.elapsed(), final_dwa.states.len());
    crate::debug!(4, "Stats for final DWA:\n{}", final_dwa.stats());

    crate::debug!(3, "Total precompute4 time: {:?}", now_total.elapsed());
    final_dwa
}
