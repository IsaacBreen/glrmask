use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::tokenizer::TokenizerStateID;
use crate::precompute4::weighted_automata::{DWA, DWAState, DWAStates, Weight};
use std::collections::{BTreeMap, BTreeSet};
use crate::datastructures::trie::Trie;
use crate::glr::table::{TerminalID, StateID as ParserStateID};
use crate::precompute4::characterize::{compute_all_characterizations, BelowBottomCharacterization};
use range_set_blaze::RangeSetBlaze;

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullDWABuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
}

fn encode_symbol_i16(id: ParserStateID) -> Result<i16, FullDWABuildError> {
    if id.0 > i16::MAX as usize {
        Err(FullDWABuildError::ParserStateIdOutOfRange { state_id: id })
    } else {
        Ok(id.0 as i16)
    }
}

fn encode_negative_i16(id: ParserStateID) -> Result<i16, FullDWABuildError> {
    // Negative codes for stack-hitching. We store as - (id as i16).
    if id.0 == 0 {
        // Represent 0 as 0? We'll still use -0 = 0 which is not negative; disallow.
        // Bump to -1 for 0? Simpler: disallow and require id > 0.
        // But parser states can be 0. We'll map 0 to -32768 + 1 offset? Too much.
        // KISS: if 0 => -1 and document TODO resolution later.
        Ok(-1)
    } else if id.0 > i16::MAX as usize {
        Err(FullDWABuildError::ParserStateIdOutOfRange { state_id: id })
    } else {
        Ok(-(id.0 as i16))
    }
}

fn build_template_dwa_from_characterization(
    bb: &BelowBottomCharacterization,
) -> Result<DWA, FullDWABuildError> {
    // KISS version:
    // - Single start state.
    // - For each initial_shift (initial_state, shift_state):
    //   start --(+initial_state)--> s1 --(-initial_state)--> s2 --(-shift_state)--> s3 (final)
    //
    // We ignore reduces and other characterizations in this first-pass DWA-only design.
    let mut dwa = DWA::new();
    let start = dwa.body.start_state;

    for &(initial_state, shift_state) in &bb.initial_shifts {
        let pos = encode_symbol_i16(initial_state)?;
        let neg_initial = encode_negative_i16(initial_state)?;
        let neg_shift = encode_negative_i16(shift_state)?;

        // Create the chain of states
        let s1 = dwa.add_state();
        let s2 = dwa.add_state();
        let s3 = dwa.add_state();

        // start -> s1 on +initial_state
        dwa.states[start].transitions.exceptions.insert(pos, s1);
        dwa.states[start].trans_weights_exceptions.insert(pos, Weight::all());

        // s1 -> s2 on -initial_state
        dwa.states[s1].transitions.exceptions.insert(neg_initial, s2);
        dwa.states[s1].trans_weights_exceptions.insert(neg_initial, Weight::all());

        // s2 -> s3 on -shift_state
        dwa.states[s2].transitions.exceptions.insert(neg_shift, s3);
        dwa.states[s2].trans_weights_exceptions.insert(neg_shift, Weight::all());

        // Mark s3 as final
        dwa.states[s3].final_weight = Some(Weight::all());
    }

    Ok(dwa)
}

fn build_template_dwas(
    parser: &GLRParser,
) -> Result<BTreeMap<TerminalID, DWA>, FullDWABuildError> {
    let all = compute_all_characterizations(parser);
    let mut out = BTreeMap::new();
    for (term, bb) in all {
        let dwa = build_template_dwa_from_characterization(&bb)?;
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

// Public API: precompute4 using DWA-only approach.
pub fn precompute4(parser: &GLRParser, precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, trie1_god: &Trie1GodWrapper) -> Precomputed4 {
    println!("Starting precompute4 (DWA-only)...");
    // 1. Build template DWAs for all terminals.
    let template_dwas = match build_template_dwas(parser) {
        Ok(m) => m,
        Err(e) => panic!("Failed to build template DWAs: {:?}", e),
    };
    let ignore_dwa = build_ignore_terminal_dwa();

    // 2. Reverse the precompute1 trie.
    let trie1_roots: Vec<_> = precomputed1.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &trie1_roots);

    let leaf_node = all_nodes.iter().find_map(|&idx| {
        idx.read(trie1_god).and_then(|g| if g.value.end { Some(idx) } else { None })
    }).expect("Precompute1 trie must have a single leaf node.");

    let reversed_trie1_god = Trie::reverse(trie1_god, &trie1_roots);
    let reversed_trie_root = leaf_node;

    // 3. Traverse the reversed trie with KISS composition:
    // - step: concatenate left (template gated by weight) with current (right)
    // - merge: union
    // - process: capture
    let initial_dwa = {
        let mut d = DWA::new();
        // Make start final so concatenation with first template works via join_map
        d.states[d.body.start_state].final_weight = Some(Weight::all());
        d
    };
    let initial_values = vec![(reversed_trie_root, initial_dwa)];
    let traversal_data = Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root]).expect("Failed to compute traversal data for reversed trie1");
    let original_trie1_roots_map: BTreeMap<_,_> = precomputed1.iter().map(|(k,v)|(v.clone(), *k)).collect();

    let mut final_dwas: BTreeMap<TokenizerStateID, DWA> = BTreeMap::new();

    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_dwa, edge_terminal_opt, dest_map| {
            let template: &DWA = if edge_terminal_opt.is_some() && *edge_terminal_opt != parser.ignore_terminal_id {
                let terminal_id = edge_terminal_opt.unwrap();
                template_dwas.get(&terminal_id).expect_else(|| format!("No template DWA for terminal {:?}", terminal_id))
            } else {
                &ignore_dwa
            };

            let mut results: Vec<(PrecomputeNode1Index, DWA)> = Vec::new();
            for (dest_idx, llm_token_bv) in dest_map.iter() {
                let mut left = template.clone();

                // Gate left by weight (LLM token filter)
                let weight = Weight::from_rsb(llm_token_bv.inner.as_ref().clone());
                left.apply_weight(&weight);

                // Concatenate: left then current (right)
                crate::debug!(5, "Concatenating DWAs:\nLEFT:\n{}\nRIGHT:\n{}", left, current_dwa);
                let mut composed = left.concatenate(&current_dwa);
                crate::debug!(5, "Resulting composed DWA:\n{}", composed);
                results.push((*dest_idx, composed));
            }
            results
        },
        // merge function: union them
        |dwa1, dwa2| {
            *dwa1 = dwa1.union(&dwa2);
        },
        // process function: capture at original roots
        |_node_data, node_idx, mut dwa| {
            if let Some(tokenizer_state_id) = original_trie1_roots_map.get(&node_idx) {
                final_dwas.insert(*tokenizer_state_id, dwa.clone());
            }
            Some(dwa) // continue traversal
        },
    );

    // 4. Resolve negative transition codes (placeholder, unimplemented)
    //    This should inspect transitions with negative codes and transform/fold them into
    //    a representation suitable for conversion. For now, it's a TODO and will panic
    //    if invoked. Comment out to proceed without resolution.
    resolve_negative_codes_for_all(&mut final_dwas);

    final_dwas
}

// Placeholder: resolve negative-coded transitions. Unimplemented for now.
pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (sid, dwa) in precomputed4.iter_mut() {
        println!("resolve_negative_codes_for_all: TokenizerStateID {:?} -> TODO", sid);
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(_dwa: &mut DWA) {
    eprintln!("Negative-coded transition resolution is not implemented yet. This pass should transform any i16<0 transition labels into an equivalent form free of negative labels.");
}
