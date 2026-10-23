//! Conservative, grammar-aware text forcing with exact model-token events.
//!
//! Ordinary text is forced only while every live parser/lexer alternative has
//! the same next byte. Its tokenization is longest admitted spelling, then
//! lowest original ID, and is withheld if an admitted longer token could extend
//! the proved prefix. This is a library canonicalization policy, not a claim
//! that every allowed model-token sequence has the same IDs or probabilities.
//!
//! Exact-token events are never reinterpreted as bytes. An event competes with
//! text; only a unique admitted original ID may pass that boundary. Every
//! returned ID is committed through the full ordinary/event successor before
//! any more output is considered. Query limits shorten results, never weaken
//! their proof. No normal mask/commit hot path depends on this optional query.

use super::{advance_special_terminal_paths, commit_bytes_impl, expand_runtime_product_states};
use crate::automata::lexer::Lexer;
use crate::runtime::constraint::Constraint;
use crate::runtime::state::{frontier_is_accepting, CommitBuffers, ConstraintState, ParserStateMap};

const MAX_OUTPUT_BYTES: usize = 10_000;
const MAX_OUTPUT_IDS: usize = 10_000;
// Bounds checked byte/event probes, not CPU, allocator work, or mask cost.
const MAX_PROBES: usize = 100_000;
const REJECTED: &str = "commit rejected: no valid parser states remain";

struct Budget(usize);
impl Budget {
    fn probe(&mut self) -> Result<(), ()> {
        if self.0 == 0 { return Err(()); }
        self.0 -= 1;
        Ok(())
    }
}

fn bit(mask: &[u32], id: u32) -> bool {
    mask.get(id as usize / 32).is_some_and(|word| word & (1 << (id % 32)) != 0)
}
fn ids(mask: &[u32]) -> impl Iterator<Item = u32> + '_ {
    mask.iter().enumerate().flat_map(|(word_index, &word)| {
        let mut pending = word;
        std::iter::from_fn(move || {
            if pending == 0 { return None; }
            let index = pending.trailing_zeros();
            pending &= pending - 1;
            Some(word_index as u32 * 32 + index)
        })
    })
}
fn singleton(mask: &[u32]) -> Option<u32> {
    let mut iter = ids(mask);
    let first = iter.next()?;
    iter.next().is_none().then_some(first)
}

fn same_spelling_policy(states: &[ConstraintState<'_>]) -> bool {
    let Some(first) = states.first() else { return false; };
    states[1..].iter().all(|state| {
        let a = first.constraint;
        let b = state.constraint;
        std::ptr::eq(a, b) || (a.end_tokens.as_ref() == b.end_tokens.as_ref()
            && a.token_bytes_count() == b.token_bytes_count()
            && a.token_bytes_iter().all(|(id, bytes)| b.token_bytes_for_id(id) == Some(bytes)))
    })
}

fn masks(states: &[ConstraintState<'_>], dynamic: Option<bool>) -> (Vec<Vec<u32>>, Vec<u32>) {
    let individual = states.iter().map(|state| {
        if state.state.is_empty() { return Vec::new(); }
        state.mask_for_forced(dynamic.unwrap_or_else(|| state.constraint.uses_dynamic_runtime()))
    }).collect::<Vec<_>>();
    let mut union = vec![0; individual.iter().map(Vec::len).max().unwrap_or(0)];
    for mask in &individual {
        for (out, &word) in union.iter_mut().zip(mask) { *out |= word; }
    }
    (individual, union)
}

fn event_possible(
    constraints: &[&Constraint], frontiers: &[ParserStateMap], budget: &mut Budget,
) -> Result<bool, ()> {
    for (&constraint, state) in constraints.iter().zip(frontiers) {
        if state.is_empty() { continue; }
        for special in &constraint.special_token_terminals {
            if constraint.is_late_grammar_placeholder_terminal(special.terminal_id) { continue; }
            let count = u64::from(special.token_id_end) - u64::from(special.token_id) + 1;
            let end_count = constraint.end_tokens.iter()
                .filter(|&&id| special.contains(id)).count() as u64;
            // Reserved root end IDs are not ordinary event transitions.
            if count == end_count { continue; }
            budget.probe()?;
            if advance_special_terminal_paths(constraint, state, special.terminal_id).is_some() {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// A physical transition bound is used only when there can be no zero-byte
/// lexer finalization/dispatch. All other representations use the full alphabet.
fn byte_candidates(constraints: &[&Constraint], frontiers: &[ParserStateMap]) -> [bool; 256] {
    let mut bytes = [false; 256];
    for (&constraint, frontier) in constraints.iter().zip(frontiers) {
        if frontier.is_empty() { continue; }
        if constraint.uses_compact_segmented_parser_runtime() { return [true; 256]; }
        let lexer = &constraint.tokenizer;
        for &state in frontier.keys() {
            if state >= lexer.num_states() || lexer.state_has_epsilon_transitions(state)
                || !lexer.matched_terminal_bitset(state).is_empty()
            { return [true; 256]; }
            for (byte, _) in lexer.transitions_from(state) { bytes[byte as usize] = true; }
        }
    }
    bytes
}

fn byte_successor(
    constraints: &[&Constraint], frontiers: &[ParserStateMap], scratch: &mut [CommitBuffers],
    byte: u8, budget: &mut Budget,
) -> Result<Option<Vec<ParserStateMap>>, ()> {
    let mut next = Vec::with_capacity(frontiers.len());
    let mut live = false;
    for ((&constraint, frontier), buffers) in constraints.iter().zip(frontiers).zip(scratch) {
        if frontier.is_empty() { next.push(ParserStateMap::default()); continue; }
        budget.probe()?;
        let mut state = frontier.clone();
        buffers.reset_all();
        // A proof/metadata/resource failure is NOT evidence of rejection.
        // Only the commit engine's explicit grammatical-rejection result can
        // discard this lane. Other errors withhold the unproved continuation.
        match commit_bytes_impl(constraint, &mut state, &[byte], buffers) {
            Ok(()) => { live |= !state.is_empty(); next.push(state); }
            Err(message) if message == REJECTED => next.push(ParserStateMap::default()),
            Err(_) => return Err(()),
        }
    }
    Ok(live.then_some(next))
}

fn byte_prefix(
    states: &[ConstraintState<'_>], max_bytes: usize, budget: &mut Budget,
) -> Vec<u8> {
    byte_prefix_with_bounds(states, max_bytes, budget, true)
}

fn byte_prefix_with_bounds(
    states: &[ConstraintState<'_>], max_bytes: usize, budget: &mut Budget, physical_bounds: bool,
) -> Vec<u8> {
    let constraints = states.iter().map(|state| state.constraint).collect::<Vec<_>>();
    let mut frontier = states.iter().map(|state| state.state.clone()).collect::<Vec<_>>();
    let mut scratch = constraints.iter().map(|c| CommitBuffers::for_constraint(c)).collect::<Vec<_>>();
    let mut prefix = Vec::new();
    while prefix.len() < max_bytes {
        for (&constraint, state) in constraints.iter().zip(&mut frontier) {
            expand_runtime_product_states(constraint, state);
        }
        if constraints.iter().zip(&frontier).any(|(&c, s)| frontier_is_accepting(c, s, false)) { break; }
        if !matches!(event_possible(&constraints, &frontier, budget), Ok(false)) { break; }
        let candidates = if physical_bounds { byte_candidates(&constraints, &frontier) } else { [true; 256] };
        let mut unique = None;
        for byte in 0..=u8::MAX {
            if !candidates[byte as usize] { continue; }
            match byte_successor(&constraints, &frontier, &mut scratch, byte, budget) {
                Ok(Some(next)) => {
                    if unique.is_some() { return prefix; }
                    unique = Some((byte, next));
                }
                Ok(None) => {}
                Err(()) => return prefix,
            }
        }
        let Some((byte, next)) = unique else { break; };
        prefix.push(byte);
        frontier = next;
    }
    prefix
}

fn canonical_token(constraint: &Constraint, mask: &[u32], remaining: &[u8]) -> Option<(u32, usize)> {
    if remaining.is_empty() { return None; }
    let mut best = None;
    for id in ids(mask) {
        if constraint.end_tokens.binary_search(&id).is_ok() { continue; }
        let Some(bytes) = constraint.token_bytes_for_id(id).filter(|bytes| !bytes.is_empty()) else { continue; };
        if bytes.len() > remaining.len() && bytes.starts_with(remaining) {
            // This is an actually admitted longer token, not an arbitrary
            // vocabulary spelling. Its future can change the canonical choice.
            return None;
        }
        if remaining.starts_with(bytes) && best.is_none_or(|(old_id, old_len)| {
            bytes.len() > old_len || (bytes.len() == old_len && id < old_id)
        }) { best = Some((id, bytes.len())); }
    }
    best
}

fn commit_selected(
    states: &mut Vec<ConstraintState<'_>>, individual: &[Vec<u32>], id: u32, dynamic: Option<bool>,
) -> bool {
    for (state, mask) in states.iter_mut().zip(individual) {
        if !bit(mask, id) { state.state.clear(); continue; }
        let result = if dynamic.unwrap_or_else(|| state.constraint.uses_dynamic_runtime()) {
            state.commit_token_dynamic(id)
        } else { state.commit_token_raw(id) };
        if result.is_err() { return false; }
    }
    states.retain(|state| !state.state.is_empty());
    !states.is_empty()
}

pub(crate) fn force(states: &[ConstraintState<'_>], dynamic: Option<bool>) -> Vec<u32> {
    force_result(states, dynamic)
}

/// Public-query adapter for the canonical reference implementation. A proof or
/// resource limit returns the empty prefix.
pub(crate) fn force_result(
    states: &[ConstraintState<'_>], dynamic: Option<bool>,
) -> Vec<u32> {
    glrmask_invariant::__private::catch_compilation_resource_limit(|| force_inner(states, dynamic))
        .unwrap_or_default()
}

/// Strict token-ID forcing: no canonical spelling selection. Each next ID is
/// the only one admitted by the union's authoritative token mask, then committed
/// through every applicable owner. Stop at any accepting boundary or ambiguity.
pub(crate) fn force_token_ids(states: &[ConstraintState<'_>], dynamic: Option<bool>) -> Vec<u32> {
    glrmask_invariant::__private::catch_compilation_resource_limit(|| {
        let mut states = states.to_vec();
        states.retain(|state| !state.state.is_empty());
        let mut output = Vec::new();
        while !states.is_empty() && output.len() < MAX_OUTPUT_IDS {
            if states.iter().any(ConstraintState::frontier_complete) { break; }
            let (individual, union) = masks(&states, dynamic);
            let Some(id) = singleton(&union) else { break; };
            if !commit_selected(&mut states, &individual, id, dynamic) { break; }
            output.push(id);
        }
        output
    }).unwrap_or_default()
}

fn force_inner(states: &[ConstraintState<'_>], dynamic: Option<bool>) -> Vec<u32> {
    if states.is_empty() || states.iter().any(ConstraintState::frontier_complete) || !same_spelling_policy(states) {
        return Vec::new();
    }
    let mut states = states.to_vec();
    states.retain(|state| !state.state.is_empty());
    let mut budget = Budget(MAX_PROBES);
    let mut output = Vec::new();
    let mut output_bytes = 0;
    while !states.is_empty() && output.len() < MAX_OUTPUT_IDS && output_bytes < MAX_OUTPUT_BYTES {
        if states.iter().any(ConstraintState::frontier_complete) { break; }
        let (individual, mask) = masks(&states, dynamic);
        let constraints = states.iter().map(|s| s.constraint).collect::<Vec<_>>();
        let mut frontiers = states.iter().map(|s| s.state.clone()).collect::<Vec<_>>();
        for (&c, s) in constraints.iter().zip(&mut frontiers) { expand_runtime_product_states(c, s); }
        let event = match event_possible(&constraints, &frontiers, &mut budget) {
            Ok(event) => event,
            Err(()) => break,
        };
        if event {
            let Some(id) = singleton(&mask) else { break; };
            let width = states[0].constraint.token_bytes_for_id(id).map_or(0, <[u8]>::len);
            if width > MAX_OUTPUT_BYTES - output_bytes { break; }
            if !commit_selected(&mut states, &individual, id, dynamic) { break; }
            output.push(id);
            output_bytes += width;
            continue;
        }
        let prefix = byte_prefix(&states, MAX_OUTPUT_BYTES - output_bytes, &mut budget);
        if prefix.is_empty() { break; }
        let mut offset = 0;
        let mut current_masks = individual;
        let mut current_mask = mask;
        while offset < prefix.len() {
            if states.iter().any(ConstraintState::is_accepting) || output.len() >= MAX_OUTPUT_IDS { return output; }
            let Some((id, width)) = canonical_token(states[0].constraint, &current_mask, &prefix[offset..]) else { return output; };
            if !commit_selected(&mut states, &current_masks, id, dynamic) { return output; }
            output.push(id);
            output_bytes += width;
            offset += width;
            if offset < prefix.len() {
                // The full ID successor, not a byte-shadow cursor, owns the next proof.
                let constraints = states.iter().map(|s| s.constraint).collect::<Vec<_>>();
                let mut next = states.iter().map(|s| s.state.clone()).collect::<Vec<_>>();
                for (&c, s) in constraints.iter().zip(&mut next) { expand_runtime_product_states(c, s); }
                if !matches!(event_possible(&constraints, &next, &mut budget), Ok(false)) { return output; }
                (current_masks, current_mask) = masks(&states, dynamic);
            }
        }
    }
    output
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BuildOptions, Grammar, MaskStrategy, TokenSet, Vocab};

    #[test]
    fn union_forcing_returns_common_prefix_not_equal_whole_vectors() {
        let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"c".to_vec())]);
        for mode in [MaskStrategy::TokenTrie, MaskStrategy::TokenClassTrie, MaskStrategy::ParserDwa] {
            let options=BuildOptions::default().optimization(mode);
            let first=Grammar::from_ebnf("start ::= \"ab\"").unwrap().compile_with_build_options(&vocab,options.clone()).unwrap();
            let second=Grammar::from_ebnf("start ::= \"abc\"").unwrap().compile_with_build_options(&vocab,options).unwrap();
            let states=[first.start(),second.start()];
            assert_eq!(force(&states,None),[0,1],"{mode:?}");
        }
    }

    #[test]
    fn union_with_accepting_owner_must_not_force_a_continuation() {
        let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec())]);
        let first=Grammar::from_ebnf("start ::= \"a\"?").unwrap().compile(&vocab).unwrap();
        let second=Grammar::from_ebnf("start ::= \"ab\"").unwrap().compile(&vocab).unwrap();
        assert!(force(&[first.start(),second.start()],None).is_empty());
    }

    #[test]
    fn different_owner_spelling_maps_cannot_be_canonicalized_together() {
        let a=Vocab::new(vec![(0,b"a".to_vec())]);
        let b=Vocab::new(vec![(0,b"b".to_vec())]);
        let first=Grammar::from_ebnf("start ::= \"a\"").unwrap().compile(&a).unwrap();
        let second=Grammar::from_ebnf("start ::= \"b\"").unwrap().compile(&b).unwrap();
        assert!(force(&[first.start(),second.start()],None).is_empty());
    }

    #[test]
    fn greedy_ab_stops_before_optional_event_after_a() {
        let vocab=Vocab::new(vec![(0,b"a".to_vec()),(1,b"ab".to_vec()),(2,b"b".to_vec())]).with_special_tokens([9]);
        let ordinary=Grammar::from_glrm("start root nt root=\"ab\"").unwrap();
        let guarded=Grammar::from_glrm(
            "extern token MARK start root nt root=\"ab\" | \"a\" MARK \"b\"").unwrap()
            .bind_snapshot("MARK",TokenSet::new([9])).unwrap();
        for mode in [MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa] {
            let options=BuildOptions::default().optimization(mode);
            let baseline=ordinary.compile_with_build_options(&vocab,options.clone()).unwrap();
            assert_eq!(baseline.start().forced_tokens(),[1],"{mode:?}");
            let event=guarded.compile_with_build_options(&vocab,options).unwrap();
            assert!(event.start().forced_tokens().is_empty(),"{mode:?}");
        }
    }

    #[test]
    fn physical_byte_bound_matches_complete_alphabet_probe() {
        let v=Vocab::new((0..=255).map(|i| (i,vec![i as u8])).collect());
        for source in [
            "glrm 1; start root; nt root = \"abc\";",
            "glrm 1; start root; t A=/a+/; nt root = A \"b\";",
            "glrm 1; start root; t W=/[ \\t]+/; ignore W; nt root=\"a\" \"b\";",
            "glrm 1; start root; nt root = \"\u{e9}\u{e9}\";",
        ] {
            for mode in [MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa] {
                let c=Grammar::from_glrm(source).unwrap().compile_with_build_options(&v,BuildOptions::default().optimization(mode)).unwrap();
                let seed=[c.start()];
                let fast=byte_prefix_with_bounds(&seed,32,&mut Budget(MAX_PROBES),true);
                let full=byte_prefix_with_bounds(&seed,32,&mut Budget(MAX_PROBES),false);
                assert_eq!(fast,full,"{mode:?}: {source}");
                assert!(byte_prefix(&seed,32,&mut Budget(0)).is_empty());
            }
        }
    }
}
