//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

pub(super) fn merge_parser_state(
    states: &mut ParserStatesByTokenizer,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    states
        .entry(tokenizer_state)
        .and_modify(|existing| *existing = existing.merge(&gss))
        .or_insert(gss);
}

pub(super) fn queue_parser_state(
    processing_queue: &mut [ParserStatesByTokenizer],
    pending_state: &mut ParserStatesByTokenizer,
    new_offset: usize,
    total_len: usize,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    if new_offset == total_len {
        merge_parser_state(pending_state, tokenizer_state, gss);
    } else {
        merge_parser_state(&mut processing_queue[new_offset], tokenizer_state, gss);
    }
}

pub(super) fn finalize_pending_state(mut pending_state: ParserStatesByTokenizer) -> BTreeMap<u32, ParserGSS> {
    match pending_state.len() {
        0 => BTreeMap::new(),
        1 => {
            let (tokenizer_state, parser_state) = pending_state.drain().next().unwrap();
            let fused = parser_state.fuse(Some(1));
            if fused.is_empty() {
                BTreeMap::new()
            } else {
                let mut new_state = BTreeMap::new();
                new_state.insert(tokenizer_state, fused);
                new_state
            }
        }
        _ => {
            let mut new_state: BTreeMap<u32, ParserGSS> = pending_state.into_iter().collect();
            for parser_state in new_state.values_mut() {
                *parser_state = parser_state.fuse(Some(1));
            }
            new_state.retain(|_, parser_state| !parser_state.is_empty());
            new_state
        }
    }
}

pub(super) fn merge_small_parser_state(
    states: &mut SmallVec<[(u32, ParserGSS); 4]>,
    tokenizer_state: u32,
    gss: ParserGSS,
) {
    for (existing_state, existing_gss) in states.iter_mut() {
        if *existing_state == tokenizer_state {
            *existing_gss = existing_gss.merge(&gss);
            return;
        }
    }
    states.push((tokenizer_state, gss));
}

