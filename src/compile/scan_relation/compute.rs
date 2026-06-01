//! Public scan-relation construction entry points used by the compile pipeline.
//!
//! This is the only module in `compile::scan_relation` that should be called by
//! pipeline phases.  It wires together ordered vocabulary artifacts, root
//! collection, vocabulary materialization, optional CanMatch vocabulary
//! quotienting, and final mapped-artifact construction.

use super::prelude::*;
use super::collector;
use super::ordered_vocab::{
    build_internal_token_bytes_from_groups,
    emit_ordered_vocab_cache_profile,
    get_ordered_vocab_trie_artifacts,
    get_ordered_vocab_trie_artifacts_for_vocab,
    OrderedVocabCacheProfile,
    OrderedVocabTrieArtifacts,
};
use super::root_collect::{
    collect_sparse_root_can_match,
    root_terminal_union_count,
    sparse_root_collect_enabled,
    sparse_root_state_limit,
    sparse_root_terminal_limit,
};
use super::types::*;
use super::vocab_equivalence::{
    compute_scan_relation_vocab_equivalence_map,
    compute_scan_relation_vocab_equivalence_map_fast,
    scan_relation_vocab_equiv_enabled,
};
use super::vocab_materialize::build_scan_relation_vocab_and_weights_from_interval_maps;
use crate::compile::scan_relation::profile::elapsed_ms;

pub(crate) fn compute_scan_relation(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
    _config: ScanRelationConfig,
) -> ScanRelationComputation {
    compute_scan_relation_with_artifacts(
        tokenizer,
        token_bytes.len(),
        get_ordered_vocab_trie_artifacts(token_bytes),
        None,
    )
}

fn compute_scan_relation_with_artifacts(
    tokenizer: &Tokenizer,
    original_token_count: usize,
    artifacts_and_profile: (OrderedVocabTrieArtifacts, OrderedVocabCacheProfile),
    initial_vocab_map: Option<&ManyToOneIdMap>,
) -> ScanRelationComputation {
    let scan_relation_started_at = Instant::now();

    let (artifacts, ordered_vocab_cache_profile) = artifacts_and_profile;
    emit_ordered_vocab_cache_profile(ordered_vocab_cache_profile);
    let ordered_vocab = artifacts.ordered_vocab;
    let trie = artifacts.trie;

    let trie_build_states: Vec<u32> = (0..tokenizer.num_states()).collect();

    let root_terminal_union = root_terminal_union_count(tokenizer, &trie_build_states);
    let use_sparse_root_collect = sparse_root_collect_enabled()
        && trie_build_states.len() <= sparse_root_state_limit()
        && root_terminal_union <= sparse_root_terminal_limit();

    let trie_class_result = if use_sparse_root_collect {
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][trie_build_sparse_root] states={} terminals={} max_states={} max_terminals={}",
                trie_build_states.len(),
                root_terminal_union,
                sparse_root_state_limit(),
                sparse_root_terminal_limit(),
            );
        }
        collect_sparse_root_can_match(
            tokenizer,
            &trie.root,
            &trie_build_states,
            None,
        )
    } else {
        collector::collect_can_match_interval_trie_class_build_with_classes(
            tokenizer,
            &trie.root,
            &trie_build_states,
            None,
        )
        .0
    };

    let scan_relation_collect_ms = elapsed_ms(scan_relation_started_at);

    let scan_relation_vocab_started_at = Instant::now();
    let (scan_relation_vocab, can_match) = build_scan_relation_vocab_and_weights_from_interval_maps(&trie_class_result.class_maps, &trie_class_result.state_classes, ordered_vocab.as_ref());

    let local_vocab_map = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
        scan_relation_vocab.original_to_internal.clone(),
        scan_relation_vocab.internal_to_originals.len() as u32,
    );
    let vocab_tokens = if let Some(initial_vocab_map) = initial_vocab_map {
        initial_vocab_map.compose(&local_vocab_map)
    } else {
        local_vocab_map
    };

    let scan_relation_id_map = InternalIdMap {
        tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            trie_class_result.state_classes.clone(),
            trie_class_result.state_classes.iter().copied().filter(|&class_id| class_id != u32::MAX).max().map(|class_id| class_id + 1).unwrap_or(0),
        ),
        vocab_tokens,
    };

    if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some() || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
        eprintln!("[glrmask/profile][scan_relation_vocab] original_tokens={} ordered_byte_tokens={} can_match_tokens={}", original_token_count, ordered_vocab.ordered_to_originals.len(), scan_relation_id_map.vocab_tokens.internal_to_originals.len());
    }

    let scan_relation_vocab_ms = elapsed_ms(scan_relation_vocab_started_at);

    ScanRelationComputation {
        mapped_can_match: MappedArtifact::new(can_match, scan_relation_id_map),
        profile: ScanRelationProfile { scan_relation_collect_ms, scan_relation_vocab_ms },
    }
}

pub(crate) fn compute_scan_relation_for_vocab(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _config: ScanRelationConfig,
) -> ScanRelationComputation {
    if scan_relation_vocab_equiv_enabled() {
        let (full_artifacts, full_profile) = get_ordered_vocab_trie_artifacts_for_vocab(vocab);
        emit_ordered_vocab_cache_profile(full_profile);
        let vocab_equiv_started_at = Instant::now();
        let use_naive = std::env::var("GLRMASK_SCAN_RELATION_VOCAB_EQUIV_NAIVE")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let can_match_vocab_map = if use_naive {
            compute_scan_relation_vocab_equivalence_map(
                tokenizer,
                full_artifacts.ordered_vocab.as_ref(),
                full_artifacts.trie.as_ref(),
            )
        } else {
            compute_scan_relation_vocab_equivalence_map_fast(tokenizer, full_artifacts.ordered_vocab.as_ref())
        };
        if std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
            || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
        {
            eprintln!(
                "[glrmask/profile][scan_relation_vocab_equiv] original_tokens={} can_match_vocab_classes={} mode={} ms={:.3}",
                vocab.entries.len(),
                can_match_vocab_map.internal_to_originals.len(),
                if use_naive { "naive" } else { "fast" },
                elapsed_ms(vocab_equiv_started_at),
            );
        }
        let compact_token_bytes =
            build_internal_token_bytes_from_groups(vocab, &can_match_vocab_map.internal_to_originals);
        return compute_scan_relation_with_artifacts(
            tokenizer,
            vocab.entries.len(),
            get_ordered_vocab_trie_artifacts(&compact_token_bytes),
            Some(&can_match_vocab_map),
        );
    }

    compute_scan_relation_with_artifacts(
        tokenizer,
        vocab.entries.len(),
        get_ordered_vocab_trie_artifacts_for_vocab(vocab),
        None,
    )
}

pub(crate) fn prepare_vocab_for_scan_relation(vocab: &Vocab) {
    let _ = get_ordered_vocab_trie_artifacts_for_vocab(vocab);
}
