//! Vocabulary partitioning for Terminal-DWA construction.
//!
//! The Terminal DWA relation is defined over the full vocabulary.  Partitioning
//! is an implementation strategy: build local automata over smaller token sets,
//! then merge their id maps and weights.  This module contains exactly that
//! strategy choice so that the top-level builder can read like the paper.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::parser::glr::analysis::AnalyzedGrammar;
use crate::compile::terminal_dwa::classify::{
    self,
    classify_vocab_char_type,
    SharedClassifyCache,
};
use crate::compile::terminal_dwa::options::{
    self,
    VocabPartitionScheme,
};
use crate::compile::terminal_dwa::types::compile_profile_enabled;
use crate::ds::bitset::BitSet;
use crate::Vocab;

#[derive(Debug)]
struct CharTypeSubVocabs {
    sub_vocabs: Arc<[Vocab]>,
}

impl crate::vocab::VocabDerivedArtifact for CharTypeSubVocabs {}

pub(crate) fn vocab_from_token_partitions(vocab: &Vocab, token_partitions: Vec<Vec<u32>>) -> Arc<[Vocab]> {
    token_partitions
        .into_iter()
        .map(|token_ids| {
            let entries = token_ids
                .into_iter()
                .filter_map(|token_id| vocab.entries.get(&token_id).map(|bytes| (token_id, bytes.clone())))
                .collect();
            Vocab::new(entries, None)
        })
        .collect::<Vec<_>>()
        .into()
}

pub(crate) fn build_char_type_sub_vocabs(vocab: &Vocab) -> Arc<[Vocab]> {
    if let Some(cached) = vocab.vocab_derived_cache_get::<CharTypeSubVocabs>() {
        return Arc::clone(&cached.sub_vocabs);
    }

    let mut partition_entries: Vec<Vec<(u32, Vec<u8>)>> = (0..7).map(|_| Vec::new()).collect();
    for (&token_id, bytes) in vocab.entries.iter() {
        let idx = classify_vocab_char_type(bytes) as usize;
        partition_entries[idx].push((token_id, bytes.clone()));
    }
    let sub_vocabs: Arc<[Vocab]> = partition_entries
        .into_iter()
        .map(|entries| Vocab::new(entries, None))
        .collect::<Vec<_>>()
        .into();
    vocab.vocab_derived_cache_set(Arc::new(CharTypeSubVocabs {
        sub_vocabs: Arc::clone(&sub_vocabs),
    }));
    sub_vocabs
}

/// Precompute the vocabulary-derived artifacts used by Terminal-DWA builders.
///
/// This is intentionally a cache warmer rather than a semantic phase.  If it is
/// omitted, the same data is constructed lazily during the build.
pub(crate) fn prepare_vocab_for_terminal_dwa(vocab: &Vocab) {
    classify::prepare_vocab_for_terminal_classification(vocab);
    super::direct_partition::prepare_direct_partition_identity_vocab_order(vocab);

    if options::vocab_partition_scheme_from_env() == VocabPartitionScheme::CharType {
        for sub_vocab in build_char_type_sub_vocabs(vocab).iter() {
            classify::prepare_vocab_for_terminal_classification(sub_vocab);
            super::direct_partition::prepare_direct_partition_identity_vocab_order(sub_vocab);
        }
    }
}

/// Choose the local sub-vocabularies used to build and later merge Terminal DWAs.
///
/// The result is a partition of the original vocabulary IDs.  The choice affects
/// compile time and intermediate automaton size, but the merged Terminal DWA must
/// denote the same relation as building once over the full vocabulary.
pub(crate) fn choose_terminal_dwa_sub_vocabs(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    shared_classify_cache: &SharedClassifyCache,
) -> Arc<[Vocab]> {
    match options::vocab_partition_scheme_from_env() {
        VocabPartitionScheme::CharType => build_char_type_sub_vocabs(vocab),
        VocabPartitionScheme::PairPartitionCost => choose_cost_partitioned_sub_vocabs(
            tokenizer,
            vocab,
            grammar,
            disallowed_follows,
            shared_classify_cache,
        ),
        VocabPartitionScheme::AutoPairPartitionCost => choose_auto_partitioned_sub_vocabs(
            tokenizer,
            vocab,
            grammar,
            disallowed_follows,
            shared_classify_cache,
        ),
    }
}

fn choose_cost_partitioned_sub_vocabs(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    shared_classify_cache: &SharedClassifyCache,
) -> Arc<[Vocab]> {
    let cost_fn = options::pair_partition_cost_fn_from_env();
    let objective = options::pair_partition_objective_from_env();
    let num_partitions = options::pair_partition_count_from_env();
    let bytesets = shared_classify_cache.get_or_init(|| {
        classify::SharedClassifyBytesets::build(tokenizer, grammar.num_terminals)
    });
    let partitioning = classify::partition_vocab_by_pair_partition_cost(
        vocab,
        bytesets,
        disallowed_follows,
        grammar.num_terminals,
        num_partitions,
        cost_fn,
        objective,
    );

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][pair_partition_cost_partitioning] cost_fn={} objective={} partitions={} estimated_costs={:?} estimated_pair_partition_terminals={:?} objective_score={:.3}",
            cost_fn.as_str(),
            objective.as_str(),
            num_partitions,
            partitioning.estimated_partition_costs,
            partitioning.estimated_pair_partition_terminals,
            partitioning.objective_score,
        );
    }

    vocab_from_token_partitions(vocab, partitioning.partitions)
}

fn choose_auto_partitioned_sub_vocabs(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    shared_classify_cache: &SharedClassifyCache,
) -> Arc<[Vocab]> {
    let cost_fn = options::pair_partition_cost_fn_from_env();
    let objective = options::pair_partition_objective_from_env();
    let num_partitions = options::pair_partition_count_from_env();
    let min_grammar_terminals_limit = options::pair_partition_auto_min_grammar_terminals_from_env();
    let char_token_partitions = classify::partition_vocab_char_type_tokens(vocab);
    let char_partition_sizes = char_token_partitions
        .iter()
        .map(|partition| partition.len())
        .collect::<Vec<_>>();

    if (grammar.num_terminals as usize) < min_grammar_terminals_limit {
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][auto_pair_partition_partition] cost_fn={} objective={} pair_partition_partitions={} grammar_terminals={} disallowed_follows_len={} min_grammar_terminals_limit={} char_partition_sizes={:?} chosen=char_type reason=low_grammar_terminal_count",
                cost_fn.as_str(),
                objective.as_str(),
                num_partitions,
                grammar.num_terminals,
                disallowed_follows.len(),
                min_grammar_terminals_limit,
                char_partition_sizes,
            );
        }
        return vocab_from_token_partitions(vocab, char_token_partitions);
    }

    let bytesets = shared_classify_cache.get_or_init(|| {
        classify::SharedClassifyBytesets::build(tokenizer, grammar.num_terminals)
    });
    let second_largest_limit = options::pair_partition_auto_second_largest_limit_from_env();
    let max_estimated_pair_partition_terminals_limit =
        options::pair_partition_auto_max_estimated_pair_partition_terminals_from_env();
    let min_estimated_pair_partition_terminals_limit =
        options::pair_partition_auto_min_estimated_pair_partition_terminals_from_env();

    let (pair_partition_partitioning, token_pair_partition_map) =
        classify::partition_vocab_by_pair_partition_cost_with_token_map(
            vocab,
            bytesets,
            disallowed_follows,
            grammar.num_terminals,
            num_partitions,
            cost_fn,
            objective,
        );

    let pair_partition_partition_sizes = pair_partition_partitioning
        .partitions
        .iter()
        .map(|token_ids| token_ids.len())
        .collect::<Vec<_>>();
    let mut sorted_sizes = pair_partition_partition_sizes.clone();
    sorted_sizes.sort_unstable_by(|left, right| right.cmp(left));
    let second_largest = sorted_sizes.get(1).copied().unwrap_or(0);
    let max_estimated_pair_partition_terminals = pair_partition_partitioning
        .estimated_pair_partition_terminals
        .iter()
        .copied()
        .max()
        .unwrap_or(0);

    let mut char_costs = Vec::new();
    let mut char_pair_partition_terminals = Vec::new();
    let mut char_score = f64::INFINITY;

    let use_pair_partition = if second_largest <= second_largest_limit
        && max_estimated_pair_partition_terminals >= min_estimated_pair_partition_terminals_limit
        && max_estimated_pair_partition_terminals <= max_estimated_pair_partition_terminals_limit
    {
        let (computed_char_costs, computed_char_pair_partition_terminals, computed_char_score) =
            classify::estimate_pair_partition_objective_for_token_partitions(
                &char_token_partitions,
                &token_pair_partition_map,
                cost_fn,
                objective,
            );
        char_costs = computed_char_costs;
        char_pair_partition_terminals = computed_char_pair_partition_terminals;
        char_score = computed_char_score;
        pair_partition_partitioning.objective_score < char_score
    } else {
        false
    };

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][auto_pair_partition_partition] cost_fn={} objective={} pair_partition_partitions={} pair_partition_score={:.3} char_score={:.3} second_largest={} second_largest_limit={} disallowed_follows_len={} max_estimated_pair_partition_terminals={} min_estimated_pair_partition_terminals_limit={} max_estimated_pair_partition_terminals_limit={} char_partition_sizes={:?} chosen={} pair_partition_sizes={:?} pair_partition_costs={:?} char_costs={:?} pair_partition_pair_partition_terminals={:?} char_pair_partition_terminals={:?}",
            cost_fn.as_str(),
            objective.as_str(),
            num_partitions,
            pair_partition_partitioning.objective_score,
            char_score,
            second_largest,
            second_largest_limit,
            disallowed_follows.len(),
            max_estimated_pair_partition_terminals,
            min_estimated_pair_partition_terminals_limit,
            max_estimated_pair_partition_terminals_limit,
            char_partition_sizes,
            if use_pair_partition { VocabPartitionScheme::PairPartitionCost.as_str() } else { VocabPartitionScheme::CharType.as_str() },
            pair_partition_partition_sizes,
            pair_partition_partitioning.estimated_partition_costs,
            char_costs,
            pair_partition_partitioning.estimated_pair_partition_terminals,
            char_pair_partition_terminals,
        );
    }

    if use_pair_partition {
        vocab_from_token_partitions(vocab, pair_partition_partitioning.partitions)
    } else {
        vocab_from_token_partitions(vocab, char_token_partitions)
    }
}
