//! Exact projected L1 terminal DWA construction.

pub mod implementations;

use std::sync::{Arc, OnceLock};

use super::types::{
    LocalIdMapTerminalDwa, LocalTerminalBuild, TerminalColoring, TerminalDwaPhaseProfile,
};
use crate::Vocab;
use crate::automata::lexer::{Lexer, tokenizer::Tokenizer};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::grammar::flat::TerminalID;

const L1_GENERIC_NFA_TOKEN_BOUNDED_MAX_VOCAB: usize = 20_000;

pub use crate::vocab::LexicalVocabulary as L1IdentityVocabOrder;

impl crate::vocab::VocabDerivedArtifact
    for super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie
{
}

fn l1_token_bounded_analysis_trie(
    vocab: &Vocab,
) -> Option<Arc<super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie>>
{
    if vocab.len() > L1_GENERIC_NFA_TOKEN_BOUNDED_MAX_VOCAB {
        return None;
    }
    if let Some(cached) = vocab.vocab_derived_cache_get::<
        super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >() {
        return Some(cached);
    }
    let order = l1_identity_vocab_order(vocab);
    let trie = Arc::new(
        super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie::from_radix(
            order.trie(),
        ),
    );
    vocab.vocab_derived_cache_set(Arc::clone(&trie));
    Some(trie)
}

pub fn prepare_l1_token_bounded_analysis_trie(vocab: &Vocab) {
    let _ = l1_token_bounded_analysis_trie(vocab);
}

pub fn prepared_l1_token_bounded_analysis_trie(
    vocab: &Vocab,
) -> Option<Arc<super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie>>
{
    l1_token_bounded_analysis_trie(vocab)
}

/// Reuse a token-bounded analysis trie only when some earlier vocabulary
/// preparation or branch has already paid to build it.
///
/// Routing callers frequently need only a tiny token subset. Forcing the
/// whole-partition trie on a cold path can cost much more than the exact
/// subset analysis it is intended to accelerate.
pub(crate) fn cached_l1_token_bounded_analysis_trie(
    vocab: &Vocab,
) -> Option<Arc<super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie>>
{
    vocab.vocab_derived_cache_get::<
        super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >()
}

fn l1_identity_vocab_order(vocab: &Vocab) -> Arc<L1IdentityVocabOrder> {
    vocab.lexical_vocabulary()
}
pub fn prepare_l1_identity_vocab_order(vocab: &Vocab) {
    let _ = l1_identity_vocab_order(vocab);
}

pub fn prepared_l1_identity_vocab_order(vocab: &Vocab) -> Arc<L1IdentityVocabOrder> {
    l1_identity_vocab_order(vocab)
}

/// Reuse an already-prepared parent order without forcing a full-vocabulary
/// sort solely to accelerate a much smaller subset build.
pub(crate) fn cached_l1_identity_vocab_order(vocab: &Vocab) -> Option<Arc<L1IdentityVocabOrder>> {
    vocab.vocab_derived_cache_get::<L1IdentityVocabOrder>()
}

/// Resolve exact token-byte groups from the already-prepared identity order.
///
/// This never forces preparation: tiny grammar-specific fast paths may use it
/// when CFA/runtime vocabulary preparation has already paid for the byte order,
/// but cold standalone callers should remain free to choose a cheaper fallback.
pub fn cached_l1_exact_token_groups(
    vocab: &Vocab,
    byte_strings: &[Vec<u8>],
) -> Option<Vec<Vec<u32>>> {
    let order = vocab.vocab_derived_cache_get::<L1IdentityVocabOrder>()?;
    Some(byte_strings.iter().map(|bytes| {
        order.ordered_token_bytes.binary_search_by(|word| word.as_ref().cmp(bytes.as_slice()))
            .ok().map(|index| order.ordered_to_originals[index].clone()).unwrap_or_default()
    }).collect())
}
/// Reuse the parent L1 byte order for an L2P boundary subset when it is already cached.
pub(crate) fn inherit_l2p_lexical_entry_order(
    parent_vocab: &Vocab,
    child_vocab: &Vocab,
    child_token_ids_in_entry_order: &[u32],
) {
    let Some(parent_order) = parent_vocab.vocab_derived_cache_get::<L1IdentityVocabOrder>() else {
        return;
    };
    debug_assert_eq!(child_token_ids_in_entry_order.len(), child_vocab.len());
    let mut entry_indices = (0..child_token_ids_in_entry_order.len() as u32).collect::<Vec<_>>();
    entry_indices.sort_unstable_by_key(|&entry_index| {
        let token_id = child_token_ids_in_entry_order[entry_index as usize] as usize;
        (parent_order.original_to_ordered()[token_id], token_id)
    });
    child_vocab.vocab_derived_cache_set(Arc::new(
        super::l2p::equivalence_analysis::shared::VocabLexicalEntryOrder {
            entry_indices: entry_indices.into(),
        },
    ));
}
pub fn prepare_l1_finite_vocab_projection(vocab: &Vocab) {
    implementations::prepare_finite_vocab_projection(vocab);
}

pub fn fast_projected_l1_id_map_max_tsids() -> usize {
    static MAX_TSID: OnceLock<usize> = OnceLock::new();
    *MAX_TSID.get_or_init(|| {
        std::env::var("GLRMASK_L1_FAST_PROJECTED_ID_MAP_MAX_TSIDS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(16_384)
    })
}

/// Build an L1 id map and terminal DWA using the projected-residual compiler.
pub fn build_l1_id_map_and_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_generic_nfa_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_generic_nfa_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
) -> Option<LocalIdMapTerminalDwa> {
    build_l1_id_map_and_terminal_dwa_mode(
        partition_label,
        tokenizer,
        vocab,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        grammar,
        active_terminals,
        flat_trans,
        transitions_by_byte,
        initial_state_map,
        shared_generic_nfa_topology,
        shared_generic_nfa_trie,
        subset_parent_order,
        false,
        false,
    ).map(LocalTerminalBuild::into_weighted)
}

pub fn build_l1_id_map_and_terminal_dwa_mode(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
    flat_trans: &Arc<[u32]>,
    transitions_by_byte: Option<&[u32]>,
    initial_state_map: Option<&ManyToOneIdMap>,
    shared_generic_nfa_topology: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTopology,
    >,
    shared_generic_nfa_trie: Option<
        &super::l2p::equivalence_analysis::state_equivalence::nfa::TokenBoundedAnalysisTrie,
    >,
    subset_parent_order: Option<&L1IdentityVocabOrder>,
    initial_state_domain_is_exact: bool,
    id_map_only: bool,
) -> Option<LocalTerminalBuild> {
    let input = implementations::BuildInput {
        partition_label,
        tokenizer,
        vocab,
        terminal_coloring,
        use_terminal_coloring,
        ignore_terminal,
        grammar,
        active_terminals,
        flat_trans,
        transitions_by_byte,
        initial_state_map,
        shared_generic_nfa_topology,
        shared_generic_nfa_trie,
        subset_parent_order,
        initial_state_domain_is_exact,
        id_map_only,
    };
    if id_map_only {
        // Vocabulary-partition compilation consumes only the exact L1 token
        // quotient.  Use the dedicated vocab-equivalence kernel rather than
        // constructing tokenizer-state/DWA-shaped artifacts and discarding
        // them immediately afterwards.
        let result = implementations::build_projected_vocab_equivalence(input)?;
        return Some(LocalTerminalBuild::VocabMap {
            map: result.vocab_map,
            profile: TerminalDwaPhaseProfile {
                id_map_ms: result.total_wall_ms,
                compact_ms: result.compact_ms,
                ..TerminalDwaPhaseProfile::default()
            },
        });
    }
    let mut result = implementations::build(input)?;
    if initial_state_domain_is_exact {
        super::scope::complete_with_continuation_singletons(&mut result.id_map.tokenizer_states);
    }
    Some(LocalTerminalBuild::Weighted(result))
}

fn collect_active_terminal_signature(
    tokenizer: &Tokenizer,
    state: u32,
    active_terminals: &[bool],
) -> Vec<u32> {
    let mut signature = Vec::<u32>::new();
    for tid in tokenizer.matched_terminals_iter(state) {
        if active_terminals.get(tid as usize).copied().unwrap_or(false) {
            signature.push(tid);
        }
    }
    for tid in tokenizer.tokens_accessible_from_state(state).iter() {
        if active_terminals.get(tid).copied().unwrap_or(false) {
            signature.push(tid as u32);
        }
    }
    signature.sort_unstable();
    signature.dedup();
    signature
}

pub fn build_flat_transition_table(tokenizer: &Tokenizer) -> Vec<u32> {
    let mut flat_trans = vec![0; tokenizer.num_states() as usize * 256];
    for (state_idx, row) in flat_trans.as_chunks_mut::<256>().0.iter_mut().enumerate() {
        tokenizer.fill_transition_row(state_idx as u32, row);
    }
    flat_trans
}
