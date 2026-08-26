use std::collections::BTreeMap;
use std::sync::Arc;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::constraint_possible_matches::ConstraintPossibleMatchesComputation;
use crate::grammar::flat::{DirectRegularAutomaton, GrammarDef, Symbol, Terminal, TerminalID};
use crate::Vocab;

use crate::runtime::{Constraint, ConstraintState, DynamicMaskVocab, SpecialTokenTerminal};

const DYNAMIC_CONSTRAINT_MAGIC: [u8; 8] = *b"GLRDYN\0\0";
const DYNAMIC_CONSTRAINT_VERSION_V12: u16 = 12;
const DYNAMIC_CONSTRAINT_VERSION_V13: u16 = 13;
const DYNAMIC_CONSTRAINT_VERSION: u16 = 14;
const DYNAMIC_CONSTRAINT_HEADER_LEN: usize = DYNAMIC_CONSTRAINT_MAGIC.len() + 2 + 8;
const DYNAMIC_TRANSFER_MAGIC: [u8; 8] = *b"GLRDXF\0\0";
const DYNAMIC_TRANSFER_VERSION_V1: u16 = 1;
const DYNAMIC_TRANSFER_VERSION_V2: u16 = 2;
const DYNAMIC_TRANSFER_VERSION_V3: u16 = 3;
const DYNAMIC_TRANSFER_VERSION: u16 = 4;

mod compressed_terminal_exprs_serde {
    use super::Expr;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(exprs: &Option<Vec<Expr>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error as _;
        let compressed = match exprs {
            None => None,
            Some(exprs) => {
                let raw = bincode::serialize(exprs).map_err(S::Error::custom)?;
                Some(zstd::bulk::compress(&raw, 1).map_err(S::Error::custom)?)
            }
        };
        compressed.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<Expr>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;
        let compressed = Option::<Vec<u8>>::deserialize(deserializer)?;
        let Some(compressed) = compressed else {
            return Ok(None);
        };
        let raw = zstd::stream::decode_all(compressed.as_slice()).map_err(D::Error::custom)?;
        let exprs = bincode::deserialize(&raw).map_err(D::Error::custom)?;
        Ok(Some(exprs))
    }
}

#[derive(Debug)]
struct CompactTransferTokenizer(Tokenizer);

impl serde::Serialize for CompactTransferTokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        crate::automata::lexer::tokenizer::compact_artifact_serde::serialize(&self.0, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for CompactTransferTokenizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        crate::automata::lexer::tokenizer::compact_artifact_serde::deserialize(deserializer)
            .map(Self)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
    #[serde(default, with = "compressed_terminal_exprs_serde")]
    terminal_exprs: Option<Vec<Expr>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    eos_token_id: Option<u32>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV2 {
    v1: DynamicConstraintPayloadV1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV2 {
    v1: LegacyDynamicConstraintPayloadV1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV10V1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV10V2 {
    v1: LegacyDynamicConstraintPayloadV10V1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV10V3 {
    alternatives: Vec<LegacyDynamicConstraintPayloadV10V2>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV9V1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV9 {
    v1: LegacyDynamicConstraintPayloadV9V1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV3 {
    alternatives: Vec<DynamicConstraintPayloadV2>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV4Alternative {
    constraint: DynamicConstraintPayloadV2,
    late_grammar_slots: Vec<crate::runtime::LateGrammarSlot>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV4 {
    alternatives: Vec<DynamicConstraintPayloadV4Alternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV11V1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV11V2 {
    v1: LegacyDynamicConstraintPayloadV11V1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV11V3 {
    alternatives: Vec<LegacyDynamicConstraintPayloadV11V2>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV12V1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    #[serde(default)]
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV12V2 {
    v1: LegacyDynamicConstraintPayloadV12V1,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV12V3 {
    alternatives: Vec<LegacyDynamicConstraintPayloadV12V2>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferAlternativeV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
    #[serde(default, with = "compressed_terminal_exprs_serde")]
    terminal_exprs: Option<Vec<Expr>>,
    mask_tokenizer: Option<CompactTransferTokenizer>,
    full_to_mask_state: Vec<u32>,

}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV1 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV1>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferAlternativeV3 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
    mask_tokenizer: Option<CompactTransferTokenizer>,
    full_to_mask_state: Vec<u32>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferPayloadV3 {
    alternatives: Vec<LegacyDynamicConstraintTransferAlternativeV3>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferAlternativeV2 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    #[serde(default)]
    ignore_expr: Option<Expr>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferPayloadV2 {
    alternatives: Vec<LegacyDynamicConstraintTransferAlternativeV2>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferAlternativeV1 {
    table: GLRTable,
    terminal_display_names: Vec<String>,
    #[serde(with = "crate::automata::lexer::tokenizer::compact_artifact_serde")]
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintTransferPayloadV1 {
    alternatives: Vec<LegacyDynamicConstraintTransferAlternativeV1>,
}

/// A constraint optimized for low compilation latency.
///
/// Unlike [`Constraint`], this omits terminal-DWA, possible-match, parser-DWA,
/// token-remapping, and dense-mask compilation. It produces the same masks as
/// [`Constraint`] but performs more work during mask generation.
#[derive(Debug, Clone)]
pub struct DynamicConstraint {
    pub(crate) inner: Constraint,
    alternatives: Vec<Constraint>,
    // Retained only in memory so a freshly compiled dynamic artifact can be
    // materialized as an exact static composer input without bloating its
    // serialized representation. Entries align with inner + alternatives.
    composition_grammars: Vec<Option<GrammarDef>>,
}

impl DynamicConstraint {
    pub(crate) fn from_parts(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        direct_regular_automaton: Option<DirectRegularAutomaton>,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
    ) -> Self {
        let dynamic_mask_vocab =
            crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab);
        Self::from_parts_with_dynamic_vocab(
            table,
            terminal_display_names,
            tokenizer,
            direct_regular_automaton,
            ignore_terminal,
            special_token_terminals,
            vocab,
            dynamic_mask_vocab,
        )
    }

    pub(crate) fn from_parts_with_dynamic_vocab(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        direct_regular_automaton: Option<DirectRegularAutomaton>,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Self {
        let ignore_expr = ignore_terminal
            .and_then(|terminal| tokenizer.terminal_expr(terminal).cloned());
        let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);
        Self::from_payload_v2_with_dynamic_vocab(
            DynamicConstraintPayloadV2 {
                v1: DynamicConstraintPayloadV1 {
                    table,
                    terminal_display_names,
                    tokenizer,
                    ignore_terminal,
                    direct_regular_automaton,
                    token_bytes: vocab.entries_arc(),
                    ignore_expr,
                    terminal_exprs,
                },
                special_token_terminals,
            },
            dynamic_mask_vocab,
        )
    }

    pub(crate) fn from_parts_with_dynamic_vocab_unfinalized(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        direct_regular_automaton: Option<DirectRegularAutomaton>,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Self {
        let ignore_expr = ignore_terminal
            .and_then(|terminal| tokenizer.terminal_expr(terminal).cloned());
        let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);
        let payload = DynamicConstraintPayloadV2 {
            v1: DynamicConstraintPayloadV1 {
                table,
                terminal_display_names,
                tokenizer,
                ignore_terminal,
                direct_regular_automaton,
                token_bytes: vocab.entries_arc(),
                ignore_expr,
                terminal_exprs,
            },
            special_token_terminals,
        };
        Self {
            inner: Self::constraint_from_payload_v2_with_dynamic_vocab(
                payload,
                dynamic_mask_vocab,
            ),
            alternatives: Vec::new(),
            composition_grammars: vec![None],
        }
    }

    pub(crate) fn from_parts_with_possible_matches(
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
        computation: ConstraintPossibleMatchesComputation,
    ) -> Self {
        let ConstraintPossibleMatchesComputation {
            mapped_possible_matches,
            runtime_dynamic_vocab,
            complete,
            profile: _,
        } = computation;
        let (possible_matches, mut id_map) = mapped_possible_matches.into_parts();
        id_map.materialize_deferred_vocab_singletons();
        let ignore_expr = ignore_terminal
            .and_then(|terminal| tokenizer.terminal_expr(terminal).cloned());
        let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);

        let mut result = Self::from_payload_v2_with_dynamic_vocab(
            DynamicConstraintPayloadV2 {
                v1: DynamicConstraintPayloadV1 {
                    table,
                    terminal_display_names,
                    tokenizer,
                    ignore_terminal,
                    direct_regular_automaton: None,
                    token_bytes: vocab.entries_arc(),
                    ignore_expr,
                    terminal_exprs,
                },
                special_token_terminals,
            },
            runtime_dynamic_vocab.vocab,
        );
        result.inner.possible_matches = possible_matches;
        result.inner.possible_matches_complete = complete;
        result.inner.state_to_internal_tsid = id_map.tokenizer_states.original_to_internal;
        result.inner.internal_tsid_to_states = id_map.tokenizer_states.internal_to_originals;
        result.inner.original_token_to_internal = id_map.vocab_tokens.original_to_internal;
        result.inner.internal_token_to_tokens = id_map.vocab_tokens.internal_to_originals;
        result
    }

    fn migrate_legacy_v1(
        payload: LegacyDynamicConstraintPayloadV1,
    ) -> crate::Result<DynamicConstraintPayloadV1> {
        if payload.eos_token_id.is_some() {
            return Err(crate::GlrMaskError::Serialization(
                "legacy dynamic constraint artifact embeds Vocab-level EOS semantics; rebuild it with grammar-level end tokens"
                    .to_owned(),
            ));
        }
        Ok(DynamicConstraintPayloadV1 {
            table: payload.table,
            terminal_display_names: payload.terminal_display_names,
            tokenizer: payload.tokenizer,
            ignore_terminal: payload.ignore_terminal,
            direct_regular_automaton: None,
            token_bytes: payload.token_bytes,
            ignore_expr: None,
            terminal_exprs: None,
        })
    }

    fn from_legacy_payload_v1(payload: LegacyDynamicConstraintPayloadV1) -> crate::Result<Self> {
        Ok(Self::from_payload_v2(DynamicConstraintPayloadV2 {
            v1: Self::migrate_legacy_v1(payload)?,
            special_token_terminals: Vec::new(),
        }))
    }

    fn from_legacy_payload_v2(payload: LegacyDynamicConstraintPayloadV2) -> crate::Result<Self> {
        Ok(Self::from_payload_v2(DynamicConstraintPayloadV2 {
            v1: Self::migrate_legacy_v1(payload.v1)?,
            special_token_terminals: payload.special_token_terminals,
        }))
    }

    fn from_payload_v2(payload: DynamicConstraintPayloadV2) -> Self {
        Self::from_payload_v2_with_dynamic_vocab(payload, DynamicMaskVocab::default())
    }

    fn from_payload_v2_with_dynamic_vocab(
        payload: DynamicConstraintPayloadV2,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Self {
        let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
            payload,
            dynamic_mask_vocab,
        );
        inner.rebuild_dynamic_runtime_caches();
        Self {
            inner,
            alternatives: Vec::new(),
            composition_grammars: vec![None],
        }
    }

    fn constraint_from_payload_v2_with_dynamic_vocab(
        payload: DynamicConstraintPayloadV2,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> Constraint {
        let DynamicConstraintPayloadV2 {
            v1: mut payload,
            special_token_terminals,
        } = payload;
        payload
            .tokenizer
            .restore_terminal_exprs(payload.terminal_exprs.take())
            .expect("dynamic payload terminal expressions must match tokenizer terminal count");
        let max_token_id = payload
            .token_bytes
            .keys()
            .next_back()
            .copied()
            .into_iter()
            .chain(special_token_terminals.iter().map(|special| special.token_id))
            .max()
            .unwrap_or(0);
        let inner = Constraint {
            runtime_backend: crate::runtime::ConstraintRuntimeBackend::Dynamic,
            static_dynamic_overlay: None,
            late_grammar_slots: Vec::new(),
            late_bind_vocab: std::sync::OnceLock::new(),
            scoped_ignore_only_tokens: Vec::new(),
            scoped_ignore_prefix_fusions: Vec::new(),
            parser_dwa: DWA::new(payload.tokenizer.num_states(), max_token_id),
            packed_parser_dwa: None,
            parser_start_final_override: None,
            parser_top_accept: BTreeMap::new(),
            parser_top_accept_parts: BTreeMap::new(),
            direct_regular_l1_complete_by_terminal: BTreeMap::new(),
            packed_non_dwa_weights: None,
            direct_regular_wide_frontier_acceptance: Vec::new(),
            direct_regular_dynamic_hot_frontiers: Vec::new(),
            direct_regular_parser_state_acceptance: Vec::new(),
            direct_regular_automaton: payload.direct_regular_automaton,
            table: payload.table,
            terminal_display_names: payload.terminal_display_names,
            tokenizer: payload.tokenizer,
            tokenizer_has_epsilon_transitions: false,
            ignore_terminal: payload.ignore_terminal,
            special_token_terminals,
            dynamic_mask_vocab,
            lazy_dynamic_mask_vocab: std::sync::OnceLock::new(),
            possible_matches: BTreeMap::new(),
            possible_matches_complete: false,
            state_to_internal_tsid: Vec::new(),
            internal_tsid_to_states: Vec::new(),
            deferred_internal_tsid_to_states: Default::default(),
            composition_reset_tokens_by_terminal: Vec::new(),
            unbound_grammar_placeholders: BTreeMap::new(),
            composition_parser_templates_by_terminal: Vec::new(),
            composition_parser_characterizations_by_terminal: Vec::new(),
            composition_grammar_summary: None,
            terminal_live_states: Vec::new(),
            state_internal_tsid_offsets: Vec::new(),
            state_internal_tsids: Vec::new(),
            runtime_source_state_offset: None,
            runtime_product_source_offsets: Vec::new(),
            runtime_product_source_states: Vec::new(),
            runtime_product_exact_source_states: Vec::new(),
            runtime_product_state_by_source_subset: Default::default(),
            template_dfas_by_terminal: Vec::new(),
            fast_template_dfas_by_terminal: Vec::new(),
            original_token_to_internal: Vec::new(),
            packed_original_token_to_internal: None,
            deferred_original_token_to_internal: std::sync::OnceLock::new(),
            internal_token_to_tokens: Vec::new(),
            deferred_internal_token_to_tokens: std::sync::OnceLock::new(),
            token_bytes: payload.token_bytes,
            packed_token_bytes: None,
            internal_token_bytes: BTreeMap::new(),
            token_bytes_dense: Vec::new(),
            internal_token_buf_masks: Vec::new(),
            word_group_buf_masks: Vec::new(),
            pair_word_group_buf_masks: Default::default(),
            quad_word_group_buf_masks: Default::default(),
            super_word_group_buf_masks: Default::default(),
            mega_word_group_buf_masks: Default::default(),
            giga_word_group_buf_masks: Default::default(),
            word_group_sparse_masks: Vec::new(),
            word_group_prefix_buf_masks: Default::default(),
            word_group_sparse_prefix_entries: Vec::new(),
            quad_group_sparse_masks: Vec::new(),
            quad_group_dense_masks: Vec::new(),
            byte_group_sparse_masks: Vec::new(),
            byte_group_dense_masks: Vec::new(),
            word_group_sparse_total_entries: 0,
            word_group_sparse_max_entries: 0,
            all_tokens_buf_mask: Box::new([]),
            internal_token_dense_words: 0,
            weight_token_dense_masks: Default::default(),
            packed_dwa_token_dense_masks: Default::default(),
            weight_token_buf_masks: Default::default(),
            weight_token_sparse_buf_masks: Default::default(),
            direct_sparse_weight_token_sets: Default::default(),
            seed_terminal_dense: Default::default(),
            seed_terminal_dense_fallback: Default::default(),
            seed_universe_dense: Arc::from(Vec::<u64>::new().into_boxed_slice()),
            dwa_fast_transitions: Default::default(),
            parser_runtime_caches_prebuilt: false,
            indexed_dag_dense_transitions: Vec::new(),
            indexed_dag_dense_finals: Vec::new(),
            tokenizer_fast_transitions: Default::default(),
            heavy_token_dense_masks: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            backed_internal_token_buf_flat: None,
            internal_token_buf_offsets: Box::new([]),
            total_internal_buf_cost: 0,
            heavy_token_indices: Vec::new(),
            heavy_total_cost: 0,
            light_avg_cost_x256: 0,
            internal_token_buf_op_costs: Vec::new(),
            word_group_buf_op_costs: Vec::new(),
            final_mask_mapping: Default::default(),
            parser_state_domain_labels: Vec::new(),
            ignore_expr: payload.ignore_expr,
            serialized_artifact_cache: None,
            deferred_terminal_exprs_blob: None,
            deferred_terminal_exprs: Default::default(),
            deferred_composition_metadata_blob: None,
            composition_link_metadata_materialized: true,
            deferred_table_rules_blob: None,
            deferred_table_rules: Default::default(),
        };
        inner
    }

    pub(crate) fn from_alternatives(mut alternatives: Vec<Self>) -> Self {
        assert!(!alternatives.is_empty(), "dynamic union requires at least one alternative");
        let first = alternatives.remove(0);
        let mut result = Self {
            inner: first.inner,
            alternatives: first.alternatives,
            composition_grammars: first.composition_grammars,
        };
        for alternative in alternatives {
            result.alternatives.push(alternative.inner);
            result.alternatives.extend(alternative.alternatives);
            result.composition_grammars.extend(alternative.composition_grammars);
        }
        result
    }

    pub(crate) fn from_constraints(mut constraints: Vec<Constraint>) -> Self {
        assert!(!constraints.is_empty(), "dynamic union requires at least one alternative");
        let inner = constraints.remove(0);
        let composition_grammars = vec![None; constraints.len() + 1];
        Self { inner, alternatives: constraints, composition_grammars }
    }

    pub(crate) fn clone_constraints(&self) -> Vec<Constraint> {
        std::iter::once(&self.inner).chain(&self.alternatives).cloned().collect()
    }

    pub(crate) fn targets_vocab(&self, vocab: &Vocab) -> bool {
        std::iter::once(&self.inner)
            .chain(&self.alternatives)
            .all(|constraint| constraint.token_bytes_match_vocab(vocab))
    }

    pub(crate) fn attach_late_grammar_placeholders(
        &mut self,
        placeholders: &[(u32, String)],
    ) -> crate::Result<()> {
        for constraint in std::iter::once(&mut self.inner).chain(&mut self.alternatives) {
            constraint.late_grammar_slots.clear();
            for (placeholder_token_id, binding_name) in placeholders {
                let mut matching = constraint
                    .special_token_terminals
                    .iter()
                    .filter(|special| special.token_id == *placeholder_token_id)
                    .map(|special| special.terminal_id);
                let Some(terminal_id) = matching.next() else {
                    // Dynamic alternatives can omit an unreachable choice.
                    continue;
                };
                if matching.next().is_some() {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "compiled GLRM external subgrammar {binding_name:?} has multiple hidden linker terminals",
                    )));
                }
                constraint.late_grammar_slots.push(crate::runtime::LateGrammarSlot {
                    name: binding_name.clone(),
                    terminal_id,
                });
            }
        }
        Ok(())
    }

    pub(crate) fn set_composition_grammar(&mut self, grammar: GrammarDef) {
        assert_eq!(self.composition_grammars.len(), 1);
        self.composition_grammars[0] = Some(grammar);
    }

    pub(crate) fn reconstruct_composition_grammar(constraint: &Constraint) -> crate::Result<GrammarDef> {
        let exprs = constraint.tokenizer.terminal_exprs().ok_or_else(|| {
            crate::GlrMaskError::Compilation(
                "this legacy dynamic artifact does not retain terminal expressions required for compiled-child composition; rebuild it".to_owned(),
            )
        })?;
        let num_terminals = constraint.tokenizer.num_terminals() as usize;
        if exprs.len() != num_terminals {
            return Err(crate::GlrMaskError::Compilation(format!(
                "dynamic artifact terminal-expression count {} does not match tokenizer terminal count {num_terminals}",
                exprs.len(),
            )));
        }

        let special_by_terminal = constraint
            .special_token_terminals
            .iter()
            .map(|special| (special.terminal_id, special.token_id))
            .collect::<BTreeMap<_, _>>();
        let terminals = exprs
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, expr)| {
                let id = id as u32;
                special_by_terminal.get(&id).copied().map_or(
                    Terminal::Expr { id, expr },
                    |token_id| Terminal::SpecialToken { id, token_id },
                )
            })
            .collect::<Vec<_>>();
        let terminal_names = constraint
            .terminal_display_names
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, name)| (id as u32, name))
            .collect::<BTreeMap<_, _>>();

        let (start, rules, nonterminal_names) = if constraint.direct_regular_automaton.is_some() {
            (0, Vec::new(), BTreeMap::new())
        } else {
            let augmented = constraint.table.rules.first().ok_or_else(|| {
                crate::GlrMaskError::Compilation(
                    "dynamic artifact has no augmented-start rule for compiled-child composition"
                        .to_owned(),
                )
            })?;
            let start = match augmented.rhs.as_slice() {
                [Symbol::Nonterminal(start)] => *start,
                rhs => {
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "dynamic artifact augmented-start rule must contain one nonterminal, found {rhs:?}",
                    )));
                }
            };
            let nonterminal_names = constraint
                .table
                .nonterminal_display_names
                .iter()
                .cloned()
                .enumerate()
                .filter(|(id, _)| *id as u32 != augmented.lhs)
                .map(|(id, name)| (id as u32, name))
                .collect::<BTreeMap<_, _>>();
            (start, constraint.table.rules[1..].to_vec(), nonterminal_names)
        };

        Ok(GrammarDef {
            rules,
            start,
            terminals,
            nonterminal_names,
            terminal_names,
            ignore_terminal: constraint.ignore_terminal,
            lexer_partitions: BTreeMap::new(),
            residual_isolation_classes: BTreeMap::new(),
            requires_global_terminal_observation: true,
            direct_regular_automaton: constraint.direct_regular_automaton.clone(),
        })
    }

    pub(crate) fn bind_vocab_exact(&mut self, vocab: &Vocab) -> Result<(), String> {
        self.inner.bind_vocab_exact(vocab)?;
        for alternative in &mut self.alternatives {
            alternative.bind_vocab_exact(vocab)?;
        }
        Ok(())
    }

    pub(crate) fn into_constraint(self) -> Constraint {
        assert!(
            self.alternatives.is_empty(),
            "a union dynamic constraint cannot be converted to one Constraint",
        );
        self.inner
    }

    fn payload_for_constraint(constraint: &Constraint) -> DynamicConstraintPayloadV2 {
        let terminal_exprs = constraint.tokenizer.terminal_exprs().map(ToOwned::to_owned);
        DynamicConstraintPayloadV2 {
            v1: DynamicConstraintPayloadV1 {
                table: constraint.table.clone(),
                terminal_display_names: constraint.terminal_display_names.clone(),
                tokenizer: constraint.tokenizer.clone(),
                ignore_terminal: constraint.ignore_terminal,
                direct_regular_automaton: constraint.direct_regular_automaton.clone(),
                token_bytes: Arc::clone(&constraint.token_bytes),
                ignore_expr: constraint.ignore_expr.clone(),
                terminal_exprs,
            },
            special_token_terminals: constraint.special_token_terminals.clone(),
        }
    }

    fn transfer_payload_from_constraint_owned(
        constraint: Constraint,
    ) -> DynamicConstraintTransferAlternativeV1 {
        let terminal_exprs = constraint.tokenizer.terminal_exprs().map(ToOwned::to_owned);
        let mask_quotient = constraint
            .dynamic_mask_vocab
            .mask_tokenizer_quotient_for_transfer();

        DynamicConstraintTransferAlternativeV1 {
            table: constraint.table,
            terminal_display_names: constraint.terminal_display_names,
            tokenizer: constraint.tokenizer,
            ignore_terminal: constraint.ignore_terminal,
            direct_regular_automaton: constraint.direct_regular_automaton,
            special_token_terminals: constraint.special_token_terminals,
            ignore_expr: constraint.ignore_expr,
            terminal_exprs,
            mask_tokenizer: mask_quotient
                .as_ref()
                .map(|(tokenizer, _)| CompactTransferTokenizer(tokenizer.clone())),
            full_to_mask_state: mask_quotient.map_or_else(Vec::new, |(_, mapping)| mapping),

        }
    }

    pub(crate) fn into_saved(self) -> Vec<u8> {
        let payload = DynamicConstraintTransferPayloadV1 {
            alternatives: std::iter::once(self.inner)
                .chain(self.alternatives)
                .map(Self::transfer_payload_from_constraint_owned)
                .collect(),
        };
        let payload = bincode::serialize(&payload)
            .expect("DynamicConstraint transfer serialization should succeed");
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn migrate_legacy_v10_payload(
        payload: LegacyDynamicConstraintPayloadV10V3,
    ) -> DynamicConstraintPayloadV3 {
        DynamicConstraintPayloadV3 {
            alternatives: payload
                .alternatives
                .into_iter()
                .map(|alternative| DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: alternative.v1.table,
                        terminal_display_names: alternative.v1.terminal_display_names,
                        tokenizer: alternative.v1.tokenizer,
                        ignore_terminal: alternative.v1.ignore_terminal,
                        direct_regular_automaton: alternative.v1.direct_regular_automaton,
                        token_bytes: alternative.v1.token_bytes,
                        ignore_expr: None,
                        terminal_exprs: None,
                    },
                    special_token_terminals: alternative.special_token_terminals,
                })
                .collect(),
        }
    }

    fn migrate_legacy_v11_payload(
        payload: LegacyDynamicConstraintPayloadV11V3,
    ) -> DynamicConstraintPayloadV3 {
        DynamicConstraintPayloadV3 {
            alternatives: payload
                .alternatives
                .into_iter()
                .map(|alternative| DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: alternative.v1.table,
                        terminal_display_names: alternative.v1.terminal_display_names,
                        tokenizer: alternative.v1.tokenizer,
                        ignore_terminal: alternative.v1.ignore_terminal,
                        direct_regular_automaton: alternative.v1.direct_regular_automaton,
                        token_bytes: alternative.v1.token_bytes,
                        ignore_expr: None,
                        terminal_exprs: None,
                    },
                    special_token_terminals: alternative.special_token_terminals,
                })
                .collect(),
        }
    }

    fn migrate_legacy_v12_payload(
        payload: LegacyDynamicConstraintPayloadV12V3,
    ) -> DynamicConstraintPayloadV3 {
        DynamicConstraintPayloadV3 {
            alternatives: payload
                .alternatives
                .into_iter()
                .map(|alternative| DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: alternative.v1.table,
                        terminal_display_names: alternative.v1.terminal_display_names,
                        tokenizer: alternative.v1.tokenizer,
                        ignore_terminal: alternative.v1.ignore_terminal,
                        direct_regular_automaton: alternative.v1.direct_regular_automaton,
                        token_bytes: alternative.v1.token_bytes,
                        ignore_expr: alternative.v1.ignore_expr,
                        terminal_exprs: None,
                    },
                    special_token_terminals: alternative.special_token_terminals,
                })
                .collect(),
        }
    }

    fn from_payload_v3(payload: DynamicConstraintPayloadV3) -> crate::Result<Self> {
        let mut alternatives = payload
            .alternatives
            .into_iter()
            .map(Self::from_payload_v2)
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(std::mem::take(&mut alternatives)))
    }

    fn from_payload_v4(payload: DynamicConstraintPayloadV4) -> crate::Result<Self> {
        let mut alternatives = payload
            .alternatives
            .into_iter()
            .map(|alternative| {
                let mut constraint = Self::from_payload_v2(alternative.constraint);
                constraint.inner.late_grammar_slots = alternative.late_grammar_slots;
                constraint
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(std::mem::take(&mut alternatives)))
    }


    fn from_payload_v3_with_vocab(
        payload: DynamicConstraintPayloadV3,
        vocab: &Vocab,
    ) -> crate::Result<Self> {
        let mut alternatives = payload
            .alternatives
            .into_iter()
            .map(|alternative| {
                Self::from_payload_v2_with_dynamic_vocab(
                    alternative,
                    crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(
                        vocab,
                    ),
                )
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(std::mem::take(&mut alternatives)))
    }

    /// Serialize this dynamic constraint to a versioned binary artifact.
    pub fn save(&self) -> Vec<u8> {
        if std::env::var_os("GLRMASK_PROFILE_DYNAMIC_ARTIFACT").is_some() {
            for (index, constraint) in std::iter::once(&self.inner)
                .chain(self.alternatives.iter())
                .enumerate()
            {
                let (finalizer_bits, future_bits, max_finalizers, max_futures) =
                    constraint.tokenizer.artifact_metadata_stats();
                eprintln!(
                    "[glrmask/profile][dynamic_artifact_metadata] alternative={} states={} terminals={} finalizer_bits={} future_bits={} max_finalizers={} max_futures={}",
                    index,
                    constraint.tokenizer.num_states(),
                    constraint.tokenizer.num_terminals(),
                    finalizer_bits,
                    future_bits,
                    max_finalizers,
                    max_futures,
                );
            }
        }
        let payload = DynamicConstraintPayloadV4 {
            alternatives: std::iter::once(&self.inner)
                .chain(self.alternatives.iter())
                .map(|constraint| DynamicConstraintPayloadV4Alternative {
                    constraint: Self::payload_for_constraint(constraint),
                    late_grammar_slots: constraint.late_grammar_slots.clone(),
                })
                .collect(),
        };
        let payload = bincode::serialize(&payload)
            .expect("DynamicConstraint serialization should succeed");
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// Load either a self-contained artifact or a transfer artifact whose
    /// vocabulary bytes are supplied out of band.
    pub(crate) fn load_with_vocab(bytes: &[u8], vocab: &Vocab) -> crate::Result<Self> {
        if !bytes.starts_with(&DYNAMIC_TRANSFER_MAGIC) {
            let mut loaded = Self::load(bytes)?;
            loaded
                .bind_vocab_exact(vocab)
                .map_err(crate::GlrMaskError::Serialization)?;
            return Ok(loaded);
        }
        if bytes.len() < DYNAMIC_CONSTRAINT_HEADER_LEN {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic transfer artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if !matches!(
            version,
            DYNAMIC_TRANSFER_VERSION_V1
                | DYNAMIC_TRANSFER_VERSION_V2
                | DYNAMIC_TRANSFER_VERSION_V3
                | DYNAMIC_TRANSFER_VERSION

        ) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "unsupported dynamic transfer artifact version {version}",
            )));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[10..18]
                .try_into()
                .expect("dynamic transfer header has fixed width"),
        ))
        .map_err(|_| {
            crate::GlrMaskError::Serialization(
                "dynamic transfer payload length does not fit this platform".to_owned(),
            )
        })?;
        if bytes.len() != DYNAMIC_CONSTRAINT_HEADER_LEN.saturating_add(payload_len) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic transfer artifact payload length".to_owned(),
            ));
        }
        let payload = match version {
            DYNAMIC_TRANSFER_VERSION => bincode::deserialize::<DynamicConstraintTransferPayloadV1>(
                &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
            )
            .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?,
            DYNAMIC_TRANSFER_VERSION_V3 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV3>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                DynamicConstraintTransferPayloadV1 {
                    alternatives: legacy.alternatives.into_iter().map(|alternative| {
                        DynamicConstraintTransferAlternativeV1 {
                            table: alternative.table,
                            terminal_display_names: alternative.terminal_display_names,
                            tokenizer: alternative.tokenizer,
                            ignore_terminal: alternative.ignore_terminal,
                            direct_regular_automaton: alternative.direct_regular_automaton,
                            special_token_terminals: alternative.special_token_terminals,
                            ignore_expr: alternative.ignore_expr,
                            terminal_exprs: None,
                            mask_tokenizer: alternative.mask_tokenizer,
                            full_to_mask_state: alternative.full_to_mask_state,
                        }
                    }).collect(),
                }
            }
            DYNAMIC_TRANSFER_VERSION_V2 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV2>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                DynamicConstraintTransferPayloadV1 {
                    alternatives: legacy.alternatives.into_iter().map(|alternative| {
                        DynamicConstraintTransferAlternativeV1 {
                            table: alternative.table,
                            terminal_display_names: alternative.terminal_display_names,
                            tokenizer: alternative.tokenizer,
                            ignore_terminal: alternative.ignore_terminal,
                            direct_regular_automaton: alternative.direct_regular_automaton,
                            special_token_terminals: alternative.special_token_terminals,
                            ignore_expr: alternative.ignore_expr,
                            terminal_exprs: None,
                            mask_tokenizer: None,
                            full_to_mask_state: Vec::new(),
                        }
                    }).collect(),
                }
            }
            DYNAMIC_TRANSFER_VERSION_V1 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV1>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                DynamicConstraintTransferPayloadV1 {
                    alternatives: legacy.alternatives.into_iter().map(|alternative| {
                        DynamicConstraintTransferAlternativeV1 {
                            table: alternative.table,
                            terminal_display_names: alternative.terminal_display_names,
                            tokenizer: alternative.tokenizer,
                            ignore_terminal: alternative.ignore_terminal,
                            direct_regular_automaton: alternative.direct_regular_automaton,
                            special_token_terminals: alternative.special_token_terminals,
                            ignore_expr: None,
                            terminal_exprs: None,
                            mask_tokenizer: None,
                            full_to_mask_state: Vec::new(),
                        }
                    }).collect(),
                }
            }
            _ => unreachable!("transfer version was validated above"),
        };
        let token_bytes = vocab.entries_arc();
        let mut alternatives = payload
            .alternatives
            .into_iter()
            .map(|alternative| {
                let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
                    DynamicConstraintPayloadV2 {
                        v1: DynamicConstraintPayloadV1 {
                            table: alternative.table,
                            terminal_display_names: alternative.terminal_display_names,
                            tokenizer: alternative.tokenizer,
                            ignore_terminal: alternative.ignore_terminal,
                            direct_regular_automaton: alternative.direct_regular_automaton,
                            token_bytes: Arc::clone(&token_bytes),
                            ignore_expr: alternative.ignore_expr,
                            terminal_exprs: alternative.terminal_exprs,
                        },
                        special_token_terminals: alternative.special_token_terminals,
                    },
                    crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(
                        vocab,
                    ),
                );
                if let Some(mask_tokenizer) = alternative.mask_tokenizer {
                    inner.dynamic_mask_vocab.set_mask_tokenizer_quotient(
                        mask_tokenizer.0,
                        alternative.full_to_mask_state,
                    );
                }
                inner.rebuild_dynamic_runtime_caches();
                Self {
                    inner,
                    alternatives: Vec::new(),
                    composition_grammars: vec![None],
                }
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union transfer artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(std::mem::take(&mut alternatives)))
    }

    /// Load an artifact produced by [`DynamicConstraint::save`].
    pub fn load(bytes: &[u8]) -> crate::Result<Self> {
        if bytes.len() < DYNAMIC_CONSTRAINT_HEADER_LEN
            || !bytes.starts_with(&DYNAMIC_CONSTRAINT_MAGIC)
        {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic constraint artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if !matches!(
            version,
            1 | 2
                | 7
                | 8
                | 9
                | 10
                | 11
                | DYNAMIC_CONSTRAINT_VERSION_V12
                | DYNAMIC_CONSTRAINT_VERSION_V13
                | DYNAMIC_CONSTRAINT_VERSION
        ) {
            return Err(crate::GlrMaskError::Serialization(format!(
                "unsupported dynamic constraint artifact version {version}",
            )));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[10..18]
                .try_into()
                .expect("dynamic constraint header has fixed width"),
        ))
        .map_err(|_| {
            crate::GlrMaskError::Serialization(
                "dynamic constraint payload length does not fit this platform".to_owned(),
            )
        })?;
        if bytes.len() != DYNAMIC_CONSTRAINT_HEADER_LEN.saturating_add(payload_len) {
            return Err(crate::GlrMaskError::Serialization(
                "invalid dynamic constraint artifact payload length".to_owned(),
            ));
        }
        match version {
            1 => {
                let payload: LegacyDynamicConstraintPayloadV1 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v1(payload)
            }
            2 => {
                let payload: LegacyDynamicConstraintPayloadV2 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v2(payload)
            }
            7 | 8 => Err(crate::GlrMaskError::Serialization(
                "dynamic constraint artifact contains removed precompiled token programs; rebuild it"
                    .to_owned(),
            )),
            9 => {
                let payload: LegacyDynamicConstraintPayloadV9 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Ok(Self::from_payload_v2(DynamicConstraintPayloadV2 {
                    v1: DynamicConstraintPayloadV1 {
                        table: payload.v1.table,
                        terminal_display_names: payload.v1.terminal_display_names,
                        tokenizer: payload.v1.tokenizer,
                        ignore_terminal: payload.v1.ignore_terminal,
                        direct_regular_automaton: None,
                        token_bytes: payload.v1.token_bytes,
                        ignore_expr: None,
                        terminal_exprs: None,
                    },
                    special_token_terminals: payload.special_token_terminals,
                }))
            }
            10 => {
                let payload: LegacyDynamicConstraintPayloadV10V3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(Self::migrate_legacy_v10_payload(payload))
            }
            11 => {
                let payload: LegacyDynamicConstraintPayloadV11V3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(Self::migrate_legacy_v11_payload(payload))
            }
            DYNAMIC_CONSTRAINT_VERSION_V12 => {
                let payload: LegacyDynamicConstraintPayloadV12V3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(Self::migrate_legacy_v12_payload(payload))
            }
            DYNAMIC_CONSTRAINT_VERSION_V13 => {
                let payload: DynamicConstraintPayloadV3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(payload)
            }
            DYNAMIC_CONSTRAINT_VERSION => {
                let payload: DynamicConstraintPayloadV4 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v4(payload)
            }
            _ => unreachable!("version was validated above"),
        }
    }

    /// Return the number of `u32` words required for a packed token mask.
    pub fn mask_len(&self) -> usize {
        std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .map(Constraint::mask_len)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn max_original_token_id(&self) -> Option<u32> {
        std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .filter_map(Constraint::max_original_token_id)
            .max()
    }

    /// Create a fresh state for one generated sequence.
    pub fn start(&self) -> DynamicConstraintState<'_> {
        DynamicConstraintState {
            alternatives: std::iter::once(&self.inner)
                .chain(self.alternatives.iter())
                .map(Constraint::start)
                .collect(),
            mask_len: self.mask_len(),
        }
    }
}

/// Mutable per-sequence state for a [`DynamicConstraint`].
#[derive(Clone)]
pub struct DynamicConstraintState<'a> {
    alternatives: Vec<ConstraintState<'a>>,
    mask_len: usize,
}

impl<'a> DynamicConstraintState<'a> {
    fn retain_committing(
        &mut self,
        mut commit: impl FnMut(&mut ConstraintState<'a>) -> Result<(), String>,
    ) -> Result<(), String> {
        self.alternatives.retain_mut(|state| commit(state).is_ok());
        if self.alternatives.is_empty() {
            Err("commit rejected: no valid parser states remain".to_owned())
        } else {
            Ok(())
        }
    }

    /// Advance the state by raw bytes.
    pub fn commit_bytes(&mut self, bytes: &[u8]) -> crate::Result<()> {
        self.commit_bytes_raw(bytes).map_err(crate::Error::State)
    }

    fn commit_bytes_raw(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.retain_committing(|state| state.commit_bytes_raw(bytes))
    }

    /// Advance the state by one model token ID.
    pub fn commit_token(&mut self, token_id: u32) -> crate::Result<()> {
        if self
            .alternatives
            .iter()
            .all(|state| !state.knows_token_id(token_id))
        {
            return Err(crate::Error::State(format!(
                "commit_token: token_id {token_id} not in vocabulary or special-token terminals"
            )));
        }
        self.commit_token_raw(token_id).map_err(crate::Error::State)
    }

    fn commit_token_raw(&mut self, token_id: u32) -> Result<(), String> {
        self.retain_committing(|state| state.commit_token_raw(token_id))
    }

    /// Fill `buf` with the allowed-token mask as a packed bitset.
    pub fn fill_mask(&self, buf: &mut [u32]) {
        assert!(buf.len() >= self.mask_len, "mask buffer is smaller than constraint mask");
        buf.fill(0);
        let Some((first, rest)) = self.alternatives.split_first() else {
            return;
        };
        first.fill_mask(buf);
        if rest.is_empty() {
            return;
        }
        let mut scratch = vec![0u32; buf.len()];
        for state in rest {
            state.fill_mask(&mut scratch);
            for (target, source) in buf.iter_mut().zip(&scratch) {
                *target |= *source;
            }
            scratch.fill(0);
        }
    }

    /// Return a forced token sequence when one can be determined.
    pub fn forced(&self) -> Vec<u32> {
        let Some((first, rest)) = self.alternatives.split_first() else {
            return Vec::new();
        };
        let forced = first.forced();
        (!forced.is_empty()
            && rest.iter().all(|state| state.forced() == forced))
            .then_some(forced)
            .unwrap_or_default()
    }

    /// Return whether the committed prefix is currently accepted by the grammar.
    ///
    /// An accepting prefix may still admit additional tokens.
    pub fn is_accepting(&self) -> bool {
        self.alternatives.iter().any(ConstraintState::is_accepting)
    }

    /// Return whether the committed prefix has been irrecoverably rejected.
    pub fn is_rejected(&self) -> bool {
        self.alternatives.is_empty()
    }

    /// Return the allowed-token mask as a packed `u32` bitset.
    pub fn mask(&self) -> Vec<u32> {
        let mut mask = vec![0u32; self.mask_len];
        self.fill_mask(&mut mask);
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compressed_lark_grammar(source: &str) -> crate::grammar::flat::GrammarDef {
        let mut named = crate::import::lark::parse_lark_to_named_uncompressed(source).unwrap();
        assert!(crate::grammar::right_linear::compress_large_right_linear_grammar(
            &mut named
        ));
        let factored = crate::grammar::factoring::factor_named_grammar(named);
        crate::grammar::ast::lower(&factored).unwrap()
    }

    fn compile_compressed_static(source: &str, vocab: &Vocab) -> crate::Constraint {
        crate::compiler::pipeline::compile_owned_with_table_construction(
            compressed_lark_grammar(source),
            vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
    }

    fn compile_compressed_dynamic(source: &str, vocab: &Vocab) -> DynamicConstraint {
        crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            compressed_lark_grammar(source),
            vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
    }

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"aa".to_vec()),
            (4, b" ".to_vec()),
        ])
    }

    #[test]
    fn dynamic_constraint_matches_constraint_masks_and_commits() {
        let vocab = vocab();
        let grammar = r#"
            start start;
            t A ::= 'a'+;
            t B ::= 'b';
            nt start ::= A B;
        "#;
        let normal = crate::Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut normal_state = normal.start();
        let mut dynamic_state = dynamic.start();

        assert_eq!(normal_state.mask(), dynamic_state.mask());
        normal_state.commit_token(3).unwrap();
        dynamic_state.commit_token(3).unwrap();
        assert_eq!(normal_state.mask(), dynamic_state.mask());
        normal_state.commit_token(1).unwrap();
        dynamic_state.commit_token(1).unwrap();
        assert_eq!(normal_state.is_accepting(), dynamic_state.is_accepting());
        assert_eq!(normal_state.mask(), dynamic_state.mask());
    }

    #[test]
    fn dynamic_constraint_save_load_round_trip() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a"+ "b";
            "#,
            &vocab,
        )
        .unwrap();
        let loaded = DynamicConstraint::load(&constraint.save()).unwrap();
        assert!(constraint.inner.ignore_expr.is_some());
        assert_eq!(loaded.inner.ignore_expr, constraint.inner.ignore_expr);
        assert_eq!(
            loaded.inner.tokenizer.terminal_exprs(),
            constraint.inner.tokenizer.terminal_exprs(),
        );
        assert_eq!(constraint.mask_len(), loaded.mask_len());
        assert_eq!(constraint.start().mask(), loaded.start().mask());
    }

    #[test]
    fn dynamic_constraint_loads_v11_payload_without_ignore_descriptor() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let original_mask = constraint.start().mask();
        let legacy = LegacyDynamicConstraintPayloadV11V3 {
            alternatives: vec![LegacyDynamicConstraintPayloadV11V2 {
                v1: LegacyDynamicConstraintPayloadV11V1 {
                    table: constraint.inner.table.clone(),
                    terminal_display_names: constraint.inner.terminal_display_names.clone(),
                    tokenizer: constraint.inner.tokenizer.clone(),
                    ignore_terminal: constraint.inner.ignore_terminal,
                    direct_regular_automaton: constraint.inner.direct_regular_automaton.clone(),
                    token_bytes: Arc::clone(&constraint.inner.token_bytes),
                },
                special_token_terminals: constraint.inner.special_token_terminals.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&11u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(loaded.inner.ignore_expr.is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }


    #[test]
    fn dynamic_constraint_loads_v12_payload_without_terminal_exprs() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            "start start; nt start ::= \"a\" \"b\";",
            &vocab,
        )
        .unwrap();
        let original_mask = constraint.start().mask();
        let legacy = LegacyDynamicConstraintPayloadV12V3 {
            alternatives: vec![LegacyDynamicConstraintPayloadV12V2 {
                v1: LegacyDynamicConstraintPayloadV12V1 {
                    table: constraint.inner.table.clone(),
                    terminal_display_names: constraint.inner.terminal_display_names.clone(),
                    tokenizer: constraint.inner.tokenizer.clone(),
                    ignore_terminal: constraint.inner.ignore_terminal,
                    direct_regular_automaton: constraint.inner.direct_regular_automaton.clone(),
                    token_bytes: Arc::clone(&constraint.inner.token_bytes),
                    ignore_expr: constraint.inner.ignore_expr.clone(),
                },
                special_token_terminals: constraint.inner.special_token_terminals.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&12u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(loaded.inner.tokenizer.terminal_exprs().is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }


    #[test]
    fn dynamic_transfer_load_uses_prepared_vocab_and_matches_original() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let original_mask = original.start().mask();
        let transfer = original.into_saved();
        assert!(transfer.starts_with(&DYNAMIC_TRANSFER_MAGIC));

        let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(original_mask, loaded.start().mask());
    }

    #[test]
    fn self_contained_dynamic_load_with_vocab_shares_exact_vocab() {
        let vocab = vocab();
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let loaded = DynamicConstraint::load_with_vocab(&original.save(), &vocab).unwrap();

        assert!(std::sync::Arc::ptr_eq(
            &loaded.inner.token_bytes,
            &vocab.entries_arc(),
        ));
        assert!(std::sync::Arc::ptr_eq(
            &loaded
                .inner
                .late_bind_vocab
                .get()
                .expect("dynamic load_with_vocab should seed late-bind vocab")
                .entries_arc(),
            &vocab.entries_arc(),
        ));
    }

    #[test]
    fn dynamic_transfer_loads_v1_payload_without_ignore_descriptor() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                ignore WS;
                t WS ::= " "+;
                nt start ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let original_mask = original.start().mask();
        let legacy = LegacyDynamicConstraintTransferPayloadV1 {
            alternatives: vec![LegacyDynamicConstraintTransferAlternativeV1 {
                table: original.inner.table.clone(),
                terminal_display_names: original.inner.terminal_display_names.clone(),
                tokenizer: original.inner.tokenizer.clone(),
                ignore_terminal: original.inner.ignore_terminal,
                direct_regular_automaton: original.inner.direct_regular_automaton.clone(),
                special_token_terminals: original.inner.special_token_terminals.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(loaded.inner.ignore_expr.is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }


    #[test]
    fn dynamic_transfer_loads_v2_payload_without_terminal_exprs() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let original_mask = original.start().mask();
        let legacy = LegacyDynamicConstraintTransferPayloadV2 {
            alternatives: vec![LegacyDynamicConstraintTransferAlternativeV2 {
                table: original.inner.table.clone(),
                terminal_display_names: original.inner.terminal_display_names.clone(),
                tokenizer: original.inner.tokenizer.clone(),
                ignore_terminal: original.inner.ignore_terminal,
                direct_regular_automaton: original.inner.direct_regular_automaton.clone(),
                special_token_terminals: original.inner.special_token_terminals.clone(),
                ignore_expr: original.inner.ignore_expr.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(loaded.inner.tokenizer.terminal_exprs().is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }


    #[test]
    fn dynamic_transfer_loads_v3_mask_quotient_payload_without_terminal_exprs() {
        let vocab = vocab();
        crate::compiler::constraint_possible_matches::prepare_vocab_for_dynamic_mask(&vocab);
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let original_mask = original.start().mask();
        let mask_quotient = original
            .inner
            .dynamic_mask_vocab
            .mask_tokenizer_quotient_for_transfer();
        let legacy = LegacyDynamicConstraintTransferPayloadV3 {
            alternatives: vec![LegacyDynamicConstraintTransferAlternativeV3 {
                table: original.inner.table.clone(),
                terminal_display_names: original.inner.terminal_display_names.clone(),
                tokenizer: original.inner.tokenizer.clone(),
                ignore_terminal: original.inner.ignore_terminal,
                direct_regular_automaton: original.inner.direct_regular_automaton.clone(),
                special_token_terminals: original.inner.special_token_terminals.clone(),
                ignore_expr: original.inner.ignore_expr.clone(),
                mask_tokenizer: mask_quotient
                    .as_ref()
                    .map(|(tokenizer, _)| CompactTransferTokenizer(tokenizer.clone())),
                full_to_mask_state: mask_quotient.map_or_else(Vec::new, |(_, mapping)| mapping),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_VERSION_V3.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(loaded.inner.tokenizer.terminal_exprs().is_none());
        assert_eq!(loaded.start().mask(), original_mask);
    }

    #[test]
    fn precompiled_dynamic_artifacts_require_rebuild() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let mut bytes = constraint.save();
        bytes[8..10].copy_from_slice(&8u16.to_le_bytes());
        let error = DynamicConstraint::load(&bytes).unwrap_err().to_string();
        assert!(error.contains("removed precompiled token programs"));
    }

    #[test]
    fn direct_regular_dynamic_constraint_uses_dynamic_runtime() {
        let vocab = vocab();
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"b\"\n");

        let normal = compile_compressed_static(&grammar, &vocab);
        let dynamic = compile_compressed_dynamic(&grammar, &vocab);
        assert_eq!(dynamic.inner.table.num_rules, 0);
        assert!(dynamic.inner.uses_dynamic_runtime());
        assert_eq!(normal.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn compressed_static_unions_mixed_l1_l2p_token_lengths() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"a\"\n");

        let static_constraint = compile_compressed_static(&grammar, &vocab);
        let dynamic_constraint = compile_compressed_dynamic(&grammar, &vocab);
        assert!(!static_constraint.uses_dynamic_runtime());
        assert_eq!(static_constraint.table.num_rules, 0);
        assert!(
            !static_constraint.parser_top_accept_parts.is_empty(),
            "regression must exercise the direct parser-acceptance summaries",
        );

        let static_mask = static_constraint.start().mask();
        assert_ne!(static_mask[0] & (1 << 0), 0, "single-byte token must be allowed");
        assert_ne!(static_mask[0] & (1 << 1), 0, "two-byte token must be allowed");
        assert_eq!(static_mask, dynamic_constraint.start().mask());
    }

    #[test]
    fn compressed_right_linear_plus_loop_commits_terminal_at_token_boundary() {
        let mut grammar = String::from("start: H s0\nplus_line: PLUS_LINE\n");
        let n = 15;
        for i in 0..n {
            grammar.push_str(&format!(
                "s{i}: line{i}{}\n",
                if i + 1 < n {
                    format!(" | s{}", i + 1)
                } else {
                    String::new()
                }
            ));
        }
        for i in 0..n {
            grammar.push_str(&format!(
                "line{i}: plus_line* SRC_{i} plus_line* {}\n",
                if i + 1 < n {
                    format!("line{}?", i + 1)
                } else {
                    String::new()
                }
            ));
        }
        grammar.push_str("H: \"h\\n\"\nPLUS_LINE: /\\+[^\\n\\r]*\\n/\n");
        for i in 0..n {
            grammar.push_str(&format!("SRC_{i}: \" a{i}\\n\"\n"));
        }
        let vocab = Vocab::new(vec![
            (0, b"h\n".to_vec()),
            (1, b"+x".to_vec()),
            (2, b"\n".to_vec()),
            (3, b" a0\n".to_vec()),
        ]);
        let constraint = compile_compressed_static(&grammar, &vocab);
        assert!(!constraint.uses_dynamic_runtime());
        let mut state = constraint.start();

        state.commit_token(0).unwrap();
        state.commit_token(1).unwrap();
        assert_ne!(state.mask()[0] & (1 << 2), 0);
        state.commit_token(2).unwrap();
        assert_ne!(state.mask()[0] & (1 << 3), 0);
        state.commit_token(3).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn direct_regular_static_constraint_roundtrips_and_matches_dynamic() {
        let vocab = vocab();
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"b\"\n");

        let constraint = crate::Constraint::from_lark(&grammar, &vocab).unwrap();
        assert!(!constraint.uses_dynamic_runtime());
        assert!(constraint.possible_matches_complete);
        let dynamic = compile_compressed_dynamic(&grammar, &vocab);
        assert!(!dynamic.inner.possible_matches_complete);

        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();
        assert_eq!(static_state.mask(), dynamic_state.mask());
        assert_eq!(static_state.forced(), dynamic_state.forced());

        for token in [3, 3] {
            static_state.commit_token(token).unwrap();
            dynamic_state.commit_token(token).unwrap();
            assert_eq!(static_state.mask(), dynamic_state.mask());
            assert_eq!(static_state.is_accepting(), dynamic_state.is_accepting());
        }

        let before_third = static_state.mask();
        let checkpoint = static_state.clone();
        static_state.commit_token(0).unwrap();
        dynamic_state.commit_token(0).unwrap();
        let after_third = static_state.mask();
        assert_eq!(after_third, dynamic_state.mask());
        static_state = checkpoint;
        assert_eq!(static_state.mask(), before_third);
        static_state.commit_token(0).unwrap();
        assert_eq!(static_state.mask(), after_third);

        let loaded = crate::Constraint::load(&constraint.save()).unwrap();
        assert!(!loaded.uses_dynamic_runtime());
        let mut loaded_state = loaded.start();
        let mut original_state = constraint.start();
        assert_eq!(loaded_state.mask(), original_state.mask());
        for token in [3, 3, 0] {
            loaded_state.commit_token(token).unwrap();
            original_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.mask(), original_state.mask());
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
        }
    }

    #[test]
    fn non_regular_constraint_keeps_static_backend() {
        let vocab = vocab();
        let constraint = crate::Constraint::from_ebnf(
            "start ::= 'a' start 'b' | ''",
            &vocab,
        )
        .unwrap();
        assert!(!constraint.uses_dynamic_runtime());
    }

    #[test]
    fn dynamic_forced_uses_dynamic_masks() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'", &vocab).unwrap();
        assert_eq!(constraint.start().forced(), vec![0]);
    }
}
