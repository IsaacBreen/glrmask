// PRE-RELEASE COMPATIBILITY POLICY
// ================================
// GLRMask is still pre-release. Historical DynamicConstraint/transfer wire
// layouts are NOT a compatibility contract and must not constrain the current
// architecture. Prefer the cleanest exact current representation; legacy
// readers below are best-effort development conveniences only and may be
// deleted or broken whenever preserving them would complicate correctness,
// performance, or the wire format. Do not add migration machinery unless
// compatibility becomes an explicit product requirement.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::{Tokenizer, VirtualTokenizerRuntimeMetadata};
use crate::automata::regex::Expr;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::constraint_possible_matches::ConstraintPossibleMatchesComputation;
use crate::grammar::flat::{DirectRegularAutomaton, GrammarDef, Symbol, Terminal, TerminalID};
use crate::Vocab;

use crate::runtime::{Constraint, ConstraintState, DynamicMaskVocab, SpecialTokenTerminal};

const DYNAMIC_CONSTRAINT_MAGIC: [u8; 8] = *b"GLRDYN\0\0";
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V12: u16 = 12;
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V13: u16 = 13;
// v14 is the glrmask-main composition artifact that first persisted late
// grammar slots. The residual-runtime branch independently used v14 during
// development, but that layout was never published; do not conflate the two.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V14_MAIN: u16 = 14;
// v15 is the residual-runtime artifact produced by the completed feature
// branch before integration with the composition work.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V15_RESIDUAL: u16 = 15;
// v16 combines the v14 composition metadata and v15 residual-runtime metadata.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V16: u16 = 16;
// v17 additionally persists exact terminal-observation quotient certificates.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V17: u16 = 17;
// v18 additionally persists the initialized dynamic-mask vocabulary trie so
// self-contained loads do not rebuild the full vocabulary index from token bytes.
const LEGACY_DYNAMIC_CONSTRAINT_VERSION_V18: u16 = 18;
// v19 is the first post-integration wire: it combines residual-runtime, terminal-
// observation, persisted-vocabulary, late-grammar, and boundary-trigger metadata.
const DYNAMIC_CONSTRAINT_VERSION: u16 = 19;
const DYNAMIC_CONSTRAINT_HEADER_LEN: usize = DYNAMIC_CONSTRAINT_MAGIC.len() + 2 + 8;
const DYNAMIC_TRANSFER_MAGIC: [u8; 8] = *b"GLRDXF\0\0";
const DYNAMIC_TRANSFER_VERSION_V1: u16 = 1;
const DYNAMIC_TRANSFER_VERSION_V2: u16 = 2;
const DYNAMIC_TRANSFER_VERSION_V3: u16 = 3;
const DYNAMIC_TRANSFER_VERSION_V4: u16 = 4;
const LEGACY_DYNAMIC_TRANSFER_VERSION_V5: u16 = 5;
// v6 is the integrated residual-runtime transfer layout.
const LEGACY_DYNAMIC_TRANSFER_VERSION_V6: u16 = 6;
// v7 additionally carries exact terminal-observation quotient certificates.
const LEGACY_DYNAMIC_TRANSFER_VERSION_V7: u16 = 7;
// v8 also carries reusable composition boundary-trigger metadata.
const DYNAMIC_TRANSFER_VERSION: u16 = 8;

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
struct LegacyDynamicConstraintPayloadV14MainAlternative {
    constraint: DynamicConstraintPayloadV2,
    late_grammar_slots: Vec<crate::runtime::LateGrammarSlot>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV14Main {
    alternatives: Vec<LegacyDynamicConstraintPayloadV14MainAlternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV15ResidualAlternative {
    v2: DynamicConstraintPayloadV2,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LegacyDynamicConstraintPayloadV15Residual {
    alternatives: Vec<LegacyDynamicConstraintPayloadV15ResidualAlternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV4Alternative {
    v2: DynamicConstraintPayloadV2,
    late_grammar_slots: Vec<crate::runtime::LateGrammarSlot>,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV4 {
    alternatives: Vec<DynamicConstraintPayloadV4Alternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV5Alternative {
    base: DynamicConstraintPayloadV4Alternative,
    terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV5 {
    alternatives: Vec<DynamicConstraintPayloadV5Alternative>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV6 {
    alternatives: Vec<DynamicConstraintPayloadV5Alternative>,
    /// Vocabulary-only runtime index shared by every union alternative.
    dynamic_mask_vocab: Option<crate::runtime::DynamicMaskVocabArtifact>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum DynamicBoundaryTriggerWire {
    None,
    // Legacy/conservative uniform token summary. Keep this variant in place so
    // previously serialized dynamic artifacts retain their discriminant.
    Tokens(Vec<u32>),
    Exact(DWA),
    TokenTsids {
        tokens: Vec<u32>,
        tsids: Vec<u32>,
    },
}

impl DynamicBoundaryTriggerWire {
    fn from_trigger(trigger: &crate::runtime::BoundaryTrigger) -> Self {
        match trigger {
            crate::runtime::BoundaryTrigger::None => Self::None,
            crate::runtime::BoundaryTrigger::Tokens(tokens) => {
                if let Some(tsids) = tokens.explicit_tsids() {
                    Self::TokenTsids {
                        tokens: tokens.token_summary().to_vec(),
                        tsids: tsids.to_vec(),
                    }
                } else {
                    Self::Tokens(tokens.token_summary().to_vec())
                }
            }
            crate::runtime::BoundaryTrigger::Exact(dwa) => Self::Exact((**dwa).clone()),
        }
    }

    fn into_trigger(self) -> crate::runtime::BoundaryTrigger {
        match self {
            Self::None => crate::runtime::BoundaryTrigger::None,
            Self::Tokens(tokens) => crate::runtime::BoundaryTrigger::Tokens(
                crate::runtime::BoundaryTokenTrigger::all_tsids(tokens),
            ),
            Self::Exact(dwa) => crate::runtime::BoundaryTrigger::Exact(Arc::new(dwa)),
            Self::TokenTsids { tokens, tsids } => crate::runtime::BoundaryTrigger::Tokens(
                crate::runtime::BoundaryTokenTrigger::token_tsids(tokens, tsids),
            ),
        }
    }
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV7Alternative {
    base: DynamicConstraintPayloadV5Alternative,
    boundary_trigger: DynamicBoundaryTriggerWire,
    /// Present only for recursively composed alternatives. The ordinary
    /// dynamic fields above remain the compact standalone representation; the
    /// universal Constraint artifact is authoritative for the recursive
    /// provider/component tree and its lazy compiler views.
    recursive_constraint_artifact: Option<Vec<u8>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintPayloadV7 {
    alternatives: Vec<DynamicConstraintPayloadV7Alternative>,
    /// Vocabulary-only runtime index shared by every union alternative.
    dynamic_mask_vocab: Option<crate::runtime::DynamicMaskVocabArtifact>,
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
struct DynamicConstraintTransferAlternativeV2 {
    v1: DynamicConstraintTransferAlternativeV1,
    virtual_runtimes: Vec<VirtualTokenizerRuntimeMetadata>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV2 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV2>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferAlternativeV3 {
    base: DynamicConstraintTransferAlternativeV2,
    terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV3 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV3>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferAlternativeV4 {
    base: DynamicConstraintTransferAlternativeV3,
    boundary_trigger: DynamicBoundaryTriggerWire,
    recursive_constraint_artifact: Option<Vec<u8>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DynamicConstraintTransferPayloadV4 {
    alternatives: Vec<DynamicConstraintTransferAlternativeV4>,
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
            boundary_trigger: crate::runtime::BoundaryTrigger::None,
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

    pub(crate) fn constraints_mut(&mut self) -> impl Iterator<Item = &mut Constraint> {
        std::iter::once(&mut self.inner).chain(&mut self.alternatives)
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

    fn payload_v4_for_constraint(constraint: &Constraint) -> DynamicConstraintPayloadV4Alternative {
        DynamicConstraintPayloadV4Alternative {
            v2: Self::payload_for_constraint(constraint),
            late_grammar_slots: constraint.late_grammar_slots.clone(),
            virtual_runtimes: constraint.tokenizer.virtual_runtime_metadata(),
        }
    }

    fn payload_v5_for_constraint(constraint: &Constraint) -> DynamicConstraintPayloadV5Alternative {
        DynamicConstraintPayloadV5Alternative {
            base: Self::payload_v4_for_constraint(constraint),
            terminal_observation_classes: constraint
                .dynamic_mask_vocab
                .terminal_observation_classes_for_artifact(),
        }
    }

    fn payload_v7_for_constraint(constraint: &Constraint) -> DynamicConstraintPayloadV7Alternative {
        DynamicConstraintPayloadV7Alternative {
            base: Self::payload_v5_for_constraint(constraint),
            boundary_trigger: DynamicBoundaryTriggerWire::from_trigger(
                &constraint.boundary_trigger,
            ),
            recursive_constraint_artifact: constraint
                .uses_compact_segmented_parser_runtime()
                .then(|| constraint.save()),
        }
    }

    fn restore_terminal_observation_classes(
        constraint: &mut Constraint,
        rows: Vec<(TerminalID, Vec<u32>)>,
    ) -> crate::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut seen = BTreeSet::<TerminalID>::new();
        let expected_states = constraint.tokenizer.num_states() as usize;
        let mut restored = Vec::with_capacity(rows.len());
        for (terminal, classes) in rows {
            if terminal >= constraint.tokenizer.num_terminals() {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "terminal-observation certificate references terminal {terminal}, but tokenizer has {} terminals",
                    constraint.tokenizer.num_terminals(),
                )));
            }
            if !seen.insert(terminal) {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "terminal-observation certificate repeats terminal {terminal}",
                )));
            }
            if classes.len() != expected_states {
                return Err(crate::GlrMaskError::Serialization(format!(
                    "terminal-observation certificate for terminal {terminal} has {} states, expected {expected_states}",
                    classes.len(),
                )));
            }
            restored.push((terminal, Arc::from(classes)));
        }
        constraint
            .dynamic_mask_vocab
            .set_terminal_observation_classes(restored);
        Ok(())
    }

    fn transfer_payload_v3_from_constraint_owned(
        constraint: Constraint,
    ) -> DynamicConstraintTransferAlternativeV3 {
        let terminal_exprs = constraint.tokenizer.terminal_exprs().map(ToOwned::to_owned);
        let virtual_runtimes = constraint.tokenizer.virtual_runtime_metadata();
        let mask_quotient = constraint
            .dynamic_mask_vocab
            .mask_tokenizer_quotient_for_transfer();
        let terminal_observation_classes = constraint
            .dynamic_mask_vocab
            .terminal_observation_classes_for_artifact();

        DynamicConstraintTransferAlternativeV3 {
            base: DynamicConstraintTransferAlternativeV2 {
                v1: DynamicConstraintTransferAlternativeV1 {
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
                },
                virtual_runtimes,
            },
            terminal_observation_classes,
        }
    }

    fn transfer_payload_from_constraint_owned(
        constraint: Constraint,
    ) -> DynamicConstraintTransferAlternativeV4 {
        let boundary_trigger = DynamicBoundaryTriggerWire::from_trigger(&constraint.boundary_trigger);
        let recursive_constraint_artifact = constraint
            .uses_compact_segmented_parser_runtime()
            .then(|| constraint.save());
        DynamicConstraintTransferAlternativeV4 {
            base: Self::transfer_payload_v3_from_constraint_owned(constraint),
            boundary_trigger,
            recursive_constraint_artifact,
        }
    }

    pub(crate) fn into_saved(self) -> Vec<u8> {
        let payload = DynamicConstraintTransferPayloadV4 {
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

    fn from_payload_v4(
        payload: DynamicConstraintPayloadV4,
        allow_legacy_exact_dead_residual_roots: bool,
    ) -> crate::Result<Self> {
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for mut alternative in payload.alternatives {
            let exprs = alternative.v2.v1.terminal_exprs.clone();
            alternative
                .v2
                .v1
                .tokenizer
                .restore_terminal_exprs_with_virtual_runtime_metadata(
                    exprs,
                    &alternative.virtual_runtimes,
                    allow_legacy_exact_dead_residual_roots,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            let mut constraint = Self::from_payload_v2(alternative.v2);
            constraint.inner.late_grammar_slots = alternative.late_grammar_slots;
            alternatives.push(constraint);
        }
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(alternatives))
    }

    fn from_payload_v5(
        payload: DynamicConstraintPayloadV5,
        allow_legacy_exact_dead_residual_roots: bool,
    ) -> crate::Result<Self> {
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for alternative in payload.alternatives {
            let DynamicConstraintPayloadV5Alternative {
                mut base,
                terminal_observation_classes,
            } = alternative;
            let exprs = base.v2.v1.terminal_exprs.clone();
            base.v2
                .v1
                .tokenizer
                .restore_terminal_exprs_with_virtual_runtime_metadata(
                    exprs,
                    &base.virtual_runtimes,
                    allow_legacy_exact_dead_residual_roots,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
                base.v2,
                DynamicMaskVocab::default(),
            );
            inner.late_grammar_slots = base.late_grammar_slots;
            Self::restore_terminal_observation_classes(
                &mut inner,
                terminal_observation_classes,
            )?;
            inner.rebuild_dynamic_runtime_caches();
            alternatives.push(Self {
                inner,
                alternatives: Vec::new(),
                composition_grammars: vec![None],
            });
        }
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(alternatives))
    }

    fn from_payload_v6(payload: DynamicConstraintPayloadV6) -> crate::Result<Self> {
        if payload.dynamic_mask_vocab.is_some() {
            let mut alternatives = payload.alternatives.iter();
            if let Some(first) = alternatives.next() {
                let token_bytes = &first.base.v2.v1.token_bytes;
                if alternatives.any(|alternative| alternative.base.v2.v1.token_bytes != *token_bytes) {
                    return Err(crate::GlrMaskError::Serialization(
                        "dynamic artifact shares a vocabulary index across alternatives with different token bytes"
                            .to_owned(),
                    ));
                }
            }
        }
        let shared_dynamic_vocab = payload
            .dynamic_mask_vocab
            .map(DynamicMaskVocab::from_artifact)
            .transpose()
            .map_err(crate::GlrMaskError::Serialization)?;
        if let (Some(vocab), Some(first)) = (&shared_dynamic_vocab, payload.alternatives.first()) {
            if !vocab.matches_token_bytes_exact(&first.base.v2.v1.token_bytes) {
                return Err(crate::GlrMaskError::Serialization(
                    "dynamic artifact vocabulary index does not match serialized token bytes"
                        .to_owned(),
                ));
            }
        }
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for alternative in payload.alternatives {
            let DynamicConstraintPayloadV5Alternative {
                mut base,
                terminal_observation_classes,
            } = alternative;
            let exprs = base.v2.v1.terminal_exprs.clone();
            base.v2
                .v1
                .tokenizer
                .restore_terminal_exprs_with_virtual_runtime_metadata(
                    exprs,
                    &base.virtual_runtimes,
                    false,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            let dynamic_mask_vocab = shared_dynamic_vocab
                .as_ref()
                .map(DynamicMaskVocab::fresh_runtime_instance)
                .unwrap_or_default();
            let mut inner = Self::constraint_from_payload_v2_with_dynamic_vocab(
                base.v2,
                dynamic_mask_vocab,
            );
            inner.late_grammar_slots = base.late_grammar_slots;
            Self::restore_terminal_observation_classes(
                &mut inner,
                terminal_observation_classes,
            )?;
            inner.rebuild_dynamic_runtime_caches();
            alternatives.push(Self {
                inner,
                alternatives: Vec::new(),
                composition_grammars: vec![None],
            });
        }
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(alternatives))
    }

    fn from_payload_v7(payload: DynamicConstraintPayloadV7) -> crate::Result<Self> {
        let metadata = payload
            .alternatives
            .iter()
            .map(|alternative| {
                (
                    alternative.boundary_trigger.clone(),
                    alternative.recursive_constraint_artifact.clone(),
                )
            })
            .collect::<Vec<_>>();
        let base = DynamicConstraintPayloadV6 {
            alternatives: payload
                .alternatives
                .into_iter()
                .map(|alternative| alternative.base)
                .collect(),
            dynamic_mask_vocab: payload.dynamic_mask_vocab,
        };
        let mut constraint = Self::from_payload_v6(base)?;
        if constraint.constraints_mut().count() != metadata.len() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic artifact alternative metadata count mismatch".to_owned(),
            ));
        }
        for (inner, (trigger, recursive_artifact)) in
            constraint.constraints_mut().zip(metadata)
        {
            if let Some(recursive_artifact) = recursive_artifact {
                let loaded = Constraint::load(recursive_artifact)?;
                let same_vocab = loaded.token_bytes_count() == inner.token_bytes_count()
                    && inner.token_bytes.iter().all(|(&token_id, bytes)| {
                        loaded.token_bytes_for_id(token_id) == Some(bytes.as_slice())
                    });
                if !same_vocab {
                    return Err(crate::GlrMaskError::Serialization(
                        "recursive dynamic alternative artifact targets a different vocabulary"
                            .to_owned(),
                    ));
                }
                *inner = loaded;
            }
            inner.boundary_trigger = trigger.into_trigger();
        }
        Ok(constraint)
    }

    fn from_legacy_payload_v14_main(
        payload: LegacyDynamicConstraintPayloadV14Main,
    ) -> crate::Result<Self> {
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

    fn from_legacy_payload_v15_residual(
        payload: LegacyDynamicConstraintPayloadV15Residual,
    ) -> crate::Result<Self> {
        let mut alternatives = Vec::with_capacity(payload.alternatives.len());
        for mut alternative in payload.alternatives {
            let exprs = alternative.v2.v1.terminal_exprs.clone();
            alternative
                .v2
                .v1
                .tokenizer
                .restore_terminal_exprs_with_virtual_runtime_metadata(
                    exprs,
                    &alternative.virtual_runtimes,
                    false,
                )
                .map_err(crate::GlrMaskError::Serialization)?;
            alternatives.push(Self::from_payload_v2(alternative.v2));
        }
        if alternatives.is_empty() {
            return Err(crate::GlrMaskError::Serialization(
                "dynamic union artifact has no alternatives".to_owned(),
            ));
        }
        Ok(Self::from_alternatives(alternatives))
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
        let constraints = std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .collect::<Vec<_>>();
        let share_vocab = constraints.first().is_none_or(|first| {
            constraints
                .iter()
                .skip(1)
                .all(|constraint| constraint.token_bytes == first.token_bytes)
        });
        let dynamic_mask_vocab = share_vocab
            .then(|| {
                constraints
                    .iter()
                    .find_map(|constraint| constraint.dynamic_mask_vocab.to_vocab_artifact())
            })
            .flatten();
        let payload = DynamicConstraintPayloadV7 {
            alternatives: constraints
                .into_iter()
                .map(Self::payload_v7_for_constraint)
                .collect(),
            dynamic_mask_vocab,
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
                | DYNAMIC_TRANSFER_VERSION_V4
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V5
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V6
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V7
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
        let exact_virtual_metadata = matches!(
            version,
            LEGACY_DYNAMIC_TRANSFER_VERSION_V5
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V6
                | LEGACY_DYNAMIC_TRANSFER_VERSION_V7
                | DYNAMIC_TRANSFER_VERSION
        );
        let allow_legacy_exact_dead_residual_roots =
            version == LEGACY_DYNAMIC_TRANSFER_VERSION_V5;
        let payload_alternatives: Vec<(
            DynamicConstraintTransferAlternativeV2,
            Vec<(TerminalID, Vec<u32>)>,
            DynamicBoundaryTriggerWire,
            Option<Vec<u8>>,
        )> = match version {
            DYNAMIC_TRANSFER_VERSION => {
                bincode::deserialize::<DynamicConstraintTransferPayloadV4>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
                .alternatives
                .into_iter()
                .map(|alternative| {
                    (
                        alternative.base.base,
                        alternative.base.terminal_observation_classes,
                        alternative.boundary_trigger,
                        alternative.recursive_constraint_artifact,
                    )
                })
                .collect()
            }
            LEGACY_DYNAMIC_TRANSFER_VERSION_V7 => {
                bincode::deserialize::<DynamicConstraintTransferPayloadV3>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
                .alternatives
                .into_iter()
                .map(|alternative| {
                    (
                        alternative.base,
                        alternative.terminal_observation_classes,
                        DynamicBoundaryTriggerWire::None,
                        None,
                    )
                })
                .collect()
            }
            LEGACY_DYNAMIC_TRANSFER_VERSION_V5 | LEGACY_DYNAMIC_TRANSFER_VERSION_V6 => {
                bincode::deserialize::<DynamicConstraintTransferPayloadV2>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?
                .alternatives
                .into_iter()
                .map(|alternative| (alternative, Vec::new(), DynamicBoundaryTriggerWire::None, None))
                .collect()
            }
            DYNAMIC_TRANSFER_VERSION_V4 => {
                let legacy = bincode::deserialize::<DynamicConstraintTransferPayloadV1>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                legacy
                    .alternatives
                    .into_iter()
                    .map(|v1| {
                        (
                            DynamicConstraintTransferAlternativeV2 {
                                v1,
                                virtual_runtimes: Vec::new(),
                            },
                            Vec::new(),
                            DynamicBoundaryTriggerWire::None,
                            None,
                        )
                    })
                    .collect()
            }
            DYNAMIC_TRANSFER_VERSION_V3 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV3>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                legacy.alternatives.into_iter().map(|alternative| {
                    (
                        DynamicConstraintTransferAlternativeV2 {
                            v1: DynamicConstraintTransferAlternativeV1 {
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
                            },
                            virtual_runtimes: Vec::new(),
                        },
                        Vec::new(),
                        DynamicBoundaryTriggerWire::None,
                        None,
                    )
                }).collect()
            }
            DYNAMIC_TRANSFER_VERSION_V2 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV2>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                legacy.alternatives.into_iter().map(|alternative| {
                    (
                        DynamicConstraintTransferAlternativeV2 {
                            v1: DynamicConstraintTransferAlternativeV1 {
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
                            },
                            virtual_runtimes: Vec::new(),
                        },
                        Vec::new(),
                        DynamicBoundaryTriggerWire::None,
                        None,
                    )
                }).collect()
            }
            DYNAMIC_TRANSFER_VERSION_V1 => {
                let legacy = bincode::deserialize::<LegacyDynamicConstraintTransferPayloadV1>(
                    &bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..],
                )
                .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                legacy.alternatives.into_iter().map(|alternative| {
                    (
                        DynamicConstraintTransferAlternativeV2 {
                            v1: DynamicConstraintTransferAlternativeV1 {
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
                            },
                            virtual_runtimes: Vec::new(),
                        },
                        Vec::new(),
                        DynamicBoundaryTriggerWire::None,
                        None,
                    )
                }).collect()
            }
            _ => unreachable!("transfer version was validated above"),
        };
        let token_bytes = vocab.entries_arc();
        let mut alternatives = payload_alternatives
            .into_iter()
            .map(|(mut alternative, terminal_observation_classes, boundary_trigger, recursive_artifact)| -> crate::Result<Self> {
                if let Some(recursive_artifact) = recursive_artifact {
                    let mut inner = Constraint::load_with_vocab(recursive_artifact, vocab)?;
                    inner.boundary_trigger = boundary_trigger.into_trigger();
                    return Ok(Self {
                        inner,
                        alternatives: Vec::new(),
                        composition_grammars: vec![None],
                    });
                }
                if exact_virtual_metadata {
                    let exprs = alternative.v1.terminal_exprs.clone();
                    alternative
                        .v1
                        .tokenizer
                        .restore_terminal_exprs_with_virtual_runtime_metadata(
                            exprs,
                            &alternative.virtual_runtimes,
                            allow_legacy_exact_dead_residual_roots,
                        )
                        .map_err(crate::GlrMaskError::Serialization)?;
                }
                let alternative = alternative.v1;
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
                Self::restore_terminal_observation_classes(
                    &mut inner,
                    terminal_observation_classes,
                )?;
                inner.boundary_trigger = boundary_trigger.into_trigger();
                inner.rebuild_dynamic_runtime_caches();
                Ok(Self {
                    inner,
                    alternatives: Vec::new(),
                    composition_grammars: vec![None],
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
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
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V12
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V13
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V14_MAIN
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V15_RESIDUAL
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V16
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V17
                | LEGACY_DYNAMIC_CONSTRAINT_VERSION_V18
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
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V12 => {
                let payload: LegacyDynamicConstraintPayloadV12V3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(Self::migrate_legacy_v12_payload(payload))
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V13 => {
                let payload: DynamicConstraintPayloadV3 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v3(payload)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V14_MAIN => {
                let payload: LegacyDynamicConstraintPayloadV14Main =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v14_main(payload)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V15_RESIDUAL => {
                let payload: LegacyDynamicConstraintPayloadV15Residual =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_legacy_payload_v15_residual(payload)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V16 => {
                let payload: DynamicConstraintPayloadV4 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v4(payload, false)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V17 => {
                let payload: DynamicConstraintPayloadV5 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v5(payload, false)
            }
            LEGACY_DYNAMIC_CONSTRAINT_VERSION_V18 => {
                let payload: DynamicConstraintPayloadV6 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v6(payload)
            }
            DYNAMIC_CONSTRAINT_VERSION => {
                let payload: DynamicConstraintPayloadV7 =
                    bincode::deserialize(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..])
                        .map_err(|err| crate::GlrMaskError::Serialization(err.to_string()))?;
                Self::from_payload_v7(payload)
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

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word)
            .is_some_and(|bits| bits & (1u32 << bit) != 0)
    }

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
        .unwrap()
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
    fn dynamic_json_schema_bounded_pattern_uses_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"\\".to_vec()),
            (5, b"u".to_vec()),
            (6, b"0".to_vec()),
            (7, b"6".to_vec()),
            (8, b"1".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(
            !dynamic
                .inner
                .dynamic_mask_vocab_for_runtime()
                .has_terminal_observation_classes(),
            "physical terminal-observation quotients must stay disabled for virtual residual runtimes",
        );
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "the importer-generated pattern/length intersection must carry the certified prefix-code liveness oracle",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""a""#));
        assert!(accepts(br#""aa""#));
        assert!(
            accepts(br#""\u0061\u0061""#),
            "two escaped spellings of decoded 'a' must count as two JSON characters",
        );
        assert!(!accepts(br#""ab""#));

        let mut at_limit = Vec::with_capacity(5002);
        at_limit.push(b'"');
        at_limit.extend(std::iter::repeat_n(b'a', 5000));
        at_limit.push(b'"');
        assert!(accepts(&at_limit));

        let mut too_long = Vec::with_capacity(5003);
        too_long.push(b'"');
        too_long.extend(std::iter::repeat_n(b'a', 5001));
        too_long.push(b'"');
        assert!(!accepts(&too_long));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            !loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .has_terminal_observation_classes(),
            "save/load must not attach a physical observation quotient to a virtual residual runtime",
        );
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the certified liveness oracle from the retained terminal expression",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_bounded_format_uses_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b".".to_vec()),
            (4, b"-".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "format": "hostname",
            "minLength": 3,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "format/length intersections must carry the same certified JSON decoded-length liveness oracle as pattern/length intersections",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""aa""#));
        assert!(accepts(br#""aaa""#));
        assert!(accepts(br#""a.b""#));
        assert!(!accepts(br#""a..b""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the bounded format liveness oracle",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_pattern_format_and_length_share_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b".".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "format": "hostname",
            "minLength": 2,
            "maxLength": 128
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "nested pattern/format/length intersections below the generic giant-repeat threshold must still use the certified residual representation",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""a""#));
        assert!(accepts(br#""aa""#));
        assert!(accepts(br#""bb""#));
        assert!(!accepts(br#""a.b""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let encode_current = |payload: DynamicConstraintPayloadV5| {
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&LEGACY_DYNAMIC_CONSTRAINT_VERSION_V17.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        };

        let mut missing_owner = DynamicConstraintPayloadV5 {
            alternatives: vec![DynamicConstraint::payload_v5_for_constraint(&dynamic.inner)],
        };
        assert_eq!(missing_owner.alternatives[0].base.virtual_runtimes.len(), 1);
        missing_owner.alternatives[0].base.virtual_runtimes.clear();
        let error = DynamicConstraint::load(&encode_current(missing_owner)).unwrap_err();
        assert!(
            error.to_string().contains("terminal ownership mismatch"),
            "dropping a below-threshold residual owner from its physical proxy artifact must fail closed: {error}",
        );

        let mut forged_owner = DynamicConstraintPayloadV5 {
            alternatives: vec![DynamicConstraint::payload_v5_for_constraint(&dynamic.inner)],
        };
        let terminal = forged_owner.alternatives[0].base.virtual_runtimes[0].terminal as usize;
        forged_owner.alternatives[0]
            .base
            .v2
            .v1
            .terminal_exprs
            .as_mut()
            .expect("current dynamic artifact retains terminal expressions")[terminal] =
            Expr::U8Seq(b"a".to_vec());
        let error = DynamicConstraint::load(&encode_current(forged_owner)).unwrap_err();
        assert!(
            error.to_string().contains("certified bounded-code residual"),
            "a below-threshold residual owner cannot be forged for an uncertified expression: {error}",
        );
    }

    #[test]
    fn dynamic_json_schema_cross_branch_bounded_string_constraints_share_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"bb".to_vec()),
            (5, b".".to_vec()),
            (6, b"-".to_vec()),
        ]);
        let schemas = [
            r#"{
                "allOf": [
                    {"type":"string","format":"hostname","minLength":2,"maxLength":5000},
                    {"type":"string","pattern":"^(?:a|bb)+$","minLength":3,"maxLength":5000},
                    {"type":"string","pattern":"^(?:a|bbb)+$","maxLength":5000}
                ]
            }"#,
            r#"{
                "allOf": [
                    {"type":"string","pattern":"^(?:a|bb)+$","minLength":2,"maxLength":6000},
                    {"type":"string","pattern":"^(?:a|bbb)+$","minLength":3,"maxLength":5000},
                    {"type":"string","format":"hostname","maxLength":5500}
                ]
            }"#,
        ];

        for schema in schemas {
            let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
            assert!(
                dynamic.inner.tokenizer.has_virtual_residual_runtime(),
                "expected virtual residual runtime for cross-branch schema: {schema}",
            );
            assert!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count()
                    > 0,
                "cross-branch bounded string constraints must flatten to one common JSON decoded-length envelope plus finite constraint operands: {schema}",
            );
            assert!(
                !dynamic
                    .inner
                    .dynamic_mask_vocab_for_runtime()
                    .has_terminal_observation_classes(),
                "virtual residual constraints must not attach a physical observation quotient",
            );

            let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
            assert!(
                loaded
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count()
                    > 0,
                "save/load must reconstruct the cross-branch bounded-code oracle: {schema}",
            );
            assert_eq!(loaded.start().mask(), dynamic.start().mask());
        }
    }

    #[test]
    fn dynamic_json_schema_allof_bounded_patterns_share_exact_code_liveness_oracle() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"c".to_vec()),
            (5, b"\\".to_vec()),
            (6, b"u".to_vec()),
            (7, b"0".to_vec()),
            (8, b"6".to_vec()),
            (9, b"1".to_vec()),
        ]);
        let schema = r#"{
            "allOf": [
                {
                    "type": "string",
                    "pattern": "^(?:a|bb)+$",
                    "minLength": 2,
                    "maxLength": 5000
                },
                {
                    "type": "string",
                    "pattern": "^(?:a|cc)+$",
                    "minLength": 3,
                    "maxLength": 4000
                }
            ]
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "allOf branches with the same JSON length envelope language should share one exact bounded-code oracle",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts(br#""aa""#));
        assert!(accepts(br#""aaa""#));
        assert!(accepts(br#""\u0061\u0061\u0061""#));
        assert!(!accepts(br#""bbb""#));
        assert!(!accepts(br#""ccc""#));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the coalesced allOf oracle",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_bounded_unicode_pattern_keeps_raw_and_escaped_spellings_exact() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, "é".as_bytes().to_vec()),
            (2, br"\u00e9".to_vec()),
            (3, br"\u00E9".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:é|xx)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes).is_ok() && state.is_accepting()
        };
        assert!(!accepts("\"é\"".as_bytes()));
        assert!(accepts("\"éé\"".as_bytes()));
        assert!(accepts(br#""\u00e9\u00E9""#));
        assert!(accepts("\"é\\u00e9\"".as_bytes()));
        assert!(!accepts("\"éx\"".as_bytes()));
    }

    #[test]
    fn static_constraint_artifact_preserves_dynamic_runtime_sidecars() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"x".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let original = dynamic.clone().into_constraint();
        assert!(!original.tokenizer.virtual_runtime_metadata().is_empty());
        let loaded = Constraint::load(original.save()).unwrap();
        assert_eq!(
            loaded.tokenizer.virtual_runtime_metadata(),
            original.tokenizer.virtual_runtime_metadata(),
        );
        assert_eq!(loaded.start().mask(), original.start().mask());
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
    fn dynamic_v19_persists_vocab_and_validates_shared_vocab() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"b".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a+/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();

        let current = constraint.save();
        assert_eq!(
            u16::from_le_bytes([current[8], current[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let payload: DynamicConstraintPayloadV7 =
            bincode::deserialize(&current[DYNAMIC_CONSTRAINT_HEADER_LEN..]).unwrap();
        assert!(payload.dynamic_mask_vocab.is_some());
        let current_loaded = DynamicConstraint::load(&current).unwrap();
        assert_eq!(current_loaded.start().mask(), constraint.start().mask());

        let mut mismatched_shared_vocab = payload.clone();
        let mut second = mismatched_shared_vocab.alternatives[0].clone();
        second.base.base.v2.v1.token_bytes =
            Arc::new(BTreeMap::from([(0, b"z".to_vec())]));
        mismatched_shared_vocab.alternatives.push(second);
        let mismatched_shared_vocab = bincode::serialize(&mismatched_shared_vocab).unwrap();
        let mut malformed =
            Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + mismatched_shared_vocab.len());
        malformed.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        malformed.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
        malformed.extend_from_slice(&(mismatched_shared_vocab.len() as u64).to_le_bytes());
        malformed.extend_from_slice(&mismatched_shared_vocab);
        let error = DynamicConstraint::load(&malformed).unwrap_err();
        assert!(error
            .to_string()
            .contains("shares a vocabulary index across alternatives with different token bytes"));

    }

    #[test]
    fn dynamic_virtual_unit_repeat_save_load_round_trip() {
        let vocab = Vocab::new(vec![
            (0, b"b".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"aaa".to_vec()),
            (4, b"baaa".to_vec()),
            (5, b"baaaaa".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let grammar = r#"
            start start;
            t A ::= /a{0,1000000000}/;
            t B ::= 'b';
            nt start ::= B A;
        "#;
        let constraint = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );

        let bytes = constraint.save();
        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_zero_min_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "load must reconstruct the exact arithmetic lexer sidecar",
        );
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(4).unwrap();
        loaded_state.commit_token(4).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_virtual_runtime_metadata_is_exact_and_fail_closed() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                t B ::= /b{0,9000}/;
                nt start ::= A | B;
            "#,
            &vocab,
        )
        .unwrap();
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 2);

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata().len(), 2);
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata().len(), 2);
        assert_eq!(constraint.start().mask(), transferred.start().mask());

        fn encode_payload(base: DynamicConstraintPayloadV4Alternative) -> Vec<u8> {
            let payload = DynamicConstraintPayloadV7 {
                alternatives: vec![DynamicConstraintPayloadV7Alternative {
                    base: DynamicConstraintPayloadV5Alternative {
                        base,
                        terminal_observation_classes: Vec::new(),
                    },
                    boundary_trigger: DynamicBoundaryTriggerWire::None,
                    recursive_constraint_artifact: None,
                }],
                dynamic_mask_vocab: None,
            };
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        }

        let mut missing = DynamicConstraint::payload_v4_for_constraint(&constraint.inner);
        missing.virtual_runtimes.pop();
        let error = DynamicConstraint::load(&encode_payload(missing)).unwrap_err();
        assert!(
            error.to_string().contains("terminal ownership mismatch"),
            "unexpected missing-runtime error: {error}",
        );

        let mut duplicate_root = DynamicConstraint::payload_v4_for_constraint(&constraint.inner);
        let root = duplicate_root.virtual_runtimes[0].root_state;
        duplicate_root.virtual_runtimes[1].root_state = root;
        let error = DynamicConstraint::load(&encode_payload(duplicate_root)).unwrap_err();
        assert!(
            error.to_string().contains("invalid terminal/root ownership"),
            "unexpected duplicate-root error: {error}",
        );

        let mut mismatched_support = DynamicConstraint::payload_v4_for_constraint(&constraint.inner);
        let terminal = mismatched_support.virtual_runtimes[0].terminal as usize;
        if let Some(exprs) = mismatched_support.v2.v1.terminal_exprs.as_mut() {
            exprs[terminal] = Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"z".to_vec())),
                min: 0,
                max: Some(10_000),
            };
        }
        let error = DynamicConstraint::load(&encode_payload(mismatched_support)).unwrap_err();
        assert!(
            error.to_string().contains("byte support"),
            "unexpected byte-support mismatch error: {error}",
        );
    }

    #[test]
    fn dynamic_v19_terminal_observation_certificate_is_shape_validated() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a+/;
                t B ::= /b+/;
                nt start ::= A | B;
            "#,
            &vocab,
        )
        .unwrap();
        let states = constraint.inner.tokenizer.num_states() as usize;
        let terminals = constraint.inner.tokenizer.num_terminals();

        let encode = |terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>| {
            let mut alternative = DynamicConstraint::payload_v5_for_constraint(&constraint.inner);
            alternative.terminal_observation_classes = terminal_observation_classes;
            let payload = DynamicConstraintPayloadV7 {
                alternatives: vec![DynamicConstraintPayloadV7Alternative {
                    base: alternative,
                    boundary_trigger: DynamicBoundaryTriggerWire::None,
                    recursive_constraint_artifact: None,
                }],
                dynamic_mask_vocab: None,
            };
            let payload = bincode::serialize(&payload).unwrap();
            let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
            bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_VERSION.to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&payload);
            bytes
        };

        let bad_terminal = DynamicConstraint::load(&encode(vec![(
            terminals,
            vec![1; states],
        )]))
        .unwrap_err();
        assert!(bad_terminal
            .to_string()
            .contains("terminal-observation certificate references terminal"));

        let duplicate = DynamicConstraint::load(&encode(vec![
            (0, vec![1; states]),
            (0, vec![1; states]),
        ]))
        .unwrap_err();
        assert!(duplicate
            .to_string()
            .contains("terminal-observation certificate repeats terminal"));

        let bad_len = DynamicConstraint::load(&encode(vec![(
            0,
            vec![1; states.saturating_sub(1)],
        )]))
        .unwrap_err();
        assert!(bad_len
            .to_string()
            .contains("terminal-observation certificate for terminal"));
    }

    #[test]
    fn dynamic_standalone_virtual_unit_v19_and_transfer_v8_round_trip() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].kind, crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeKind::UnitRepeat);
        assert_eq!(metadata[0].root_state, 0);

        let saved = constraint.save();
        assert_eq!(u16::from_le_bytes([saved[8], saved[9]]), DYNAMIC_CONSTRAINT_VERSION);
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(u16::from_le_bytes([transfer[8], transfer[9]]), DYNAMIC_TRANSFER_VERSION);
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert_eq!(transferred.start().mask(), constraint.start().mask());

        for token in [0u32, 1, 2] {
            let mut original = constraint.start();
            let mut loaded_state = loaded.start();
            let mut transferred_state = transferred.start();
            original.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            transferred_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original.is_accepting());
            assert_eq!(transferred_state.is_accepting(), original.is_accepting());
            assert_eq!(loaded_state.mask(), original.mask());
            assert_eq!(transferred_state.mask(), original.mask());
        }

    }

    #[test]
    fn dynamic_v19_combines_composition_and_residual_runtime_metadata() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let mut constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        constraint.inner.late_grammar_slots = vec![crate::runtime::LateGrammarSlot {
            name: "child".to_owned(),
            terminal_id: 0,
        }];
        constraint.inner.boundary_trigger = crate::runtime::BoundaryTrigger::Tokens(
            crate::runtime::BoundaryTokenTrigger::token_tsids(vec![1u32], vec![0u32]),
        );
        let state_count = constraint.inner.tokenizer.num_states() as u32;
        constraint.inner.dynamic_mask_vocab.set_terminal_observation_classes(vec![(
            0,
            Arc::from((1..=state_count).collect::<Vec<_>>().into_boxed_slice()),
        )]);
        assert!(constraint.inner.dynamic_mask_vocab.to_vocab_artifact().is_some());
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert!(!metadata.is_empty());

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_CONSTRAINT_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.inner.late_grammar_slots, constraint.inner.late_grammar_slots);
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert!(loaded.inner.dynamic_mask_vocab.has_terminal_observation_classes());
        assert!(loaded.inner.dynamic_mask_vocab.to_vocab_artifact().is_some());
        assert_eq!(
            loaded.inner.boundary_trigger.token_summary().map(|tokens| tokens.to_vec()),
            Some(vec![1]),
        );
        let crate::runtime::BoundaryTrigger::Tokens(loaded_trigger) =
            &loaded.inner.boundary_trigger
        else {
            panic!("expected Tokens trigger after dynamic save/load");
        };
        assert_eq!(loaded_trigger.explicit_tsids(), Some(&[0u32][..]));
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert_eq!(transferred.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert!(transferred.inner.dynamic_mask_vocab.has_terminal_observation_classes());
        assert_eq!(
            transferred
                .inner
                .boundary_trigger
                .token_summary()
                .map(|tokens| tokens.to_vec()),
            Some(vec![1]),
        );
        let crate::runtime::BoundaryTrigger::Tokens(transferred_trigger) =
            &transferred.inner.boundary_trigger
        else {
            panic!("expected Tokens trigger after dynamic transfer round-trip");
        };
        assert_eq!(transferred_trigger.explicit_tsids(), Some(&[0u32][..]));
        assert_eq!(transferred.start().mask(), constraint.start().mask());

    }

    #[test]
    #[ignore = "pre-release legacy dynamic artifact compatibility is not an acceptance requirement"]
    fn dynamic_main_v14_late_slot_artifact_still_loads() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let mut constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= "a";
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        constraint.inner.late_grammar_slots = vec![crate::runtime::LateGrammarSlot {
            name: "child".to_owned(),
            terminal_id: 0,
        }];
        let legacy = LegacyDynamicConstraintPayloadV14Main {
            alternatives: vec![LegacyDynamicConstraintPayloadV14MainAlternative {
                constraint: DynamicConstraint::payload_for_constraint(&constraint.inner),
                late_grammar_slots: constraint.inner.late_grammar_slots.clone(),
            }],
        };
        let payload = bincode::serialize(&legacy).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&LEGACY_DYNAMIC_CONSTRAINT_VERSION_V14_MAIN.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert_eq!(loaded.inner.late_grammar_slots, constraint.inner.late_grammar_slots);
        assert_eq!(loaded.start().mask(), constraint.start().mask());
    }

    #[test]
    #[ignore = "pre-release legacy dynamic artifact compatibility is not an acceptance requirement"]
    fn dynamic_virtual_runtime_legacy_v13_and_transfer_v4_still_load() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A ::= /a{0,10000}/;
                nt start ::= A;
            "#,
            &vocab,
        )
        .unwrap();
        let original_mask = constraint.start().mask();

        let legacy_persistent = DynamicConstraintPayloadV3 {
            alternatives: vec![DynamicConstraint::payload_for_constraint(&constraint.inner)],
        };
        let payload = bincode::serialize(&legacy_persistent).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_CONSTRAINT_MAGIC);
        bytes.extend_from_slice(&LEGACY_DYNAMIC_CONSTRAINT_VERSION_V13.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection()
            || loaded.inner.tokenizer.virtual_zero_min_unit_repeat_mask_tokenizer(2).is_some());
        assert_eq!(loaded.start().mask(), original_mask);

        let legacy_transfer = DynamicConstraintTransferPayloadV1 {
            alternatives: vec![DynamicConstraintTransferAlternativeV1 {
                table: constraint.inner.table.clone(),
                terminal_display_names: constraint.inner.terminal_display_names.clone(),
                tokenizer: constraint.inner.tokenizer.clone(),
                ignore_terminal: constraint.inner.ignore_terminal,
                direct_regular_automaton: constraint.inner.direct_regular_automaton.clone(),
                special_token_terminals: constraint.inner.special_token_terminals.clone(),
                ignore_expr: constraint.inner.ignore_expr.clone(),
                terminal_exprs: constraint.inner.tokenizer.terminal_exprs().map(ToOwned::to_owned),
                mask_tokenizer: None,
                full_to_mask_state: Vec::new(),
            }],
        };
        let payload = bincode::serialize(&legacy_transfer).unwrap();
        let mut bytes = Vec::with_capacity(DYNAMIC_CONSTRAINT_HEADER_LEN + payload.len());
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_VERSION_V4.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), original_mask);
    }

    fn variable_width_repeat_intersection_grammar(
        left_max: usize,
        right_max: usize,
    ) -> crate::grammar::flat::GrammarDef {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let left_body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"bb".to_vec()),
        ]);
        let right_body = Expr::Choice(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ]);
        GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(left_body),
                        min: 0,
                        max: Some(left_max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(right_body),
                        min: 0,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        }
    }

    fn variable_width_repeat_intersection_vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"abb".to_vec()),
            (3, b"aabb".to_vec()),
            (4, b"bbbb".to_vec()),
            (5, b"ab".to_vec()),
            (6, b"b".to_vec()),
            (7, b"x".to_vec()),
        ])
    }

    #[test]
    fn dynamic_billion_by_billion_repeat_intersection_is_lazy_end_to_end() {
        let vocab = variable_width_repeat_intersection_vocab();
        let grammar = variable_width_repeat_intersection_grammar(
            1_000_000_000,
            900_000_000,
        );
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            variable_width_repeat_intersection_grammar(32, 32),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            constraint.inner.tokenizer.num_states() < 32,
            "physical tokenizer must not scale with either billion-sized bound",
        );
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                <= 4,
            "build-time exact residual discovery must stay constant and independent of N*M",
        );
        let mask_tokenizer = constraint
            .inner
            .dynamic_mask_vocab
            .mask_projection_tokenizer()
            .expect("lazy exact product must install a finite mask tokenizer");
        assert!(
            mask_tokenizer.num_states() < 2_000,
            "mask tokenizer must scale with vocab horizon/body DFAs, not N*M",
        );

        let start_mask = constraint.start().mask();
        assert_eq!(
            start_mask,
            oracle.start().mask(),
            "far from either upper bound, the billion-scale lazy lexer must have the same finite-vocabulary observations as a materialized 32x32 oracle",
        );
        assert!(!token_allowed(&start_mask, 7), "x cannot begin either repeat language");

        let mut state = constraint.start();
        let mut oracle_state = oracle.start();
        state.commit_token(2).unwrap(); // "abb" = "a" + "bb"
        oracle_state.commit_token(2).unwrap();
        assert!(state.is_accepting());
        let after = state.mask();
        assert_eq!(after, oracle_state.mask());
        assert!(
            constraint
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                < 32,
            "a short commit must discover only a short exact residual path",
        );
    }

    #[test]
    fn dynamic_repeat_intersection_matches_materialized_boundary_oracle() {
        let vocab = variable_width_repeat_intersection_vocab();
        let grammar = variable_width_repeat_intersection_grammar(4, 5);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar.clone(),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let ordinary = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());

        let mut dynamic_state = dynamic.start();
        let mut ordinary_state = ordinary.start();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        dynamic_state.commit_token(2).unwrap();
        ordinary_state.commit_token(2).unwrap();
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        dynamic_state.commit_token(1).unwrap();
        ordinary_state.commit_token(1).unwrap();
        assert_eq!(dynamic_state.is_accepting(), ordinary_state.is_accepting());
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
    }

    #[test]
    fn dynamic_virtual_repeat_intersection_save_load_round_trip() {
        let vocab = variable_width_repeat_intersection_vocab();
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            variable_width_repeat_intersection_grammar(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());

        let bytes = constraint.save();
        let loaded = DynamicConstraint::load(&bytes).unwrap();
        assert!(
            loaded.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "load must reconstruct the lazy exact product sidecar",
        );
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_binary_repeat_intersection_interned_state_count()
                <= 4,
            "load-time residual reconstruction must stay constant and independent of N*M",
        );
        assert_eq!(constraint.start().mask(), loaded.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(2).unwrap();
        loaded_state.commit_token(2).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_virtual_repeat_intersection_rejects_dead_common_prefix() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                        min: 0,
                        max: Some(1_000_000_000),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ac".to_vec())),
                        min: 0,
                        max: Some(900_000_000),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ac".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let constraint = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle_grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ab".to_vec())),
                        min: 0,
                        max: Some(8),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"ac".to_vec())),
                        min: 0,
                        max: Some(8),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            oracle_grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(constraint.inner.tokenizer.has_virtual_binary_repeat_intersection());
        let mask = constraint.start().mask();
        assert_eq!(mask, oracle.start().mask());
        assert!(
            !token_allowed(&mask, 0),
            "the common first byte 'a' is dead: left requires b next while right requires c",
        );
        assert!(mask.iter().all(|&word| word == 0));
    }

    #[test]
    fn dynamic_billion_bound_variable_width_repeat_is_lazy_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"ab".to_vec()),
                        Expr::U8Seq(b"ac".to_vec()),
                    ])),
                    min: 0,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"ac".to_vec()),
            (3, b"abab".to_vec()),
            (4, b"acab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(16),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [3, 4, 1, 2] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(0).unwrap();
        loaded_state.commit_token(0).unwrap();
        assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
        assert_eq!(loaded_state.mask(), original_state.mask());
    }

    #[test]
    fn dynamic_billion_nonzero_min_variable_width_repeat_is_lazy_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min, max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"ab".to_vec()),
                        Expr::U8Seq(b"c".to_vec()),
                    ])),
                    min,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"cab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(999_999_997, 1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        // A small ordinary oracle with the same local lower/upper-bound
        // geometry is enough for the first few model-token walks. Both starts
        // are farther than one vocabulary horizon from either boundary.
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(10, 13),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [1u32, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert!(!dynamic_state.is_accepting());
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_binary_repeat_intersections_mask_tokenizer(vocab.max_token_byte_len())
                .is_some()
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [1u32, 2, 3] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_same_body_nonzero_repeat_intersection_factors_end_to_end() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_min, left_max, right_min, right_max| {
            let body = Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"c".to_vec()),
            ]);
            GrammarDef {
                start: 0,
                rules: vec![Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                }],
                terminals: vec![Terminal::Expr {
                    id: 0,
                    expr: Expr::Intersect {
                        expr: Box::new(Expr::Repeat {
                            expr: Box::new(body.clone()),
                            min: left_min,
                            max: Some(left_max),
                        }),
                        intersect: Box::new(Expr::Repeat {
                            expr: Box::new(body),
                            min: right_min,
                            max: Some(right_max),
                        }),
                    },
                }],
                ..GrammarDef::default()
            }
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"cab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(900_000_000, 1_000_000_000, 999_999_997, 999_999_999),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(5, 13, 10, 12),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(dynamic.inner.tokenizer.num_states() < 32);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [1u32, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [1u32, 2, 3] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_billion_nonzero_min_unit_repeat_uses_arithmetic_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min, max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(500_000_000, 1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "one-byte nonzero-min repeats should use the O(1) arithmetic sidecar",
        );

        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(10, 20),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [2u32, 1, 0] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(!loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [2u32, 1, 0] {
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_aligned_nonzero_unit_intersection_uses_arithmetic_runtime() {
        use crate::automata::regex::Expr;
        use crate::ds::u8set::U8Set;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_min, left_max, right_min, right_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                        min: left_min,
                        max: Some(left_max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
                        min: right_min,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"b".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"bbb".to_vec()),
            (3, b"a".to_vec()),
            (4, b"c".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(500_000_000, 1_000_000_000, 600_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(5, 20, 10, 15),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
            "aligned one-byte intersection should factor to the arithmetic repeat runtime",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [2u32, 1, 0] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(!loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                .is_some(),
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_giant_aligned_unit_empty_intersection_normalizes_before_validation() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |min| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min,
                        max: Some(1_000_000_000),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min,
                        max: Some(1_000_000_000),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);

        for min in [0usize, 1] {
            let dynamic =
                crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
                    grammar_for(min),
                    &vocab,
                    crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
                )
                .unwrap();
            assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
            assert!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_unit_repeat_mask_tokenizer(vocab.max_token_byte_len())
                    .is_none(),
                "empty/epsilon normalization should remove the giant repeat entirely",
            );
            let saved = dynamic.save();
            let loaded = DynamicConstraint::load(&saved).unwrap();
            assert_eq!(loaded.start().mask(), dynamic.start().mask());
        }
    }

    #[test]
    fn dynamic_distinct_nonzero_min_giant_intersection_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 3,
                        max: Some(max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"aa".to_vec())),
                        min: 2,
                        max: Some(max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [
            vec![0u32],
            vec![1u32],
            vec![0u32, 0, 0],
            vec![1u32, 1],
            vec![0u32, 1, 0],
        ] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_multiple_large_plain_repeats_use_exact_virtual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"b".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "both large terminals must stay on exact lazy virtual runtimes",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![0u32, 1], vec![0u32, 0, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_non_prefix_free_large_repeat_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    // `a | aa` is not prefix-free, so one scalar repeat
                    // coordinate is not a complete exact residual.
                    expr: Box::new(Expr::Choice(vec![
                        Expr::U8Seq(b"a".to_vec()),
                        Expr::U8Seq(b"aa".to_vec()),
                    ])),
                    min: 0,
                    max: Some(max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![0u32, 1], vec![1u32, 1, 0]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_general_residual_prunes_semantically_dead_token_prefix() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| {
            let giant_branch = |suffix: &[u8]| {
                Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"x".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(suffix.to_vec()),
                ])
            };
            GrammarDef {
                start: 0,
                rules: vec![Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                }],
                terminals: vec![Terminal::Expr {
                    id: 0,
                    expr: Expr::Intersect {
                        expr: Box::new(Expr::Choice(vec![giant_branch(b"ab"), Expr::U8Seq(b"c".to_vec())])),
                        intersect: Box::new(Expr::Choice(vec![
                            giant_branch(b"ac"),
                            Expr::U8Seq(b"c".to_vec()),
                        ])),
                    },
                }],
                ..GrammarDef::default()
            }
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"c".to_vec()),
            (2, b"x".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let start_mask = dynamic.start().mask();
        let allowed = |token: u32| {
            let word = token as usize / 32;
            let bit = token % 32;
            start_mask[word] & (1u32 << bit) != 0
        };
        assert!(!allowed(0), "a leads to the dead residual b ∩ c and must be pruned");
        assert!(allowed(1), "c is the one common accepted word");
        assert!(!allowed(2), "entering the giant x* branches can never reach a common suffix");

        let mut rejected = dynamic.start();
        assert!(rejected.commit_token(0).is_err());
        let mut accepted = dynamic.start();
        accepted.commit_token(1).unwrap();
        assert!(accepted.is_accepting());

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), start_mask);
    }

    #[test]
    fn dynamic_nested_large_repeat_suffix_compiles_lazily_and_exactly() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"b".to_vec()),
                ]),
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![1u32], vec![0u32, 1], vec![0u32, 0, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_nested_large_repeat_in_lazy_intersection_remains_supported() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::U8Seq(b"[".to_vec()),
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"]".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Seq(vec![
                        Expr::U8Seq(b"[".to_vec()),
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(7),
                        },
                        Expr::U8Seq(b"]".to_vec()),
                    ])),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"[".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"]".to_vec()),
            (4, b"[a]".to_vec()),
            (5, b"[aa]".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            dynamic.inner.tokenizer.num_states() < 128,
            "nested giant repeat must stay on the existing lazy intersection path",
        );
        assert!(
            !dynamic
                .inner
                .tokenizer
                .has_virtual_binary_repeat_intersection(),
            "this regression must exercise the nested repeat-with-suffix lane, not the top-level virtual repeat product",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        for token in [0, 2, 3] {
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }
    }

    #[test]
    fn dynamic_general_residual_repeat_supports_u32_max_bound() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Seq(vec![
                    Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"aa".to_vec()),
                        ])),
                        min: 0,
                        max: Some(max),
                    },
                    Expr::U8Seq(b"z".to_vec()),
                ]),
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"z".to_vec()),
            (3, b"aaz".to_vec()),
        ]);

        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(u32::MAX as usize),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![2u32], vec![3u32], vec![0u32, 2], vec![1u32, 2]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_hybrid_unit_repeat_near_state_id_limit_uses_repeat_product() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar = GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(((1u64 << 31) - 1) as usize),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::U8Seq(b"b".to_vec()),
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "hybrid physical states leave too little high-bit state-ID space for the arithmetic unit lane",
        );
        assert!(dynamic.inner.tokenizer.num_states() < 64);

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_top_level_large_repeat_inside_finite_intersection_stays_exact() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"ab".to_vec()),
                            Expr::U8Seq(b"c".to_vec()),
                        ])),
                        min: 3,
                        max: Some(max),
                    }),
                    // `abcabc` has the unique body factorization
                    // `ab · c · ab · c`, so it is accepted at count four.
                    // The finite right coordinate bounds generic product
                    // discovery independently of the giant repeat maximum.
                    intersect: Box::new(Expr::U8Seq(b"abcabc".to_vec())),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"abcabc".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"c".to_vec()),
            (3, b"abc".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            !dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "this case should use the generic product's virtual repeat coordinate, not a runtime sidecar",
        );
        assert!(dynamic.inner.tokenizer.num_states() < 64);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for sequence in [vec![0u32], vec![1u32, 2, 1, 2], vec![3u32, 3]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        let mut loaded_state = loaded.start();
        loaded_state.commit_token(0).unwrap();
        assert!(loaded_state.is_accepting());
    }

    #[test]
    fn dynamic_top_level_large_repeat_inside_cyclic_intersection_uses_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    }),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: None,
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![1u32], vec![1u32, 1], vec![0u32, 1, 0]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_nested_giant_with_budgeted_other_repeat_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::ds::u8set::U8Set;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(left_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    // This root repeat is above the generic product's direct
                    // threshold and would normally stay virtual. Its exact DFA
                    // is nevertheless tiny enough for the lazy intersection's
                    // explicitly budgeted ordinary side.
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                        min: 0,
                        max: Some(100),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"b".to_vec()),
            (3, b"ab".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(128),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(
            dynamic.inner.tokenizer.num_states() < 512,
            "the physical tokenizer must stay small independently of the giant max",
        );
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert!(
            dynamic.inner.tokenizer.has_virtual_residual_runtime(),
            "nested giant intersections outside the arithmetic fast paths must use the general residual runtime",
        );
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for sequence in [vec![2u32], vec![3u32], vec![1u32, 2], vec![0u32, 0, 2]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let mut dynamic_boundary = dynamic.start();
        let mut oracle_boundary = oracle.start();
        for _ in 0..49 {
            dynamic_boundary.commit_token(1).unwrap();
            oracle_boundary.commit_token(1).unwrap();
        }
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());
        assert!(!dynamic_boundary.is_accepting());

        // At 98 leading 'a' bytes, one more 'a' is still live but another
        // two-byte "aa" token would overshoot the right-hand 100-byte cap once
        // the required trailing 'b' is included.
        let mut dynamic_overrun = dynamic_boundary.clone();
        let mut oracle_overrun = oracle_boundary.clone();
        assert!(dynamic_overrun.commit_token(1).is_err());
        assert!(oracle_overrun.commit_token(1).is_err());

        dynamic_boundary.commit_token(0).unwrap();
        oracle_boundary.commit_token(0).unwrap();
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());
        assert!(!dynamic_boundary.is_accepting());
        dynamic_boundary.commit_token(2).unwrap();
        oracle_boundary.commit_token(2).unwrap();
        assert!(dynamic_boundary.is_accepting());
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved).unwrap();
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        assert!(
            loaded.inner.tokenizer.has_virtual_residual_runtime(),
            "save/load must reconstruct the general residual runtime",
        );
        let mut loaded_state = loaded.start();
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        loaded_state.commit_token(3).unwrap();
        dynamic_state.commit_token(3).unwrap();
        oracle_state.commit_token(3).unwrap();
        assert!(loaded_state.is_accepting());
        assert_eq!(loaded_state.is_accepting(), dynamic_state.is_accepting());
        assert_eq!(loaded_state.is_accepting(), oracle_state.is_accepting());
        assert_eq!(loaded_state.mask(), dynamic_state.mask());
        assert_eq!(loaded_state.mask(), oracle_state.mask());
    }

    #[test]
    fn dynamic_nested_giant_with_large_finite_other_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let mut long_body = vec![b'a'; 31];
        long_body.push(b'b');
        let grammar_for = |left_max, right_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(left_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(long_body.clone())),
                        min: 0,
                        max: Some(right_max),
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, long_body.clone()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000, 4_095),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(64, 4),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        dynamic_state.commit_token(2).unwrap();
        oracle_state.commit_token(2).unwrap();
        assert!(dynamic_state.is_accepting());
        assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
        assert_eq!(dynamic_state.mask(), oracle_state.mask());

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_empty_giant_cyclic_intersection_becomes_dead_residual_proxy() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    intersect: Box::new(Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: None,
                    }),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        assert!(dynamic.start().mask().iter().all(|&word| word == 0));

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_two_nested_large_repeat_components_with_disjoint_delimiters_factor_to_empty() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let repeated_suffix = |suffix| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min: 0,
                    max: Some(10_000),
                },
                Expr::U8Seq(vec![suffix]),
            ])
        };
        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(repeated_suffix(b'b')),
                    intersect: Box::new(repeated_suffix(b'c')),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let start_mask = dynamic.start().mask();
        assert!(
            start_mask.iter().all(|&word| word == 0),
            "different uniquely-delimited literal suffixes make the intersection empty",
        );

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert_eq!(loaded.start().mask(), start_mask);
    }

    #[test]
    fn dynamic_positive_min_giant_suffix_with_disjoint_counts_factors_to_empty() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let component = |min, max| {
            Expr::Seq(vec![
                Expr::Repeat {
                    expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                    min,
                    max: Some(max),
                },
                Expr::U8Seq(b"b".to_vec()),
            ])
        };
        let grammar = GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Intersect {
                    expr: Box::new(component(5, 10_000)),
                    intersect: Box::new(component(1, 3)),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        assert!(dynamic.start().mask().iter().all(|&word| word == 0));
    }

    #[test]
    fn dynamic_multiple_giant_terminals_share_exact_virtual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |a_max, b_max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(a_max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"b".to_vec()),
                        ])),
                        min: 0,
                        max: Some(b_max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8, 7),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(
            dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection(),
            "multiple giant terminals must stay on shared lazy exact runtimes",
        );
        let exact_after_a = dynamic.inner.tokenizer.run(b"a");
        assert_eq!(
            exact_after_a.len(),
            2,
            "overlapping virtual terminals must retain distinct exact states",
        );
        let matched_after_a = exact_after_a
            .iter()
            .flat_map(|&state| dynamic.inner.tokenizer.matched_terminals(state))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(matched_after_a, [0u32, 1u32].into_iter().collect());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());

        for token in [0u32, 1, 2, 3, 4] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(dynamic.start().mask(), loaded.start().mask());
        for token in [0u32, 1] {
            let mut original_state = dynamic.start();
            let mut loaded_state = loaded.start();
            original_state.commit_token(token).unwrap();
            loaded_state.commit_token(token).unwrap();
            assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
            assert_eq!(original_state.mask(), loaded_state.mask());
        }

        let transfer = dynamic.clone().into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .tokenizer
                .has_virtual_binary_repeat_intersection(),
            "v6 transfer load must reconstruct all shared lazy repeat runtimes",
        );
        assert_eq!(dynamic.start().mask(), transferred.start().mask());
        for token in [0u32, 1] {
            let mut original_state = dynamic.start();
            let mut transferred_state = transferred.start();
            original_state.commit_token(token).unwrap();
            transferred_state.commit_token(token).unwrap();
            assert_eq!(
                original_state.is_accepting(),
                transferred_state.is_accepting()
            );
            assert_eq!(original_state.mask(), transferred_state.mask());
        }
    }

    #[test]
    fn dynamic_mixed_specialized_and_general_giants_share_residual_runtime_family() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                        min: 0,
                        max: Some(max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Seq(vec![
                        Expr::Repeat {
                            // Prefix ambiguity keeps this out of the simple
                            // bounded-repeat descriptor fast path.
                            expr: Box::new(Expr::Choice(vec![
                                Expr::U8Seq(b"x".to_vec()),
                                Expr::U8Seq(b"xx".to_vec()),
                            ])),
                            min: 0,
                            max: Some(max),
                        },
                        Expr::U8Seq(b"z".to_vec()),
                    ]),
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"x".to_vec()),
            (2, b"xx".to_vec()),
            (3, b"z".to_vec()),
            (4, b"xz".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert!(!dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![0u32], vec![3u32], vec![4u32], vec![2u32, 4]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_multiple_variable_width_giant_terminals_match_small_oracle() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |left_max, right_max| GrammarDef {
            start: 0,
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            terminals: vec![
                Terminal::Expr {
                    id: 0,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"bb".to_vec()),
                        ])),
                        min: 0,
                        max: Some(left_max),
                    },
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Repeat {
                        expr: Box::new(Expr::Choice(vec![
                            Expr::U8Seq(b"a".to_vec()),
                            Expr::U8Seq(b"cc".to_vec()),
                        ])),
                        min: 0,
                        max: Some(right_max),
                    },
                },
            ],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"bb".to_vec()),
            (2, b"cc".to_vec()),
            (3, b"abb".to_vec()),
            (4, b"acc".to_vec()),
            (5, b"x".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(1_000_000_000, 900_000_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(8, 7),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );

        assert!(dynamic.inner.tokenizer.has_virtual_binary_repeat_intersection());
        let exact_after_a = dynamic.inner.tokenizer.run(b"a");
        assert_eq!(exact_after_a.len(), 2);
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for token in 0u32..=4 {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            dynamic_state.commit_token(token).unwrap();
            oracle_state.commit_token(token).unwrap();
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert_eq!(dynamic.start().mask(), loaded.start().mask());
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        original_state.commit_token(3).unwrap();
        loaded_state.commit_token(3).unwrap();
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_nested_giant_repeat_body_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
        use crate::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};

        let grammar_for = |inner_max, outer_max| GrammarDef {
            start: 0,
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            terminals: vec![Terminal::Expr {
                id: 0,
                expr: Expr::Repeat {
                    expr: Box::new(Expr::Seq(vec![
                        Expr::Repeat {
                            expr: Box::new(Expr::U8Seq(b"a".to_vec())),
                            min: 0,
                            max: Some(inner_max),
                        },
                        Expr::U8Seq(b"b".to_vec()),
                    ])),
                    min: 0,
                    max: Some(outer_max),
                },
            }],
            ..GrammarDef::default()
        };
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"aab".to_vec()),
        ]);
        let dynamic = crate::compiler::pipeline::compile_dynamic_owned_with_table_construction(
            grammar_for(10_000, 10_000),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
        .unwrap();
        let oracle = crate::compiler::pipeline::compile_owned_with_table_construction(
            grammar_for(4, 4),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        );
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(dynamic.start().mask(), oracle.start().mask());
        for sequence in [vec![1u32], vec![2u32], vec![3u32], vec![4u32], vec![2u32, 1]] {
            let mut dynamic_state = dynamic.start();
            let mut oracle_state = oracle.start();
            for token in sequence {
                dynamic_state.commit_token(token).unwrap();
                oracle_state.commit_token(token).unwrap();
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let loaded = DynamicConstraint::load(&dynamic.save()).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    #[ignore = "pre-release legacy dynamic artifact compatibility is not an acceptance requirement"]
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
    #[ignore = "pre-release legacy dynamic artifact compatibility is not an acceptance requirement"]
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
