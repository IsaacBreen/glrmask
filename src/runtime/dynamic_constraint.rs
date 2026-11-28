use crate::ErrorExt as _;
// PRE-RELEASE COMPATIBILITY POLICY
// ================================
// GLRMask is still pre-release. Historical DynamicConstraint/transfer wire
// layouts are NOT a compatibility contract and must not constrain the current
// architecture. Prefer the cleanest exact current representation; legacy
// readers are not retained when preserving them would complicate correctness,
// performance, or the wire format. Do not add migration machinery unless
// compatibility becomes an explicit product requirement.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[path = "build_remainder_trace.rs"]
pub(crate) mod remainder_trace;
#[cfg(test)]
#[path = "dynamic_constraint_remainder_tests.rs"]
mod remainder_tests;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::{
    TerminalProjectedQuotient, Tokenizer,
};
use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::constraint_possible_matches::ConstraintPossibleMatchesComputation;
use crate::grammar::flat::{DirectRegularAutomaton, GrammarDef, Symbol, Terminal, TerminalID};
use crate::Vocab;

use crate::runtime::{
    dynamic_mask_profile_enabled, CommitProfile, Constraint, ConstraintState, DynamicMaskVocab,
    SpecialTokenTerminal,
};

/// Exact grammar-dependent execution metadata, separate from the shared
/// vocabulary. Both LR and native programs use the same coordinate proofs.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct DynamicExecutionMetadata {
    terminal_observation_classes: Vec<(TerminalID, Vec<u32>)>,
    projected_terminal_quotients: Vec<(TerminalID, TerminalProjectedQuotient)>,
    projected_terminal_quotients_prepared: bool,
    prepared_master_proofs: crate::runtime::PreparedMasterProofArtifact,
    mask_quotient: Option<(CompactTransferTokenizer, Vec<u32>)>,
    pub(crate) residual_oracles: Vec<(TerminalID, Vec<u8>)>,
    pub(crate) master_slices: Vec<crate::automata::lexer::tokenizer::VirtualResidualMasterSliceArtifact>,
}

impl DynamicExecutionMetadata {
    pub(crate) fn from_constraint(constraint: &Constraint) -> Self {
        Self {
            terminal_observation_classes: constraint.dynamic_mask_vocab.terminal_observation_classes_for_artifact(),
            projected_terminal_quotients: constraint.dynamic_mask_vocab.projected_terminal_quotients_for_artifact(),
            projected_terminal_quotients_prepared: constraint.dynamic_mask_vocab.projected_terminal_quotients_prepared(),
            prepared_master_proofs: constraint.dynamic_mask_vocab.prepared_master_proof_artifact(),
            mask_quotient: if constraint.dynamic_mask_vocab.is_grammar_quotiented() {
                // The vocabulary quotient artifact already owns this map.
                None
            } else {
                constraint.dynamic_mask_vocab.mask_tokenizer_quotient_for_transfer()
                    .map(|(tokenizer, map)| (CompactTransferTokenizer(tokenizer), map))
            },
            residual_oracles: if crate::env::var("GLRMASK_DYNAMIC_TRANSFER_RESIDUAL_ORACLE_SIDECAR")
                .ok().is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
            {
                constraint.tokenizer.virtual_residual_runtime_oracles()
            } else {
                Vec::new()
            },
            master_slices: constraint.tokenizer.virtual_residual_master_slice_artifacts(),
        }
    }

    pub(crate) fn restore_mask_quotient(&mut self, constraint: &mut Constraint) -> crate::Result<()> {
        if let Some((CompactTransferTokenizer(tokenizer), map)) = self.mask_quotient.take() {
            if map.len() != constraint.tokenizer.num_states() as usize
                || map.is_empty()
                || map.iter().any(|&state| state >= tokenizer.num_states())
            {
                return Err(crate::Error::Serialization(
                    "invalid dynamic mask-tokenizer quotient".into(),
                ));
            }
            constraint.dynamic_mask_vocab.set_mask_tokenizer_quotient(tokenizer, map);
        }
        Ok(())
    }

    pub(crate) fn restore(mut self, constraint: &mut Constraint) -> crate::Result<()> {
        self.restore_mask_quotient(constraint)?;
        constraint.tokenizer.restore_virtual_residual_master_slice_artifacts(
            std::mem::take(&mut self.master_slices),
        ).map_err(crate::Error::Serialization)?;
        DynamicConstraint::restore_terminal_observation_classes(constraint, self.terminal_observation_classes)?;
        if self.projected_terminal_quotients_prepared {
            DynamicConstraint::restore_projected_terminal_quotients(constraint, self.projected_terminal_quotients)?;
        } else if !self.projected_terminal_quotients.is_empty() {
            return Err(crate::Error::Serialization("unprepared projected-terminal certificate has quotient rows".to_owned()));
        }
        let source_states = constraint.tokenizer.num_states() as usize;
        constraint.dynamic_mask_vocab.restore_prepared_master_proof_artifact(self.prepared_master_proofs, source_states)
            .map_err(crate::Error::Serialization)
    }
}

const DYNAMIC_TRANSFER_MAGIC: [u8; 8] = *b"GLRDXF\0\0";
const DYNAMIC_CONSTRAINT_HEADER_LEN: usize = DYNAMIC_TRANSFER_MAGIC.len() + 2 + 8;
const DYNAMIC_TRANSFER_VERSION: u16 = 18;
const DYNAMIC_PAYLOAD_HEADER_LEN: usize = 32 + 4;

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

struct DynamicRuntimeParts {
    table: crate::runtime::parser_backend::ParserExecutable,
    terminal_display_names: Vec<String>,
    tokenizer: Tokenizer,
    ignore_terminal: Option<TerminalID>,
    direct_regular_automaton: Option<DirectRegularAutomaton>,
    token_bytes: Arc<BTreeMap<u32, Vec<u8>>>,
    ignore_expr: Option<Expr>,
    terminal_exprs: Option<Vec<Expr>>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
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
    /// Canonical current external-vocabulary artifact prepared as part of
    /// constraint finalization (or retained verbatim by load).  `save()` is a
    /// persistence operation, not a second compilation/finalization pass.
    external_vocab_artifact_cache: Option<Arc<Vec<u8>>>,
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
    ) -> crate::Result<Self> {
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
    ) -> crate::Result<Self> {
        let mut result = Self::from_parts_with_dynamic_vocab_unfinalized(
            table, terminal_display_names, tokenizer, direct_regular_automaton,
            ignore_terminal, special_token_terminals, vocab, dynamic_mask_vocab)?;
        result.inner.rebuild_dynamic_runtime_caches();
        Ok(result)
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
    ) -> crate::Result<Self> {
        Self::from_parts_with_dynamic_vocab_backend_unfinalized(
            crate::runtime::parser_backend::DynamicParserBackend::Native,
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

    /// Shared unfinalized ordinary-dynamic assembly. Native and retained-LR
    /// construction differ only at the representation boundary: the native arm
    /// derives the template parser and drops the table, while the LR arm keeps
    /// the supplied table through [`ParserExecutable::explicit`] and never
    /// derives templates, embeddings, completion domains or stack metadata.
    /// The grammar, tokenizer and dynamic mask vocabulary work is shared by the
    /// callers before this point.
    pub(crate) fn from_parts_with_dynamic_vocab_backend_unfinalized(
        backend: crate::runtime::parser_backend::DynamicParserBackend,
        table: GLRTable,
        terminal_display_names: Vec<String>,
        tokenizer: Tokenizer,
        direct_regular_automaton: Option<DirectRegularAutomaton>,
        ignore_terminal: Option<TerminalID>,
        special_token_terminals: Vec<SpecialTokenTerminal>,
        vocab: &Vocab,
        dynamic_mask_vocab: DynamicMaskVocab,
    ) -> crate::Result<Self> {
        match backend {
            crate::runtime::parser_backend::DynamicParserBackend::Native => {
                let prepared = crate::runtime::parser_backend::PreparedTemplateParser::from_compiler_parts(
                    &table, direct_regular_automaton.as_ref(), None, ignore_terminal, false,
                )?;
                drop(table);
                let mut inner = Self::from_template_runtime_parts_unfinalized(
                    tokenizer, terminal_display_names, ignore_terminal,
                    Arc::new(prepared.parser), vocab, dynamic_mask_vocab,
                );
                inner.special_token_terminals = special_token_terminals;
                inner.direct_regular_automaton = direct_regular_automaton;
                Ok(Self { inner, alternatives: Vec::new(), composition_grammars: vec![None],
                    external_vocab_artifact_cache: None })
            }
            crate::runtime::parser_backend::DynamicParserBackend::LrTable => {
                let ignore_expr = ignore_terminal
                    .and_then(|terminal| tokenizer.terminal_expr(terminal).cloned());
                let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);
                let payload = DynamicRuntimeParts {
                    table: crate::runtime::parser_backend::ParserExecutable::explicit(table),
                    terminal_display_names,
                    tokenizer,
                    ignore_terminal,
                    direct_regular_automaton,
                    token_bytes: vocab.entries_arc(),
                    ignore_expr,
                    terminal_exprs,
                    special_token_terminals,
                };
                let inner = Self::constraint_from_runtime_parts(payload, dynamic_mask_vocab, Some(vocab));
                // Retain the supplied vocabulary so the external-vocabulary
                // artifact can embed the exact canonical content digest for
                // load validation.
                let _ = inner.late_bind_vocab.set(vocab.clone());
                assert!(inner.table.is_present(), "retained LR constructor must retain its LR table");
                assert!(inner.template_parser().is_none(), "retained LR constructor must not install a template parser");
                Ok(Self { inner, alternatives: Vec::new(), composition_grammars: vec![None],
                    external_vocab_artifact_cache: None })
            }
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
    ) -> crate::Result<Self> {
        let ConstraintPossibleMatchesComputation {
            mapped_possible_matches,
            runtime_dynamic_vocab,
            complete,
            profile: _,
        } = computation;
        let (possible_matches, mut id_map) = mapped_possible_matches.into_parts();
        id_map.materialize_deferred_vocab_singletons();
        let mut result = Self::from_parts_with_dynamic_vocab(
            table, terminal_display_names, tokenizer, None, ignore_terminal,
            special_token_terminals, vocab, runtime_dynamic_vocab.vocab,
        )?;
        result.inner.non_dwa_weights.materialized_mut().possible_matches = possible_matches;
        result.inner.possible_matches_complete = complete;
        result.inner.tokenizer_tsids = crate::runtime::TokenizerTsidMap::partition(
            id_map.tokenizer_states.original_to_internal, id_map.tokenizer_states.internal_to_originals,
        );
        result.inner.token_classes = crate::runtime::TokenClasses::mapped(id_map.vocab_tokens.original_to_internal, id_map.vocab_tokens.internal_to_originals);
        Ok(result)
    }

    /// Shared assembly accepts exactly the selected executable parser, before
    /// any runtime-derived caches are initialized.
    fn constraint_from_runtime_parts(
        mut payload: DynamicRuntimeParts,
        dynamic_mask_vocab: DynamicMaskVocab,
        source_vocab: Option<&Vocab>,
    ) -> Constraint {
        payload
            .tokenizer
            .restore_terminal_exprs(payload.terminal_exprs.take())
            .expect("dynamic payload terminal expressions must match tokenizer terminal count");
        // Compile callers own precisely this Vocab's immutable byte inventory.
        // Loaded payloads have no such identity witness and must derive the
        // summary from their authoritative decoded bytes instead.
        let empty_byte_token_ids = if let Some(vocab) = source_vocab {
            assert!(Arc::ptr_eq(&payload.token_bytes, &vocab.entries_arc()),
                "compiled dynamic payload must retain its source vocabulary byte inventory");
            crate::compiler::compile::vocab_empty_byte_token_ids(vocab)
        } else {
            payload.token_bytes.iter()
                .filter_map(|(&id, bytes)| bytes.is_empty().then_some(id))
                .collect::<Vec<u32>>().into()
        };
        
        Constraint::from_data(crate::runtime::ConstraintData {
            end_tokens: std::sync::Arc::from([]),
            marker_queries: None,
            runtime_backend: crate::runtime::ConstraintRuntimeBackend::Dynamic,
            static_dynamic_overlay: None,
            boundary_trigger: crate::runtime::BoundaryTrigger::None,
            boundary_candidate_summary: std::sync::OnceLock::new(),
            late_grammar_slots: Vec::new(),
            late_bind_vocab: std::sync::OnceLock::new(),
            pending_vocab_fingerprint: None,
            pending_special_token_ids: None,
            parser_dwa: crate::runtime::ParserDwaStorage::Absent,
            non_dwa_weights: Default::default(),
            direct_regular_wide_frontier_acceptance: Vec::new(),
            direct_regular_dynamic_hot_frontiers: Vec::new(),
            direct_regular_parser_state_acceptance: Vec::new(),
            direct_regular_automaton: payload.direct_regular_automaton,
            table: payload.table,
            terminal_display_names: payload.terminal_display_names,
            tokenizer: payload.tokenizer.into(),
            tokenizer_has_epsilon_transitions: false,
            ignore_terminal: payload.ignore_terminal,
            special_token_terminals: payload.special_token_terminals,
            dynamic_mask_vocab,
            lazy_dynamic_mask_vocab: std::sync::OnceLock::new(),
            empty_byte_token_ids,
            possible_matches_complete: false,
            tokenizer_tsids: Default::default(),
                composition_grammar_summary: None,
            token_classes: Default::default(),
            token_bytes: crate::runtime::TokenByteDomain::from_entries(payload.token_bytes),
            token_projection: Default::default(),
            internal_token_dense_words: 0,
            weight_token_masks: Default::default(),
            packed_dwa_token_dense_masks: Default::default(),
            seed_terminal_dense: Default::default(),
            seed_terminal_dense_fallback: Default::default(),
            seed_universe_dense: Arc::from(Vec::<u64>::new().into_boxed_slice()),
            dwa_fast_transitions: Default::default(),
            parser_runtime_caches_prebuilt: false,
            indexed_dag_dense_transitions: Vec::new(),
            indexed_dag_dense_finals: Vec::new(),
            tokenizer_fast_transitions: Default::default(),
            parser_state_domain_labels: Vec::new(),
            ignore_expr: payload.ignore_expr,
            serialized_artifact_cache: None,
            deferred_terminal_exprs_blob: None,
            deferred_terminal_exprs: Default::default(),
            deferred_composition_metadata_blob: None,
                deferred_table_rules_blob: None,
            deferred_table_rules: Default::default(),
        })
    }

    pub(crate) fn from_template_runtime_parts(
        tokenizer: Tokenizer,
        terminal_display_names: Vec<String>,
        ignore_terminal: Option<TerminalID>,
        parser: Arc<crate::runtime::parser_backend::TemplateParser>,
        vocab: &Vocab,
    ) -> Constraint {
        let dynamic_vocab = crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab);
        let mut inner = Self::from_template_runtime_parts_unfinalized(
            tokenizer, terminal_display_names, ignore_terminal, parser, vocab, dynamic_vocab,
        );
        inner.rebuild_dynamic_runtime_caches();
        inner
    }

    pub(crate) fn from_template_runtime_parts_unfinalized(
        tokenizer: Tokenizer,
        terminal_display_names: Vec<String>,
        ignore_terminal: Option<TerminalID>,
        parser: Arc<crate::runtime::parser_backend::TemplateParser>,
        vocab: &Vocab,
        dynamic_vocab: DynamicMaskVocab,
    ) -> Constraint {
        let ignore_expr = ignore_terminal.and_then(|t|tokenizer.terminal_expr(t).cloned());
        let terminal_exprs = tokenizer.terminal_exprs().map(ToOwned::to_owned);
        let payload = DynamicRuntimeParts {
            table: crate::runtime::parser_backend::ParserExecutable::Native(parser),
            terminal_display_names, tokenizer, ignore_terminal,
            direct_regular_automaton: None, token_bytes: vocab.entries_arc(),
            ignore_expr, terminal_exprs,
            special_token_terminals: Vec::new(),
        };
        let mut inner = Self::constraint_from_runtime_parts(payload, dynamic_vocab, Some(vocab));
        let _ = inner.late_bind_vocab.set(vocab.clone());
        assert!(!inner.table.is_present(), "data-only constructor created an LR table");
        inner
    }

    pub(crate) fn from_alternatives(mut alternatives: Vec<Self>) -> Self {
        assert!(!alternatives.is_empty(), "dynamic union requires at least one alternative");
        let first = alternatives.remove(0);
        let mut external_vocab_artifact_cache = first.external_vocab_artifact_cache;
        let mut result = Self {
            inner: first.inner,
            alternatives: first.alternatives,
            composition_grammars: first.composition_grammars,
            external_vocab_artifact_cache: None,
        };
        for alternative in alternatives {
            // Combining alternatives changes the serialized payload; none of
            // the per-alternative cached artifacts represents the new union.
            external_vocab_artifact_cache = None;
            result.alternatives.push(alternative.inner);
            result.alternatives.extend(alternative.alternatives);
            result.composition_grammars.extend(alternative.composition_grammars);
        }
        result.external_vocab_artifact_cache = external_vocab_artifact_cache;
        result
    }

    pub(crate) fn from_constraints(mut constraints: Vec<Constraint>) -> Self {
        assert!(!constraints.is_empty(), "dynamic union requires at least one alternative");
        let inner = constraints.remove(0);
        let composition_grammars = vec![None; constraints.len() + 1];
        Self { inner, alternatives: constraints, composition_grammars, external_vocab_artifact_cache: None }
    }

    pub(crate) fn clone_constraints(&self) -> Vec<Constraint> {
        std::iter::once(&self.inner).chain(&self.alternatives).cloned().collect()
    }

    pub(crate) fn into_constraints(self) -> Vec<Constraint> {
        std::iter::once(self.inner)
            .chain(self.alternatives)
            .collect()
    }

    pub(crate) fn constraints_mut(&mut self) -> impl Iterator<Item = &mut Constraint> {
        self.external_vocab_artifact_cache = None;
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
        self.external_vocab_artifact_cache = None;
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
                    return Err(crate::Error::Compilation(format!(
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
        self.external_vocab_artifact_cache = None;
        assert_eq!(self.composition_grammars.len(), 1);
        self.composition_grammars[0] = Some(grammar);
    }

    pub(crate) fn reconstruct_composition_grammar(constraint: &Constraint) -> crate::Result<GrammarDef> {
        let exprs = constraint.tokenizer.terminal_exprs().ok_or_else(|| {
            crate::Error::Compilation(
                "this legacy dynamic artifact does not retain terminal expressions required for compiled-child composition; rebuild it".to_owned(),
            )
        })?;
        let num_terminals = constraint.tokenizer.num_terminals() as usize;
        if exprs.len() != num_terminals {
            return Err(crate::Error::Compilation(format!(
                "dynamic artifact terminal-expression count {} does not match tokenizer terminal count {num_terminals}",
                exprs.len(),
            )));
        }

        let special_by_terminal = constraint
            .special_token_terminals
            .iter()
            .map(|special| (special.terminal_id, (special.token_id, special.token_id_end)))
            .collect::<BTreeMap<_, _>>();
        let markers_by_terminal = constraint.marker_queries.as_deref().map(|queries|
            queries.terminals.iter().map(|marker| (marker.terminal_id, marker.marker_id)).collect::<BTreeMap<_, _>>()
        ).unwrap_or_default();
        let terminals = exprs
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, expr)| {
                let id = id as u32;
                if let Some(&marker_id) = markers_by_terminal.get(&id) {
                    return Terminal::MarkerProbe { id, marker_id };
                }
                special_by_terminal.get(&id).copied().map_or(
                    Terminal::Expr { id, expr },
                    |(start, end)| if start == end {
                        Terminal::SpecialToken { id, token_id: start }
                    } else { Terminal::SpecialTokenRange { id, start, end } },
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
                crate::Error::Compilation(
                    "dynamic artifact has no augmented-start rule for compiled-child composition"
                        .to_owned(),
                )
            })?;
            let start = match augmented.rhs.as_slice() {
                [Symbol::Nonterminal(start)] => *start,
                rhs => {
                    return Err(crate::Error::Compilation(format!(
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
        self.external_vocab_artifact_cache = None;
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
                return Err(crate::Error::Serialization(format!(
                    "terminal-observation certificate references terminal {terminal}, but tokenizer has {} terminals",
                    constraint.tokenizer.num_terminals(),
                )));
            }
            if !seen.insert(terminal) {
                return Err(crate::Error::Serialization(format!(
                    "terminal-observation certificate repeats terminal {terminal}",
                )));
            }
            if classes.len() != expected_states {
                return Err(crate::Error::Serialization(format!(
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

    fn restore_projected_terminal_quotients(
        constraint: &mut Constraint,
        rows: Vec<(TerminalID, TerminalProjectedQuotient)>,
    ) -> crate::Result<()> {
        let mut seen = BTreeSet::<TerminalID>::new();
        for (terminal, quotient) in &rows {
            if !seen.insert(*terminal) {
                return Err(crate::Error::Serialization(format!(
                    "projected-terminal quotient repeats terminal {terminal}"
                )));
            }
            quotient
                .validate_exact_projection(&constraint.tokenizer, *terminal)
                .map_err(crate::Error::Serialization)?;
        }
        // Calling the setter even for an empty row set is intentional: v20/v9
        // use empty+prepared to mean compile-time analysis completed and found
        // no useful quotient, so runtime load must not repeat that work.
        constraint
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(rows);
        Ok(())
    }

    fn build_external_vocab_artifact_bytes(&self) -> Vec<u8> {
        let count = self.alternatives.len().checked_add(1)
            .expect("dynamic alternative count");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_MAGIC);
        bytes.extend_from_slice(&DYNAMIC_TRANSFER_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&self.inner.artifact_vocab_digest());
        bytes.extend_from_slice(
            &u32::try_from(count).expect("dynamic alternative count fits u32").to_le_bytes()
        );
        for constraint in std::iter::once(&self.inner).chain(&self.alternatives) {
            // The closed-root artifact includes the exact root vocabulary policy.
            // Each alternative uses the same body codec, regardless of parser.
            let body = constraint.save();
            bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&body);
        }
        let payload_len = bytes.len() - DYNAMIC_CONSTRAINT_HEADER_LEN;
        bytes[10..18].copy_from_slice(&(payload_len as u64).to_le_bytes());
        bytes
    }

    /// Materialize the current external-vocabulary artifact while the
    /// constraint is still inside compilation/finalization. This keeps
    /// persistence honest: `save()` later copies bytes rather than running
    /// another serializer/finalizer whose cost would be hidden outside build.
    pub(crate) fn cache_external_vocab_artifact_for_save(&mut self) {
        if self.external_vocab_artifact_cache.is_some() {
            return;
        }
        let bytes = self.build_external_vocab_artifact_bytes();
        self.external_vocab_artifact_cache = Some(Arc::new(bytes));
    }

    /// Compact transfer artifact that deliberately omits vocabulary bytes.
    /// Pair with `load_with_vocab`. This is the natural persisted format for
    /// APIs (such as Python) whose load operation already requires a Vocab.
    pub fn save_without_vocab(&self) -> Vec<u8> {
        if let Some(bytes) = &self.external_vocab_artifact_cache {
            return bytes.as_ref().clone();
        }
        self.build_external_vocab_artifact_bytes()
    }

    pub(crate) fn into_saved(mut self) -> Vec<u8> {
        match self.external_vocab_artifact_cache.take() {
            Some(bytes) => Arc::try_unwrap(bytes)
                .unwrap_or_else(|bytes| bytes.as_ref().clone()),
            None => self.build_external_vocab_artifact_bytes(),
        }
    }

    fn load_alternatives(payload: &[u8], vocab: &Vocab) -> crate::Result<Self> {
        let error = |message: &str| crate::Error::Serialization(message.into());
        let header = payload.get(..DYNAMIC_PAYLOAD_HEADER_LEN)
            .ok_or_else(|| error("truncated dynamic vocabulary/count header"))?;
        if header[..32] != crate::compiler::compile::vocab_content_digest(vocab) {
            return Err(error("dynamic transfer artifact does not match supplied vocabulary"));
        }
        let count = u32::from_le_bytes(header[32..36].try_into().unwrap()) as usize;
        if count == 0 || count > payload.len().saturating_sub(header.len()) / 8 {
            return Err(error("invalid dynamic alternative count"));
        }
        let mut rest = &payload[header.len()..];
        let mut constraints = Vec::with_capacity(count);
        for _ in 0..count {
            let length = rest.get(..8)
                .ok_or_else(|| error("truncated dynamic alternative length"))?;
            let length = usize::try_from(u64::from_le_bytes(length.try_into().unwrap()))
                .map_err(|_| error("dynamic alternative length does not fit this platform"))?;
            let end = 8usize.checked_add(length)
                .ok_or_else(|| error("dynamic alternative length overflow"))?;
            let body = rest.get(8..end)
                .ok_or_else(|| error("truncated dynamic alternative"))?;
            let mut constraint = Constraint::load_external(body, vocab)?;
            if !constraint.has_template_parser() {
                constraint.materialize_composition_metadata_for_compilation()
                    .map_err(crate::Error::Serialization)?;
            }
            constraints.push(constraint);
            rest = &rest[end..];
        }
        if !rest.is_empty() {
            return Err(error("trailing dynamic transfer bytes"));
        }
        Ok(Self::from_constraints(constraints))
    }

    /// Serialize this dynamic constraint without embedding the vocabulary.
    ///
    /// Load with [`Self::load`] and the exact original vocabulary.
    pub fn save(&self) -> Vec<u8> {
        self.save_without_vocab()
    }

    pub(crate) fn dynamic_vocab_from_transfer_artifact(
        artifact: Option<crate::runtime::DynamicMaskVocabArtifact>,
        vocab: &Vocab,
    ) -> crate::Result<DynamicMaskVocab> {
        let Some(artifact) = artifact else {
            return Ok(
                crate::compiler::constraint_possible_matches::runtime_dynamic_vocab_for_vocab(vocab),
            );
        };
        let profile = crate::env::var_os("GLRMASK_PROFILE_DYNAMIC_LOAD").is_some();
        let from_started = profile.then(std::time::Instant::now);
        let full_vocab_template =
            crate::compiler::constraint_possible_matches::prepared_runtime_dynamic_vocab_for_vocab(
                vocab,
            );
        let quotient = DynamicMaskVocab::from_external_vocab_artifact(
            artifact,
            full_vocab_template.as_ref(),
        )
        .map_err(crate::Error::Serialization)?;
        let from_ms = from_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let validate_started = profile.then(std::time::Instant::now);
        if quotient.source_vocab_digest()
            != Some(crate::compiler::compile::vocab_content_digest(vocab))
        {
            return Err(crate::Error::Serialization(
                "dynamic transfer grammar-quotient vocabulary does not match supplied vocabulary"
                    .to_owned(),
            ));
        }
        let validate_ms = validate_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_transfer_quotient_assemble] from_artifact_ms={from_ms:.3} validate_ms={validate_ms:.3}",
            );
        }
        Ok(quotient)
    }

    /// Load a vocabulary-free transfer artifact with its exact vocabulary.
    pub(crate) fn load_with_vocab(bytes: &[u8], vocab: &Vocab) -> crate::Result<Self> {
        let mut loaded = Self::load_with_vocab_inner(bytes, vocab)?;
        if std::iter::once(&loaded.inner)
            .chain(&loaded.alternatives)
            .any(|constraint| !constraint.late_grammar_slots.is_empty())
        {
            return Err(crate::Error::Serialization(
                "dynamic constraint artifact has unresolved slots".to_owned(),
            ));
        }
        loaded.seed_late_bind_vocab(vocab);
        Ok(loaded)
    }

    fn seed_late_bind_vocab(&mut self, vocab: &Vocab) {
        for constraint in std::iter::once(&mut self.inner).chain(&mut self.alternatives) {
            if constraint.late_bind_vocab.get().is_none() {
                constraint.late_bind_vocab = std::sync::OnceLock::from(vocab.clone());
            }
        }
    }

    fn load_with_vocab_inner(bytes: &[u8], vocab: &Vocab) -> crate::Result<Self> {
        if bytes.len() < DYNAMIC_CONSTRAINT_HEADER_LEN || !bytes.starts_with(&DYNAMIC_TRANSFER_MAGIC) {
            return Err(crate::Error::Serialization(
                "invalid dynamic transfer artifact header".to_owned(),
            ));
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if version != DYNAMIC_TRANSFER_VERSION {
            return Err(crate::Error::Serialization(format!(
                "unsupported dynamic transfer artifact version {version}: recompile the pre-release artifact")));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[10..18]
                .try_into()
                .expect("dynamic transfer header has fixed width"),
        ))
        .map_err(|_| {
            crate::Error::Serialization(
                "dynamic transfer payload length does not fit this platform".to_owned(),
            )
        })?;
        if bytes.len() != DYNAMIC_CONSTRAINT_HEADER_LEN.saturating_add(payload_len) {
            return Err(crate::Error::Serialization(
                "invalid dynamic transfer artifact payload length".to_owned(),
            ));
        }
        let mut loaded = Self::load_alternatives(&bytes[DYNAMIC_CONSTRAINT_HEADER_LEN..], vocab)?;
        loaded.external_vocab_artifact_cache = Some(Arc::new(bytes.to_vec()));
        Ok(loaded)
    }

    /// Load an artifact produced by [`DynamicConstraint::save`].
    ///
    /// `vocab` must be the exact vocabulary the constraint was compiled
    /// against; the stored fingerprint covers the ID-to-bytes mapping.
    pub fn load(bytes: &[u8], vocab: &Vocab) -> crate::Result<Self> {
        Self::load_with_vocab(bytes, vocab)
    }

    /// One more than the largest token ID.
    /// Packed buffers need `mask_bit_len().div_ceil(32)` 32-bit words.
    pub fn mask_bit_len(&self) -> usize {
        std::iter::once(&self.inner)
            .chain(self.alternatives.iter())
            .map(Constraint::mask_bit_len)
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
    pub fn start(&self) -> DynamicConstraintState {
        DynamicConstraintState {
            alternatives: std::iter::once(&self.inner)
                .chain(self.alternatives.iter())
                .map(Constraint::start_dynamic)
                .collect(),
            mask_bit_len: self.mask_bit_len(),
        }
    }
}

/// Mutable per-sequence state for a [`DynamicConstraint`].
#[derive(Clone)]
pub struct DynamicConstraintState {
    alternatives: Vec<ConstraintState>,
    mask_bit_len: usize,
}

#[cfg(test)]
mod parser_replacement_compile_tests {
    use super::*;

    #[test]
    fn parser_replacement_omits_only_the_discarded_lr_snapshot() {
        let words = (0..40).map(|i| format!("k{i:02};")).collect::<Vec<_>>();
        let vocab = Vocab::from_entries_unchecked(words.iter().enumerate()
            .map(|(i, word)| (i as u32, word.as_bytes().to_vec())).collect());
        let mut source = String::from("start root;\n");
        for (i, word) in words.iter().enumerate() {
            source.push_str(&format!("t T{i} = \"{word}\";\n"));
        }
        source.push_str("nt root = ");
        source.push_str(&(0..words.len()).map(|i| format!("T{i}"))
            .collect::<Vec<_>>().join(" "));
        source.push(';');
        let named = crate::grammar::glrm::from_glrm(&source).unwrap();
        let grammar = crate::grammar::ast::lower(&named).unwrap();
        let grammar = crate::compiler::grammar::transforms::prepare_grammar_transforms_only(grammar);
        let construction = crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged;
        let mut ordinary = crate::compiler::pipeline::
            compile_dynamic_owned_with_vocab_partition_with_table_construction(
                grammar.clone(), &vocab, construction,
            ).unwrap();
        let mut replacement = crate::compiler::pipeline::
            compile_dynamic_owned_with_vocab_partition_for_parser_replacement(
                grammar, &vocab, construction,
            ).unwrap();

        assert!(ordinary.external_vocab_artifact_cache.is_some(),
            "the reference fixture must actually build a non-tiny transfer snapshot");
        assert!(replacement.external_vocab_artifact_cache.is_none());
        assert!(ordinary.inner.table.as_lr().is_some(),
            "the LR finalization must retain its table");
        assert!(replacement.inner.table.as_lr().is_none(),
            "the template core must be installed before quotient finalization returns");
        assert_eq!(ordinary.inner.dynamic_mask_vocab.canonical_token_count(),
                   replacement.inner.dynamic_mask_vocab.canonical_token_count());
        for dynamic in [&mut ordinary, &mut replacement] {
            for constraint in dynamic.constraints_mut() {
                constraint.install_template_parser().unwrap();
                assert!(constraint.table.as_lr().is_none());
            }
            assert!(dynamic.external_vocab_artifact_cache.is_none(),
                "parser replacement must invalidate the original LR snapshot");
        }
        let reference = ordinary.into_constraints().pop().unwrap();
        let candidate = replacement.into_constraints().pop().unwrap();
        let restored = Constraint::load(candidate.save(), &vocab).unwrap();
        let external = Constraint::load_body_external_artifact(
            candidate.save(), &vocab,
        ).unwrap();
        for constraint in [&candidate, &restored, &external] {
            assert!(constraint.table.as_lr().is_none());
            let mut left = reference.start();
            let mut right = constraint.start();
            let mut a = vec![0; reference.mask_bit_len().div_ceil(32)];
            let mut b = vec![0; constraint.mask_bit_len().div_ceil(32)];
            for token in 0..words.len() as u32 {
                left.fill_mask(&mut a);
                right.fill_mask(&mut b);
                assert_eq!(a, b, "mask before token {token}");
                assert_eq!(left.is_accepting(), right.is_accepting());
                assert_ne!(a[token as usize / 32] & (1 << (token % 32)), 0);
                assert!(left.commit_token(token));
                assert!(right.commit_token(token));
            }
            assert!(left.is_accepting());
            assert!(right.is_accepting());
            left.fill_mask(&mut a);
            right.fill_mask(&mut b);
            assert_eq!(a, b);
        }
    }
}

impl DynamicConstraintState {
    fn retain_committing(
        &mut self,
        mut commit: impl FnMut(&mut ConstraintState) -> Result<(), String>,
    ) -> Result<(), String> {
        self.alternatives.retain_mut(|state| commit(state).is_ok());
        if self.alternatives.is_empty() {
            Err("commit rejected: no valid parser states remain".to_owned())
        } else {
            Ok(())
        }
    }

    /// Returns `false`, and ends the state, if the bytes aren't allowed.
    #[must_use]
    pub fn commit_bytes(&mut self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return !self.is_terminated() || self.is_accepting();
        }
        self.commit_bytes_raw(bytes).is_ok()
    }

    fn commit_bytes_raw(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.retain_committing(|state| state.commit_bytes_raw(bytes))
    }

    /// Returns `false`, and ends the state, if the token isn't allowed.
    #[must_use]
    pub fn commit_token(&mut self, token_id: u32) -> bool {
        if self
            .alternatives
            .iter()
            .all(|state| !state.knows_token_id(token_id))
        {
            // An unknown ID is a failed commit: the state is dead.
            self.reject_after_failed_commit();
            return false;
        }
        self.commit_token_raw(token_id).is_ok()
    }

    pub(crate) fn reject_after_failed_commit(&mut self) {
        self.alternatives.clear();
    }

    /// Returns the number of tokens committed. If it is less than `token_ids.len()`, the token at that index was not allowed and the state is now terminated (as after `commit_token` returns false). To try tokens without changing this state, commit them on a clone.
    #[must_use]
    pub fn commit_tokens(&mut self, token_ids: &[u32]) -> usize {
        for (i, &token_id) in token_ids.iter().enumerate() {
            if !self.commit_token(token_id) {
                return i;
            }
        }
        token_ids.len()
    }

    fn commit_token_raw(&mut self, token_id: u32) -> Result<(), String> {
        self.retain_committing(|state| state.commit_token_raw(token_id))
    }

    /// Diagnostic commit profile for the overwhelmingly common single-
    /// alternative dynamic constraint. Multi-alternative dynamic constraints
    /// deliberately keep using the ordinary commit path rather than inventing
    /// an ambiguous aggregate profile.
    #[doc(hidden)]
    pub fn commit_token_profiled(&mut self, token_id: u32) -> Result<CommitProfile, String> {
        let [state] = self.alternatives.as_mut_slice() else {
            return Err(format!(
                "profiled dynamic commit requires exactly one alternative (have {})",
                self.alternatives.len()
            ));
        };
        state.commit_token_profiled(token_id)
    }

    /// Bounded diagnostic mirror of the common state profiler. No timing
    /// benchmark should use this stack-enumerating inspection API.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn commit_token_per_advance(&mut self, token_id: u32) -> Result<
        (Vec<crate::runtime::PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile),
        String,
    > {
        let [state] = self.alternatives.as_mut_slice() else {
            return Err("per-advance profiling requires one dynamic alternative".into());
        };
        state.commit_token_per_advance(token_id)
    }

    /// Fill `buf` with the allowed-token mask as a packed bitset.
    pub fn fill_mask(&self, buf: &mut [u32]) {
        assert!(buf.len() >= self.mask_bit_len.div_ceil(32), "mask buffer is smaller than constraint mask");
        let Some((first, rest)) = self.alternatives.split_first() else {
            buf.fill(0);
            return;
        };
        let profile = dynamic_mask_profile_enabled(first.generation);
        let total_started = profile.then(std::time::Instant::now);
        let first_started = profile.then(std::time::Instant::now);
        first.fill_mask(buf);
        if profile {
            eprintln!(
                "[glrmask/profile][dynamic_mask_union_alt] alt=0 alternatives={} ms={:.3}",
                self.alternatives.len(),
                first_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        if rest.is_empty() {
            if let Some(started) = total_started {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_union] alternatives=1 total_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return;
        }
        let mut scratch = vec![0u32; buf.len()];
        for (index, state) in rest.iter().enumerate() {
            let started = profile.then(std::time::Instant::now);
            state.fill_mask(&mut scratch);
            for (target, source) in buf.iter_mut().zip(&scratch) {
                *target |= *source;
            }
            scratch.fill(0);
            if profile {
                eprintln!(
                    "[glrmask/profile][dynamic_mask_union_alt] alt={} alternatives={} ms={:.3}",
                    index + 1,
                    self.alternatives.len(),
                    started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                );
            }
        }
        if let Some(started) = total_started {
            eprintln!(
                "[glrmask/profile][dynamic_mask_union] alternatives={} total_ms={:.3}",
                self.alternatives.len(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }


    /// The bytes the grammar requires next. Stops where a token from a token set, the end of the grammar, or more than one byte could come next. May return only part of a long run; call again after committing.
    pub fn forced_bytes(&self) -> Vec<u8> {
        crate::runtime::forced_bytes_for_states(&self.alternatives)
    }
    pub(crate) fn mask_if_at_most(&self,max_allowed:usize)->crate::Result<Option<Vec<u32>>> {
        crate::runtime::mask_for_states_at_most(&self.alternatives,max_allowed)
    }
    #[cfg(any(test,feature="internal-api"))]
    #[doc(hidden)]
    pub fn forced_canonical_reference(&self)->Vec<u32> {
        crate::runtime::force_for_states(&self.alternatives,Some(true))
    }
    #[cfg(any(test,feature="internal-api"))]
    #[doc(hidden)]
    pub fn forced_token_ids_reference(&self)->Vec<u32> {
        crate::runtime::force_ids_for_states(&self.alternatives,Some(true))
    }

    /// Whether what's committed is a complete match. If so, allow EOS (don't commit it).
    pub fn is_accepting(&self) -> bool {
        self.alternatives.iter().any(ConstraintState::is_accepting)
    }

    /// Whether nothing more can be committed.
    #[must_use]
    pub fn is_terminated(&self) -> bool {
        self.alternatives.iter().all(ConstraintState::is_terminated)
    }

    pub(crate) fn frontier_complete(&self) -> bool {
        self.alternatives.iter().any(ConstraintState::frontier_complete)
    }

    /// One more than the largest token ID.
    /// Packed buffers need `mask_bit_len().div_ceil(32)` 32-bit words.
    pub fn mask_bit_len(&self) -> usize { self.mask_bit_len }

    /// Return the allowed-token mask as a packed `u32` bitmask.
    pub fn mask(&self) -> Vec<u32> {
        let mut mask = vec![0u32; self.mask_bit_len.div_ceil(32)];
        self.fill_mask(&mut mask);
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins `TEST_COMPAT_MODE` to JsonSchema for full-JSON tests, restoring
    /// the prior mode on drop. Mirrors `EnvVarGuard`'s compat handling in the
    /// json_schema tests; the mode cell is the live control (env only seeds
    /// fresh threads), so no env change is needed.
    struct CompatModeGuard {
        original: crate::import::json_schema::string::JsonStringCompatMode,
    }

    impl CompatModeGuard {
        fn json_schema() -> Self {
            Self::set_mode(crate::import::json_schema::string::JsonStringCompatMode::JsonSchema)
        }

        fn native() -> Self {
            Self::set_mode(
                crate::import::json_schema::string::JsonStringCompatMode::LlGuidanceNative,
            )
        }

        fn set_mode(mode: crate::import::json_schema::string::JsonStringCompatMode) -> Self {
            use crate::import::json_schema::string::TEST_COMPAT_MODE;
            let original = TEST_COMPAT_MODE.with(|cell| cell.get());
            TEST_COMPAT_MODE.with(|cell| cell.set(mode));
            Self { original }
        }
    }

    impl Drop for CompatModeGuard {
        fn drop(&mut self) {
            use crate::import::json_schema::string::TEST_COMPAT_MODE;
            TEST_COMPAT_MODE.with(|cell| cell.set(self.original));
        }
    }

    mod projected_quotient_fixture {
        use crate as glrmask;

        include!("../../tests/fixtures/snowplow_hostname.rsinc");

        pub(super) fn schema_and_vocab() -> (&'static str, Vocab) {
            (SNOWPLOW_SCHEMA, snowplow_vocab())
        }

        pub(super) fn replay_ids() -> &'static [u32] {
            SNOWPLOW_REPLAY_IDS
        }
    }

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

    fn compile_compressed_static(source: &str, vocab: &Vocab) -> crate::runtime::Constraint {
        crate::compiler::pipeline::compile_owned_with_table_construction(
            compressed_lark_grammar(source),
            vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        )
    }

    fn compile_compressed_dynamic(source: &str, vocab: &Vocab) -> DynamicConstraint {
        crate::compiler::pipeline::compile_dynamic_owned_with_backend(
            compressed_lark_grammar(source),
            vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
            crate::runtime::parser_backend::DynamicParserBackend::Native,
        )
        .unwrap()
    }

    fn vocab() -> Vocab {
        Vocab::from_entries_unchecked(vec![
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
            t A = 'a'+;
            t B = 'b';
            nt start = A B;
        "#;
        let normal = crate::runtime::Constraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let dynamic = DynamicConstraint::from_glrm_grammar(grammar, &vocab).unwrap();
        let mut normal_state = normal.start();
        let mut dynamic_state = dynamic.start();

        assert_eq!(normal_state.mask(), dynamic_state.mask());
        assert!(normal_state.commit_token(3));
        assert!(dynamic_state.commit_token(3));
        assert_eq!(normal_state.mask(), dynamic_state.mask());
        assert!(normal_state.commit_token(1));
        assert!(dynamic_state.commit_token(1));
        assert_eq!(normal_state.is_accepting(), dynamic_state.is_accepting());
        assert_eq!(normal_state.mask(), dynamic_state.mask());
    }

    #[test]
    fn dynamic_json_schema_ordinary_bounded_array_item_stays_a_lazy_terminal() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"[".to_vec()),
            (1, b"]".to_vec()),
            (2, b",".to_vec()),
            (3, b"\"".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b"aa".to_vec()),
            (7, b"\"a\"".to_vec()),
            (8, b"\"b\"".to_vec()),
            (9, b" ".to_vec()),
        ]);
        let schema = r#"{
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1024
            }
        }"#;
        let dynamic = DynamicConstraint::from_json_schema(schema, &vocab).unwrap();
        assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(
            dynamic
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count(),
            1,
        );
        assert!(
            dynamic.inner.tokenizer.num_states() < 1_000,
            "the 1024-character item bound must not be materialized into the physical tokenizer",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes) && state.is_accepting()
        };
        assert!(accepts(br#"["a", "bb"]"#));
        assert!(!accepts(br#"[""]"#));

        let mut at_limit = Vec::with_capacity(1028);
        at_limit.extend_from_slice(b"[\"");
        at_limit.extend(std::iter::repeat_n(b'a', 1024));
        at_limit.extend_from_slice(b"\"]");
        assert!(accepts(&at_limit));

        let mut too_long = Vec::with_capacity(1029);
        too_long.extend_from_slice(b"[\"");
        too_long.extend(std::iter::repeat_n(b'a', 1025));
        too_long.extend_from_slice(b"\"]");
        assert!(!accepts(&too_long));

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut dynamic_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for (step, token) in [0u32, 7, 2, 9, 8, 1]
        .into_iter()
        .enumerate()
        {
            assert_eq!(dynamic_state.is_accepting(), loaded_state.is_accepting());
            let dynamic_mask = dynamic_state.mask();
            let loaded_mask = loaded_state.mask();
            assert_eq!(dynamic_mask, loaded_mask, "save/load mask mismatch at step {step}");
            assert!(
                token_allowed(&dynamic_mask, token),
                "fixture token {token} must be admitted at step {step}",
            );
            assert!(dynamic_state
                .commit_token(token));
            assert!(loaded_state
                .commit_token(token));
        }
        assert_eq!(dynamic_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(dynamic_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_json_schema_ordinary_bounded_string_bound_is_symbolic() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
        ]);
        let mut physical_state_counts = Vec::new();
        for max_length in [255usize, 1024, 1_000_000_000_000] {
            let schema = format!(r#"{{"type":"string","maxLength":{max_length}}}"#);
            let dynamic = DynamicConstraint::from_json_schema(&schema, &vocab).unwrap();
            assert!(dynamic.inner.tokenizer.has_virtual_residual_runtime());
            assert_eq!(
                dynamic
                    .inner
                    .tokenizer
                    .virtual_residual_bounded_code_liveness_oracle_count(),
                1,
            );
            physical_state_counts.push(dynamic.inner.tokenizer.num_states());
        }
        assert!(
            physical_state_counts.windows(2).all(|pair| pair[0] == pair[1]),
            "physical tokenizer size must not depend on the numeric bounded-string maximum: {physical_state_counts:?}",
        );
    }

    #[test]
    fn dynamic_json_schema_bounded_pattern_rejects_escaped_spellings_in_native_mode() {
        // Native-default counterpart to the JsonSchema-mode escaped-spelling
        // tests: the LlGuidanceNative lexical language deliberately does not
        // materialize `\uXXXX` spellings (production default for automata
        // size); raw spellings keep working. Same schema as the bounded
        // liveness-oracle test; representative single pin, not a clone set.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::native();
        let vocab = Vocab::from_entries_unchecked(vec![
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

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes) && state.is_accepting()
        };
        assert!(accepts(br#""aa""#));
        assert!(!accepts(br#""\u0061\u0061""#));
        assert!(!accepts(br#""a\u0061""#));
    }

    #[test]
    fn dynamic_json_schema_bounded_pattern_uses_exact_code_liveness_oracle() {
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::from_entries_unchecked(vec![
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
            state.commit_bytes(bytes) && state.is_accepting()
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

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            "annotation-only formats must retain the certified JSON decoded-length liveness oracle",
        );

        let accepts = |bytes: &[u8]| {
            let mut state = dynamic.start();
            state.commit_bytes(bytes) && state.is_accepting()
        };
        assert!(!accepts(br#""aa""#));
        assert!(accepts(br#""aaa""#));
        assert!(accepts(br#""a.b""#));
        assert!(accepts(br#""a..b""#), "format is an annotation; length remains asserted");

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
            "save/load must reconstruct the bounded length liveness oracle",
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_json_schema_pattern_format_and_length_share_exact_code_liveness_oracle() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
            state.commit_bytes(bytes) && state.is_accepting()
        };
        assert!(!accepts(br#""a""#));
        assert!(accepts(br#""aa""#));
        assert!(accepts(br#""bb""#));
        assert!(!accepts(br#""a.b""#));

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(
            loaded
                .inner
                .tokenizer
                .virtual_residual_bounded_code_liveness_oracle_count()
                > 0,
        );
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let metadata = dynamic.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(),1);
        let exprs = dynamic.inner.retained_terminal_exprs().unwrap().to_vec();
        let mut tokenizer = dynamic.inner.tokenizer.as_ref().clone();
        let error = tokenizer.restore_terminal_exprs_with_virtual_runtime_metadata(Some(exprs.clone()), &[], false).unwrap_err();
        assert!(error.contains("terminal ownership mismatch"), "{error}");
        let mut forged_exprs = exprs;
        forged_exprs[metadata[0].terminal as usize] = Expr::U8Seq(b"a".to_vec());
        let mut tokenizer = dynamic.inner.tokenizer.as_ref().clone();
        let error = tokenizer.restore_terminal_exprs_with_virtual_runtime_metadata(Some(forged_exprs), &metadata, false).unwrap_err();
        assert!(error.contains("certified bounded-code residual"), "{error}");
    }

    #[test]
    fn dynamic_json_schema_cross_branch_bounded_string_constraints_share_exact_code_liveness_oracle() {
        let vocab = Vocab::from_entries_unchecked(vec![
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

            let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
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
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::from_entries_unchecked(vec![
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
            state.commit_bytes(bytes) && state.is_accepting()
        };
        assert!(!accepts(br#""aa""#));
        assert!(accepts(br#""aaa""#));
        assert!(accepts(br#""\u0061\u0061\u0061""#));
        assert!(!accepts(br#""bbb""#));
        assert!(!accepts(br#""ccc""#));

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
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
        // Pin JsonSchema mode: full-JSON expectations predate the native default (assertions unchanged).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _compat = CompatModeGuard::json_schema();
        let vocab = Vocab::from_entries_unchecked(vec![
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
            state.commit_bytes(bytes) && state.is_accepting()
        };
        assert!(!accepts("\"é\"".as_bytes()));
        assert!(accepts("\"éé\"".as_bytes()));
        assert!(accepts(br#""\u00e9\u00E9""#));
        assert!(accepts("\"é\\u00e9\"".as_bytes()));
        assert!(!accepts("\"éx\"".as_bytes()));
    }

    #[test]
    fn static_constraint_artifact_preserves_dynamic_runtime_sidecars() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"x".to_vec()),
        ]);
        let dynamic = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A = /a{0,10000}/;
                nt start = A;
            "#,
            &vocab,
        )
        .unwrap();
        let original = dynamic.clone().into_constraint();
        assert!(!original.tokenizer.virtual_runtime_metadata().is_empty());
        let loaded = Constraint::load(original.save(), &vocab).unwrap();
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
                t WS = " "+;
                nt start = "a"+ "b";
            "#,
            &vocab,
        )
        .unwrap();
        let loaded = DynamicConstraint::load(&constraint.save(), &vocab).unwrap();
        assert!(constraint.inner.ignore_expr.is_some());
        assert_eq!(loaded.inner.ignore_expr, constraint.inner.ignore_expr);
        assert_eq!(
            loaded.inner.retained_terminal_exprs(),
            constraint.inner.retained_terminal_exprs(),
        );
        assert_eq!(constraint.mask_bit_len().div_ceil(32), loaded.mask_bit_len().div_ceil(32));
        assert_eq!(constraint.start().mask(), loaded.start().mask());
    }

    #[test]
    fn dynamic_v20_persists_vocab_and_validates_shared_vocab() {
        let vocab = Vocab::from_entries_unchecked(vec![(0,b"a".to_vec()),(1,b"ab".to_vec()),(2,b"b".to_vec())]);
        let constraint = DynamicConstraint::from_ebnf_with_bounded_template_parser("start ::= 'a'+", &vocab).unwrap();
        let current = constraint.save();
        assert_eq!(u16::from_le_bytes([current[8],current[9]]), DYNAMIC_TRANSFER_VERSION);
        let loaded = DynamicConstraint::load(&current, &vocab).unwrap();
        assert!(loaded.inner.dynamic_mask_vocab.to_artifact().is_some());
        assert_eq!(loaded.start().mask(), constraint.start().mask());
        let other_vocab = Vocab::from_entries_unchecked(vec![(0,b"z".to_vec())]);
        let other = DynamicConstraint::from_ebnf_with_bounded_template_parser("start ::= 'z'+", &other_vocab).unwrap();
        let malformed = DynamicConstraint::from_constraints(vec![constraint.into_constraint(), other.into_constraint()]).save();
        let error = DynamicConstraint::load(&malformed, &vocab).unwrap_err();
        assert!(error.to_string().contains("vocabulary"), "{error}");
    }

    #[test]
    fn dynamic_virtual_unit_repeat_save_load_round_trip() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
            t A = /a{0,1000000000}/;
            t B = 'b';
            nt start = B A;
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
        let loaded = DynamicConstraint::load(&bytes, &vocab).unwrap();
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
        assert!(original_state.commit_token(4));
        assert!(loaded_state.commit_token(4));
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_virtual_runtime_metadata_is_exact_and_fail_closed() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_with_bounded_template_parser(
            r#"
                start start;
                t A = /a{0,10000}/;
                t B = /b{0,9000}/;
                nt start = A | B;
            "#,
            &vocab,
        )
        .unwrap();
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 2);

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
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

        let validate = |metadata: &[VirtualTokenizerRuntimeMetadata], exprs: Option<Vec<Expr>>| {
            let mut tokenizer = constraint.inner.tokenizer.as_ref().clone();
            tokenizer.restore_terminal_exprs_with_virtual_runtime_metadata(exprs, metadata, false)
        };
        let exprs = constraint.inner.retained_terminal_exprs().unwrap().to_vec();
        let mut missing = metadata.clone();
        missing.pop();
        let error = validate(&missing, Some(exprs.clone())).unwrap_err();
        assert!(error.contains("terminal ownership mismatch"), "{error}");
        let mut duplicate_root = metadata.clone();
        duplicate_root[1].root_state = duplicate_root[0].root_state;
        let error = validate(&duplicate_root, Some(exprs.clone())).unwrap_err();
        assert!(error.contains("invalid terminal/root ownership"), "{error}");
        let mut mismatched_exprs = exprs;
        let terminal = metadata[0].terminal as usize;
        mismatched_exprs[terminal] = Expr::Repeat { expr: Box::new(Expr::U8Seq(b"z".to_vec())), min:0, max:Some(10000) };
        let error = validate(&metadata, Some(mismatched_exprs)).unwrap_err();
        assert!(error.contains("byte support"), "{error}");
    }

    #[test]
    fn current_dynamic_transfer_virtual_residuals_round_trip() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"b".to_vec()),
            (4, b"bb".to_vec()),
        ]);
        let schema = r#"{
            "type": "string",
            "pattern": "^(?:a|bb)+$",
            "minLength": 2,
            "maxLength": 5000
        }"#;
        let constraint = DynamicConstraint::from_json_schema_with_bounded_template_parser(schema, &vocab).unwrap();
        assert!(constraint.inner.tokenizer.has_any_virtual_runtime());

        // Test current V13 save and load.
        let v13_bytes = constraint.save_without_vocab();
        assert_eq!(u16::from_le_bytes([v13_bytes[8], v13_bytes[9]]), DYNAMIC_TRANSFER_VERSION);
        assert_eq!(DYNAMIC_TRANSFER_VERSION, 18);

        let v13_loaded = DynamicConstraint::load_with_vocab(&v13_bytes, &vocab).unwrap();
        // Loaded constraint should carry the mask tokenizer projection directly from the wire
        assert!(v13_loaded.inner.dynamic_mask_vocab.mask_projection_tokenizer().is_some());
        assert_eq!(v13_loaded.start().mask(), constraint.start().mask());

    }

    #[test]
    fn current_dynamic_transfer_defers_terminal_expression_trees() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A = /a+/;
                t B = /b+/;
                nt start = A B;
            "#,
            &vocab,
        )
        .unwrap();
        let expected = constraint
            .inner
            .tokenizer
            .terminal_exprs()
            .expect("fresh constraint retains terminal expressions")
            .to_vec();
        assert!(constraint.inner.tokenizer.virtual_runtime_metadata().is_empty());

        let transfer = constraint.into_saved();
        let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            loaded.inner.tokenizer.terminal_exprs().is_none(),
            "ordinary transfer load must not eagerly materialize source expression trees",
        );
        assert!(matches!(
            loaded.inner.deferred_terminal_exprs_blob,
            Some(crate::runtime::DeferredTerminalExprBytes::Backed { .. }) | Some(crate::runtime::DeferredTerminalExprBytes::CompressedBacked { .. })
        ));
        assert_eq!(
            loaded
                .inner
                .retained_terminal_exprs()
                .expect("composition must recover deferred source expressions"),
            expected,
        );
        assert_eq!(
            loaded
                .inner
                .retained_terminal_exprs()
                .expect("decoded expressions should stay cached"),
            expected,
        );
    }

    #[test]
    fn dynamic_v20_terminal_observation_certificate_is_shape_validated() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
                start start;
                t A = /a+/;
                t B = /b+/;
                nt start = A | B;
            "#,
            &vocab,
        )
        .unwrap();
        let states = constraint.inner.tokenizer.num_states() as usize;
        let terminals = constraint.inner.tokenizer.num_terminals();

        let decode = |rows: Vec<(TerminalID, Vec<u32>)>| {
            let mut proofs = DynamicExecutionMetadata::from_constraint(&constraint.inner);
            proofs.terminal_observation_classes = rows;
            let bytes = bincode::serialize(&proofs).unwrap();
            let restored: DynamicExecutionMetadata = bincode::deserialize(&bytes).unwrap();
            restored.restore(&mut constraint.inner.clone())
        };

        let bad_terminal = decode(vec![(
            terminals,
            vec![1; states],
        )])
        .unwrap_err();
        assert!(bad_terminal
            .to_string()
            .contains("terminal-observation certificate references terminal"));

        let duplicate = decode(vec![
            (0, vec![1; states]),
            (0, vec![1; states]),
        ])
        .unwrap_err();
        assert!(duplicate
            .to_string()
            .contains("terminal-observation certificate repeats terminal"));

        let bad_len = decode(vec![(
            0,
            vec![1; states.saturating_sub(1)],
        )])
        .unwrap_err();
        assert!(bad_len
            .to_string()
            .contains("terminal-observation certificate for terminal"));
    }

    #[test]
    fn dynamic_standalone_virtual_unit_v20_and_transfer_v9_round_trip() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"x".to_vec()),
        ]);
        let constraint = DynamicConstraint::from_glrm_with_bounded_template_parser(
            r#"
                start start;
                t A = /a{0,10000}/;
                nt start = A;
            "#,
            &vocab,
        )
        .unwrap();
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].kind, crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeKind::UnitRepeat);
        assert_eq!(metadata[0].root_state, 0);

        let saved = constraint.save();
        assert_eq!(u16::from_le_bytes([saved[8], saved[9]]), DYNAMIC_TRANSFER_VERSION);
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
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
            assert!(original.commit_token(token));
            assert!(loaded_state.commit_token(token));
            assert!(transferred_state.commit_token(token));
            assert_eq!(loaded_state.is_accepting(), original.is_accepting());
            assert_eq!(transferred_state.is_accepting(), original.is_accepting());
            assert_eq!(loaded_state.mask(), original.mask());
            assert_eq!(transferred_state.mask(), original.mask());
        }

    }

    #[test]
    fn dynamic_v20_combines_composition_and_residual_runtime_metadata() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let mut constraint = crate::import::compile_dynamic_glrm_fixture(
            r#"
                start start;
                t A = /a{0,10000}/;
                nt start = A;
            "#,
            &vocab,
            crate::runtime::parser_backend::DynamicParserBackend::Native,
        )
        .unwrap();
        constraint.inner.late_grammar_slots = vec![crate::runtime::LateGrammarSlot {
            name: "child".to_owned(),
            terminal_id: 0,
        }];
        constraint.inner.boundary_trigger = crate::runtime::BoundaryTrigger::TokenSet(
            Arc::from(vec![1u32].into_boxed_slice()),
        );
        let state_count = constraint.inner.tokenizer.num_states() as u32;
        constraint.inner.dynamic_mask_vocab.set_terminal_observation_classes(vec![(
            0,
            Arc::from((1..=state_count).collect::<Vec<_>>().into_boxed_slice()),
        )]);
        assert!(constraint.inner.dynamic_mask_vocab.to_vocab_artifact().is_some());
        let metadata = constraint.inner.tokenizer.virtual_runtime_metadata();
        assert!(!metadata.is_empty());

        // This test mutates private persistence metadata after public
        // compilation has already frozen the canonical transfer artifact.
        // Refresh that cache explicitly so the transfer half of the test
        // continues to cover the wire representation produced from the
        // mutated fixture. Production constraints are immutable after build.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        assert_eq!(u32::from_le_bytes(saved[50..54].try_into().unwrap()), 1);
        let mut loaded = DynamicConstraint::from_constraints(vec![Constraint::load_body_external_artifact(&saved[62..], &vocab).unwrap()]);
        loaded.inner.materialize_composition_metadata_for_compilation().unwrap();
        assert_eq!(loaded.inner.late_grammar_slots, constraint.inner.late_grammar_slots);
        assert_eq!(loaded.inner.tokenizer.virtual_runtime_metadata(), metadata);
        assert!(loaded.inner.dynamic_mask_vocab.has_terminal_observation_classes());
        assert!(loaded.inner.dynamic_mask_vocab.to_vocab_artifact().is_some());
        assert_eq!(
            loaded.inner.boundary_trigger.token_summary().map(|tokens| tokens.to_vec()),
            Some(vec![1]),
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        assert!(DynamicConstraint::load_with_vocab(&transfer, &vocab).is_err());
        let mut transferred = DynamicConstraint::from_constraints(vec![Constraint::load_body_external_artifact(&transfer[62..], &vocab).unwrap()]);
        transferred.inner.materialize_composition_metadata_for_compilation().unwrap();
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
        assert_eq!(transferred.start().mask(), constraint.start().mask());

    }

    fn variable_width_repeat_intersection_grammar(
        left_max: usize,
        right_max: usize,
    ) -> crate::grammar::flat::GrammarDef {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        Vocab::from_entries_unchecked(vec![
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
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .mask_projection_tokenizer()
                .is_some(),
            "finite mask projection is an immutable runtime accelerator and must be prepared during build finalization",
        );

        let mask_tokenizer = constraint
            .inner
            .dynamic_mask_vocab
            .mask_projection_tokenizer()
            .expect("build finalization must prepare the finite exact product");
        let start_mask = constraint.start().mask();
        assert!(
            constraint.inner.lazy_dynamic_mask_vocab.get().is_none(),
            "first mask must not hide projection construction in a lazy runtime cache",
        );
        assert!(
            mask_tokenizer.num_states() < 2_000,
            "mask tokenizer must scale with vocab horizon/body DFAs, not N*M",
        );

        assert_eq!(
            start_mask,
            oracle.start().mask(),
            "far from either upper bound, the billion-scale lazy lexer must have the same finite-vocabulary observations as a materialized 32x32 oracle",
        );
        assert!(!token_allowed(&start_mask, 7), "x cannot begin either repeat language");

        let mut state = constraint.start();
        let mut oracle_state = oracle.start();
        assert!(state.commit_token(2)); // "abb" = "a" + "bb"
        assert!(oracle_state.commit_token(2));
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
        assert!(dynamic_state.commit_token(2));
        assert!(ordinary_state.commit_token(2));
        assert_eq!(dynamic_state.mask(), ordinary_state.mask());
        assert!(dynamic_state.commit_token(1));
        assert!(ordinary_state.commit_token(1));
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
        let loaded = DynamicConstraint::load(&bytes, &vocab).unwrap();
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
        assert!(original_state.commit_token(2));
        assert!(loaded_state.commit_token(2));
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_virtual_repeat_intersection_rejects_dead_common_prefix() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            assert!(dynamic_state.commit_token(token));
            assert!(oracle_state.commit_token(token));
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());

        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        assert!(original_state.commit_token(0));
        assert!(loaded_state.commit_token(0));
        assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
        assert_eq!(loaded_state.mask(), original_state.mask());
    }

    #[test]
    fn dynamic_billion_nonzero_min_variable_width_repeat_is_lazy_end_to_end() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            assert!(dynamic_state.commit_token(token));
            assert!(oracle_state.commit_token(token));
            assert!(!dynamic_state.is_accepting());
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
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
            assert!(original_state.commit_token(token));
            assert!(loaded_state.commit_token(token));
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_same_body_nonzero_repeat_intersection_factors_end_to_end() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            assert!(dynamic_state.commit_token(token));
            assert!(oracle_state.commit_token(token));
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        for token in [1u32, 2, 3] {
            assert!(original_state.commit_token(token));
            assert!(loaded_state.commit_token(token));
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_billion_nonzero_min_unit_repeat_uses_arithmetic_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            assert!(dynamic_state.commit_token(token));
            assert!(oracle_state.commit_token(token));
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
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
            assert!(original_state.commit_token(token));
            assert!(loaded_state.commit_token(token));
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
            assert_eq!(loaded_state.mask(), original_state.mask());
        }
    }

    #[test]
    fn dynamic_aligned_nonzero_unit_intersection_uses_arithmetic_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            assert!(dynamic_state.commit_token(token));
            assert!(oracle_state.commit_token(token));
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
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
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);

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
            let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
            assert_eq!(loaded.start().mask(), dynamic.start().mask());
        }
    }

    #[test]
    fn dynamic_distinct_nonzero_min_giant_intersection_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_multiple_large_plain_repeats_use_exact_virtual_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_non_prefix_free_large_repeat_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_general_residual_prunes_semantically_dead_token_prefix() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
        assert!(!rejected.commit_token(0));
        let mut accepted = dynamic.start();
        assert!(accepted.commit_token(1));
        assert!(accepted.is_accepting());

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), start_mask);
    }

    #[test]
    fn dynamic_nested_large_repeat_suffix_compiles_lazily_and_exactly() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_nested_large_repeat_in_lazy_intersection_remains_supported() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            assert!(dynamic_state.commit_token(token));
            assert!(oracle_state.commit_token(token));
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }
    }

    #[test]
    fn dynamic_general_residual_repeat_supports_u32_max_bound() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_hybrid_unit_repeat_near_state_id_limit_uses_repeat_product() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_top_level_large_repeat_inside_finite_intersection_stays_exact() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        let mut loaded_state = loaded.start();
        assert!(loaded_state.commit_token(0));
        assert!(loaded_state.is_accepting());
    }

    #[test]
    fn dynamic_top_level_large_repeat_inside_cyclic_intersection_uses_residual_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }
    }

    #[test]
    fn dynamic_nested_giant_with_budgeted_other_repeat_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let mut dynamic_boundary = dynamic.start();
        let mut oracle_boundary = oracle.start();
        for _ in 0..49 {
            assert!(dynamic_boundary.commit_token(1));
            assert!(oracle_boundary.commit_token(1));
        }
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());
        assert!(!dynamic_boundary.is_accepting());

        // At 98 leading 'a' bytes, one more 'a' is still live but another
        // two-byte "aa" token would overshoot the right-hand 100-byte cap once
        // the required trailing 'b' is included.
        let mut dynamic_overrun = dynamic_boundary.clone();
        let mut oracle_overrun = oracle_boundary.clone();
        assert!(!dynamic_overrun.commit_token(1));
        assert!(!oracle_overrun.commit_token(1));

        assert!(dynamic_boundary.commit_token(0));
        assert!(oracle_boundary.commit_token(0));
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());
        assert!(!dynamic_boundary.is_accepting());
        assert!(dynamic_boundary.commit_token(2));
        assert!(oracle_boundary.commit_token(2));
        assert!(dynamic_boundary.is_accepting());
        assert_eq!(dynamic_boundary.is_accepting(), oracle_boundary.is_accepting());
        assert_eq!(dynamic_boundary.mask(), oracle_boundary.mask());

        let saved = dynamic.save();
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
        assert!(
            loaded.inner.tokenizer.has_virtual_residual_runtime(),
            "save/load must reconstruct the general residual runtime",
        );
        let mut loaded_state = loaded.start();
        let mut dynamic_state = dynamic.start();
        let mut oracle_state = oracle.start();
        assert!(loaded_state.commit_token(3));
        assert!(dynamic_state.commit_token(3));
        assert!(oracle_state.commit_token(3));
        assert!(loaded_state.is_accepting());
        assert_eq!(loaded_state.is_accepting(), dynamic_state.is_accepting());
        assert_eq!(loaded_state.is_accepting(), oracle_state.is_accepting());
        assert_eq!(loaded_state.mask(), dynamic_state.mask());
        assert_eq!(loaded_state.mask(), oracle_state.mask());
    }

    #[test]
    fn dynamic_nested_giant_with_large_finite_other_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
        assert!(dynamic_state.commit_token(2));
        assert!(oracle_state.commit_token(2));
        assert!(dynamic_state.is_accepting());
        assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
        assert_eq!(dynamic_state.mask(), oracle_state.mask());

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_empty_giant_cyclic_intersection_becomes_dead_residual_proxy() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
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

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_two_nested_large_repeat_components_with_disjoint_delimiters_factor_to_empty() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert_eq!(loaded.start().mask(), start_mask);
    }

    #[test]
    fn dynamic_positive_min_giant_suffix_with_disjoint_counts_factors_to_empty() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"b".to_vec())]);
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
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            assert!(dynamic_state.commit_token(token));
            assert!(oracle_state.commit_token(token));
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_binary_repeat_intersection());
        assert_eq!(dynamic.start().mask(), loaded.start().mask());
        for token in [0u32, 1] {
            let mut original_state = dynamic.start();
            let mut loaded_state = loaded.start();
            assert!(original_state.commit_token(token));
            assert!(loaded_state.commit_token(token));
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
            assert!(original_state.commit_token(token));
            assert!(transferred_state.commit_token(token));
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
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn dynamic_multiple_variable_width_giant_terminals_match_small_oracle() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
            assert!(dynamic_state.commit_token(token));
            assert!(oracle_state.commit_token(token));
            assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
            assert_eq!(dynamic_state.mask(), oracle_state.mask());
        }

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert_eq!(dynamic.start().mask(), loaded.start().mask());
        let mut original_state = dynamic.start();
        let mut loaded_state = loaded.start();
        assert!(original_state.commit_token(3));
        assert!(loaded_state.commit_token(3));
        assert_eq!(original_state.is_accepting(), loaded_state.is_accepting());
        assert_eq!(original_state.mask(), loaded_state.mask());
    }

    #[test]
    fn dynamic_nested_giant_repeat_body_uses_general_residual_runtime() {
        use crate::automata::regex::Expr;
#[cfg(test)]
use crate::automata::lexer::tokenizer::VirtualTokenizerRuntimeMetadata;
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
                assert!(dynamic_state.commit_token(token));
                assert!(oracle_state.commit_token(token));
                assert_eq!(dynamic_state.is_accepting(), oracle_state.is_accepting());
                assert_eq!(dynamic_state.mask(), oracle_state.mask());
            }
        }

        let loaded = DynamicConstraint::load(&dynamic.save(), &vocab).unwrap();
        assert!(loaded.inner.tokenizer.has_virtual_residual_runtime());
        assert_eq!(loaded.start().mask(), dynamic.start().mask());
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
    fn dynamic_v20_round_trips_minimal_nonempty_projected_quotients() {
        // Serialization-unit intent: a minimal deterministic nonempty
        // projected-quotient sidecar round-trips through save/load/transfer
        // and preserves mask semantics exactly. The quotients come from the
        // existing per-terminal containment builder (same exactness proofs as
        // the shared-component selector) on a two-literal fixture — no
        // reset-dispatcher sharing, no optimizer-selection assumptions, no
        // JSON lowering, no compat-mode sensitivity.
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"d".to_vec()),
            (4, b"e".to_vec()),
            (5, b"f".to_vec()),
            (6, b"g".to_vec()),
        ]);
        let mut constraint = crate::compiler::pipeline::compile_dynamic_owned_with_backend(
            crate::grammar::ast::lower(&crate::grammar::factoring::factor_named_grammar(
                crate::import::parse_glrm_to_named(
                    "start start; t A = \"abcdef\"; t B = \"abcdeg\"; nt start = A | B;",
                ).unwrap(),
            )).unwrap(),
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
            crate::runtime::parser_backend::DynamicParserBackend::Native,
        )
        .unwrap();
        let twin = constraint.clone();
        assert!(
            !twin
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "reference twin must be quotient-free (unprepared)"
        );
        assert!(
            !twin
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients(),
            "reference twin must be quotient-free (empty)"
        );
        let quotients = constraint
            .inner
            .tokenizer
            .build_terminal_projected_quotients_for_containment();
        assert_eq!(
            quotients.len(),
            2,
            "both terminals of the minimal fixture must yield exact quotients"
        );
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(quotients);
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            constraint
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        let expected = constraint
            .inner
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();
        assert!(!expected.is_empty());

        // See the Snowplow sidecar test below: this is a test-only mutation of
        // otherwise immutable post-build persistence metadata.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        assert_eq!(
            bincode::serialize(
                &loaded
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .has_projected_terminal_quotients()
        );
        assert_eq!(
            bincode::serialize(
                &transferred
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );

        // Semantic exactness: prepared quotients must not change the accepted
        // language or any mask. Differential of the quotient-free twin against
        // the quotient-bearing original plus both reloaded artifacts over
        // accept/reject paths.
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![0],
            vec![0, 1],
            vec![0, 1, 2, 3, 4, 5],
            vec![0, 1, 2, 3, 4, 6],
            vec![1],
            vec![0, 0],
        ];
        for path in &paths {
            let mut states = [
                twin.start(),
                constraint.start(),
                loaded.start(),
                transferred.start(),
            ];
            let mut failed = false;
            for &token in path {
                let masks: Vec<_> = states.iter().map(|state| state.mask()).collect();
                for (index, mask) in masks.iter().enumerate().skip(1) {
                    assert_eq!(
                        *mask, masks[0],
                        "mask agreement at {path:?} before token {token} (state {index})",
                    );
                }
                let results: Vec<bool> = states
                    .iter_mut()
                    .map(|state| state.commit_token(token))
                    .collect();
                for (index, result) in results.iter().enumerate().skip(1) {
                    assert_eq!(
                        *result, results[0],
                        "commit agreement at {path:?} token {token} (state {index})",
                    );
                }
                if !results[0] {
                    // A failed commit leaves every variant dead.
                    for (index, state) in states.iter().enumerate() {
                        assert!(
                            state.mask().iter().all(|&word| word == 0),
                            "state {index} still allows tokens after a failed commit at {path:?} token {token}",
                        );
                    }
                    failed = true;
                    break;
                }
            }
            if failed {
                continue;
            }
            // Every commit succeeded: compare the post-path masks (initial and
            // after-every-successful-commit coverage) and acceptance.
            let masks: Vec<_> = states.iter().map(|state| state.mask()).collect();
            for (index, mask) in masks.iter().enumerate().skip(1) {
                assert_eq!(
                    *mask, masks[0],
                    "final mask agreement at {path:?} (state {index})",
                );
            }
            let accepting: Vec<bool> =
                states.iter().map(|state| state.is_accepting()).collect();
            for (index, value) in accepting.iter().enumerate().skip(1) {
                assert_eq!(
                    *value, accepting[0],
                    "acceptance agreement at {path:?} (state {index})",
                );
            }
        }
        // Explicit acceptance/rejection pins on every artifact (not only
        // cross-artifact equality), so the differential above is not vacuous.
        for (label, artifact) in [
            ("twin", &twin),
            ("original", &constraint),
            ("loaded", &loaded),
            ("transferred", &transferred),
        ] {
            for path in [vec![0u32, 1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4, 6]] {
                let mut state = artifact.start();
                for &token in &path {
                    assert!(state.commit_token(token));
                }
                assert!(state.is_accepting(), "{label} must accept {path:?}");
            }
            let mut state = artifact.start();
            assert!(
                !state.is_accepting(),
                "{label} must not accept the empty prefix"
            );
            assert!(
                !state.commit_token(1),
                "{label} must reject a leading b"
            );
            let mut state = artifact.start();
            assert!(state.commit_token(0));
            assert!(
                !state.is_accepting(),
                "{label} must not accept the proper prefix [a]"
            );
            assert!(
                !state.commit_token(0),
                "{label} must reject the aa divergence"
            );
        }
    }

    #[test]
    fn dynamic_v20_round_trips_snowplow_optimizer_selected_quotients() {
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let (schema, vocab) = projected_quotient_fixture::schema_and_vocab();
        // Integration tolerance: whatever the shared-component optimizer
        // selects for Snowplow (currently nothing — the lexer isolates this
        // fixture's large terminals), the sidecar round-trips and masks agree
        // end to end. Nonempty wire-format coverage lives in the minimal
        // unit test above; this test must NOT pin the optimizer's selection.
        // Do not restore production sharing just to make Snowplow nonempty.
        let schema_value: serde_json::Value = serde_json::from_str(schema).unwrap();
        let named = crate::import::json_schema::schema_to_named_grammar(&schema_value).unwrap();
        let mut factored = crate::grammar::factoring::factor_named_grammar(named);
        crate::import::json_schema::prepare_named_grammar(&mut factored).unwrap();
        let grammar = crate::grammar::ast::lower(&factored).unwrap();
        let mut constraint = crate::compiler::pipeline::compile_dynamic_owned_with_backend(
            grammar,
            &vocab,
            crate::compiler::glr::table::GlrTableConstruction::LegacyRowBisim,
            crate::runtime::parser_backend::DynamicParserBackend::Native,
        )
        .unwrap();
        let quotients = constraint
            .inner
            .tokenizer
            .build_shared_component_terminal_projected_quotients(256);
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(quotients);
        let expected = constraint
            .inner
            .dynamic_mask_vocab
            .projected_terminal_quotients_for_artifact();

        // The private mutation above occurs after public finalization froze the
        // canonical transfer artifact. Refresh it explicitly for wire-format
        // coverage; ordinary callers cannot mutate a compiled constraint.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let saved = constraint.save();
        assert_eq!(
            u16::from_le_bytes([saved[8], saved[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let loaded = DynamicConstraint::load(&saved, &vocab).unwrap();
        assert!(loaded.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert_eq!(
            bincode::serialize(
                &loaded
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        assert_eq!(
            u16::from_le_bytes([transfer[8], transfer[9]]),
            DYNAMIC_TRANSFER_VERSION,
        );
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert_eq!(
            bincode::serialize(
                &transferred
                    .inner
                    .dynamic_mask_vocab
                    .projected_terminal_quotients_for_artifact()
            )
            .unwrap(),
            bincode::serialize(&expected).unwrap(),
        );
        assert_eq!(transferred.start().mask(), constraint.start().mask());

        let mut original_state = constraint.start();
        let mut loaded_state = loaded.start();
        let mut transferred_state = transferred.start();
        for &token in projected_quotient_fixture::replay_ids() {
            let expected = original_state.mask();
            assert_eq!(loaded_state.mask(), expected, "self-contained load before token {token}");
            assert_eq!(
                transferred_state.mask(),
                expected,
                "transfer load before token {token}"
            );
            assert!(original_state.commit_token(token));
            assert!(loaded_state.commit_token(token));
            assert!(transferred_state.commit_token(token));
        }
        assert_eq!(loaded_state.mask(), original_state.mask());
        assert_eq!(transferred_state.mask(), original_state.mask());
    }

    #[test]
    fn dynamic_v20_round_trips_prepared_empty_projected_terminal_quotients() {
        let vocab = vocab();
        let mut constraint = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        constraint
            .inner
            .dynamic_mask_vocab
            .set_projected_terminal_quotients(Vec::new());
        assert!(constraint.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert!(!constraint.inner.dynamic_mask_vocab.has_projected_terminal_quotients());

        // See the nonempty sidecar test above: this is a test-only mutation of
        // otherwise immutable post-build persistence metadata.
        constraint.external_vocab_artifact_cache = None;
        constraint.cache_external_vocab_artifact_for_save();

        let loaded = DynamicConstraint::load(&constraint.save(), &vocab).unwrap();
        assert!(loaded.inner.dynamic_mask_vocab.projected_terminal_quotients_prepared());
        assert!(!loaded.inner.dynamic_mask_vocab.has_projected_terminal_quotients());
        assert_eq!(loaded.start().mask(), constraint.start().mask());

        let transfer = constraint.clone().into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared()
        );
        assert!(!transferred.inner.dynamic_mask_vocab.has_projected_terminal_quotients());
        assert_eq!(transferred.start().mask(), constraint.start().mask());
    }

    #[test]
    fn current_dynamic_transfer_preserves_unprepared_projected_terminal_quotients() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_glrm_grammar(
            r#"
start start;
t A = /a{0,1000000000}/;
nt start = A;
"#,
            &vocab,
        )
        .unwrap();
        assert!(constraint.inner.tokenizer.has_any_virtual_runtime());
        assert!(
            !constraint
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "fresh dynamic compilation should defer projected-terminal proof artifacts",
        );

        let transfer = constraint.into_saved();
        let transferred = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(
            !transferred
                .inner
                .dynamic_mask_vocab
                .projected_terminal_quotients_prepared(),
            "transfer load must preserve unprepared-empty rather than converting it to prepared-empty",
        );
    }

    #[test]
    fn self_contained_dynamic_load_with_vocab_shares_exact_vocab() {
        let vocab = vocab();
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let loaded = DynamicConstraint::load_with_vocab(&original.save(), &vocab).unwrap();

        assert!(std::sync::Arc::ptr_eq(
            &loaded.inner.token_bytes.entries_arc(),
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
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let mut bytes = original.save_without_vocab();
        bytes[8..10].copy_from_slice(&1u16.to_le_bytes());
        assert!(DynamicConstraint::load_with_vocab(&bytes, &vocab).is_err(), "retired LR transfer must fail closed");
    }

    #[test]
    fn dynamic_transfer_loads_v2_payload_without_terminal_exprs() {
        let vocab = vocab();
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let mut bytes = original.save_without_vocab();
        bytes[8..10].copy_from_slice(&2u16.to_le_bytes());
        assert!(DynamicConstraint::load_with_vocab(&bytes, &vocab).is_err(), "retired LR transfer must fail closed");
    }

    #[test]
    fn dynamic_transfer_loads_v3_mask_quotient_payload_without_terminal_exprs() {
        let vocab = vocab();
        let original = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let mut bytes = original.save_without_vocab();
        bytes[8..10].copy_from_slice(&3u16.to_le_bytes());
        assert!(DynamicConstraint::load_with_vocab(&bytes, &vocab).is_err(), "retired LR transfer must fail closed");
    }

    #[test]
    fn precompiled_dynamic_artifacts_require_rebuild() {
        let vocab = vocab();
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'+ 'b'", &vocab).unwrap();
        let mut bytes = constraint.save();
        bytes[8..10].copy_from_slice(&8u16.to_le_bytes());
        let error = DynamicConstraint::load(&bytes, &vocab).unwrap_err().to_string();
        assert!(error.contains("unsupported dynamic transfer artifact version 8"));
        assert!(error.contains("recompile"));
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
        assert!(!dynamic.inner.table.is_present());
        assert!(dynamic.inner.direct_regular_automaton.is_some());
        assert!(dynamic.inner.uses_dynamic_runtime());
        assert_eq!(normal.start().mask(), dynamic.start().mask());
    }

    #[test]
    fn compressed_static_unions_mixed_l1_l2p_token_lengths() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let mut grammar = String::from("start: r0\n");
        for index in 0..63 {
            grammar.push_str(&format!("r{index}: \"a\" r{}\n", index + 1));
        }
        grammar.push_str("r63: \"a\"\n");

        let static_constraint = compile_compressed_static(&grammar, &vocab);
        let dynamic_constraint = compile_compressed_dynamic(&grammar, &vocab);
        assert!(!static_constraint.uses_dynamic_runtime());
        assert!(!static_constraint.table.is_present());
        assert!(static_constraint.direct_regular_automaton.is_some());
        assert!(
            !static_constraint.non_dwa_weights.materialized_for_compilation().unwrap().parser_top_accept_parts.is_empty(),
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
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"h\n".to_vec()),
            (1, b"+x".to_vec()),
            (2, b"\n".to_vec()),
            (3, b" a0\n".to_vec()),
        ]);
        let constraint = compile_compressed_static(&grammar, &vocab);
        assert!(!constraint.uses_dynamic_runtime());
        let mut state = constraint.start();

        assert!(state.commit_token(0));
        assert!(state.commit_token(1));
        assert_ne!(state.mask()[0] & (1 << 2), 0);
        assert!(state.commit_token(2));
        assert_ne!(state.mask()[0] & (1 << 3), 0);
        assert!(state.commit_token(3));
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

        let constraint = crate::runtime::Constraint::from_lark(&grammar, &vocab).unwrap();
        assert!(!constraint.uses_dynamic_runtime());
        assert!(constraint.possible_matches_complete);
        let dynamic = compile_compressed_dynamic(&grammar, &vocab);
        assert!(!dynamic.inner.possible_matches_complete);

        let mut static_state = constraint.start();
        let mut dynamic_state = dynamic.start();
        assert_eq!(static_state.mask(), dynamic_state.mask());
        assert_eq!(static_state.forced_canonical_reference(), dynamic_state.forced_canonical_reference());

        for token in [3, 3] {
            assert!(static_state.commit_token(token));
            assert!(dynamic_state.commit_token(token));
            assert_eq!(static_state.mask(), dynamic_state.mask());
            assert_eq!(static_state.is_accepting(), dynamic_state.is_accepting());
        }

        let before_third = static_state.mask();
        let checkpoint = static_state.clone();
        assert!(static_state.commit_token(0));
        assert!(dynamic_state.commit_token(0));
        let after_third = static_state.mask();
        assert_eq!(after_third, dynamic_state.mask());
        static_state = checkpoint;
        assert_eq!(static_state.mask(), before_third);
        assert!(static_state.commit_token(0));
        assert_eq!(static_state.mask(), after_third);

        let loaded = crate::runtime::Constraint::load(constraint.save(), &vocab).unwrap();
        assert!(!loaded.uses_dynamic_runtime());
        let mut loaded_state = loaded.start();
        let mut original_state = constraint.start();
        assert_eq!(loaded_state.mask(), original_state.mask());
        for token in [3, 3, 0] {
            assert!(loaded_state.commit_token(token));
            assert!(original_state.commit_token(token));
            assert_eq!(loaded_state.mask(), original_state.mask());
            assert_eq!(loaded_state.is_accepting(), original_state.is_accepting());
        }
    }

    #[test]
    fn non_regular_constraint_keeps_static_backend() {
        let vocab = vocab();
        let constraint = crate::runtime::Constraint::from_ebnf(
            "start ::= 'a' start 'b' | ''",
            &vocab,
        )
        .unwrap();
        assert!(!constraint.uses_dynamic_runtime());
    }

    #[test]
    fn dynamic_forced_uses_dynamic_masks() {
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec())]);
        let constraint = DynamicConstraint::from_ebnf("start ::= 'a'", &vocab).unwrap();
        assert_eq!(constraint.start().forced_canonical_reference(), vec![0]);
    }
}
