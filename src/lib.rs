#![recursion_limit = "256"]

//! Extremely fast grammar-constrained decoding for LLMs.
//!
//! # Pre-release compatibility policy
//!
//! GLRMask is still pre-release. Serialized constraint formats, internal runtime
//! layouts, and compiler artifacts are **not** required to remain compatible
//! with earlier development versions. Architecture, correctness, simplicity,
//! and performance take priority over loading old artifacts. Compatibility code
//! may be removed whenever it obstructs a cleaner design; do not preserve an old
//! representation unless there is a separate current reason to keep it.
//!
//! GLRMask compiles a [`Grammar`] together with a model vocabulary into a reusable
//! [`Constraint`]. A mutable [`ConstraintState`] tracks one generated sequence:
//! obtain the next-token mask, sample a token, then commit that token to advance
//! the parser state.
//!
//! [`DynamicConstraint`] compiles faster than [`Constraint`] and produces masks
//! more slowly. [`DynamicConstraintState`] has the same decoding operations as
//! [`ConstraintState`].
//!
//! # Quickstart
//!
//! ```
//! use glrmask::{Constraint, Grammar, Vocab};
//!
//! let vocab = Vocab::new(vec![
//!     (0, b"hello".to_vec()),
//!     (1, b" ".to_vec()),
//!     (2, b"world".to_vec()),
//! ]);
//! let grammar = Grammar::ebnf(r#"start ::= "hello" " " "world""#);
//! let constraint = Constraint::compile(grammar, &vocab).unwrap();
//!
//! let mut state = constraint.start();
//! assert_ne!(state.mask()[0] & (1 << 0), 0);
//! state.commit_token(0).unwrap();
//! assert_ne!(state.mask()[0] & (1 << 1), 0);
//! state.commit_token(1).unwrap();
//! state.commit_token(2).unwrap();
//! assert!(state.is_accepting());
//! ```
//!
//! Masks in the Rust API are packed `u32` bitsets. Bit `token_id % 32` of word
//! `token_id / 32` indicates whether that token is allowed.
//!
//! # Grammar inputs
//!
//! [`Grammar`] accepts JSON Schema, GLRM, Lark, or EBNF. GLRM source uses the
//! literal `glrm 1;` header. Bind source subgrammars with
//! [`Grammar::bind_grammar`]. Use [`ConstraintSpec::builder`] for exact-token or
//! compiled-subgrammar bindings.
//!
//! # Persistence
//!
//! Use [`Constraint::save`] and [`Constraint::load`] to cache compiled
//! constraints across requests or process restarts.
//!
//! # GLRM external bindings
//!
//! GLRM declares exact model-token terminals with `extern token NAME;`. Bind
//! their IDs with [`ConstraintSpecBuilder::bind_token`].
//!
//! GLRM declares child grammars with `extern grammar name;`. Bind a source child
//! with [`Grammar::bind_grammar`], bind a source/spec/compiled child before
//! compilation with [`ConstraintSpecBuilder::bind_grammar`], or compile an
//! unresolved reusable parent and later attach a compiled child with
//! [`Constraint::bind_grammar`].
//!
//! See the repository's Python guide and README for model integration examples,
//! grammar syntax, special tokens, and benchmarks.

#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
mod public_api;
pub(crate) use glrmask_grammar::__private::grammar;
pub(crate) mod import;
pub(crate) mod programmatic_js;
pub(crate) mod runtime;
#[path = "runtime/dynamic_constraint.rs"]
mod dynamic_constraint;
pub(crate) use glrmask_vocab::__private as vocab;

pub use dynamic_constraint::{DynamicConstraint, DynamicConstraintState};
pub use runtime::{BoundaryTriggerDetail, Constraint, ConstraintState};
pub use glrmask_vocab::Vocab;
pub use error::{Error, Result};
pub use public_api::{ConstraintSpec, ConstraintSpecBuilder, Grammar, VocabPartition};

#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod grammar_cross_tests;
#[cfg(test)]
mod terminal_dwa_cross_tests;

pub(crate) use error::GlrMaskError;

/// Compile a Constraint from a serialized GrammarDef JSON + vocab.
/// This runs the full compile pipeline (equivalence analysis, terminal DWA, parser DWA).
pub(crate) fn compile_grammar_def_json(
    grammar_def_json: &str,
    vocab: &Vocab,
) -> Result<Constraint> {
    let gdef: grammar::flat::GrammarDef = serde_json::from_str(grammar_def_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid GrammarDef JSON: {e}")))?;
    error::catch_internal_invariant(|| {
        compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
            compiler::compile_owned(gdef, vocab)
        })
    })
}

/// Populate compile-time artifacts that are pure functions of the vocabulary.
///
/// This intentionally does not compile any grammar/schema-dependent artifact.
pub(crate) fn prepare_vocab_for_compile(vocab: &Vocab) {
    compiler::compile::prepare_vocab_for_compile(vocab);
}

/// Build (and, if configured, start the keepalive for) the terminal
/// interchangeability certification thread pool ahead of first use.
///
/// Calling this at Python module import warms the pool so discovery does not
/// pay the first-use worker-wake handoff (a large latency on macOS).
pub(crate) fn warm_ti_pool() {
    compiler::stages::id_map_and_terminal_dwa::l2p::warm_ti_pool();
}

/// Dump the imported JSON Schema grammar in GLRM format.
///
/// This intentionally preserves exact subtraction syntax so dumps reflect the
/// source-level structure. The compile/import pipeline may still apply exact
/// subtraction lowering.
pub(crate) fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid JSON: {e}")))?;
    let named = import::json_schema::schema_to_named_grammar(&schema)?;
    let mut factored = grammar::factoring::factor_named_grammar(named);
    import::json_schema::prepare_named_grammar_for_dump(&mut factored)?;
    Ok(grammar::glrm::to_glrm(&factored))
}

pub(crate) fn set_test_compat_mode(enabled: bool) {
    glrmask_json_schema::__private::set_test_compat_mode(enabled);
}

#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    #[derive(Debug, Clone, Copy, Default)]
    pub struct CompilerCacheStats {
        pub token_set_entries: usize,
        pub live_token_set_entries: usize,
        pub weight_buckets: usize,
        pub weight_entries: usize,
        pub live_weight_entries: usize,
        pub current_thread_weight_ops: usize,
        pub current_thread_token_set_ops: usize,
        pub current_thread_public_intersections: usize,
        pub current_thread_weight_hashes: usize,
        pub weight_op_generation: u64,
        pub weight_hash_generation: u64,
        pub vocab_artifacts: usize,
    }

    pub fn compiler_cache_stats(vocab: Option<&crate::Vocab>) -> CompilerCacheStats {
        let stats = crate::ds::weight::weight_cache_stats();
        CompilerCacheStats {
            token_set_entries: stats.token_set_entries,
            live_token_set_entries: stats.live_token_set_entries,
            weight_buckets: stats.weight_buckets,
            weight_entries: stats.weight_entries,
            live_weight_entries: stats.live_weight_entries,
            current_thread_weight_ops: stats.current_thread_weight_ops,
            current_thread_token_set_ops: stats.current_thread_token_set_ops,
            current_thread_public_intersections: stats.current_thread_public_intersections,
            current_thread_weight_hashes: stats.current_thread_weight_hashes,
            weight_op_generation: stats.weight_op_generation,
            weight_hash_generation: stats.weight_hash_generation,
            vocab_artifacts: vocab.map_or(0, crate::Vocab::compiler_cache_entry_count),
        }
    }

    pub use crate::compiler::glr::table::TableAmbiguity;
    pub use crate::error::Error;
    pub use crate::runtime::{
        AdvanceTrace,
        AdvanceTraceStep,
        CommitProfile,
        GssProfileSummary,
        MaskProfile,
        PerAdvanceEntry,
    };

    use crate::{Constraint, ConstraintState, DynamicConstraint, Vocab};

    pub type Result<T> = std::result::Result<T, Error>;

    /// Internal release-only benchmark for exact o21137 subgrammar decomposition.
    pub fn run_o21137_subgrammar_benchmark(mode: &str) {
        crate::compiler::o21137_subgrammar_bench::run(mode);
    }

    pub trait ConstraintExt: Sized {
        fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Self>;
        fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String>;
        fn profile_json_schema_import(schema_json: &str) -> Result<()>;
        fn warm_ti_pool();
        fn clear_stale_weights();
        fn clear_weight_interners();
        fn clear_weight_op_caches();
        fn set_test_compat_mode(enabled: bool);
        fn prepare_composition_grammar_summary(&mut self) -> Result<()>;

        fn bind_vocab_exact(&mut self, vocab: &Vocab) -> std::result::Result<(), String>;
        fn prepare_for_composition(&mut self, vocab: &Vocab) -> Result<()>;

        fn num_parser_states(&self) -> u32;
        fn num_tokenizer_states(&self) -> usize;
        fn compute_forced_minimized_tokenizer_state_count(&self) -> usize;
        fn max_original_token_id(&self) -> Option<u32>;
        fn final_internal_token_count(&self) -> usize;
        fn final_original_token_map(&self) -> Vec<u32>;
        fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity>;
        fn table_has_ambiguity(&self) -> bool;
        fn terminal_display_names(&self) -> &[String];
        fn terminal_display_name(&self, terminal_id: u32) -> Option<&str>;
        /// Internal benchmark/debug bridge for linking already-compiled
        /// subgrammar artifacts after the legacy public composition API was
        /// removed.  Production callers should express subgrammars in GLRM.
        fn compose_compiled_subgrammars(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> Result<Self>;
        fn compose_compiled_subgrammars_shared(
            self,
            children: &[(&str, std::sync::Arc<Constraint>)],
            vocab: &Vocab,
        ) -> Result<Self>;
    }

    pub trait DynamicConstraintExt: Sized {
        fn compile_ebnf_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<(Vec<u8>, u64, u64)>;
        fn compile_lark_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<(Vec<u8>, u64, u64)>;
        fn compile_json_schema_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<(Vec<u8>, u64, u64)>;
        fn compile_glrm_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<(Vec<u8>, u64, u64)>;
        fn compile_ebnf_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>>;
        fn compile_lark_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>>;
        fn compile_json_schema_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>>;
        fn compile_glrm_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>>;
        fn load_with_vocab(bytes: &[u8], vocab: &Vocab) -> Result<Self>;
        fn max_original_token_id(&self) -> Option<u32>;
    }

    impl ConstraintExt for Constraint {
        fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Self> {
            crate::compile_grammar_def_json(grammar_def_json, vocab)
        }

        fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String> {
            crate::dump_json_schema_grammar_glrm(schema_json)
        }

        fn profile_json_schema_import(schema_json: &str) -> Result<()> {
            crate::import::__profile_json_schema_import(schema_json)
        }

        fn warm_ti_pool() {
            crate::warm_ti_pool();
        }

        fn bind_vocab_exact(&mut self, vocab: &Vocab) -> std::result::Result<(), String> {
            Constraint::bind_vocab_exact(self, vocab)
        }

        fn prepare_for_composition(&mut self, vocab: &Vocab) -> Result<()> {
            self.prepare_for_composition_internal(vocab)
        }

        fn compose_compiled_subgrammars(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> Result<Self> {
            use crate::compiler::constraint_compose::{
                CompiledSubgrammarInput, compose_constraints_owned_parent,
            };
            use std::collections::BTreeSet;

            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = self
                    .terminal_display_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        Error::Compilation(format!(
                            "parent has no subgrammar placeholder terminal {name:?}",
                        ))
                    })? as u32;
                if !seen.insert(placeholder_terminal) {
                    return Err(Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent(self, &inputs, vocab)
                .map(|composition| composition.constraint)
                .map_err(Error::Compilation)
        }

        fn compose_compiled_subgrammars_shared(
            self,
            children: &[(&str, std::sync::Arc<Constraint>)],
            vocab: &Vocab,
        ) -> Result<Self> {
            use crate::compiler::constraint_compose::{
                CompiledSubgrammarInput, compose_constraints_owned_parent_shared,
            };
            use std::collections::BTreeSet;
            use std::sync::Arc;

            let mut inputs = Vec::with_capacity(children.len());
            let mut shared = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for (name, child) in children {
                let placeholder_terminal = self
                    .terminal_display_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        Error::Compilation(format!(
                            "parent has no subgrammar placeholder terminal {name:?}",
                        ))
                    })? as u32;
                if !seen.insert(placeholder_terminal) {
                    return Err(Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child.as_ref(),
                });
                shared.push(Arc::clone(child));
            }
            compose_constraints_owned_parent_shared(self, &inputs, &shared, vocab)
                .map(|composition| composition.constraint)
                .map_err(Error::Compilation)
        }

        fn clear_stale_weights() {
            crate::ds::weight::clear_stale_weights();
        }

        fn clear_weight_interners() {
            crate::ds::weight::clear_weight_interners();
        }

        fn clear_weight_op_caches() {
            crate::ds::weight::clear_weight_op_caches();
        }

        fn set_test_compat_mode(enabled: bool) {
            crate::set_test_compat_mode(enabled);
        }

        fn prepare_composition_grammar_summary(&mut self) -> Result<()> {
            if self.composition_grammar_summary.is_some() {
                return Ok(());
            }
            let augmented_start = self
                .table
                .rules
                .first()
                .map(|rule| rule.lhs)
                .ok_or_else(|| Error::Compilation("constraint table has no augmented-start rule".to_string()))?;
            let analyzed = crate::compiler::glr::analysis::AnalyzedGrammar::from_composed_rules(
                self.table.rules.clone(),
                self.table.num_terminals,
                self.terminal_display_names.clone(),
                self.table.nonterminal_display_names.clone(),
                augmented_start,
            );
            self.composition_grammar_summary = Some(
                crate::compiler::pipeline::composition_grammar_summary_from_analysis(&analyzed),
            );
            Ok(())
        }


        fn num_parser_states(&self) -> u32 {
            Constraint::num_parser_states(self)
        }

        fn num_tokenizer_states(&self) -> usize {
            Constraint::num_tokenizer_states(self)
        }

        fn compute_forced_minimized_tokenizer_state_count(&self) -> usize {
            Constraint::compute_forced_minimized_tokenizer_state_count(self)
        }

        fn max_original_token_id(&self) -> Option<u32> {
            Constraint::max_original_token_id(self)
        }

        fn final_internal_token_count(&self) -> usize {
            self.internal_token_count()
        }

        fn final_original_token_map(&self) -> Vec<u32> {
            self.original_token_map().to_vec()
        }

        fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity> {
            Constraint::table_ambiguous_actions(self)
        }

        fn table_has_ambiguity(&self) -> bool {
            Constraint::table_has_ambiguity(self)
        }

        fn terminal_display_names(&self) -> &[String] {
            Constraint::terminal_display_names(self)
        }

        fn terminal_display_name(&self, terminal_id: u32) -> Option<&str> {
            Constraint::terminal_display_name(self, terminal_id)
        }

    }

    impl DynamicConstraintExt for DynamicConstraint {
        fn compile_ebnf_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<(Vec<u8>, u64, u64)> {
            DynamicConstraint::compile_ebnf_serialized_profiled_with_end_tokens(
                source,
                vocab,
                end_token_ids,
            )
        }

        fn compile_lark_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<(Vec<u8>, u64, u64)> {
            DynamicConstraint::compile_lark_serialized_profiled_with_end_tokens(
                source,
                vocab,
                end_token_ids,
            )
        }

        fn compile_json_schema_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<(Vec<u8>, u64, u64)> {
            DynamicConstraint::compile_json_schema_serialized_profiled_with_end_tokens(
                source,
                vocab,
                end_token_ids,
            )
        }

        fn compile_glrm_serialized_profiled_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<(Vec<u8>, u64, u64)> {
            DynamicConstraint::compile_glrm_serialized_profiled_with_end_tokens(
                source,
                vocab,
                end_token_ids,
            )
        }

        fn compile_ebnf_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>> {
            DynamicConstraint::compile_ebnf_serialized_with_end_tokens(source, vocab, end_token_ids)
        }

        fn compile_lark_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>> {
            DynamicConstraint::compile_lark_serialized_with_end_tokens(source, vocab, end_token_ids)
        }

        fn compile_json_schema_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>> {
            DynamicConstraint::compile_json_schema_serialized_with_end_tokens(
                source,
                vocab,
                end_token_ids,
            )
        }

        fn compile_glrm_serialized_with_end_tokens(
            source: &str,
            vocab: &Vocab,
            end_token_ids: &[u32],
        ) -> Result<Vec<u8>> {
            DynamicConstraint::compile_glrm_serialized_with_end_tokens(source, vocab, end_token_ids)
        }

        fn load_with_vocab(bytes: &[u8], vocab: &Vocab) -> Result<Self> {
            DynamicConstraint::load_with_vocab(bytes, vocab)
        }

        fn max_original_token_id(&self) -> Option<u32> {
            DynamicConstraint::max_original_token_id(self)
        }
    }

    pub trait ConstraintStateExt {
        fn commit_token_timed_ns(&mut self, token_id: u32) -> std::result::Result<u64, String>;
        fn commit_token_profiled(
            &mut self,
            token_id: u32,
        ) -> std::result::Result<CommitProfile, String>;
        fn commit_token_per_advance(
            &mut self,
            token_id: u32,
        ) -> std::result::Result<
            (Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile),
            String,
        >;
        fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)>;
        fn fill_mask_profiled(&self, buf: &mut [u32]) -> MaskProfile;
        fn fill_mask_timed_ns(&self, buf: &mut [u32]) -> u64;
        fn has_parser_ambiguity(&self) -> bool;
        fn parser_path_count(&self, limit: usize) -> usize;
        fn parser_root_count(&self) -> usize;
    }

    impl ConstraintStateExt for ConstraintState<'_> {
        fn commit_token_timed_ns(&mut self, token_id: u32) -> std::result::Result<u64, String> {
            ConstraintState::commit_token_timed_ns(self, token_id)
        }

        fn commit_token_profiled(
            &mut self,
            token_id: u32,
        ) -> std::result::Result<CommitProfile, String> {
            ConstraintState::commit_token_profiled(self, token_id)
        }

        fn commit_token_per_advance(
            &mut self,
            token_id: u32,
        ) -> std::result::Result<
            (Vec<PerAdvanceEntry>, Vec<(u32, Vec<Vec<u32>>)>, CommitProfile),
            String,
        > {
            ConstraintState::commit_token_per_advance(self, token_id)
        }

        fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
            ConstraintState::debug_parser_stacks(self)
        }

        fn fill_mask_profiled(&self, buf: &mut [u32]) -> MaskProfile {
            ConstraintState::fill_mask_profiled(self, buf)
        }

        fn fill_mask_timed_ns(&self, buf: &mut [u32]) -> u64 {
            ConstraintState::fill_mask_timed_ns(self, buf)
        }

        fn has_parser_ambiguity(&self) -> bool {
            ConstraintState::has_parser_ambiguity(self)
        }

        fn parser_path_count(&self, limit: usize) -> usize {
            ConstraintState::parser_path_count(self, limit)
        }

        fn parser_root_count(&self) -> usize {
            ConstraintState::parser_root_count(self)
        }

    }

    pub trait VocabExt {
        fn prepare_for_compile(&self);
    }

    impl VocabExt for Vocab {
        fn prepare_for_compile(&self) {
            crate::prepare_vocab_for_compile(self);
        }
    }
}
