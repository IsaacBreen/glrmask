#![deny(warnings)]
#![allow(dead_code)]
#![allow(unused_variables)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
mod error;
pub(crate) mod grammar;
pub(crate) mod import;
pub(crate) mod runtime;
mod vocab;

pub use runtime::{Constraint, ConstraintState};
pub use vocab::Vocab;

pub(crate) use error::{GlrMaskError, Result};

/// Compile a Constraint from a serialized GrammarDef JSON + vocab.
/// This runs the full compile pipeline (equivalence analysis, terminal DWA, parser DWA).
pub(crate) fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Constraint> {
    let gdef: grammar::flat::GrammarDef = serde_json::from_str(grammar_def_json)
        .map_err(|e| GlrMaskError::GrammarParse(format!("invalid GrammarDef JSON: {e}")))?;
    Ok(compiler::stages::id_map_and_terminal_dwa::l2p::with_ti_pool(|| {
        compiler::compile_owned(gdef, vocab)
    }))
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
    if import::json_schema::simplify_grammar_enabled() {
        grammar::named_simplify::simplify_named_grammar(&mut factored);
    }
    if import::json_schema::promote_literal_choices_enabled() {
        grammar::terminal_choice_promotion::promote_choice_terminals_exact(&mut factored, false);
    }
    import::json_schema::assign_default_lexer_partitions(&mut factored);
    Ok(grammar::glrm::to_glrm(&factored))
}

pub(crate) fn set_test_compat_mode(enabled: bool) {
    crate::import::json_schema::string::TEST_COMPAT_MODE.with(|cell| {
        cell.set(if enabled {
            crate::import::json_schema::string::JsonStringCompatMode::LlGuidanceNative
        } else {
            crate::import::json_schema::string::JsonStringCompatMode::JsonSchema
        });
    });
}

#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
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

    use crate::{Constraint, ConstraintState, Vocab};

    pub type Result<T> = std::result::Result<T, Error>;

    pub trait ConstraintExt: Sized {
        fn compile_grammar_def_json(grammar_def_json: &str, vocab: &Vocab) -> Result<Self>;
        fn dump_json_schema_grammar_glrm(schema_json: &str) -> Result<String>;
        fn profile_json_schema_import(schema_json: &str) -> Result<()>;
        fn warm_ti_pool();
        fn clear_stale_weights();
        fn clear_weight_interners();
        fn clear_weight_op_caches();
        fn set_test_compat_mode(enabled: bool);

        fn save_runtime_payload_v1(&self) -> Vec<u8>;
        fn load_runtime_payload_v1(bytes: &[u8]) -> Result<Self>;
        fn save_runtime_payload_v2(&self) -> Vec<u8>;
        fn load_runtime_payload_v2(bytes: &[u8]) -> Result<Self>;
        fn mask_game_internal_to_original(&self) -> &[Vec<u32>];
        fn mask_game_original_to_internal(&self) -> &[u32];
        fn num_parser_states(&self) -> u32;
        fn num_tokenizer_states(&self) -> usize;
        fn num_forced_minimized_tokenizer_states(&self) -> usize;
        fn table_ambiguous_actions(&self) -> Vec<TableAmbiguity>;
        fn table_has_ambiguity(&self) -> bool;
        fn terminal_display_names(&self) -> &[String];
        fn terminal_display_name(&self, terminal_id: u32) -> Option<&str>;
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

        fn save_runtime_payload_v1(&self) -> Vec<u8> {
            Constraint::save_runtime_payload_v1(self)
        }

        fn load_runtime_payload_v1(bytes: &[u8]) -> Result<Self> {
            Constraint::load_runtime_payload_v1(bytes)
        }

        fn save_runtime_payload_v2(&self) -> Vec<u8> {
            Constraint::save_runtime_payload_v2(self)
        }

        fn load_runtime_payload_v2(bytes: &[u8]) -> Result<Self> {
            Constraint::load_runtime_payload_v2(bytes)
        }

        fn mask_game_internal_to_original(&self) -> &[Vec<u32>] {
            Constraint::mask_game_internal_to_original(self)
        }

        fn mask_game_original_to_internal(&self) -> &[u32] {
            Constraint::mask_game_original_to_internal(self)
        }

        fn num_parser_states(&self) -> u32 {
            Constraint::num_parser_states(self)
        }

        fn num_tokenizer_states(&self) -> usize {
            Constraint::num_tokenizer_states(self)
        }

        fn num_forced_minimized_tokenizer_states(&self) -> usize {
            Constraint::num_forced_minimized_tokenizer_states(self)
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
        fn mask_game_fill_mask_and_internal_ids(&self, buf: &mut [u32]) -> Vec<u32>;
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

        fn mask_game_fill_mask_and_internal_ids(&self, buf: &mut [u32]) -> Vec<u32> {
            ConstraintState::mask_game_fill_mask_and_internal_ids(self, buf)
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
