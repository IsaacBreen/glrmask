//! # Compilation Pipeline
//!
//! This module provides a flexible, staged pipeline for building grammar constraints.
//!
//! ## Pipeline Stages
//!
//! The compilation pipeline has three main stages:
//!
//! 1. **Parsing**: Convert grammar source (EBNF, Lark, or expressions) to `GrammarDefinition`
//! 2. **Compilation**: Build tokenizer and GLR parser → `CompiledGrammar`
//! 3. **Precomputation**: Build Parser DWA → `GrammarConstraint`
//!
//! ## Quick Start
//!
//! For most use cases, use the convenience methods:
//!
//! ```rust,ignore
//! // Simple: Load grammar and vocabulary, get a constraint
//! let constraint = Pipeline::from_ebnf_file("grammar.ebnf")?
//!     .with_vocab_url("https://...")?
//!     .build()?;
//!
//! // With customization
//! let constraint = Pipeline::from_ebnf_file("grammar.ebnf")?
//!     .with_config(PipelineConfig {
//!         optimize_grammar: false,
//!         ..Default::default()
//!     })
//!     .with_vocab_file("vocab.json")?
//!     .build()?;
//! ```
//!
//! ## Manual Stage-by-Stage Building
//!
//! For advanced use cases, build each stage separately:
//!
//! ```rust,ignore
//! // Stage 1: Parse grammar (optionally skip optimization)
//! let mut definition = GrammarDefinition::from_ebnf(source)?;
//! // Optionally skip or customize optimization
//! // definition.optimize();
//!
//! // Stage 2: Compile tokenizer + parser
//! let compiled = CompiledGrammar::from_definition(Arc::new(definition));
//!
//! // Stage 3: Precompute with vocabulary
//! let constraint = GrammarConstraint::new(&compiled, &vocab_url)?;
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::constraint::GrammarConstraint;
use crate::interface::{CompiledGrammar, GrammarDefinition, GrammarExpr};
use crate::finite_automata::Expr;
use crate::tokenizer::LLMTokenID;

/// Configuration for the grammar compilation pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Whether to optimize the grammar (remove unreachable rules, etc.).
    /// Default: true
    pub optimize_grammar: bool,
    
    /// Whether to minimize the grammar (inline single-use rules, etc.).
    /// Default: true
    pub minimize_grammar: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            optimize_grammar: true,
            minimize_grammar: true,
        }
    }
}

impl PipelineConfig {
    /// Create a config that disables all optimization.
    pub fn no_optimization() -> Self {
        Self {
            optimize_grammar: false,
            minimize_grammar: false,
        }
    }
}

/// Builder for the grammar compilation pipeline.
///
/// This provides a fluent API for building grammar constraints with various
/// configuration options.
pub struct Pipeline {
    definition: GrammarDefinition,
    config: PipelineConfig,
    vocab: Option<VocabSource>,
}

/// Source for LLM vocabulary.
#[derive(Debug, Clone)]
pub enum VocabSource {
    /// Load from a local JSON file.
    File(String),
    /// Load from a URL.
    Url(String),
    /// Use a pre-loaded vocabulary map.
    Map(BTreeMap<Vec<u8>, LLMTokenID>),
}

impl Pipeline {
    /// Create a pipeline from an EBNF source string.
    pub fn from_ebnf(source: &str) -> Result<Self, String> {
        let definition = GrammarDefinition::from_ebnf(source)?;
        Ok(Self {
            definition,
            config: PipelineConfig::default(),
            vocab: None,
        })
    }

    /// Create a pipeline from an EBNF file.
    pub fn from_ebnf_file(path: &str) -> Result<Self, String> {
        let definition = GrammarDefinition::from_ebnf_file(path)?;
        Ok(Self {
            definition,
            config: PipelineConfig::default(),
            vocab: None,
        })
    }

    /// Create a pipeline from a Lark source string.
    pub fn from_lark(source: &str) -> Result<Self, String> {
        let definition = GrammarDefinition::from_lark(source)?;
        Ok(Self {
            definition,
            config: PipelineConfig::default(),
            vocab: None,
        })
    }

    /// Create a pipeline from a Lark file.
    pub fn from_lark_file(path: &str) -> Result<Self, String> {
        let definition = GrammarDefinition::from_lark_file(path)?;
        Ok(Self {
            definition,
            config: PipelineConfig::default(),
            vocab: None,
        })
    }

    /// Create a pipeline from grammar expressions.
    pub fn from_exprs(
        grammar_exprs: Vec<(String, GrammarExpr)>,
        regex_exprs: Vec<(String, Expr)>,
    ) -> Result<Self, String> {
        let definition = GrammarDefinition::from_exprs(grammar_exprs, regex_exprs)?;
        Ok(Self {
            definition,
            config: PipelineConfig::default(),
            vocab: None,
        })
    }

    /// Create a pipeline from an existing GrammarDefinition.
    pub fn from_definition(definition: GrammarDefinition) -> Self {
        Self {
            definition,
            config: PipelineConfig::default(),
            vocab: None,
        }
    }

    /// Set the pipeline configuration.
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Disable grammar optimization.
    pub fn without_optimization(mut self) -> Self {
        self.config = PipelineConfig::no_optimization();
        self
    }

    /// Set vocabulary from a URL.
    pub fn with_vocab_url(mut self, url: &str) -> Self {
        self.vocab = Some(VocabSource::Url(url.to_string()));
        self
    }

    /// Set vocabulary from a local file.
    pub fn with_vocab_file(mut self, path: &str) -> Self {
        self.vocab = Some(VocabSource::File(path.to_string()));
        self
    }

    /// Set vocabulary from a pre-loaded map.
    pub fn with_vocab_map(mut self, map: BTreeMap<Vec<u8>, LLMTokenID>) -> Self {
        self.vocab = Some(VocabSource::Map(map));
        self
    }

    /// Access the grammar definition (for inspection or manual modification).
    pub fn definition(&self) -> &GrammarDefinition {
        &self.definition
    }

    /// Access the grammar definition mutably (for manual modification).
    pub fn definition_mut(&mut self) -> &mut GrammarDefinition {
        &mut self.definition
    }

    /// Build just the CompiledGrammar (tokenizer + parser), without vocabulary.
    ///
    /// This is useful when you want to inspect the intermediate representation
    /// or use the grammar for parsing without constraint computation.
    pub fn build_compiled(&self) -> CompiledGrammar {
        let mut definition = self.definition.clone();
        
        // Apply optimizations based on config
        if self.config.optimize_grammar {
            definition.optimize();
        }
        
        CompiledGrammar::from_definition(Arc::new(definition))
    }

    /// Build the full GrammarConstraint with precomputed Parser DWA.
    ///
    /// This requires a vocabulary to be set via `with_vocab_*` methods.
    pub fn build(&self) -> Result<GrammarConstraint, String> {
        let vocab_source = self.vocab.as_ref()
            .ok_or_else(|| "Vocabulary not set. Use with_vocab_url, with_vocab_file, or with_vocab_map.".to_string())?;

        let compiled = self.build_compiled();

        match vocab_source {
            VocabSource::Url(_url) => {
                // URL fetching requires external handling (e.g., minreq or reqwest).
                // This is intentionally not implemented here to keep dependencies minimal.
                // Use with_vocab_file or with_vocab_map instead, or fetch the vocab
                // yourself and pass it via with_vocab_map.
                Err("Direct URL fetching is not supported in the pipeline. Please download the vocab file first and use with_vocab_file, or load it yourself and use with_vocab_map.".to_string())
            }
            VocabSource::File(path) => {
                // Read file and parse as JSON vocab
                let content = std::fs::read_to_string(path)
                    .map_err(|e| format!("Failed to read vocab file '{}': {}", path, e))?;
                let vocab_map: BTreeMap<String, u32> = serde_json::from_str(&content)
                    .map_err(|e| format!("Failed to parse vocab file '{}': {}", path, e))?;
                let llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = vocab_map
                    .into_iter()
                    .map(|(k, v)| (k.into_bytes(), LLMTokenID(v as usize)))
                    .collect();
                let max_id = llm_token_map.values().map(|id| id.0).max().unwrap_or(0);
                Ok(GrammarConstraint::from_compiled_grammar(
                    compiled,
                    llm_token_map,
                    max_id,
                ))
            }
            VocabSource::Map(map) => {
                let max_id = map.values().map(|id| id.0).max().unwrap_or(0);
                Ok(GrammarConstraint::from_compiled_grammar(
                    compiled,
                    map.clone(),
                    max_id,
                ))
            }
        }
    }
}

/// Quick construction functions for common use cases.
impl GrammarConstraint {
    /// Build a constraint from an EBNF file and vocabulary URL.
    ///
    /// This is the simplest way to create a constraint:
    /// ```rust,ignore
    /// let constraint = GrammarConstraint::from_ebnf_and_vocab(
    ///     "grammar.ebnf",
    ///     "https://huggingface.co/.../vocab.json"
    /// )?;
    /// ```
    pub fn from_ebnf_and_vocab(grammar_path: &str, vocab_url: &str) -> Result<Self, String> {
        Pipeline::from_ebnf_file(grammar_path)?
            .with_vocab_url(vocab_url)
            .build()
    }

    /// Build a constraint from a Lark file and vocabulary URL.
    pub fn from_lark_and_vocab(grammar_path: &str, vocab_url: &str) -> Result<Self, String> {
        Pipeline::from_lark_file(grammar_path)?
            .with_vocab_url(vocab_url)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.optimize_grammar);
        assert!(config.minimize_grammar);
    }

    #[test]
    fn test_pipeline_config_no_optimization() {
        let config = PipelineConfig::no_optimization();
        assert!(!config.optimize_grammar);
        assert!(!config.minimize_grammar);
    }

    #[test]
    fn test_pipeline_from_ebnf() {
        let source = r#"
            start ::= "hello" ;
        "#;
        let pipeline = Pipeline::from_ebnf(source).unwrap();
        assert!(!pipeline.definition.productions.is_empty());
    }

    #[test]
    fn test_pipeline_build_compiled() {
        let source = r#"
            start ::= "hello" ;
        "#;
        let pipeline = Pipeline::from_ebnf(source).unwrap();
        let compiled = pipeline.build_compiled();
        assert!(!compiled.definition.productions.is_empty());
    }
}
