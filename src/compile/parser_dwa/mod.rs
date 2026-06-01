//! Parser DWA construction.
//!
//! # Denotation
//!
//! The Parser DWA is the compile-time weighted automaton whose input word is a
//! parser-stack prefix `rho` and whose output weight is the set of lexer-state /
//! vocabulary-token pairs accepted after that stack prefix.
//!
//! In the paper's notation, this module builds the automaton `PDWA` satisfying
//!
//! ```text
//! [[PDWA]](rho)_{q,v} = 1  iff  rho ∈ E_{q,v}.
//! ```
//!
//! The important point is that the Parser DWA does not know, or expose, the
//! internal details of the parser algorithm.  It only consumes stack-effect
//! recognizers.  Today those recognizers are produced from a GLR table; the
//! construction is intentionally organized so that a future parser backend can
//! provide the same recognizer family without changing Mask or Commit.
//!
//! # Construction shape
//!
//! The construction is a pullback/composition of two finite objects:
//!
//! 1. The Terminal DWA, which maps terminal strings to lexer-state/token-pair
//!    weights.
//! 2. Template DFAs/NWAs, one per terminal, which recognize the parser-stack
//!    prefixes that can realize that terminal's stack effect.
//!
//! For each Terminal-DWA transition bundle, we splice the corresponding
//! terminal templates in front of the continuation state of the target
//! Terminal-DWA state.  The resulting weighted NWA is then determinized over
//! parser-state labels to obtain the runtime Parser DWA.
//!
//! # File guide
//!
//! - `builder.rs`: public build entrypoints and high-level phase ordering.
//! - `compose_nwa.rs`: Terminal-DWA / template composition into a parser NWA.
//! - `terminal_projection.rs`: projection of Terminal-DWA states into terminal
//!   bundles and productive continuation summaries.
//! - `determinize.rs`: weighted subset construction and fallback/default
//!   determinization.
//! - `optimize.rs`: default-edge normalization and final-weight subtraction.
//! - `options.rs`: construction policy switches.
//! - `profiling.rs`: compile-profile records and all textual profile emission.
//! - `types.rs`: local mathematical data carriers.
//! - `labels.rs`: parser-state label interpretation.

pub(crate) mod builder;
mod compose_nwa;
mod determinize;
mod labels;
mod optimize;
mod options;
mod profiling;
mod terminal_projection;
mod types;

pub(crate) use builder::{
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    build_parser_dwa_from_terminal_dwa_with_templates,
    ParserDwaBuildInputs,
    ParserDwaBuildOutput,
};
