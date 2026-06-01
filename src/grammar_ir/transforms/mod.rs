//! Grammar IR transforms.
//!
//! Transforms rewrite named grammar IR while preserving the denoted language.
//! They run before `grammar_ir::lower` converts the grammar to flat compiler
//! productions.

pub mod exact_subtraction;
pub mod factor;
pub mod simplify;
pub mod terminal_choice;

pub use exact_subtraction::{lower_exact_subtractions, ExactSubtractionLoweringStats};
pub use factor::factor_named_grammar;
pub use simplify::simplify_named_grammar;
pub use terminal_choice::{promote_choice_terminals_exact, TerminalChoicePromotionStats};
