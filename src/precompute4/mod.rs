//! Parser DWA construction.
//!
//! This module builds the Parser DWA from template DFAs and the grammar.
//!
//! Key concepts:
//! - **Terminal DWA**: Built from tokenizer in constraint_precompute.rs, encodes valid LLM tokens.
//! - **Template DFAs**: Built from terminal characterizations, encode how each terminal type
//!   interacts with the parse stack. These are unweighted DFAs that get converted to DWAs.
//! - **Parser DWA**: Combines Terminal DWA with Template DWAs to create the final constraint
//!   automaton.
//!
//! The naming distinction is important:
//! - "Terminal" refers to terminal symbols in the grammar (e.g., identifiers, keywords)
//! - "Template" refers to the DFA templates built from terminal characterizations

pub mod characterize;
pub mod parser_dwa;
pub mod resolve_negatives;

pub(crate) mod utils;
pub mod template_dfa;
pub mod nwa_optimizations;

mod test_resolve_negatives;
