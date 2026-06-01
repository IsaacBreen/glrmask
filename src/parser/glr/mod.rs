//! GLR parser domain.
//!
//! This namespace contains the parser machinery that sits below the paper's
//! Terminal-DWA/Parser-DWA layer but above raw grammar syntax.  It is deliberately
//! not nested under `compile`: the table is constructed at compile time, while
//! stack advancement is used by runtime Mask and Commit.
//!
//! The three mathematical responsibilities are:
//!
//! - [`analysis`]: normalize a flat grammar and compute nullable/FIRST/FOLLOW and
//!   display metadata.
//! - [`table`]: construct and optimize the GLR transition table.
//! - [`advance`]: execute one terminal stack-effect step over a persistent GSS.

pub mod accumulator;
pub mod advance;
pub mod analysis;
pub mod labels;
pub mod table;
