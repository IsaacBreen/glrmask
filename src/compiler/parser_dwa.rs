//! Parser DWA construction.
//!
//! Converts the GLR parse table + tokenizer info into a weighted automaton
//! that tracks parse states as tokens are consumed.

use crate::automata::weighted::dwa::Dwa;
use crate::automata::weighted::weight::WeightTable;

/// Build a parser DWA from the GLR table and token-set mappings.
pub fn build_parser_dwa() -> Dwa {
    // TODO: Implement
    let weights = WeightTable::new(1, 1);
    Dwa::new(weights, 0, vec![true])
}
