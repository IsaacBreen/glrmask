//! Terminal DWA construction.
//!
//! Builds a DWA for each terminal symbol, tracking how much of the terminal
//! pattern has been matched by consumed bytes.

use crate::automata::weighted::dwa::Dwa;
use crate::automata::weighted::weight::WeightTable;

/// Build a terminal-tracking DWA.
pub fn build_terminal_dwa() -> Dwa {
    // TODO: Implement
    let weights = WeightTable::new(1, 1);
    Dwa::new(weights, 0, vec![true])
}
