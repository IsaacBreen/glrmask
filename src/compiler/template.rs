//! Template DWA construction.
//!
//! Builds the final composed DWA by combining parser and terminal DWAs.

use crate::automata::weighted::dwa::Dwa;
use crate::automata::weighted::weight::WeightTable;

/// Build the template (composed) DWA.
pub fn build_template_dwa() -> Dwa {
    // TODO: Implement
    let weights = WeightTable::new(1, 1);
    Dwa::new(weights, 0, vec![true])
}
