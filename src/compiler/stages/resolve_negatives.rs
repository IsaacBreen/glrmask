//! Placeholder for negative-label resolution in parser NWAs.
//!
//! The real `resolve_negatives` implementation is intentionally deferred.
//! Parser-DWA cleanup keeps the stage boundary visible, but the algorithmic
//! body requested by the human has been removed for now.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::automata::weighted::nwa::NWA;

pub(crate) fn resolve_negative_codes_in_nwa(_nwa: &mut NWA) {
    todo!("resolve_negatives is intentionally left as a placeholder during parser-DWA cleanup")
}
