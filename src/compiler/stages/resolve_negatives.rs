//! NOTE: `resolve_negatives` is intentionally deferred.
//! Keep only this stage boundary during the parser-DWA cleanup.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::automata::weighted::nwa::NWA;

pub(crate) fn resolve_negative_codes_in_nwa(_nwa: &mut NWA) {
    todo!("resolve_negatives is intentionally left as a placeholder during parser-DWA cleanup")
}
