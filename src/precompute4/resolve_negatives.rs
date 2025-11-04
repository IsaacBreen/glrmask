use crate::precompute4::full_dwa::Precomputed4;
use crate::precompute4::weighted_automata::DWA;

// Placeholder: resolve negative-coded transitions. Unimplemented for now.
pub fn resolve_negative_codes_for_all(precomputed4: &mut Precomputed4) {
    for (sid, dwa) in precomputed4.iter_mut() {
        println!("resolve_negative_codes_for_all: TokenizerStateID {:?} -> TODO", sid);
        resolve_negative_codes_in_dwa(dwa);
    }
}

fn resolve_negative_codes_in_dwa(_dwa: &mut DWA) {
    eprintln!("Negative-coded transition resolution is not implemented yet. This pass should transform any i16<0 transition labels into an equivalent form free of negative labels.");
}
