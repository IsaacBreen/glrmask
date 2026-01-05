use super::*;
use std::fs;
use serde_json;

#[cfg(test)]
mod tests {
    use super::*;

    fn run_pipeline(nwa_in: &NWA, use_rustfst_determinize: bool) -> DWA {
        let mut nwa = nwa_in.clone();
        println!("--- Pipeline ({}) ---", if use_rustfst_determinize { "RustFST" } else { "Builtin" });
        nwa.simplify_with_rustfst();
        nwa.compress_transitions();
        
        let mut dwa = if use_rustfst_determinize {
            nwa.determinize_to_dwa_with_rustfst()
        } else {
            nwa.determinize()
        };
        println!("After determinize: {} states", dwa.states.len());
        
        if !use_rustfst_determinize {
            dwa.residuated_push();
        }
        dwa.simplify_with_rustfst();
        dwa.simplify();
        
        println!("Final size: {} states", dwa.states.len());
        dwa
    }

    #[test]
    fn test_trace_minimal_nwa() {
        let content = fs::read_to_string("nwa_repro_min.json").expect("Failed to read nwa_repro_min.json");
        let nwa: NWA = serde_json::from_str(&content).expect("Failed to parse NWA");
        
        let _d_rust = run_pipeline(&nwa, true);
        let d_built = run_pipeline(&nwa, false);
        
        println!("\n--- Builtin DWA Detail ---\n");
        for i in 0..d_built.states.len() {
            let s = &d_built.states.0[i];
            println!("State {}: final={:?}, transitions={:?}", i, s.final_weight, s.transitions);
            for (lbl, target) in &s.transitions {
                let w = s.trans_weights.get(lbl).unwrap();
                println!("  {} -> {}: weight length = {}", lbl, target, w.len());
                if w.len() < 100 {
                    println!("    weight: {:?}", w);
                }
            }
        }
    }
}
