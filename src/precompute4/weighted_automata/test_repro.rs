use super::*;
use std::fs;
use serde_json;
use crate::precompute4::weighted_automata::test_weighted_automata::stochastic_equivalence_test;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precompute4::weighted_automata::common::Label;

    struct SimpleRng(u64);
    impl SimpleRng {
        fn new(seed: u64) -> Self { SimpleRng(seed) }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0
        }
        fn gen_usize(&mut self, upper: usize) -> usize {
            if upper <= 1 { 0 } else { (self.next_u64() as usize) % upper }
        }
        fn gen_bool_ratio(&mut self, num: u32, den: u32) -> bool {
            if den == 0 { true } else { (self.next_u64() % (den as u64)) < (num as u64) }
        }
    }

    fn run_pipeline(nwa_in: &NWA, use_rustfst_determinize: bool) -> DWA {
        let mut nwa = nwa_in.clone();
        nwa.simplify_with_rustfst();
        nwa.compress_transitions();
        
        let dwa = if use_rustfst_determinize {
            nwa.determinize_to_dwa_with_rustfst()
        } else {
            nwa.determinize()
        };
        let mut dwa = dwa;
        dwa.simplify_with_rustfst();
        dwa.simplify();
        dwa
    }

    #[test]
    fn test_repro_and_minimize() {
        let content = fs::read_to_string("nwa_dump.json");
        if content.is_err() {
            println!("Skipping test: nwa_dump.json not found");
            return;
        }
        let content = content.unwrap();
        let nwa: NWA = serde_json::from_str(&content).expect("Failed to parse NWA");
        
        println!("Initial NWA: {} states, {} transitions", nwa.states.len(), nwa.states.num_transitions());

        let mut rng = SimpleRng::new(42);
        let mut current_nwa = nwa;

        println!("Starting reduction loop...");

        for step in 0..10000 {
            let mut candidate = current_nwa.clone();
            
            // Ultra-aggressive pruning if large
            if candidate.states.len() > 1000 {
                // Prune 95%
                let to_prune = candidate.states.len() * 19 / 20;
                for _ in 0..to_prune {
                    let q = rng.gen_usize(candidate.states.len());
                    if !candidate.body.start_states.contains(&q) {
                        candidate.states[q].transitions.clear();
                        candidate.states[q].epsilons.clear();
                        candidate.states[q].final_weight = None;
                    }
                }
            } else if candidate.states.len() > 100 {
                let to_prune = (candidate.states.len() / 5).max(1);
                for _ in 0..to_prune {
                    let q = rng.gen_usize(candidate.states.len());
                    if !candidate.body.start_states.contains(&q) {
                        candidate.states[q].transitions.clear();
                        candidate.states[q].epsilons.clear();
                        candidate.states[q].final_weight = None;
                    }
                }
            } else {
                perturb_reduce(&mut candidate, &mut rng);
            }
            
            candidate.prune_unreachable();
            candidate.prune_dead_ends();

            if candidate.states.len() == 0 || candidate.states.num_transitions() == 0 {
                continue;
            }

            if candidate.states.len() == current_nwa.states.len() && candidate.states.num_transitions() == current_nwa.states.num_transitions() {
                continue;
            }

            // println!("Step {}: Testing NWA ({} states, {} trans)...", step, candidate.states.len(), candidate.states.num_transitions());

            // Run pipelines
            let d_rust = run_pipeline(&candidate, true);
            let d_built = run_pipeline(&candidate, false);

            if d_rust.states.len() < d_built.states.len() {
                // Success! We still have the discrepancy.
                if candidate.states.len() < current_nwa.states.len() || candidate.states.num_transitions() < current_nwa.states.num_transitions() {
                    
                    if candidate.states.len() < 500 {
                         stochastic_equivalence_test(d_rust.clone(), d_built.clone());
                    }

                    current_nwa = candidate;
                    println!("Step {}: SUCCESS! NWA: {} states, {} trans. DWA Rust: {}, Builtin: {}", 
                        step, current_nwa.states.len(), current_nwa.states.num_transitions(),
                        d_rust.states.len(), d_built.states.len());
                        
                    if current_nwa.states.len() < 300 {
                        let json = serde_json::to_string(&current_nwa).unwrap();
                        std::fs::write("nwa_repro_min.json", json).unwrap();
                    }
                }
            }
            
            if current_nwa.states.len() < 8 {
                println!("Minimal NWA found!");
                break;
            }
        }
        
        println!("Final minimized NWA: {} states, {} transitions", current_nwa.states.len(), current_nwa.states.num_transitions());
        let json = serde_json::to_string(&current_nwa).unwrap();
        std::fs::write("nwa_repro_min.json", json).unwrap();
    }

    fn perturb_reduce(nwa: &mut NWA, rng: &mut SimpleRng) {
        let n = nwa.states.len();
        if n == 0 { return; }

        let r = rng.next_u64() % 100;
        if r < 35 {
            let q = rng.gen_usize(n);
            if !nwa.body.start_states.contains(&q) {
                nwa.states[q].transitions.clear();
                nwa.states[q].epsilons.clear();
                nwa.states[q].final_weight = None;
            }
        } else if r < 95 {
            let q = rng.gen_usize(n);
            let n_trans = nwa.states[q].transitions.len();
            if n_trans > 0 {
                let to_remove_idx = rng.gen_usize(n_trans);
                let key = nwa.states[q].transitions.keys().nth(to_remove_idx).cloned().unwrap();
                nwa.states[q].transitions.remove(&key);
            }
            
            let n_eps = nwa.states[q].epsilons.len();
            if n_eps > 0 {
                let to_remove_idx = rng.gen_usize(n_eps);
                nwa.states[q].epsilons.remove(to_remove_idx);
            }
        } else {
            let q = rng.gen_usize(n);
            if let Some(fw) = &mut nwa.states[q].final_weight {
                if !fw.is_empty() {
                     let first = fw.rsb.ranges().next().unwrap();
                     *fw = Weight::from_item(*first.start());
                }
            }
            for targets in nwa.states[q].transitions.values_mut() {
                for (_, w) in targets {
                    if !w.is_empty() {
                        let first = w.rsb.ranges().next().unwrap();
                        *w = Weight::from_item(*first.start());
                    }
                }
            }
             for (_, w) in &mut nwa.states[q].epsilons {
                if !w.is_empty() {
                    let first = w.rsb.ranges().next().unwrap();
                    *w = Weight::from_item(*first.start());
                }
            }
        }
    }
}
