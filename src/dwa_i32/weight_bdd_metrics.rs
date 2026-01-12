#![allow(dead_code)]

use biodivine_lib_bdd::{Bdd, BddVariable, BddVariableSet};

use crate::dwa_i32::{DWA, Weight};

fn bits_needed(max_value: usize) -> usize {
    let bits = (usize::BITS - max_value.leading_zeros()) as usize;
    bits.max(1)
}

fn bdd_true(vars: &BddVariableSet) -> Bdd {
    vars.mk_true()
}

fn bdd_false(vars: &BddVariableSet) -> Bdd {
    vars.mk_false()
}

/// Build a BDD representing the integer interval [lo, hi] (inclusive), over `k` bits.
///
/// Variable order is MSB -> LSB, using variable indices 0..k-1.
fn bdd_interval_inclusive(vars: &BddVariableSet, lo: usize, hi: usize, k: usize) -> Bdd {
    debug_assert!(lo <= hi);
    if k == 0 {
        return if lo == 0 && hi == 0 { bdd_true(vars) } else { bdd_false(vars) };
    }

    // Full domain for k bits.
    let full_hi = if k >= usize::BITS as usize { usize::MAX } else { (1usize << k) - 1 };

    if lo == 0 && hi == full_hi {
        return bdd_true(vars);
    }

    let msb_mask = 1usize << (k - 1);
    let lo_msb = (lo & msb_mask) != 0;
    let hi_msb = (hi & msb_mask) != 0;

    // Map this recursion level (k) to variable index (MSB-first).
    let var_index: usize = vars.num_vars() as usize - k;
    let var = BddVariable::from_index(var_index);
    let x = vars.mk_var(var);
    let nx = vars.mk_not_var(var);

    match (lo_msb, hi_msb) {
        (false, false) => {
            // Stay in 0* half.
            nx.and(&bdd_interval_inclusive(vars, lo, hi, k - 1))
        }
        (true, true) => {
            // Stay in 1* half.
            x.and(&bdd_interval_inclusive(vars, lo - msb_mask, hi - msb_mask, k - 1))
        }
        (false, true) => {
            // Cross the boundary: [lo..(2^{k-1}-1)] U [2^{k-1}..hi]
            let left_hi = msb_mask - 1;
            let left = nx.and(&bdd_interval_inclusive(vars, lo, left_hi, k - 1));
            let right = x.and(&bdd_interval_inclusive(vars, 0, hi - msb_mask, k - 1));
            left.or(&right)
        }
        (true, false) => {
            // Impossible if lo <= hi.
            bdd_false(vars)
        }
    }
}

fn weight_to_bdd(vars: &BddVariableSet, weight: &Weight, domain_max: usize, k: usize) -> Bdd {
    if weight.is_empty() {
        return bdd_false(vars);
    }

    if weight.is_all_fast() {
        return bdd_true(vars);
    }

    let mut acc = bdd_false(vars);

    for r in weight.rsb.ranges() {
        let start = *r.start();
        let end = *r.end();

        if start > domain_max {
            continue;
        }
        let clipped_end = end.min(domain_max);

        let interval = bdd_interval_inclusive(vars, start, clipped_end, k);
        acc = acc.or(&interval);

        if acc.is_true() {
            break;
        }
    }

    acc
}

/// Print baseline RangeSet complexity and BDD node-count complexity for all unique (interned)
/// weights in the given DWA.
///
/// - Baseline complexity: total number of `RangeSetBlaze` ranges across unique weights.
/// - BDD complexity: total number of BDD nodes across unique weights.
///
/// Enabled only when `WEIGHT_BDD_METRICS=1`.
pub fn maybe_print_dwa_weight_bdd_metrics(dwa: &DWA, domain_max: usize, name: &str) {
    if std::env::var("WEIGHT_BDD_METRICS").map(|v| v != "1").unwrap_or(true) {
        return;
    }

    use std::collections::HashMap;
    use std::ptr;

    // Collect unique weights by Arc pointer address.
    let mut unique: HashMap<usize, (Weight, usize)> = HashMap::new(); // ptr -> (weight, num_ranges)

    for state in &dwa.states.0 {
        if let Some(fw) = &state.final_weight {
            let p = ptr::addr_of!(**fw) as usize;
            unique.entry(p).or_insert_with(|| (fw.clone(), fw.num_ranges()));
        }
        for w in state.trans_weights.values() {
            let p = ptr::addr_of!(**w) as usize;
            unique.entry(p).or_insert_with(|| (w.clone(), w.num_ranges()));
        }
    }

    let unique_weights = unique.len();
    let total_ranges_unique: usize = unique.values().map(|(_, r)| *r).sum();
    let max_ranges = unique.values().map(|(_, r)| *r).max().unwrap_or(0);

    let k = bits_needed(domain_max);
    let vars = BddVariableSet::new_anonymous(k.try_into().unwrap());

    let mut total_bdd_nodes: u64 = 0;
    let mut max_bdd_nodes: u64 = 0;
    let mut total_build_ms: u128 = 0;

    for (w, _) in unique.values().map(|(w, r)| (w, r)) {
        let start = std::time::Instant::now();
        let bdd = weight_to_bdd(&vars, w, domain_max, k);
        let elapsed = start.elapsed();
        total_build_ms += elapsed.as_millis();

        let nodes = bdd.size() as u64;
        total_bdd_nodes += nodes;
        max_bdd_nodes = max_bdd_nodes.max(nodes);
    }

    crate::debug!(5, "[WEIGHT_BDD_METRICS] {}: domain_max={}, bits={} unique_weights={} total_ranges_unique={} max_ranges={} total_bdd_nodes={} max_bdd_nodes={} build_ms_total={} (avg {:.3}ms/weight)",
        name,
        domain_max,
        k,
        unique_weights,
        total_ranges_unique,
        max_ranges,
        total_bdd_nodes,
        max_bdd_nodes,
        total_build_ms,
        if unique_weights == 0 { 0.0 } else { total_build_ms as f64 / unique_weights as f64 },
    );
}
