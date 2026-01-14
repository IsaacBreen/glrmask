#![allow(dead_code)]

use crate::dwa_i32::{DWA, Weight};

use oxidd::Manager;
use oxidd::ManagerRef;
use oxidd::Edge;
use oxidd::InnerNode;

fn bits_needed(max_value: usize) -> usize {
    let bits = (usize::BITS - max_value.leading_zeros()) as usize;
    bits.max(1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VarOrder {
    MsbFirst,
    LsbFirst,
}

impl VarOrder {
    fn from_env() -> Self {
        match std::env::var("WEIGHT_OXIDD_VAR_ORDER").as_deref() {
            Ok("lsb") | Ok("LSB") | Ok("lsb_first") => Self::LsbFirst,
            _ => Self::MsbFirst,
        }
    }

    fn var_index(self, depth_from_msb: usize, total_bits: usize) -> usize {
        match self {
            Self::MsbFirst => depth_from_msb,
            Self::LsbFirst => total_bits - 1 - depth_from_msb,
        }
    }
}

fn env_usize(name: &str, default_value: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default_value)
}

fn env_kinds() -> Vec<&'static str> {
    let s = std::env::var("WEIGHT_OXIDD_KINDS").ok();
    let Some(s) = s else {
        return vec!["bdd", "bcdd", "zbdd"];
    };
    let mut out = Vec::new();
    for part in s.split(',').map(|p| p.trim().to_ascii_lowercase()) {
        match part.as_str() {
            "bdd" => out.push("bdd"),
            "bcdd" => out.push("bcdd"),
            "zbdd" => out.push("zbdd"),
            _ => {}
        }
    }
    if out.is_empty() {
        vec!["bdd", "bcdd", "zbdd"]
    } else {
        out
    }
}

fn interval_inclusive<F: oxidd::BooleanFunction>(
    lo: usize,
    hi: usize,
    remaining_bits: usize,
    depth_from_msb: usize,
    total_bits: usize,
    order: VarOrder,
    t: &F,
    f: &F,
    vars: &[F],
    not_vars: &[F],
) -> Result<F, oxidd::util::OutOfMemory> {
    debug_assert!(lo <= hi);
    if remaining_bits == 0 {
        return if lo == 0 && hi == 0 {
            Ok(t.clone())
        } else {
            Ok(f.clone())
        };
    }

    let full_hi = if remaining_bits >= usize::BITS as usize {
        usize::MAX
    } else {
        (1usize << remaining_bits) - 1
    };

    if lo == 0 && hi == full_hi {
        return Ok(t.clone());
    }

    let msb_mask = 1usize << (remaining_bits - 1);
    let lo_msb = (lo & msb_mask) != 0;
    let hi_msb = (hi & msb_mask) != 0;

    let var_index = order.var_index(depth_from_msb, total_bits);
    let x = &vars[var_index];
    let nx = &not_vars[var_index];

    match (lo_msb, hi_msb) {
        (false, false) => {
            let child = interval_inclusive(
                lo,
                hi,
                remaining_bits - 1,
                depth_from_msb + 1,
                total_bits,
                order,
                t,
                f,
                vars,
                not_vars,
            )?;
            nx.and(&child)
        }
        (true, true) => {
            let child = interval_inclusive(
                lo - msb_mask,
                hi - msb_mask,
                remaining_bits - 1,
                depth_from_msb + 1,
                total_bits,
                order,
                t,
                f,
                vars,
                not_vars,
            )?;
            x.and(&child)
        }
        (false, true) => {
            let left_hi = msb_mask - 1;
            let left_child = interval_inclusive(
                lo,
                left_hi,
                remaining_bits - 1,
                depth_from_msb + 1,
                total_bits,
                order,
                t,
                f,
                vars,
                not_vars,
            )?;
            let right_child = interval_inclusive(
                0,
                hi - msb_mask,
                remaining_bits - 1,
                depth_from_msb + 1,
                total_bits,
                order,
                t,
                f,
                vars,
                not_vars,
            )?;

            let left = nx.and(&left_child)?;
            let right = x.and(&right_child)?;
            left.or(&right)
        }
        (true, false) => Ok(f.clone()),
    }
}

fn weight_to_dd<F: oxidd::BooleanFunction>(
    weight: &Weight,
    domain_max: usize,
    total_bits: usize,
    order: VarOrder,
    t: &F,
    f: &F,
    vars: &[F],
    not_vars: &[F],
) -> Result<F, oxidd::util::OutOfMemory> {
    if weight.is_empty() {
        return Ok(f.clone());
    }

    // Keep parity with existing biodivine metric: treat ALL as ⊤, even though
    // this includes assignments above domain_max.
    if weight.is_all_fast() {
        return Ok(t.clone());
    }

    let mut acc = f.clone();

    for r in weight.rsb().ranges() {
        let start = *r.start();
        let end = *r.end();

        if start > domain_max {
            continue;
        }
        let clipped_end = end.min(domain_max);

        let interval = interval_inclusive(
            start,
            clipped_end,
            total_bits,
            0,
            total_bits,
            order,
            t,
            f,
            vars,
            not_vars,
        )?;

        acc = acc.or(&interval)?;

        if acc.valid() {
            break;
        }
    }

    Ok(acc)
}

struct OxiddStats {
    total_fn_nodes: u64,
    max_fn_nodes: u64,
    reachable_union_nodes: u64,
    reachable_union_inner_nodes: u64,
    manager_inner_nodes: u64,
    manager_inner_nodes_approx: u64,
    build_ms_total: u128,
}

fn compute_stats_for<F>(
    name: &str,
    manager_ref: &<F as oxidd::Function>::ManagerRef,
    unique_weights: &[Weight],
    domain_max: usize,
    order: VarOrder,
) -> Result<OxiddStats, oxidd::util::OutOfMemory>
where
    F: oxidd::BooleanFunction,
{
    let total_bits = bits_needed(domain_max);

    manager_ref.with_manager_exclusive(|manager| {
        let _ = manager.add_vars(total_bits as u32);
    });

    let (t, f, vars, not_vars): (F, F, Vec<F>, Vec<F>) = manager_ref.with_manager_shared(|m| {
        let t = F::t(m);
        let f = F::f(m);
        let vars: Vec<F> = (0..total_bits)
            .map(|i| F::var(m, i as u32))
            .collect::<Result<Vec<_>, _>>()?;
        let not_vars: Vec<F> = (0..total_bits)
            .map(|i| F::not_var(m, i as u32))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((t, f, vars, not_vars))
    })?;

    let mut built: Vec<F> = Vec::with_capacity(unique_weights.len());
    let mut total_fn_nodes: u64 = 0;
    let mut max_fn_nodes: u64 = 0;
    let mut total_build_ms: u128 = 0;

    for w in unique_weights {
        let start = std::time::Instant::now();
        let func = weight_to_dd::<F>(w, domain_max, total_bits, order, &t, &f, &vars, &not_vars)?;
        total_build_ms += start.elapsed().as_millis();

        let nodes = func.node_count() as u64;
        total_fn_nodes += nodes;
        max_fn_nodes = max_fn_nodes.max(nodes);
        built.push(func);
    }

    let (reachable_union_nodes, reachable_union_inner_nodes) = manager_ref.with_manager_shared(|m| {
        fn visit<M: Manager>(
            manager: &M,
            edge: &M::Edge,
            seen: &mut std::collections::HashSet<oxidd::NodeID>,
            inner_count: &mut usize,
        ) {
            if !seen.insert(edge.node_id()) {
                return;
            }

            if let oxidd::Node::Inner(node) = manager.get_node(edge) {
                *inner_count += 1;
                for child in node.children() {
                    visit(manager, &*child, seen, inner_count);
                }
            }
        }

        // Note: We intentionally track reachability from roots, since the manager
        // does not necessarily garbage-collect dead intermediate nodes created
        // during repeated `and/or` operations.
        let mut set: std::collections::HashSet<oxidd::NodeID> = Default::default();
        let mut inner_count = 0usize;

        for func in &built {
            visit(m, func.as_edge(m), &mut set, &mut inner_count);
        }

        (set.len() as u64, inner_count as u64)
    });

    let (manager_inner_nodes, manager_inner_nodes_approx) = manager_ref.with_manager_shared(|m| {
        (m.num_inner_nodes() as u64, m.approx_num_inner_nodes() as u64)
    });

    // Keep `built` alive until after node-count query.
    std::mem::drop(built);

    crate::debug!(5,
        "[WEIGHT_OXIDD_METRICS] {} {}: domain_max={} bits={} order={:?} unique_weights={} total_fn_nodes={} max_fn_nodes={} reachable_union_nodes={} reachable_union_inner_nodes={} manager_inner_nodes={} approx={} build_ms_total={} (avg {:.3}ms/weight)",
        name,
        F::REPR_ID,
        domain_max,
        total_bits,
        order,
        unique_weights.len(),
        total_fn_nodes,
        max_fn_nodes,
        reachable_union_nodes,
        reachable_union_inner_nodes,
        manager_inner_nodes,
        manager_inner_nodes_approx,
        total_build_ms,
        if unique_weights.is_empty() { 0.0 } else { total_build_ms as f64 / unique_weights.len() as f64 },
    );

    Ok(OxiddStats {
        total_fn_nodes,
        max_fn_nodes,
        reachable_union_nodes,
        reachable_union_inner_nodes,
        manager_inner_nodes,
        manager_inner_nodes_approx,
        build_ms_total: total_build_ms,
    })
}

/// Shared-manager DD metrics for all unique (interned) weights in a DWA.
///
/// Enabled only when `WEIGHT_OXIDD_METRICS=1`.
///
/// This differs from the existing biodivine BDD metric in that it uses a single manager
/// for all weights, so node sharing across weights is visible via `manager_inner_nodes`.
pub fn maybe_print_dwa_weight_oxidd_metrics(dwa: &DWA, domain_max: usize, name: &str) {
    if std::env::var("WEIGHT_OXIDD_METRICS")
        .map(|v| v != "1")
        .unwrap_or(true)
    {
        return;
    }

    use std::collections::HashMap;
    use std::ptr;

    // Collect unique weights by Arc pointer address.
    let mut unique: HashMap<usize, Weight> = HashMap::new();

    for state in &dwa.states.0 {
        if let Some(fw) = &state.final_weight {
            let p = fw.intern_id();
            unique.entry(p).or_insert_with(|| fw.clone());
        }
        for w in state.trans_weights.values() {
            let p = w.intern_id();
            unique.entry(p).or_insert_with(|| w.clone());
        }
    }

    let unique_weights: Vec<Weight> = unique.into_values().collect();

    let total_ranges_unique: usize = unique_weights.iter().map(|w| w.num_ranges()).sum();
    let max_ranges = unique_weights.iter().map(|w| w.num_ranges()).max().unwrap_or(0);

    crate::debug!(5,
        "[WEIGHT_OXIDD_METRICS] {}: domain_max={} bits={} unique_weights={} total_ranges_unique={} max_ranges={} (kinds={:?})",
        name,
        domain_max,
        bits_needed(domain_max),
        unique_weights.len(),
        total_ranges_unique,
        max_ranges,
        env_kinds(),
    );

    let order = VarOrder::from_env();

    let inner_cap = env_usize("WEIGHT_OXIDD_INNER_CAP", 2_000_000);
    let cache_cap = env_usize("WEIGHT_OXIDD_APPLY_CACHE_CAP", 1_000_000);
    let threads = env_usize("WEIGHT_OXIDD_THREADS", 1).min(u32::MAX as usize) as u32;

    for kind in env_kinds() {
        let mut attempt_inner_cap = inner_cap;
        let mut attempt_cache_cap = cache_cap;
        for attempt in 0..3 {
            let res: Result<(), oxidd::util::OutOfMemory> = match kind {
                "bdd" => {
                    let manager_ref = oxidd::bdd::new_manager(attempt_inner_cap, attempt_cache_cap, threads);
                    compute_stats_for::<oxidd::bdd::BDDFunction>(name, &manager_ref, &unique_weights, domain_max, order)
                        .map(|_| ())
                }
                "bcdd" => {
                    let manager_ref = oxidd::bcdd::new_manager(attempt_inner_cap, attempt_cache_cap, threads);
                    compute_stats_for::<oxidd::bcdd::BCDDFunction>(name, &manager_ref, &unique_weights, domain_max, order)
                        .map(|_| ())
                }
                "zbdd" => {
                    let manager_ref = oxidd::zbdd::new_manager(attempt_inner_cap, attempt_cache_cap, threads);
                    compute_stats_for::<oxidd::zbdd::ZBDDFunction>(name, &manager_ref, &unique_weights, domain_max, order)
                        .map(|_| ())
                }
                _ => Ok(()),
            };

            match res {
                Ok(()) => break,
                Err(e) => {
                    crate::debug!(5,
                        "[WEIGHT_OXIDD_METRICS] {} {}: out of memory (attempt {}), retrying with larger caps (inner={}, cache={})",
                        name,
                        kind,
                        attempt + 1,
                        attempt_inner_cap,
                        attempt_cache_cap,
                    );
                    // Exponential-ish backoff.
                    attempt_inner_cap = attempt_inner_cap.saturating_mul(2).saturating_add(1);
                    attempt_cache_cap = attempt_cache_cap.saturating_mul(2).saturating_add(1);
                    if attempt == 2 {
                        crate::debug!(5, "[WEIGHT_OXIDD_METRICS] {} {}: giving up after 3 attempts ({:?})", name, kind, e);
                    }
                }
            }
        }
    }
}
