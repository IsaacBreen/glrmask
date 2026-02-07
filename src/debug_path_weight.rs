use rustc_hash::FxHashMap;

use crate::dwa_i32::common::{Label, NWAStateID, Weight};
use crate::dwa_i32::determinization_acyclic::{
    precompute_all_epsilon_closures_acyclic, topo_order_if_acyclic,
};
use crate::dwa_i32::dwa::DWA;
use crate::dwa_i32::nwa::NWA;

/// Check the weight of a given path through a DWA.
///
/// This is a thin wrapper around `eval_word_weight`.
pub fn check_dwa_path_weight(dwa: &DWA, path: &[Label]) -> Weight {
    dwa.eval_word_weight(path)
}

/// Check the weight of a given path through a DWA, ignoring final weights.
pub fn check_dwa_path_weight_no_final(dwa: &DWA, path: &[Label]) -> Weight {
    if dwa.states.0.is_empty() {
        return Weight::zeros();
    }
    let mut s = dwa.body.start_state;
    let mut acc = Weight::all();
    if s >= dwa.states.len() {
        return Weight::zeros();
    }
    for &ch in path {
        if s >= dwa.states.len() {
            return Weight::zeros();
        }
        if let Some((t, w)) = dwa.states[s].get_transition(ch) {
            acc &= w;
            if acc.is_empty() {
                return Weight::zeros();
            }
            s = t;
        } else {
            return Weight::zeros();
        }
    }
    acc
}

/// Parse debug path-weight settings from environment variables.
///
/// Expected:
/// - DEBUG_PATH_WEIGHT_TOKEN: usize
/// - DEBUG_PATH_WEIGHT_LABELS: comma-separated i32 labels (e.g., "34,20,4")
pub fn parse_debug_path_weight_env() -> Option<(usize, Vec<Label>)> {
    let token_id = std::env::var("DEBUG_PATH_WEIGHT_TOKEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())?;
    let labels_str = std::env::var("DEBUG_PATH_WEIGHT_LABELS").ok()?;
    let mut labels = Vec::new();
    for part in labels_str.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(label) = trimmed.parse::<Label>() {
            labels.push(label);
        }
    }
    if labels.is_empty() {
        return None;
    }
    Some((token_id, labels))
}

/// Return true if DEBUG_PATH_WEIGHT_IGNORE_FINAL is set.
pub fn debug_path_weight_ignore_final() -> bool {
    std::env::var("DEBUG_PATH_WEIGHT_IGNORE_FINAL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Check if a token ID is present in a weight (handles weight-heavy tsid encoding).
pub fn weight_contains_token(weight: &Weight, token_id: usize, num_tsids: usize) -> bool {
    if num_tsids <= 1 {
        weight.contains(token_id)
    } else {
        let start = token_id.saturating_mul(num_tsids);
        let end = start.saturating_add(num_tsids.saturating_sub(1));
        for range in weight.ranges() {
            let r_start = *range.start();
            let r_end = *range.end();
            if r_start > end {
                break;
            }
            if r_end >= start {
                return true;
            }
        }
        false
    }
}

/// Check the weight of a given path through an acyclic NWA.
///
/// Walks all paths matching the label sequence, following epsilon transitions
/// and accumulating weights across branches.
pub fn check_nwa_path_weight(nwa: &NWA, path: &[Label]) -> Weight {
    check_nwa_path_weight_with_final(nwa, path, true)
}

/// Check the weight of a given path through an acyclic NWA, ignoring final weights.
pub fn check_nwa_path_weight_no_final(nwa: &NWA, path: &[Label]) -> Weight {
    check_nwa_path_weight_with_final(nwa, path, false)
}

fn check_nwa_path_weight_with_final(nwa: &NWA, path: &[Label], include_final: bool) -> Weight {
    let topo = topo_order_if_acyclic(nwa).expect("NWA must be acyclic");
    let eps_reach = precompute_all_epsilon_closures_acyclic(&nwa.states, &topo);

    let mut current: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
    for &s in &nwa.body.start_states {
        if s < eps_reach.len() {
            for (v, w_reach) in &eps_reach[s] {
                current
                    .entry(*v)
                    .and_modify(|acc| *acc |= w_reach)
                    .or_insert_with(|| w_reach.clone());
            }
        }
    }

    for &label in path {
        let mut next: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
        for (u, w_u) in &current {
            if *u >= nwa.states.len() {
                continue;
            }
            if let Some(targets) = nwa.states[*u].transitions.get(&label) {
                for (v, w_uv) in targets {
                    let combined = w_u & w_uv;
                    if combined.is_empty() {
                        continue;
                    }
                    let entry = next.entry(*v).or_insert_with(Weight::zeros);
                    if !combined.is_subset_of(entry) {
                        *entry |= &combined;
                    }
                }
            }
        }

        if next.is_empty() {
            return Weight::zeros();
        }

        let mut expanded: FxHashMap<NWAStateID, Weight> = FxHashMap::default();
        for (v, w_v) in next {
            if v >= eps_reach.len() {
                continue;
            }
            for (v_reach, w_reach) in &eps_reach[v] {
                let combined = &w_v & w_reach;
                if combined.is_empty() {
                    continue;
                }
                let entry = expanded.entry(*v_reach).or_insert_with(Weight::zeros);
                if !combined.is_subset_of(entry) {
                    *entry |= &combined;
                }
            }
        }

        if expanded.is_empty() {
            return Weight::zeros();
        }

        current = expanded;
    }

    if !include_final {
        let mut result = Weight::zeros();
        for (_sid, w) in current {
            if w.is_empty() {
                continue;
            }
            result |= &w;
        }
        return result;
    }

    let mut result = Weight::zeros();
    for (sid, w) in current {
        if sid >= nwa.states.len() {
            continue;
        }
        if let Some(fw) = &nwa.states[sid].final_weight {
            let combined = &w & fw;
            if combined.is_empty() {
                continue;
            }
            result |= &combined;
        }
    }

    result
}
