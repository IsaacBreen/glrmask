//! Weighted-DWA minimization entry points.
//!
//! These wrappers only minimize acyclic inputs. Cyclic DWAs are returned
//! unchanged and handled by the caller.
use super::dwa::DWA;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MinimizeStrategy {
    Full,
    Fast,
    Threshold(usize),
}

fn should_skip_minimization(dwa: &DWA) -> bool {
    dwa.states().is_empty() || !dwa.is_acyclic()
}

fn minimize_if_acyclic(dwa: &DWA, minimize: impl FnOnce(&DWA) -> DWA) -> DWA {
    if should_skip_minimization(dwa) {
        return dwa.clone();
    }

    minimize(dwa)
}

pub fn minimize(dwa: &DWA) -> DWA {
    minimize_if_acyclic(dwa, super::minimize_acyclic::minimize_acyclic)
}

/// Like [`minimize`], but switches from the O(n²) incompatibility graph to
/// partition-refinement coloring when any height bucket exceeds
/// `partition_refine_threshold` candidates. Produces a slightly larger DWA
/// for those buckets but avoids the quadratic cost.
pub fn minimize_with_threshold(dwa: &DWA, partition_refine_threshold: usize) -> DWA {
    minimize_if_acyclic(dwa, |dwa| {
        super::minimize_acyclic::minimize_acyclic_with_threshold(dwa, partition_refine_threshold)
    })
}

/// Minimize using partition refinement + disjoint-domain merging, without push_weights.
/// Much faster than `minimize` for DWAs where all mergeable state pairs have disjoint
/// token domains (e.g. the global merge DWA built from partitioned vocabularies).
/// Falls back to the caller's DWA unchanged if the input is cyclic.
pub fn minimize_partition_refine(dwa: &DWA) -> DWA {
    minimize_if_acyclic(dwa, super::minimize_acyclic::minimize_acyclic_partition_refine)
}


/// O(n²) pairwise graph coloring. Produces a valid (correct) DWA that may
/// be slightly larger than the graph-coloring result (doesn't merge states
/// with overlapping needed sets). Suitable for bundle minimization where
/// the extra few states are acceptable.
pub fn minimize_fast(dwa: &DWA) -> DWA {
    minimize_if_acyclic(dwa, super::minimize_acyclic::minimize_acyclic_fast)
}

fn parse_minimize_strategy(env_var_name: &str) -> Option<MinimizeStrategy> {
    let value = match std::env::var(env_var_name) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => return None,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("{env_var_name} must be valid UTF-8")
        }
    };

    let value = value.trim();
    if value.eq_ignore_ascii_case("full") {
        return Some(MinimizeStrategy::Full);
    }
    if value.eq_ignore_ascii_case("fast") {
        return Some(MinimizeStrategy::Fast);
    }
    if let Some(rest) = value.strip_prefix("threshold:") {
        let threshold = rest.parse::<usize>().unwrap_or_else(|_| {
            panic!(
                "{env_var_name} must be one of: full, fast, threshold:<n>; got {value}"
            )
        });
        return Some(MinimizeStrategy::Threshold(threshold));
    }

    panic!(
        "{env_var_name} must be one of: full, fast, threshold:<n>; got {value}"
    );
}

pub fn minimize_from_env(
    dwa: &DWA,
    env_var_name: &str,
    default_behavior: impl FnOnce(&DWA) -> DWA,
) -> DWA {
    match parse_minimize_strategy(env_var_name) {
        Some(MinimizeStrategy::Full) => minimize(dwa),
        Some(MinimizeStrategy::Fast) => minimize_fast(dwa),
        Some(MinimizeStrategy::Threshold(threshold)) => minimize_with_threshold(dwa, threshold),
        None => default_behavior(dwa),
    }
}
