// src/constraint_precompute3_intermediate_utils.rs
use crate::constraint::{
    IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey,
    IntermediateTrie3GodWrapper, LLMTokenBV,
};

/// Normalizes a path for comparison purposes.
/// - Removes NoOp edges.
/// - Collects all CheckLLM bitvectors, intersects them, and prepends a single CheckLLM.
pub(crate) fn normalize_path(path: Vec<IntermediateTrie3EdgeKey>) -> Vec<IntermediateTrie3EdgeKey> {
    let mut combined_llm_bv = LLMTokenBV::max_ones();
    let mut has_llm_check = false;

    let mut other_ops: Vec<IntermediateTrie3EdgeKey> = path
        .into_iter()
        .filter(|ek| {
            if let IntermediateTrie3EdgeKey::CheckLLM(bv) = ek {
                combined_llm_bv &= bv;
                has_llm_check = true;
                false // remove from path
            } else {
                !matches!(ek, IntermediateTrie3EdgeKey::NoOp)
            }
        })
        .collect();

    if has_llm_check {
        other_ops.insert(0, IntermediateTrie3EdgeKey::CheckLLM(combined_llm_bv));
    }

    other_ops
}

pub fn optimize_intermediate_trie3(
    roots: &mut Vec<IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    todo!()
}

/// Runs a global optimization across all per-terminal templates.
/// Pins all (start,end) nodes to keep external references valid.
pub fn optimize_intermediate_trie3_templates_global(
    templates: &mut Vec<(
        IntermediatePrecomputeNode3Index,
        IntermediatePrecomputeNode3Index,
    )>,
    god: &IntermediateTrie3GodWrapper,
) {
    todo!()
}

fn compute_and_print_template_stats(
    templates: &[(
        IntermediatePrecomputeNode3Index,
        IntermediatePrecomputeNode3Index,
    )],
    god: &IntermediateTrie3GodWrapper,
    phase: &str,
) {
    todo!()
}

