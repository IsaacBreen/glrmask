//! Local data carriers for Parser-DWA construction.
//!
//! These types are deliberately not exported from the crate.  They name the
//! intermediate mathematical objects used by the construction so that the
//! builder does not collapse into a long procedural script.

use std::collections::BTreeMap;

use smallvec::SmallVec;

use crate::automata::weighted::dwa::DWA;
use crate::sets::weight::Weight;
use crate::grammar::flat::TerminalID;

/// A finite weighted set of terminals that all enter the same Terminal-DWA
/// target state from a single source state.
///
/// Denotationally, this is one branch of the Terminal-DWA transition relation
/// after grouping by target state.  Each terminal carries the lexer-state/token
/// pair mask supplied by the Terminal DWA.
pub(crate) type TerminalBundle = BTreeMap<TerminalID, Weight>;

/// Canonical key used to identify duplicate terminal bundles.
pub(crate) type BundleSignature = Vec<(TerminalID, Weight)>;

/// During weighted subset construction, many labels contribute to a small set
/// of target states.  SmallVec keeps that common case allocation-free.
pub(crate) type TargetContribs = SmallVec<[(u32, Weight); 4]>;

/// Add a weighted contribution to a target in a compact target-contribution
/// list, unioning weights when the target already exists.
pub(crate) fn add_target_contribution(contribs: &mut TargetContribs, target: u32, add: Weight) {
    if add.is_empty() {
        return;
    }

    if let Some((_, existing)) = contribs
        .iter_mut()
        .find(|(existing_target, _)| *existing_target == target)
    {
        *existing = existing.union(&add);
    } else {
        contribs.push((target, add));
    }
}

/// Append all contributions from `src` into `dst`, preserving target-level
/// weight unioning.
pub(crate) fn extend_target_contribs(dst: &mut TargetContribs, src: &TargetContribs) {
    for (target, weight) in src {
        add_target_contribution(dst, *target, weight.clone());
    }
}

/// A summarized Terminal-DWA branch after grouping outgoing terminal labels by
/// target state and interning equal terminal bundles.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Branch {
    pub(crate) target: u32,
    pub(crate) bundle_id: usize,
}

/// Summary of one Terminal-DWA state as seen by Parser-DWA construction.
#[derive(Debug, Clone)]
pub(crate) struct StateSummary {
    pub(crate) final_weight: Option<Weight>,
    pub(crate) branches: Vec<Branch>,
}

/// Terminal-DWA summaries plus the interned bundle table.
#[derive(Debug, Clone)]
pub(crate) struct StateSummaries {
    pub(crate) states: Vec<StateSummary>,
    pub(crate) unique_bundles: Vec<TerminalBundle>,
    pub(crate) bundle_accepts: Vec<bool>,
}

/// Result of determinizing a parser NWA while retaining, for each determinized
/// DWA state, the source NWA support set that produced it.
#[derive(Debug, Clone)]
pub(crate) struct DeterminizedDwaWithSupports {
    pub(crate) dwa: DWA,
    pub(crate) supports: Vec<Vec<u32>>,
}

/// Memoized local epsilon-closure result.
#[derive(Debug, Clone)]
pub(crate) struct CachedClosure {
    pub(crate) canon: Vec<(u32, Weight)>,
    pub(crate) edge_weight: Weight,
}

/// Compact description of which parser-state labels can leave a DWA state.
/// This is used to preserve default-edge semantics during the second
/// determinization pass.
pub(crate) enum PossibleOutgoingIds {
    Empty,
    All,
    Some(crate::sets::bitset::BitSet),
}
