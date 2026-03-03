//! Weight relaxation pass for RangeMapWeight DWAs.
//!
//! This pass reduces the number of outer range entries in RangeMapWeights
//! by computing forward reachability in the 2D RangeMapWeight space, then
//! WIDENING each weight by adding "dead" positions to merge adjacent entries.
//!
//! The approach is the 2D equivalent of consolidate_ranges.rs's gap filling:
//! - For adjacent tsid-range entries (R1,T1) and (R2,T2) in a weight:
//!   - Compute tokens_alive for R1 and R2 from forward reachability
//!   - Check if the sym_diff tokens (T1\T2 for R2, T2\T1 for R1) are dead
//!   - If all sym_diff positions are dead (not forward-reachable), merge
//!
//! This operates natively on RangeMapWeight (no flat expansion), so it's
//! fast even for weight-heavy mode where ConsolidateRanges is disabled.

use crate::datastructures::abstract_weight::AbstractWeight;
use crate::datastructures::hybrid_bitset::RangeSet;
use crate::datastructures::rangemap_weight::{intern_rangemap, RangeMapWeight};
use crate::datastructures::abstract_weight::WeightBackend;
use crate::dwa_i32::common::{Label, StateID, Weight};
use crate::dwa_i32::dwa::DWA;
use range_set_blaze::RangeSetBlaze;
use std::collections::VecDeque;
use std::sync::Arc;

impl DWA {
    /// Relax RangeMapWeights by widening entries with dead positions to enable merges.
    ///
    /// Returns true if any weights were modified.
    pub fn relax_rangemap_weights(&mut self) -> bool {
        if self.states.len() == 0 {
            return false;
        }

        // Check that we actually have RangeMapWeights
        let has_rangemap = self.states.0.iter().any(|s| {
            s.trans_weights.values().any(|w| matches!(w, AbstractWeight::RangeMap(_)))
        });
        if !has_rangemap {
            return false;
        }

        let before_ranges = self.count_rm_ranges();
        let num_tsids = self.infer_num_tsids();
        if num_tsids == 0 {
            return false;
        }

        let t0 = std::time::Instant::now();

        // Compute the "all" weight
        let all_rm = self.compute_all_rm(num_tsids);

        // Compute forward reachability only (widening needs forward, not backward)
        let forward = self.compute_forward_reach_2d(num_tsids, &all_rm);
        let t_fwd = t0.elapsed();

        // Widening pass: merge adjacent entries where sym_diff positions are dead
        let mut changed = false;
        let mut weights_modified = 0usize;
        let mut entries_saved = 0usize;
        let mut merges_blocked = 0usize;

        for state_id in 0..self.states.len() {
            let fwd = &forward[state_id];

            // Process transition weights
            let labels: Vec<Label> = self.states[state_id].transitions.keys().cloned().collect();
            for label in labels {
                if let Some(weight) = self.states[state_id].trans_weights.get(&label).cloned() {
                    if let AbstractWeight::RangeMap(ref rm) = weight {
                        let old_count = rm.map.range_values().count();
                        if old_count <= 1 {
                            continue;
                        }
                        if let Some((new_rm, saved, blocked)) = widen_merge_entries(rm, fwd) {
                            entries_saved += saved;
                            merges_blocked += blocked;
                            weights_modified += 1;
                            self.states.0[state_id].trans_weights.insert(
                                label,
                                AbstractWeight::RangeMap(intern_rangemap(new_rm)),
                            );
                            changed = true;
                        } else {
                            // No merges possible; count blocked
                        }
                    }
                }
            }

            // Process final weight
            if let Some(ref fw) = self.states[state_id].final_weight.clone() {
                if let AbstractWeight::RangeMap(ref rm) = fw {
                    let old_count = rm.map.range_values().count();
                    if old_count <= 1 {
                        continue;
                    }
                    if let Some((new_rm, saved, blocked)) = widen_merge_entries(rm, fwd) {
                        entries_saved += saved;
                        merges_blocked += blocked;
                        weights_modified += 1;
                        self.states.0[state_id].final_weight =
                            Some(AbstractWeight::RangeMap(intern_rangemap(new_rm)));
                        changed = true;
                    }
                }
            }
        }

        let after_ranges = self.count_rm_ranges();

        if std::env::var("ANALYZE_RANGEMAP_WEIGHTS").is_ok() || changed {
            eprintln!(
                "RELAX_RM: entries {} -> {} ({:.1}% reduction), {} weights modified, {} entries saved, {} merges blocked (fwd={:?})",
                before_ranges,
                after_ranges,
                if before_ranges > 0 {
                    (1.0 - after_ranges as f64 / before_ranges as f64) * 100.0
                } else {
                    0.0
                },
                weights_modified,
                entries_saved,
                merges_blocked,
                t_fwd,
            );
        }

        changed
    }

    /// Count total range entries across all RangeMapWeight instances in the DWA.
    fn count_rm_ranges(&self) -> usize {
        let mut total = 0;
        for state in &self.states.0 {
            for w in state.trans_weights.values() {
                if let AbstractWeight::RangeMap(rm) = w {
                    total += rm.map.range_values().count();
                }
            }
            if let Some(AbstractWeight::RangeMap(rm)) = &state.final_weight {
                total += rm.map.range_values().count();
            }
        }
        total
    }

    /// Infer num_tsids from the first RangeMapWeight found.
    fn infer_num_tsids(&self) -> usize {
        for state in &self.states.0 {
            for w in state.trans_weights.values() {
                if let AbstractWeight::RangeMap(rm) = w {
                    return rm.num_tsids();
                }
            }
            if let Some(AbstractWeight::RangeMap(rm)) = &state.final_weight {
                return rm.num_tsids();
            }
        }
        0
    }

    /// Compute a RangeMapWeight representing "all positions" in the DWA's domain.
    fn compute_all_rm(&self, num_tsids: usize) -> Arc<RangeMapWeight> {
        let mut max_token: usize = 0;
        for state in &self.states.0 {
            for w in state.trans_weights.values() {
                if let AbstractWeight::RangeMap(rm) = w {
                    max_token = max_token.max(rm_max_token(rm));
                }
            }
            if let Some(AbstractWeight::RangeMap(rm)) = &state.final_weight {
                max_token = max_token.max(rm_max_token(rm));
            }
        }

        let all_tokens = RangeSet::from(RangeSetBlaze::from_iter([0..=max_token]));
        let mut map = range_set_blaze::RangeMapBlaze::new();
        if RangeMapWeight::tsid_outer_enabled() {
            if num_tsids > 0 {
                map.ranges_insert(0..=(num_tsids - 1), all_tokens);
            }
        } else {
            let all_tsids = RangeSet::from(RangeSetBlaze::from_iter(
                [0..=(num_tsids.saturating_sub(1))],
            ));
            map.ranges_insert(0..=max_token, all_tsids);
        }
        intern_rangemap(RangeMapWeight::from_map(map, num_tsids))
    }

    /// Forward reachability: for each state, what positions can reach it from start.
    fn compute_forward_reach_2d(
        &self,
        num_tsids: usize,
        all_rm: &Arc<RangeMapWeight>,
    ) -> Vec<Arc<RangeMapWeight>> {
        let n = self.states.len();
        let empty = intern_rangemap(RangeMapWeight::new(num_tsids));
        let mut forward: Vec<Arc<RangeMapWeight>> = vec![empty.clone(); n];
        forward[self.body.start_state] = all_rm.clone();

        let mut queue = VecDeque::new();
        queue.push_back(self.body.start_state);
        let mut in_queue = vec![false; n];
        in_queue[self.body.start_state] = true;

        while let Some(state_id) = queue.pop_front() {
            in_queue[state_id] = false;

            if WeightBackend::is_empty(forward[state_id].as_ref()) {
                continue;
            }

            for (&label, &dest) in &self.states[state_id].transitions {
                let weight_rm = match self.states[state_id].trans_weights.get(&label) {
                    Some(AbstractWeight::RangeMap(rm)) => rm.clone(),
                    _ => all_rm.clone(),
                };

                let surviving: Arc<RangeMapWeight> = WeightBackend::intersect(&forward[state_id], &weight_rm);
                if WeightBackend::is_empty(surviving.as_ref()) {
                    continue;
                }

                let old_ptr = Arc::as_ptr(&forward[dest]);
                let new_fwd: Arc<RangeMapWeight> = WeightBackend::union(&forward[dest], &surviving);

                if Arc::as_ptr(&new_fwd) != old_ptr {
                    forward[dest] = new_fwd;
                    if !in_queue[dest] {
                        queue.push_back(dest);
                        in_queue[dest] = true;
                    }
                }
            }
        }

        forward
    }
}

/// Get the maximum token value in a RangeMapWeight.
fn rm_max_token(rm: &RangeMapWeight) -> usize {
    let mut max_t: usize = 0;
    if RangeMapWeight::tsid_outer_enabled() {
        for (_, token_set) in rm.map.range_values() {
            if let Some(r) = token_set.ranges().last() {
                max_t = max_t.max(*r.end());
            }
        }
    } else {
        if let Some((r, _)) = rm.map.range_values().last() {
            max_t = max_t.max(*r.end());
        }
    }
    max_t
}

/// Extract the union of token sets from a RangeMapWeight for all tsids in the given range.
/// In tsid-outer mode: iterate entries overlapping with tsid_range, union their token sets.
fn tokens_alive_for_tsid_range(
    rm: &RangeMapWeight,
    tsid_range: &std::ops::RangeInclusive<usize>,
) -> RangeSet {
    if RangeMapWeight::tsid_outer_enabled() {
        let mut result = RangeSet::zeros();
        for (entry_range, token_set) in rm.map.range_values() {
            // Check if entry_range overlaps with tsid_range
            if entry_range.start() <= tsid_range.end() && entry_range.end() >= tsid_range.start() {
                result |= token_set;
            }
            // Early exit: if we've passed the query range
            if entry_range.start() > tsid_range.end() {
                break;
            }
        }
        result
    } else {
        // token-outer mode: iterate all entries, check if their tsid_sets overlap
        let mut token_ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();
        for (token_range, tsid_set) in rm.map.range_values() {
            // Check if any tsid in tsid_range is in tsid_set
            let query_rs = RangeSet::from(RangeSetBlaze::from_iter([tsid_range.clone()]));
            let intersection = &query_rs & tsid_set;
            if !intersection.is_empty() {
                token_ranges.push(token_range.clone());
            }
        }
        RangeSet::from(RangeSetBlaze::from_iter(token_ranges))
    }
}

/// Try to merge adjacent entries in a RangeMapWeight by widening with dead positions.
///
/// For each pair of adjacent entries (R1,T1) and (R2,T2):
/// - T1\T2 added to R2's entry: safe if (T1\T2) is dead for all tsids in R2
/// - T2\T1 added to R1's entry: safe if (T2\T1) is dead for all tsids in R1
/// "Dead" = not in forward_rm for those tsids
///
/// Returns Some((new_weight, entries_saved, merges_blocked)) if any merges were made.
fn widen_merge_entries(
    rm: &RangeMapWeight,
    forward_rm: &Arc<RangeMapWeight>,
) -> Option<(RangeMapWeight, usize, usize)> {
    if !RangeMapWeight::tsid_outer_enabled() {
        return widen_merge_entries_token_outer(rm, forward_rm);
    }

    let entries: Vec<(std::ops::RangeInclusive<usize>, &RangeSet)> =
        rm.map.range_values().collect();

    if entries.len() <= 1 {
        return None;
    }

    // Greedy left-to-right merge
    let mut merged_entries: Vec<(std::ops::RangeInclusive<usize>, RangeSet)> = Vec::new();
    let mut merges = 0usize;
    let mut blocked = 0usize;

    // Start with first entry
    let mut current_range = entries[0].0.clone();
    let mut current_tokens = entries[0].1.clone().clone();

    for i in 1..entries.len() {
        let next_range = entries[i].0.clone();
        let next_tokens = entries[i].1.clone().clone();

        // Check if these adjacent entries can be merged
        let extra_for_current = &next_tokens - &current_tokens; // T2 \ T1: added to current
        let extra_for_next = &current_tokens - &next_tokens; // T1 \ T2: added to next

        let mut can_merge = true;

        // Check: extra_for_current must be dead for all tsids in current_range
        if !extra_for_current.is_empty() {
            let alive_current = tokens_alive_for_tsid_range(forward_rm, &current_range);
            let conflict = &extra_for_current & &alive_current;
            if !conflict.is_empty() {
                can_merge = false;
            }
        }

        // Check: extra_for_next must be dead for all tsids in next_range
        if can_merge && !extra_for_next.is_empty() {
            let alive_next = tokens_alive_for_tsid_range(forward_rm, &next_range);
            let conflict = &extra_for_next & &alive_next;
            if !conflict.is_empty() {
                can_merge = false;
            }
        }

        if can_merge {
            // Merge: extend current range, union token sets
            current_range = *current_range.start()..=*next_range.end();
            current_tokens = &current_tokens | &next_tokens;
            merges += 1;
        } else {
            // Can't merge: emit current, start new
            merged_entries.push((current_range, current_tokens));
            current_range = next_range;
            current_tokens = next_tokens;
            blocked += 1;
        }
    }
    // Emit final entry
    merged_entries.push((current_range, current_tokens));

    if merges == 0 {
        return None;
    }

    // Build new RangeMapWeight
    let mut new_map = range_set_blaze::RangeMapBlaze::new();
    for (range, tokens) in merged_entries {
        new_map.ranges_insert(range, tokens);
    }

    Some((RangeMapWeight::from_map(new_map, rm.num_tsids()), merges, blocked))
}

/// Token-outer mode variant of widen_merge_entries.
fn widen_merge_entries_token_outer(
    rm: &RangeMapWeight,
    forward_rm: &Arc<RangeMapWeight>,
) -> Option<(RangeMapWeight, usize, usize)> {
    // In token-outer mode: entries are (token_range → tsid_set)
    // Adjacent token ranges with different tsid sets can be merged if the
    // extra tsids are dead for those tokens.
    // For now, skip this mode (tsid-outer is default and where the benefit is)
    let _ = (rm, forward_rm);
    None
}
