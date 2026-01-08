//! Range consolidation pass for DWAs.
//!
//! This pass reduces the number of ranges in weights by:
//! 1. Computing forward reachability: which tokens can reach each state from start
//! 2. Computing backward reachability: which tokens can lead to acceptance from each state
//! 3. Removing ranges that don't intersect with relevant reachable tokens
//! 4. Filling gaps between ranges when safe (gap doesn't intersect reachable tokens)
//!
//! The key insight is that a range in a weight only matters if tokens in that range
//! are actually reachable through the automaton. Unreachable tokens can be safely
//! removed or added without affecting semantics.

use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use range_set_blaze::RangeSetBlaze;

impl DWA {
    /// Consolidate ranges by removing/adding ranges based on reachability analysis.
    /// 
    /// This is a semantic-preserving optimization that:
    /// 1. Computes forward reachability (tokens that can reach each state)
    /// 2. Computes backward reachability (tokens that can lead to acceptance)
    /// 3. For each weight, removes ranges that don't intersect relevant tokens
    /// 4. Fills gaps between ranges when safe
    pub fn consolidate_ranges(&mut self) -> bool {
        if self.states.len() == 0 {
            return false;
        }
        
        crate::debug!(5, "ConsolidateRanges: Analyzing {} states", self.states.len());
        
        let before_ranges = self.num_ranges_interned();
        let before_unique = self.count_unique_weights();
        
        // Run the analysis (for debugging)
        self.analyze_weights();
        self.analyze_weight_structure();
        
        // Compute forward and backward reachability
        let forward_reach = self.compute_forward_reachability();
        let backward_reach = self.compute_backward_reachability();
        
        let mut changed = false;
        let mut ranges_removed = 0usize;
        let mut gaps_filled = 0usize;
        
        // Process each state's weights
        for state_id in 0..self.states.len() {
            let reach_here = &forward_reach[state_id];
            
            // Process transition weights
            let labels: Vec<Label> = self.states[state_id].transitions.keys().cloned().collect();
            for label in labels {
                let dest = self.states[state_id].transitions[&label];
                let reach_dest = &backward_reach[dest];
                
                // For removing ranges: token must be reachable from start AND can finish from dest
                let tokens_for_removal = reach_here & reach_dest;
                // For filling gaps: token must not be reachable from start
                let tokens_for_gap_fill = reach_here;
                
                if let Some(weight) = self.states[state_id].trans_weights.get(&label).cloned() {
                    let (new_weight, removed, filled) = optimize_weight_ranges(&weight, &tokens_for_removal, tokens_for_gap_fill);
                    
                    if removed > 0 || filled > 0 {
                        ranges_removed += removed;
                        gaps_filled += filled;
                        changed = true;
                        self.states.0[state_id].trans_weights.insert(label, new_weight);
                    }
                }
            }
            
            // Process final weight
            if let Some(fw) = self.states[state_id].final_weight.clone() {
                // For final weight: forward reachability matters for both removal and gap filling
                let (new_fw, removed, filled) = optimize_weight_ranges(&fw, reach_here, reach_here);
                
                if removed > 0 || filled > 0 {
                    ranges_removed += removed;
                    gaps_filled += filled;
                    changed = true;
                    if new_fw.is_empty() {
                        self.states.0[state_id].final_weight = None;
                    } else {
                        self.states.0[state_id].final_weight = Some(new_fw);
                    }
                }
            }
        }
        
        // Remove transitions with empty weights
        for state in &mut self.states.0 {
            let empty_labels: Vec<Label> = state.trans_weights
                .iter()
                .filter(|(_, w)| w.is_empty())
                .map(|(l, _)| *l)
                .collect();
            
            for label in empty_labels {
                state.transitions.remove(&label);
                state.trans_weights.remove(&label);
                changed = true;
            }
        }
        
        let after_ranges = self.num_ranges_interned();
        let after_unique = self.count_unique_weights();
        
        crate::debug!(5, "ConsolidateRanges: {} -> {} unique-weight ranges ({} -> {} unique weights)", 
            before_ranges, after_ranges, before_unique, after_unique);
        if ranges_removed > 0 || gaps_filled > 0 {
            crate::debug!(5, "  Removed {} ranges, filled {} gaps", ranges_removed, gaps_filled);
        }
        
        changed
    }
    
    /// Compute forward reachability: for each state, which tokens can reach it from start.
    fn compute_forward_reachability(&self) -> Vec<RangeSetBlaze<usize>> {
        let n = self.states.len();
        let mut reach: Vec<RangeSetBlaze<usize>> = vec![RangeSetBlaze::new(); n];
        
        // Start state can be reached by all tokens initially
        // Actually, we need to be more careful - we track which tokens reach each state
        // Initially, start state has "all" tokens
        reach[self.body.start_state] = RangeSetBlaze::from_iter([0..=usize::MAX]);
        
        // BFS from start
        let mut queue = VecDeque::new();
        queue.push_back(self.body.start_state);
        let mut in_queue = vec![false; n];
        in_queue[self.body.start_state] = true;
        
        while let Some(state_id) = queue.pop_front() {
            in_queue[state_id] = false;
            
            let current_reach = reach[state_id].clone();
            
            for (&label, &dest) in &self.states[state_id].transitions {
                if let Some(weight) = self.states[state_id].trans_weights.get(&label) {
                    // Tokens that can reach dest through this edge
                    let edge_tokens = &current_reach & &weight.rsb;
                    
                    if !edge_tokens.is_empty() {
                        let old_reach = reach[dest].clone();
                        reach[dest] |= &edge_tokens;
                        
                        // If we added new tokens, re-process dest
                        if reach[dest] != old_reach && !in_queue[dest] {
                            queue.push_back(dest);
                            in_queue[dest] = true;
                        }
                    }
                }
            }
        }
        
        reach
    }
    
    /// Compute backward reachability: for each state, which tokens can lead to acceptance.
    fn compute_backward_reachability(&self) -> Vec<RangeSetBlaze<usize>> {
        let n = self.states.len();
        let mut reach: Vec<RangeSetBlaze<usize>> = vec![RangeSetBlaze::new(); n];
        
        // Initialize final states with their final weights
        let mut queue = VecDeque::new();
        let mut in_queue = vec![false; n];
        
        for state_id in 0..n {
            if let Some(fw) = &self.states[state_id].final_weight {
                reach[state_id] = fw.rsb.clone();
                if !in_queue[state_id] {
                    queue.push_back(state_id);
                    in_queue[state_id] = true;
                }
            }
        }
        
        // Build reverse graph
        let mut rev_edges: Vec<Vec<(StateID, Label)>> = vec![Vec::new(); n];
        for state_id in 0..n {
            for (&label, &dest) in &self.states[state_id].transitions {
                rev_edges[dest].push((state_id, label));
            }
        }
        
        // BFS backward
        while let Some(state_id) = queue.pop_front() {
            in_queue[state_id] = false;
            
            let current_reach = reach[state_id].clone();
            
            for &(src, label) in &rev_edges[state_id] {
                if let Some(weight) = self.states[src].trans_weights.get(&label) {
                    // Tokens that can reach acceptance through this edge
                    let edge_tokens = &current_reach & &weight.rsb;
                    
                    if !edge_tokens.is_empty() {
                        let old_reach = reach[src].clone();
                        reach[src] |= &edge_tokens;
                        
                        if reach[src] != old_reach && !in_queue[src] {
                            queue.push_back(src);
                            in_queue[src] = true;
                        }
                    }
                }
            }
        }
        
        reach
    }
    
    /// Count the number of unique weights in this DWA
    fn count_unique_weights(&self) -> usize {
        use std::ptr;
        let mut seen: HashSet<usize> = HashSet::new();
        
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                let ptr = ptr::addr_of!(**fw) as usize;
                seen.insert(ptr);
            }
            for w in state.trans_weights.values() {
                let ptr = ptr::addr_of!(**w) as usize;
                seen.insert(ptr);
            }
        }
        seen.len()
    }
    
    /// Analyze the structure of weights to understand fragmentation patterns
    fn analyze_weight_structure(&self) {
        use std::ptr;
        
        // Collect all unique weights with their ranges
        let mut unique_weights: HashMap<usize, (usize, usize, usize)> = HashMap::new(); // ptr -> (usage_count, num_ranges, cardinality)
        
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                let ptr = ptr::addr_of!(**fw) as usize;
                let entry = unique_weights.entry(ptr).or_insert((0, fw.num_ranges(), fw.len()));
                entry.0 += 1;
            }
            for w in state.trans_weights.values() {
                let ptr = ptr::addr_of!(**w) as usize;
                let entry = unique_weights.entry(ptr).or_insert((0, w.num_ranges(), w.len()));
                entry.0 += 1;
            }
        }
        
        // Find weights with high range count
        let mut high_range_weights: Vec<_> = unique_weights.iter()
            .filter(|(_, (_, ranges, _))| *ranges > 100)
            .map(|(ptr, (count, ranges, card))| (*ptr, *count, *ranges, *card))
            .collect();
        high_range_weights.sort_by_key(|(_, _, ranges, _)| std::cmp::Reverse(*ranges));
        
        if !high_range_weights.is_empty() {
            crate::debug!(5, "  High-range weights (>100 ranges):");
            for (_, usage_count, num_ranges, cardinality) in high_range_weights.iter().take(5) {
                crate::debug!(5, "    {} usages, {} ranges, cardinality {}", usage_count, num_ranges, cardinality);
            }
            if high_range_weights.len() > 5 {
                crate::debug!(5, "    ... and {} more", high_range_weights.len() - 5);
            }
        }
        
        // Compute potential savings from weight bucketing
        let total_current_ranges: usize = unique_weights.values().map(|(_, r, _)| r).sum();
        crate::debug!(5, "  Total ranges in unique weights: {}", total_current_ranges);
        
        // Analyze complement efficiency
        let mut complement_better_count = 0;
        let mut complement_savings = 0usize;
        
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                let complement_ranges = fw.complement().num_ranges();
                if complement_ranges < fw.num_ranges() {
                    complement_better_count += 1;
                    complement_savings += fw.num_ranges() - complement_ranges;
                }
            }
            for w in state.trans_weights.values() {
                let complement_ranges = w.complement().num_ranges();
                if complement_ranges < w.num_ranges() {
                    complement_better_count += 1;
                    complement_savings += w.num_ranges() - complement_ranges;
                }
            }
        }
        
        if complement_better_count > 0 {
            crate::debug!(5, "  Complement representation: {} weights would be smaller, saving {} ranges", 
                complement_better_count, complement_savings);
        }
    }
}

/// Optimize a weight by removing/filling ranges based on relevant tokens.
/// 
/// - `tokens_for_removal`: Ranges NOT intersecting this are removed (token can't reach here AND finish)
/// - `tokens_for_gap_fill`: Gaps NOT intersecting this can be filled (token can't reach here)
/// 
/// Returns (optimized_weight, ranges_removed, gaps_filled)
fn optimize_weight_ranges(weight: &Weight, tokens_for_removal: &RangeSetBlaze<usize>, tokens_for_gap_fill: &RangeSetBlaze<usize>) -> (Weight, usize, usize) {
    let original_ranges = weight.num_ranges();
    
    if original_ranges <= 1 {
        // Nothing to optimize for 0 or 1 range
        return (weight.clone(), 0, 0);
    }
    
    // Step 1: Remove ranges that don't intersect tokens_for_removal
    let pruned = &weight.rsb & tokens_for_removal;
    
    if pruned.is_empty() {
        return (Weight::zeros(), original_ranges, 0);
    }
    
    // Step 2: Fill gaps that don't intersect tokens_for_gap_fill
    // For each pair of adjacent ranges, check if the gap is "safe" to fill
    let ranges: Vec<_> = pruned.ranges().collect();
    
    if ranges.len() <= 1 {
        let ranges_removed = original_ranges - ranges.len();
        return (Weight::from_rsb(pruned), ranges_removed, 0);
    }
    
    let mut result = RangeSetBlaze::new();
    let mut gaps_filled = 0usize;
    
    // Start with first range
    let mut current_start = *ranges[0].start();
    let mut current_end = *ranges[0].end();
    
    for i in 1..ranges.len() {
        let next_start = *ranges[i].start();
        let next_end = *ranges[i].end();
        
        // Gap is [current_end+1, next_start-1]
        if current_end < usize::MAX && next_start > 0 && current_end + 1 < next_start {
            let gap_start = current_end + 1;
            let gap_end = next_start - 1;
            
            // Check if gap intersects with tokens_for_gap_fill
            let gap = RangeSetBlaze::from_iter([gap_start..=gap_end]);
            let gap_intersection = &gap & tokens_for_gap_fill;
            
            if gap_intersection.is_empty() {
                // Safe to fill this gap - extend current range
                current_end = next_end;
                gaps_filled += 1;
            } else {
                // Cannot fill gap - emit current range and start new one
                result |= RangeSetBlaze::from_iter([current_start..=current_end]);
                current_start = next_start;
                current_end = next_end;
            }
        } else {
            // Adjacent or overlapping - merge
            current_end = current_end.max(next_end);
        }
    }
    
    // Emit final range
    result |= RangeSetBlaze::from_iter([current_start..=current_end]);
    
    let ranges_removed = original_ranges.saturating_sub(result.ranges_len() + gaps_filled);
    
    (Weight::from_rsb(result), ranges_removed, gaps_filled)
}
