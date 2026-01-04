// src/precompute4/weighted_automata/weight_expansion.rs

//! Weight expansion for transitioning from symbol-heavy to weight-heavy regime.
//!
//! # Regime Definitions
//!
//! ## Symbol-Heavy Regime (current)
//! - Tokenizer state IDs (tsids) are encoded as transition labels at the start state
//! - Weights have length N (number of LLM tokens)
//! - Start state has transitions labeled 0..M (where M = num_tsids) to child states
//!
//! ## Weight-Heavy Regime (new)
//! - NO separate transitions per tsid at start
//! - Weights have length N×M (flattened array)
//! - Layout: weight index `i = token_id * M + tsid_id` (tsids iterate first)
//! - This way, if a single token is valid for ALL tsids, it's still ONE range: [token_id * M .. (token_id + 1) * M)
//!
//! # Benefits of Weight-Heavy
//! - Simpler DWA structure (no special initial transitions)
//! - Potentially fewer states after minimization (states that differ only in initial tsid can merge)
//! - More uniform weight handling throughout the pipeline

use super::common::{Label, StateID, Weight};
use super::dwa::{DWA, DWABody, DWAState, DWAStates};
use super::rangeset::RangeSet;
use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;

/// Parameters for weight expansion between regimes.
#[derive(Debug, Clone)]
pub struct WeightExpansionParams {
    /// Number of LLM tokens (N)
    pub num_tokens: usize,
    /// Number of tokenizer state IDs (M)
    pub num_tsids: usize,
}

impl WeightExpansionParams {
    /// Get the expanded weight length (N × M)
    #[inline]
    pub fn expanded_len(&self) -> usize {
        self.num_tokens * self.num_tsids
    }

    /// Convert a (token_id, tsid) pair to an expanded index.
    /// Layout: index = token_id * M + tsid
    #[inline]
    pub fn to_expanded_index(&self, token_id: usize, tsid: usize) -> usize {
        token_id * self.num_tsids + tsid
    }

    /// Convert an expanded index back to (token_id, tsid).
    #[inline]
    pub fn from_expanded_index(&self, index: usize) -> (usize, usize) {
        let token_id = index / self.num_tsids;
        let tsid = index % self.num_tsids;
        (token_id, tsid)
    }

    /// Expand a token-only weight (length N) to include ALL tsids.
    /// If token_id is in the input weight, then ALL indices token_id*M .. (token_id+1)*M are set.
    pub fn expand_weight_all_tsids(&self, weight: &Weight) -> Weight {
        if weight.is_all_fast() {
            // ALL tokens for ALL tsids = full expanded range
            return Weight::from_iter(0..self.expanded_len());
        }
        if weight.is_empty() {
            return Weight::zeros();
        }

        let mut rsb = RangeSetBlaze::<usize>::new();
        for range in weight.rsb.ranges() {
            let start = *range.start();
            let end = *range.end(); // inclusive

            // For each token in [start..=end], add all tsids
            // Expanded range: [start * M .. (end + 1) * M)
            let exp_start = start * self.num_tsids;
            let exp_end = (end + 1) * self.num_tsids - 1; // inclusive
            rsb.ranges_insert(exp_start..=exp_end);
        }
        RangeSet::from_rsb(rsb)
    }

    /// Expand a token-only weight (length N) to include ONLY a specific tsid.
    /// If token_id is in the input weight, then only token_id*M + tsid is set.
    pub fn expand_weight_single_tsid(&self, weight: &Weight, tsid: usize) -> Weight {
        if tsid >= self.num_tsids {
            return Weight::zeros();
        }
        if weight.is_empty() {
            return Weight::zeros();
        }

        let mut rsb = RangeSetBlaze::<usize>::new();
        for range in weight.rsb.ranges() {
            for token_id in *range.start()..=*range.end() {
                let idx = self.to_expanded_index(token_id, tsid);
                rsb.insert(idx);
            }
        }
        RangeSet::from_rsb(rsb)
    }

    /// Expand a weight by including ALL tsids for tokens that are already present.
    /// This is used for transitions that should apply equally to all tsids.
    pub fn expand_weight_for_all_tsids(&self, weight: &Weight) -> Weight {
        self.expand_weight_all_tsids(weight)
    }

    /// Create an initial weight for a specific tsid.
    /// Sets all tokens (0..N) but only for the given tsid.
    pub fn create_initial_weight_for_tsid(&self, tsid: usize) -> Weight {
        if tsid >= self.num_tsids {
            return Weight::zeros();
        }

        let mut rsb = RangeSetBlaze::<usize>::new();
        for token_id in 0..self.num_tokens {
            let idx = self.to_expanded_index(token_id, tsid);
            rsb.insert(idx);
        }
        RangeSet::from_rsb(rsb)
    }

    /// Contract an expanded weight (length N×M) back to token-only (length N).
    /// A token is set in the output if ANY tsid is set for that token in the input.
    pub fn contract_weight_any_tsid(&self, expanded_weight: &Weight) -> Weight {
        if expanded_weight.is_empty() {
            return Weight::zeros();
        }

        let mut rsb = RangeSetBlaze::<usize>::new();

        // For each range in the expanded weight, figure out which tokens are covered
        for range in expanded_weight.rsb.ranges() {
            let start = *range.start();
            let end = *range.end(); // inclusive

            // A token is covered if any index in [token*M .. (token+1)*M) is in the range
            let first_token = start / self.num_tsids;
            let last_token = end / self.num_tsids;

            for token_id in first_token..=last_token {
                // Check if this token has any bit set
                let token_start = token_id * self.num_tsids;
                let token_end = (token_id + 1) * self.num_tsids - 1;

                // If [start..=end] intersects [token_start..=token_end], the token is valid
                if start <= token_end && end >= token_start {
                    rsb.insert(token_id);
                }
            }
        }
        RangeSet::from_rsb(rsb)
    }

    /// Contract an expanded weight keeping only bits for a specific tsid.
    /// Output is token-only (length N).
    pub fn contract_weight_single_tsid(&self, expanded_weight: &Weight, tsid: usize) -> Weight {
        if tsid >= self.num_tsids || expanded_weight.is_empty() {
            return Weight::zeros();
        }

        let mut rsb = RangeSetBlaze::<usize>::new();

        for range in expanded_weight.rsb.ranges() {
            let start = *range.start();
            let end = *range.end();

            let first_token = start / self.num_tsids;
            let last_token = end / self.num_tsids;

            for token_id in first_token..=last_token {
                let idx = self.to_expanded_index(token_id, tsid);
                if idx >= start && idx <= end {
                    rsb.insert(token_id);
                }
            }
        }
        RangeSet::from_rsb(rsb)
    }
}

impl DWA {
    /// Convert a symbol-heavy DWA to weight-heavy regime.
    ///
    /// # Symbol-Heavy Structure
    /// - Start state has transitions labeled 0, 1, ..., M-1 (tsid labels)
    /// - Each leads to a child state
    /// - Weights have length N (num_tokens)
    ///
    /// # Weight-Heavy Structure  
    /// - Start state is a new state
    /// - Single epsilon-like path to merged children
    /// - Weights have length N×M
    ///
    /// # Arguments
    /// - `num_tokens`: N, the number of LLM tokens
    /// - `num_tsids`: M, the number of tokenizer state IDs (typically small, e.g., 1-10)
    ///
    /// # Returns
    /// - Converted DWA in weight-heavy regime
    /// - WeightExpansionParams for interpreting the expanded weights
    pub fn to_weight_heavy(&self, num_tokens: usize, num_tsids: usize) -> (DWA, WeightExpansionParams) {
        let params = WeightExpansionParams { num_tokens, num_tsids };
        
        let old_start = self.body.start_state;
        let old_start_state = &self.states[old_start];
        
        // Collect the tsid transitions from start state
        // These are transitions labeled 0, 1, ..., num_tsids-1
        let mut tsid_targets: BTreeMap<usize, (StateID, Weight)> = BTreeMap::new();
        for (&label, &target) in &old_start_state.transitions {
            if label >= 0 && (label as usize) < num_tsids {
                let weight = old_start_state.trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                tsid_targets.insert(label as usize, (target, weight));
            }
        }
        
        // If there are no tsid transitions or they don't go to different states,
        // the conversion is trivial - just expand all weights
        if tsid_targets.is_empty() {
            return self.expand_weights_uniformly(&params);
        }
        
        // Check if all tsid transitions lead to the same state
        let targets: Vec<StateID> = tsid_targets.values().map(|(t, _)| *t).collect();
        let all_same = targets.iter().all(|&t| t == targets[0]);
        
        if all_same {
            // Simple case: all tsids lead to same state
            return self.expand_simple_case(&params, &tsid_targets);
        }
        
        // Complex case: tsids lead to different states
        // We need to do a proper state product construction
        self.expand_complex_case(&params, &tsid_targets)
    }

    /// Expand weights uniformly when there are no tsid-specific transitions.
    fn expand_weights_uniformly(&self, params: &WeightExpansionParams) -> (DWA, WeightExpansionParams) {
        let mut new_states = DWAStates::default();
        
        for old_state in &self.states.0 {
            let mut new_state = DWAState::default();
            
            // Copy transitions
            new_state.transitions = old_state.transitions.clone();
            
            // Expand all transition weights
            for (label, weight) in &old_state.trans_weights {
                let expanded = params.expand_weight_all_tsids(weight);
                new_state.trans_weights.insert(*label, expanded);
            }
            
            // Expand final weight
            if let Some(fw) = &old_state.final_weight {
                new_state.final_weight = Some(params.expand_weight_all_tsids(fw));
            }
            
            // Expand state weight
            if let Some(sw) = &old_state.state_weight {
                new_state.state_weight = Some(params.expand_weight_all_tsids(sw));
            }
            
            new_states.add_existing_state(new_state);
        }
        
        let new_dwa = DWA {
            body: self.body.clone(),
            states: new_states,
        };
        
        (new_dwa, params.clone())
    }

    /// Handle the simple case where all tsid transitions lead to the same state.
    fn expand_simple_case(
        &self,
        params: &WeightExpansionParams,
        tsid_targets: &BTreeMap<usize, (StateID, Weight)>,
    ) -> (DWA, WeightExpansionParams) {
        let old_start = self.body.start_state;
        let common_target = tsid_targets.values().next().unwrap().0;
        
        let mut new_states = DWAStates::default();
        
        // Copy all states except start, expanding weights
        let mut state_mapping: BTreeMap<StateID, StateID> = BTreeMap::new();
        
        for (old_id, old_state) in self.states.0.iter().enumerate() {
            if old_id == old_start {
                continue; // Handle start state specially
            }
            
            let mut new_state = DWAState::default();
            
            // Copy transitions (will update targets later)
            for (&label, &target) in &old_state.transitions {
                new_state.transitions.insert(label, target);
            }
            
            // Expand weights
            for (label, weight) in &old_state.trans_weights {
                new_state.trans_weights.insert(*label, params.expand_weight_all_tsids(weight));
            }
            
            if let Some(fw) = &old_state.final_weight {
                new_state.final_weight = Some(params.expand_weight_all_tsids(fw));
            }
            
            if let Some(sw) = &old_state.state_weight {
                new_state.state_weight = Some(params.expand_weight_all_tsids(sw));
            }
            
            let new_id = new_states.add_existing_state(new_state);
            state_mapping.insert(old_id, new_id);
        }
        
        // Create new start state with combined weight for the single target
        let mut combined_weight = Weight::zeros();
        for (tsid, (_, weight)) in tsid_targets {
            let expanded = params.expand_weight_single_tsid(weight, *tsid);
            combined_weight |= &expanded;
        }
        
        let new_start_state = {
            let mut s = DWAState::default();
            let new_target = state_mapping.get(&common_target).copied().unwrap_or(common_target);
            // Use a dummy label (we'll remove tsid labels)
            s.transitions.insert(0, new_target);
            s.trans_weights.insert(0, combined_weight);
            s
        };
        
        let new_start_id = new_states.add_existing_state(new_start_state);
        
        // Update transition targets
        for state in &mut new_states.0 {
            for target in state.transitions.values_mut() {
                if let Some(&new_target) = state_mapping.get(target) {
                    *target = new_target;
                }
            }
        }
        
        let new_dwa = DWA {
            body: DWABody { start_state: new_start_id },
            states: new_states,
        };
        
        (new_dwa, params.clone())
    }

    /// Handle the complex case where tsid transitions lead to different states.
    /// This requires creating a product construction.
    fn expand_complex_case(
        &self,
        params: &WeightExpansionParams,
        tsid_targets: &BTreeMap<usize, (StateID, Weight)>,
    ) -> (DWA, WeightExpansionParams) {
        // For the complex case, we need to track which tsid we're in throughout the DWA.
        // This is done by keeping the tsid information in the weights.
        //
        // The key insight: after the initial transition, we're committed to a particular tsid.
        // So we can:
        // 1. Create a new start state
        // 2. For each tsid, create an epsilon transition (dummy label) with weight encoding that tsid
        // 3. The rest of the DWA is duplicated M times (once per tsid), or we use weights to distinguish
        //
        // Actually, a simpler approach: just expand all weights, and at the initial transitions,
        // restrict to the appropriate tsid. The rest of the DWA will naturally propagate this.
        
        let old_start = self.body.start_state;
        let mut new_states = DWAStates::default();
        
        // Create mapping for all states except old start
        let mut state_mapping: BTreeMap<StateID, StateID> = BTreeMap::new();
        
        // First pass: create all new states (except start)
        for (old_id, old_state) in self.states.0.iter().enumerate() {
            if old_id == old_start {
                continue;
            }
            
            let mut new_state = DWAState::default();
            
            // Copy structure
            new_state.transitions = old_state.transitions.clone();
            
            // Expand weights uniformly - tsid info is encoded in which bits are set
            for (label, weight) in &old_state.trans_weights {
                new_state.trans_weights.insert(*label, params.expand_weight_all_tsids(weight));
            }
            
            if let Some(fw) = &old_state.final_weight {
                new_state.final_weight = Some(params.expand_weight_all_tsids(fw));
            }
            
            if let Some(sw) = &old_state.state_weight {
                new_state.state_weight = Some(params.expand_weight_all_tsids(sw));
            }
            
            let new_id = new_states.add_existing_state(new_state);
            state_mapping.insert(old_id, new_id);
        }
        
        // Create new start state
        // It has transitions for each tsid, each leading to the appropriate target
        let mut new_start = DWAState::default();
        
        for (tsid, (target, weight)) in tsid_targets {
            let new_target = state_mapping.get(target).copied().unwrap_or(*target);
            let expanded_weight = params.expand_weight_single_tsid(weight, *tsid);
            
            // Use tsid as the label (same as before, but weight encodes tsid info)
            new_start.transitions.insert(*tsid as Label, new_target);
            new_start.trans_weights.insert(*tsid as Label, expanded_weight);
        }
        
        let new_start_id = new_states.add_existing_state(new_start);
        
        // Update all transition targets
        for state in &mut new_states.0 {
            for target in state.transitions.values_mut() {
                if let Some(&new_target) = state_mapping.get(target) {
                    *target = new_target;
                }
            }
        }
        
        let new_dwa = DWA {
            body: DWABody { start_state: new_start_id },
            states: new_states,
        };
        
        (new_dwa, params.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_expansion_params_basic() {
        let params = WeightExpansionParams {
            num_tokens: 100,
            num_tsids: 3,
        };
        
        assert_eq!(params.expanded_len(), 300);
        assert_eq!(params.to_expanded_index(0, 0), 0);
        assert_eq!(params.to_expanded_index(0, 1), 1);
        assert_eq!(params.to_expanded_index(0, 2), 2);
        assert_eq!(params.to_expanded_index(1, 0), 3);
        assert_eq!(params.to_expanded_index(1, 1), 4);
        assert_eq!(params.to_expanded_index(99, 2), 299);
        
        assert_eq!(params.from_expanded_index(0), (0, 0));
        assert_eq!(params.from_expanded_index(1), (0, 1));
        assert_eq!(params.from_expanded_index(3), (1, 0));
        assert_eq!(params.from_expanded_index(299), (99, 2));
    }

    #[test]
    fn test_expand_weight_all_tsids() {
        let params = WeightExpansionParams {
            num_tokens: 4,
            num_tsids: 3,
        };
        
        // Single token
        let w = Weight::from_item(1);
        let expanded = params.expand_weight_all_tsids(&w);
        // Token 1 with all tsids: indices 3, 4, 5
        assert!(expanded.contains(3));
        assert!(expanded.contains(4));
        assert!(expanded.contains(5));
        assert!(!expanded.contains(2)); // Token 0, tsid 2
        assert!(!expanded.contains(6)); // Token 2, tsid 0
        
        // Range of tokens
        let w2 = Weight::from_iter(1..=2);
        let expanded2 = params.expand_weight_all_tsids(&w2);
        // Token 1: indices 3, 4, 5
        // Token 2: indices 6, 7, 8
        for i in 3..=8 {
            assert!(expanded2.contains(i), "Should contain {}", i);
        }
        assert!(!expanded2.contains(0));
        assert!(!expanded2.contains(1));
        assert!(!expanded2.contains(2));
        assert!(!expanded2.contains(9));
    }

    #[test]
    fn test_expand_weight_single_tsid() {
        let params = WeightExpansionParams {
            num_tokens: 4,
            num_tsids: 3,
        };
        
        let w = Weight::from_iter(0..=3); // All 4 tokens
        let expanded = params.expand_weight_single_tsid(&w, 1); // Only tsid 1
        
        // Should have indices 1, 4, 7, 10 (token * 3 + 1)
        assert!(expanded.contains(1));
        assert!(expanded.contains(4));
        assert!(expanded.contains(7));
        assert!(expanded.contains(10));
        
        assert!(!expanded.contains(0));
        assert!(!expanded.contains(2));
        assert!(!expanded.contains(3));
        assert!(!expanded.contains(5));
    }

    #[test]
    fn test_contract_weight_any_tsid() {
        let params = WeightExpansionParams {
            num_tokens: 4,
            num_tsids: 3,
        };
        
        // Expanded weight with some bits set
        let mut rsb = RangeSetBlaze::<usize>::new();
        rsb.insert(4); // Token 1, tsid 1
        rsb.insert(8); // Token 2, tsid 2
        let expanded = RangeSet::from_rsb(rsb);
        
        let contracted = params.contract_weight_any_tsid(&expanded);
        
        assert!(contracted.contains(1));
        assert!(contracted.contains(2));
        assert!(!contracted.contains(0));
        assert!(!contracted.contains(3));
    }

    #[test]
    fn test_create_initial_weight_for_tsid() {
        let params = WeightExpansionParams {
            num_tokens: 3,
            num_tsids: 2,
        };
        
        let w0 = params.create_initial_weight_for_tsid(0);
        // Should have indices 0, 2, 4 (all tokens with tsid 0)
        assert!(w0.contains(0));
        assert!(w0.contains(2));
        assert!(w0.contains(4));
        assert!(!w0.contains(1));
        assert!(!w0.contains(3));
        assert!(!w0.contains(5));
        
        let w1 = params.create_initial_weight_for_tsid(1);
        // Should have indices 1, 3, 5 (all tokens with tsid 1)
        assert!(!w1.contains(0));
        assert!(w1.contains(1));
        assert!(!w1.contains(2));
        assert!(w1.contains(3));
        assert!(!w1.contains(4));
        assert!(w1.contains(5));
    }
}
