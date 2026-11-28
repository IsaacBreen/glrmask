//! Possible-match tables for tokenizer states and vocab-prefix subtrees.

use crate::automata::lexer::Lexer;
use crate::automata::lexer::observation::LexicalStateAction;
use std::rc::Rc;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::{Tokenizer, TokenizerStateSet};
use crate::grammar::flat::TerminalID;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;

type PossibleMatchMap = FxHashMap<TerminalID, RangeSetBlaze<u32>>;

fn reachable_u32(node: VocabPrefixTreeNode<'_>) -> RangeSetBlaze<u32> {
    let mut out = RangeSetBlaze::new();
    if let Some((first, end)) = node.ordered_interval() {
        if first < end { out.ranges_insert(first..=end - 1); }
        return out;
    }
    for range in node.reachable_token_ids().ranges() {
        out.ranges_insert(*range.start() as u32..=*range.end() as u32);
    }
    out
}

fn merge_token_ids(into: &mut RangeSetBlaze<u32>, other: &RangeSetBlaze<u32>) {
    *into |= other;
}

fn merge_possible_match_maps(into: &mut PossibleMatchMap, other: &PossibleMatchMap) {
    for (terminal, token_ids) in other {
        let existing = into.entry(*terminal).or_default();
        merge_token_ids(existing, token_ids);
    }
}

pub struct PossibleMatchesComputer<'a> {
    tokenizer: &'a Tokenizer,
    canonical_state: Option<&'a [u32]>,
    cache: FxHashMap<(usize, TokenizerStateSet), Rc<PossibleMatchMap>>,
    reachable_cache: FxHashMap<usize, Rc<RangeSetBlaze<u32>>>,
    self_loop_bytes: FxHashMap<u32, U8Set>,
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
    scalar_deterministic_scan: bool,
    state_action: Option<Box<LexicalStateAction<'a>>>,
}

impl<'a> PossibleMatchesComputer<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self::new_with_canonical_state(tokenizer, None)
    }

    pub fn new_with_canonical_state(
        tokenizer: &'a Tokenizer,
        canonical_state: Option<&'a [u32]>,
    ) -> Self {
        Self {
            tokenizer,
            canonical_state,
            cache: FxHashMap::default(),
            reachable_cache: FxHashMap::default(),
            self_loop_bytes: FxHashMap::default(),
            flat_transitions: vec![None; tokenizer.num_states() as usize],
            scalar_deterministic_scan: !tokenizer.has_epsilon_transitions()
                || tokenizer.has_scalar_deterministic_dispatch(),
            state_action: None,
        }
    }

    /// The NWA walk and its possible-match queries must share the exact
    /// source tokenizer, never merely a shape-compatible induced coordinate.
    /// PM-only sparse callers retain their allocation-free scalar/state-set
    /// path; a trie builder enables this shared lazy action when needed.
    pub fn state_action(&mut self, tokenizer: &Tokenizer) -> &mut LexicalStateAction<'a> {
        assert!(std::ptr::eq(self.tokenizer, tokenizer), "lexical action source mismatch");
        let source = self.tokenizer;
        self.state_action.get_or_insert_with(|| Box::new(LexicalStateAction::new(source)))
    }

    fn initial_states(&mut self, state: u32) -> TokenizerStateSet {
        if let Some(action) = &mut self.state_action {
            let config = action.start(state);
            TokenizerStateSet::from_slice(action.states(config))
        } else {
            let mut states = self.tokenizer.execute_from_state_end_only(&[], state);
            states.sort_unstable();
            states.dedup();
            states
        }
    }

    #[inline]
    fn advance(&mut self, states: &[u32], byte: u8) -> TokenizerStateSet {
        if states.len() == 1 && self.scalar_deterministic_scan {
            match self.fast_step(states[0], byte) {
                Some(next) => SmallVec::from_buf([next]),
                None => TokenizerStateSet::new(),
            }
        } else if let Some(action) = &mut self.state_action {
            let config = action.intern_boundary(states);
            let target = action.step(config, byte);
            TokenizerStateSet::from_slice(action.states(target))
        } else {
            self.tokenizer.step_all(states, byte)
        }
    }

    /// Observe all completed terminals along a segment, independently of its
    /// final survival. The node's descendants share every byte in this edge.
    fn observe_segment(
        &mut self, mut states: TokenizerStateSet, bytes: &[u8],
        reachable: &RangeSetBlaze<u32>, result: &mut PossibleMatchMap,
    ) -> TokenizerStateSet {
        // Retain the shared configuration coordinate across an entire edge.
        // Scalar dispatch keeps its direct row action; cold PM-only callers
        // keep the ordinary state-set representation without forcing a memo.
        let mut config = if self.scalar_deterministic_scan {
            None
        } else {
            self.state_action.as_mut().map(|action| action.intern_boundary(&states))
        };
        let tokenizer = self.tokenizer;
        for &byte in bytes {
            let observed = if let Some(config) = &mut config {
                let action = self.state_action.as_mut().expect("shared configuration action");
                *config = action.step(*config, byte);
                action.states(*config)
            } else {
                states = self.advance(&states, byte);
                states.as_slice()
            };
            if observed.is_empty() { break; }
            for &state in observed {
                for terminal in tokenizer.matched_terminals_iter(state) {
                    merge_token_ids(result.entry(terminal).or_default(), reachable);
                }
            }
        }
        if let Some(config) = config {
            TokenizerStateSet::from_slice(self.state_action.as_ref().unwrap().states(config))
        } else {
            states
        }
    }

    fn project_boundary(&self, mut states: TokenizerStateSet) -> TokenizerStateSet {
        states.retain(|state| !self.tokenizer.is_end(*state));
        if let Some(map) = self.canonical_state {
            for state in &mut states {
                *state = map.get(*state as usize).copied().unwrap_or(*state);
            }
            states.sort_unstable();
            states.dedup();
        }
        states
    }

    #[inline]
    fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state_idx = state as usize;
        if self.flat_transitions[state_idx].is_none() {
            self.flat_transitions[state_idx] = Some(self.tokenizer.transition_row(state));
        }
        let next = self.flat_transitions[state_idx].as_ref().unwrap()[byte as usize];
        if next == u32::MAX { None } else { Some(next) }
    }

    fn reachable_for_node(&mut self, node: VocabPrefixTreeNode<'_>) -> Rc<RangeSetBlaze<u32>> {
        let cache_key = node.cache_key();
        if let Some(cached) = self.reachable_cache.get(&cache_key) {
            Rc::clone(cached)
        } else {
            let reachable = Rc::new(reachable_u32(node));
            self.reachable_cache.insert(cache_key, Rc::clone(&reachable));
            reachable
        }
    }

    fn can_skip_self_loop_subtree(
        &mut self,
        node: VocabPrefixTreeNode<'_>,
        tokenizer_state: u32,
    ) -> bool {
        if !self.scalar_deterministic_scan {
            return false;
        }
        let self_loop_bytes = self
            .self_loop_bytes
            .entry(tokenizer_state)
            .or_insert_with(|| self.tokenizer.self_loop_bytes(tokenizer_state));
        U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes)
    }

    pub fn possible_matches_for_node(
        &mut self,
        node: VocabPrefixTreeNode<'_>,
        tokenizer_state: u32,
    ) -> Rc<PossibleMatchMap> {
        let states = self.initial_states(tokenizer_state);
        self.possible_matches_for_node_states(node, states)
    }

    /// Return every terminal that can match after consuming a fixed suffix
    /// shared by all tokens below `node`, starting from `tokenizer_state`.
    ///
    /// A match reached anywhere inside `suffix` invalidates the shorter
    /// boundary for every token below `node`, because every such token shares
    /// those bytes. After the suffix, ordinary trie-relative possible-match
    /// analysis handles matches in the remaining descendant bytes.
    pub fn possible_matches_for_suffix_and_node(
        &mut self,
        suffix: &[u8],
        node: VocabPrefixTreeNode<'_>,
        tokenizer_state: u32,
    ) -> Rc<PossibleMatchMap> {
        if suffix.is_empty() {
            return self.possible_matches_for_node(node, tokenizer_state);
        }

        let states = self.initial_states(tokenizer_state);
        let mut result = PossibleMatchMap::default();
        let reachable = self.reachable_for_node(node);
        let states = self.observe_segment(states, suffix, reachable.as_ref(), &mut result);
        let states = self.project_boundary(states);
        if !states.is_empty() {
            let descendant_matches = self.possible_matches_for_node_states(node, states);
            merge_possible_match_maps(&mut result, descendant_matches.as_ref());
        }
        Rc::new(result)
    }

    fn possible_matches_for_node_states(
        &mut self,
        node: VocabPrefixTreeNode<'_>,
        mut tokenizer_states: TokenizerStateSet,
    ) -> Rc<PossibleMatchMap> {
        tokenizer_states.sort_unstable();
        tokenizer_states.dedup();
        let cache_key = (
            node.cache_key(),
            tokenizer_states.clone(),
        );
        if let Some(cached) = self.cache.get(&cache_key) {
            return Rc::clone(cached);
        }

        let mut result = PossibleMatchMap::default();

        // This intentionally includes the token ending exactly at `node` before
        // recursing into child segments, so the recursive part only adds longer
        // continuations.
        if node.has_token() {
            let token_id = node.token_id() as u32;
            for &tokenizer_state in &tokenizer_states {
                for terminal in self.tokenizer.matched_terminals_iter(tokenizer_state) {
                    result.entry(terminal).or_default().insert(token_id);
                }
            }
        }

        for (segment_bytes, child) in node.iter_children() {
            let reachable = self.reachable_for_node(child);
            let states = self.observe_segment(
                tokenizer_states.clone(), segment_bytes, reachable.as_ref(), &mut result,
            );
            let current_states = self.project_boundary(states);
            if !current_states.is_empty() {
                if current_states.len() == 1
                    && self.can_skip_self_loop_subtree(child, current_states[0])
                {
                    continue;
                }
                let child_matches = self.possible_matches_for_node_states(child, current_states);
                merge_possible_match_maps(&mut result, child_matches.as_ref());
            }
        }

        let result = Rc::new(result);
        self.cache.insert(cache_key, Rc::clone(&result));
        result
    }
}
