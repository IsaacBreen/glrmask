use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use bitvec::macros::internal::funty::Fundamental;
use keyed_priority_queue::KeyedPriorityQueue;
use crate::datastructures::charmap::TrieMap;
use crate::managed_glr_parser::{ManagedGLRParserState, ManagedParseState};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};

pub type LLMTokenBV = BitVec;
pub type GrammarTokenBV = BitVec;

#[derive(Debug, Clone)]
pub struct PrecomputedFinalizer {
    pub(crate) possible_final_grammar_tokens: BTreeSet<GrammarTokenID>,
    pub(crate) compatible_llm_tokens: LLMTokenBV,
    pub(crate) tokenizer_state_ids: BTreeSet<TokenizerStateID>,
}

#[derive(Debug, Clone)]
pub struct PrecomputedNodeContents {
    pub(crate) finalizers: Vec<PrecomputedFinalizer>,
    pub(crate) possible_llm_token_ids: LLMTokenBV,
}

type Precomputed = BTreeMap<TokenizerStateID, Trie<GrammarTokenID, PrecomputedNodeContents>>;

#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    pub(crate) tokenizer: Regex,
    pub(crate) parser: GLRParser,
    pub(crate) precomputed: Precomputed,
    pub(crate) llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
    pub(crate) max_llm_token_id: usize,
}

#[derive(Debug, Clone)]
pub struct GrammarConstraintState<'a> {
    pub(crate) parent: &'a GrammarConstraint,
    pub(crate) state: ManagedGLRParserState<'a>,
}

impl GrammarConstraint {
    pub fn new(
        tokenizer: Regex,
        parser: GLRParser,
        llm_token_map: LLMTokenMap,
        max_llm_token_id: usize
    ) -> Self {
        let precomputed = GrammarConstraint::precompute(&tokenizer, &llm_token_map, max_llm_token_id);
        Self {
            tokenizer,
            parser,
            precomputed,
            llm_token_map,
            max_llm_token_id,
        }
    }

    pub fn precompute<'a>(
        tokenizer: &Regex,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        max_llm_token_id: usize,
    ) -> Precomputed {
        type VocabTrieNode = TrieMap<LLMTokenID>;
        type GrammarTokenTrieNode = Arc<Mutex<Trie<GrammarTokenID, ()>>>;
        let helper = |
            state: TokenizerStateID,
            vocab_trie_node: VocabTrieNode,
            prev_matches: BTreeMap<GrammarTokenID, GrammarTokenTrieNode>,
            merge_cache: BTreeMap<GrammarTokenTrieNode, VocabTrieNode>,
        | {

        };

        todo!()
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let glr_parser_initial_state = self.parser.init_managed_glr_parser();

        GrammarConstraintState {
            parent: self,
            state: glr_parser_initial_state,
        }
    }
}

impl GrammarConstraintState<'_> {
    pub fn get_mask(&mut self) -> LLMTokenBV {
        let mut mask = LLMTokenBV::repeat(false, self.parent.max_llm_token_id);
        for managed_parse_state in &self.state.active_states {
            mask |= managed_parse_state.llm_tokens.clone();
        }
        mask
    }

    pub fn step_with_all_llm_tokens(&mut self) {
        let all_llm_tokens = LLMTokenBV::repeat(true, self.parent.max_llm_token_id);
        self.step(&all_llm_tokens);
    }

    pub fn step_with_llm_token(&mut self, llm_token_id: LLMTokenID) {
        let mut llm_tokens = LLMTokenBV::repeat(false, self.parent.max_llm_token_id);
        llm_tokens.set(llm_token_id.0, true);
        self.step(&llm_tokens);
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        // Keep only the active states for which this LLM token is set
        self.state.active_states.retain(|managed_parse_state| managed_parse_state.llm_tokens[llm_token_id.0]);
    }

    pub fn commit_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.commit(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(Arc<Mutex<Trie<TerminalID, PrecomputedNodeContents>>>, ManagedGLRParserState)> {
        // The BTreeSet<TokenizerStateID> in each Trie node here is the set of terminal states at this node.
        // Each terminal state indicates that the path through the trie can terminate here.
        // (todo: explain this better)
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<Trie<GrammarTokenID, PrecomputedNodeContents>>>, ManagedGLRParserState)> = Vec::new();

        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, (BTreeSet<ManagedParseState>, LLMTokenBV)> = BTreeMap::new();
        for managed_parse_state in self.state.active_states.iter() {
            for tokenizer_state_id in managed_parse_state.tokenizer_state_ids.iter() {
                tokenizer_state_id_to_parse_states.entry(*tokenizer_state_id).or_default().0.insert(managed_parse_state.clone());
                tokenizer_state_id_to_parse_states.entry(*tokenizer_state_id).or_default().1 = llm_tokens.clone();
            }
        }

        for (tokenizer_state_id, (parse_states, llm_tokens)) in tokenizer_state_id_to_parse_states {
            let token_trie = self.parent.precomputed[&tokenizer_state_id].clone();
            let token_trie = Arc::new(Mutex::new(token_trie));
            let managed_glr_parser_state = GLRParser::init_managed_glr_parser_from_managed_parse_states(self.state.parser, parse_states.into_iter().collect());
            initial_nodes_and_values.push((token_trie, managed_glr_parser_state));
        }
        initial_nodes_and_values
    }

    pub fn step(&mut self, llm_tokens: &LLMTokenBV) {
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map(llm_tokens);

        let mut final_active_parse_states: Vec<ManagedParseState> = Vec::new();
        let mut final_inactive_parse_states: Vec<ManagedParseState> = Vec::new();

        Trie::special_map(
            initial_nodes_and_values,
            // step
            |managed_parse_state, grammar_token_id, child_node| {
                managed_parse_state.clone().with_step(*grammar_token_id)
            },
            // merge
            |managed_parse_state1, managed_parse_state2| {
                managed_parse_state1.merge_with(managed_parse_state2);
            },
            // process
            |node, managed_glr_parse_state| {
                // Handle finalizers
                for precomputed_finalizer in &node.value.finalizers {
                    for managed_parse_state in &managed_glr_parse_state.active_states {
                        // Ensure at least one of the final tokens parses
                        let mut valid_final_tokenizer_state_ids = BTreeSet::new();
                        for possible_final_grammar_token in &precomputed_finalizer.possible_final_grammar_tokens {
                            let mut parse_state = managed_glr_parse_state.parser.init_glr_parser_from_parse_state(ParseState::from(managed_parse_state.clone()));
                            parse_state.step(*possible_final_grammar_token);
                            if parse_state.matches_or_can_match() {
                                valid_final_tokenizer_state_ids = managed_parse_state.tokenizer_state_ids.clone();
                                break;
                            }
                        }
                        if valid_final_tokenizer_state_ids.is_empty() {
                            // If we've reached the initial state, we've matched the final token cleanly, and we can proceed without any additional tokens.
                            if precomputed_finalizer.tokenizer_state_ids.contains(&TokenizerStateID(0)) {
                                valid_final_tokenizer_state_ids.insert(TokenizerStateID(0));
                            } else {
                                continue;
                            }
                        }
                        // Compute final LLM token mask
                        let final_llm_tokens = managed_parse_state.llm_tokens.clone() | precomputed_finalizer.compatible_llm_tokens.clone();
                        if final_llm_tokens.is_empty() { continue; }
                        // Create a new managed parse state
                        let mut managed_parse_state = managed_parse_state.clone();
                        managed_parse_state.tokenizer_state_ids = valid_final_tokenizer_state_ids;
                        managed_parse_state.llm_tokens = final_llm_tokens;
                        final_active_parse_states.push(managed_parse_state);
                    }
                }
                // Update the LLM token masks for the active states
                for managed_parse_state in &mut managed_glr_parse_state.active_states {
                    // NOTE: setting state to 0 might be incorrect for the root nodes...
                    managed_parse_state.llm_tokens |= node.value.possible_llm_token_ids.clone();
                }
                !managed_glr_parse_state.active_states.is_empty()
            },
        );

        self.state.active_states = final_active_parse_states;
        self.state.inactive_states.extend(final_inactive_parse_states);
    }
}