use crate::datastructures::trie::Trie;
use crate::finite_automata::Regex;
use crate::glr::parser::{GLRParser, GLRParserState, ParseState, ParseStateKey};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use bimap::BiBTreeMap;
use bitvec::prelude::BitVec;
use bitvec::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use keyed_priority_queue::KeyedPriorityQueue;
use crate::managed_glr_parser::{ManagedGLRParserState, ManagedParseState};
use crate::types::{TerminalID as GrammarTokenID, TerminalID};

pub type LLMTokenBV = BitVec;
pub type GrammarTokenBV = BitVec;

#[derive(Debug, Clone)]
pub struct PrecomputedFinalizer {
    pub(crate) possible_final_grammar_tokens: GrammarTokenBV,
    pub(crate) compatible_llm_tokens: LLMTokenBV,
    pub(crate) tokenizer_state_id: TokenizerStateID,
}

#[derive(Debug, Clone)]
pub struct PrecomputedNodeContents {
    pub(crate) finalizers: Vec<PrecomputedFinalizer>,
    pub(crate) relevant_llm_token_ids: LLMTokenBV,
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

    /// Precomputes a map from state -> token sequence -> LLM token -> state.
    pub fn precompute<'a>(
        tokenizer: &Regex,
        llm_token_map: &BiBTreeMap<Vec<u8>, LLMTokenID>,
        max_llm_token_id: usize,
    ) -> Precomputed {
        todo!()
    }

    pub fn init(&self) -> GrammarConstraintState<'_> {
        let glr_parser_initial_state = self.parser.init_managed_glr_parser();
        let tokenizer_initial_state_id = self.tokenizer.initial_state_id();

        GrammarConstraintState {
            parent: self,
            state: glr_parser_initial_state,
        }
    }
}

impl GrammarConstraintState<'_> {
    pub fn parse(&mut self, llm_tokens: &LLMTokenBV) -> Vec<(GLRParserState, LLMTokenBV)> {
        todo!()
    }

    pub fn get_mask(&mut self) -> LLMTokenBV {
        let all_llm_tokens = LLMTokenBV::repeat(true, self.parent.max_llm_token_id);
        let mut mask = LLMTokenBV::repeat(false, self.parent.max_llm_token_id);
        let mut results = self.parse(&all_llm_tokens);
        for (_, llm_tokens) in results {
            mask |= llm_tokens;
        }
        mask
    }

    pub fn commit(&mut self, llm_token_id: LLMTokenID) {
        let mut grammar_token_trie_roots: BTreeMap<TokenizerStateID, Trie<GrammarTokenID, BTreeSet<TokenizerStateID>>> = BTreeMap::new();

    }

    pub fn commit_many(&mut self, llm_token_ids: &[LLMTokenID]) {
        for &llm_token_id in llm_token_ids {
            self.commit(llm_token_id);
        }
    }

    fn prepare_initial_nodes_and_values_for_special_map(&mut self) -> Vec<(Arc<Mutex<Trie<TerminalID, PrecomputedNodeContents>>>, (GLRParserState, LLMTokenBV))> {
        // The BTreeSet<TokenizerStateID> in each Trie node here is the set of terminal states at this node.
        // Each terminal state indicates that the path through the trie can terminate here.
        // (todo: explain this better)
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<Trie<GrammarTokenID, PrecomputedNodeContents>>>, (GLRParserState, LLMTokenBV))> = Vec::new();

        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, (BTreeSet<ParseState>, LLMTokenBV)> = BTreeMap::new();
        for managed_parse_state in self.state.active_states.iter() {
            for tokenizer_state_id in managed_parse_state.tokenizer_state_ids.iter() {
                let parse_state = ParseState::from(managed_parse_state.clone());
                tokenizer_state_id_to_parse_states.entry(*tokenizer_state_id).or_default().0.insert(parse_state);
                tokenizer_state_id_to_parse_states.entry(*tokenizer_state_id).or_default().1 |= managed_parse_state.llm_tokens.clone();
            }
        }

        for (tokenizer_state_id, (parse_states, llm_tokens)) in tokenizer_state_id_to_parse_states {
            let token_trie = self.parent.precomputed[&tokenizer_state_id].clone();
            let token_trie = Arc::new(Mutex::new(token_trie));
            let glr_parser_state = GLRParser::init_glr_parser_from_parse_states(self.state.parser, parse_states.into_iter().collect());
            initial_nodes_and_values.push((token_trie, (glr_parser_state, llm_tokens)));
        }
        initial_nodes_and_values
    }

    pub fn parse_grammar_token_trie(&mut self) {
        let initial_nodes_and_values = self.prepare_initial_nodes_and_values_for_special_map();

        let mut final_active_parse_states: Vec<ManagedParseState> = Vec::new();
        let mut final_inactive_parse_states: Vec<ManagedParseState> = Vec::new();

        Trie::special_map(
            initial_nodes_and_values,
            // step
            |(parse_state, llm_tokens), grammar_token_id, node| (parse_state.clone().with_step(*grammar_token_id), llm_tokens.clone()),
            // merge
            |(state1, llm_tokens1), (state2, llm_tokens2)| {
                state1.merge_with(state2);
                *llm_tokens1 |= llm_tokens2;
            },
            // process
            |precomputed_node_contents, (parse_state, llm_tokens)| {
                let mut tokenizer_state_ids = BTreeSet::new();
                for precomputed_finalizer in &precomputed_node_contents.finalizers {
                    tokenizer_state_ids.insert(precomputed_finalizer.tokenizer_state_id);
                }
                if !tokenizer_state_ids.is_empty() {
                    for active_state in &parse_state.active_states {
                        final_active_parse_states.push(ManagedParseState::from((active_state.clone(), tokenizer_state_ids.clone(), llm_tokens.clone())));
                    }
                    for inactive_state in &parse_state.inactive_states {
                        final_inactive_parse_states.push(ManagedParseState::from((inactive_state.clone(), tokenizer_state_ids.clone(), llm_tokens.clone())));
                    }
                }
                !parse_state.active_states.is_empty()
            },
        );

        self.state.active_states = final_active_parse_states;
        self.state.inactive_states.extend(final_inactive_parse_states);
    }
}