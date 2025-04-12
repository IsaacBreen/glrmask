use crate::datastructures::gss::BulkMerge;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID};
use crate::datastructures::gss::{GSSNode, GSSTrait};

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, Mutex};
use crate::datastructures::trie::Trie;
use crate::debug;
use crate::finite_automata::Regex;
use crate::glr::parser::{Action, GLRParser, GLRParserState, ParseState, ParseStatus, StopReason};
use crate::tokenizer::{GrammarTokenID, TokenizerStateID};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManagedParseState {
    pub tokenizer_state_ids: BTreeSet<TokenizerStateID>,
    pub stack: Arc<GSSNode<StateID>>,
    pub action_stack: Option<Arc<GSSNode<Action>>>,
    pub status: ParseStatus,
}

impl GLRParser {
    pub fn init_managed_glr_parser(&self) -> ManagedGLRParserState {
        ManagedGLRParserState {
            parser: self,
            active_states: vec![self.init_managed_parse_state()],
            inactive_states: Vec::new(),
        }
    }

    pub fn init_managed_parse_state(&self) -> ManagedParseState {
        ManagedParseState {
            tokenizer_state_ids: vec![TokenizerStateID(0)].into_iter().collect(),
            stack: Arc::new(GSSNode::new(self.start_state_id)),
            action_stack: None,
            status: ParseStatus::Active,
        }
    }

}

#[derive(Debug, Clone)]
pub struct ManagedGLRParserState<'a> {
    pub parser: &'a GLRParser,
    pub active_states: Vec<ManagedParseState>,
    pub inactive_states: Vec<ManagedParseState>,
}

impl Regex {
    pub(crate) fn make_grammar_token_trie(&self, text: &[u8], states: BTreeSet<TokenizerStateID>) -> BTreeMap<TokenizerStateID, Trie<GrammarTokenID, BTreeSet<TokenizerStateID>>> {
        todo!()
    }

}

impl<'a> ManagedGLRParserState<'a> {
    pub fn parse_grammar_token_trie(&mut self, grammar_token_trie_roots: BTreeMap<TokenizerStateID, Trie<GrammarTokenID, BTreeSet<TokenizerStateID>>>) {
        let mut initial_nodes_and_values: Vec<(Arc<Mutex<Trie<GrammarTokenID, BTreeSet<TokenizerStateID>>>>, GLRParserState)> = Vec::new();
        let mut final_active_parse_states: Vec<ManagedParseState> = Vec::new();
        let mut final_inactive_parse_states: Vec<ManagedParseState> = Vec::new();

        let mut tokenizer_state_id_to_parse_states: BTreeMap<TokenizerStateID, BTreeSet<ParseState>> = BTreeMap::new();
        for managed_parse_state in self.active_states.iter() {
            for tokenizer_state_id in managed_parse_state.tokenizer_state_ids.iter() {
                let parse_state = ParseState {
                    stack: managed_parse_state.stack.clone(),
                    action_stack: managed_parse_state.action_stack.clone(),
                    status: managed_parse_state.status,
                };
                tokenizer_state_id_to_parse_states.entry(*tokenizer_state_id).or_default().insert(parse_state);
            }
        }

        for (tokenizer_state_id, parse_states) in tokenizer_state_id_to_parse_states {
            let token_trie = grammar_token_trie_roots[&tokenizer_state_id].clone();
            let token_trie = Arc::new(Mutex::new(token_trie));
            let glr_parser_state = GLRParser::init_glr_parser_from_parse_states(self.parser, parse_states.into_iter().collect());
            initial_nodes_and_values.push((token_trie, glr_parser_state));
        }

        Trie::special_map(
            initial_nodes_and_values,
            |parse_state, grammar_token_id, grammar_token_trie| {
                let mut parse_state = parse_state.clone();
                parse_state.step(TerminalID(grammar_token_id.0));
                parse_state
            },
            |parse_state, new_parse_state| {
                todo!()
            },
            |tokenizer_state_ids, parse_state| {
                todo!()
            },
        );

        self.active_states = final_active_parse_states;
        self.inactive_states.extend(final_inactive_parse_states);
    }

    pub fn merge_with(&mut self, other: &ManagedGLRParserState) {
        assert!(std::ptr::eq(&self.parser, &other.parser));
        self.active_states.extend(other.active_states.iter().cloned());
        self.inactive_states.extend(other.inactive_states.iter().cloned());
    }

    pub fn fully_matches(&self) -> bool {
        !self.fully_matching_states().is_empty()
    }

    pub fn fully_matching_states(&self) -> Vec<&ManagedParseState> {
        self.inactive_states.iter().filter(|state| state.status == ParseStatus::Inactive(StopReason::GotoNotFound)).collect()
    }

    pub fn is_ok(&self) -> bool {
        !self.active_states.is_empty() || self.fully_matches()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManagedParseStateKey {
    tokenizer_state_ids: BTreeSet<TokenizerStateID>,
    stack: StateID,
    action_stack: Option<Action>,
}

impl ManagedParseState {
    pub fn key(&self) -> ManagedParseStateKey {
        ManagedParseStateKey {
            tokenizer_state_ids: self.tokenizer_state_ids.clone(),
            stack: *self.stack.peek(),
            action_stack: self.action_stack.peek().cloned(),
        }
    }

    pub fn merge(&mut self, other: ManagedParseState) {
        assert_eq!(self.key(), other.key());
        Arc::make_mut(&mut self.stack).merge(Arc::unwrap_or_clone(other.stack));
        match (&mut self.action_stack, other.action_stack) {
            (Some(a), Some(b)) => {
                Arc::make_mut(a).merge(Arc::unwrap_or_clone(b));
            }
            (None, None) => {}
            _ => unreachable!(),
        }
    }
}

pub trait InsertWith<K, V> {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F);
}

impl<K, V> InsertWith<K, V> for BTreeMap<K, V> where K: Eq + Ord {
    fn insert_with<F: FnOnce(&mut V, V)>(&mut self, k: K, v: V, combine: F) {
        match self.entry(k) {
            std::collections::btree_map::Entry::Occupied(mut occupied) => {
                let value = occupied.get_mut();
                combine(value, v);
            }
            std::collections::btree_map::Entry::Vacant(vacant) => {
                vacant.insert(v);
            }
        }
    }
}