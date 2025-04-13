use crate::datastructures::gss::BulkMerge;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID};
use crate::datastructures::gss::{GSSNode, GSSTrait};
use crate::types::{TerminalID as GrammarTokenID};

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, Mutex};
use crate::constraint::LLMTokenBV;
use crate::datastructures::trie::Trie;
use crate::debug;
use crate::finite_automata::Regex;
use crate::glr::parser::{Action, GLRParser, GLRParserState, ParseState, ParseStatus, StopReason};
use crate::tokenizer::{TokenizerStateID};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManagedParseState {
    pub tokenizer_state_ids: BTreeSet<TokenizerStateID>,
    pub llm_tokens: LLMTokenBV,
    pub stack: Arc<GSSNode<StateID>>,
    pub action_stack: Option<Arc<GSSNode<Action>>>,
    pub status: ParseStatus,
}

impl From<ManagedParseState> for ParseState {
    fn from(managed_parse_state: ManagedParseState) -> Self {
        ParseState {
            stack: managed_parse_state.stack,
            action_stack: managed_parse_state.action_stack,
            status: managed_parse_state.status,
        }
    }
}

impl From<(ParseState, BTreeSet<TokenizerStateID>, LLMTokenBV)> for ManagedParseState {
    fn from(x: (ParseState, BTreeSet<TokenizerStateID>, LLMTokenBV)) -> Self {
        let (parse_state, tokenizer_state_ids, llm_tokens) = x;
        ManagedParseState {
            tokenizer_state_ids,
            llm_tokens: llm_tokens.clone(),
            stack: parse_state.stack,
            action_stack: parse_state.action_stack,
            status: parse_state.status,
        }
    }
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
            llm_tokens: LLMTokenBV::repeat(true, self.terminal_map.len()),
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

impl<'a> ManagedGLRParserState<'a> {
    pub fn merge_active_states(&mut self) {
        let mut active_state_map: BTreeMap<ManagedParseStateKey, ManagedParseState> = BTreeMap::new();

        let mut new_active_states = Vec::new();

        for mut state in std::mem::take(&mut self.active_states) {
            let key = state.key();
            if let Some(existing) = active_state_map.get_mut(&key) {
                Arc::make_mut(&mut existing.stack).merge(state.stack.as_ref().clone());
                if let Some(existing_action_stack) = existing.action_stack.as_mut() {
                    Arc::make_mut(existing_action_stack).merge(state.action_stack.unwrap().as_ref().clone());
                }
            } else {
                active_state_map.insert(key, state.clone());
                new_active_states.push(state);
            }
        }
        self.active_states = new_active_states;
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
