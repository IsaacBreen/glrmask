use crate::datastructures::gss::BulkMerge;
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::items::Item;
use crate::glr::table::{NonTerminalID, ProductionID, Stage7ShiftsAndReduces, Stage7Table, StateID, TerminalID};
use crate::datastructures::gss::{GSSNode, GSSTrait};

use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use crate::debug;
use crate::glr::parser::{Action, GLRParser, ParseStatus, StopReason};
use crate::tokenizer::TokenizerStateID;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManagedParseState {
    pub tokenizer_state_ids: Vec<TokenizerStateID>,
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
            tokenizer_state_ids: vec![TokenizerStateID(0)],
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
pub struct ParseStateKey {
    stack: StateID,
    action_stack: Option<Action>,
}

impl ManagedParseState {
    pub fn key(&self) -> ParseStateKey {
        ParseStateKey {
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