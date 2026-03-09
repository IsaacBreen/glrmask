#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::possible_matches::build_possible_matches_from_token_bytes;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar_def::TerminalID;
use crate::ds::leveled_gss::LeveledGSS;

use super::state::ConstraintState;

pub(crate) type TokenizerStateID = u32;
pub(crate) type PossibleMatchesByState =
    BTreeMap<TokenizerStateID, BTreeMap<TerminalID, RangeSetBlaze<u32>>>;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct Constraint {
    pub(crate) parser_dwa: DWA,
    pub(crate) table: GLRTable,
    pub(crate) tokenizer: Tokenizer,
    #[serde(default)]
    pub(crate) ignore_terminal: Option<TerminalID>,

    #[serde(with = "crate::runtime::serde::serde_btmap_rsb")]
    pub(crate) possible_matches: PossibleMatchesByState,
    #[serde(skip, default)]
    pub(crate) possible_matches_lazy: std::sync::OnceLock<PossibleMatchesByState>,
    pub(crate) state_to_internal_tsid: Vec<u32>,
    pub(crate) internal_tsid_to_states: Vec<Vec<u32>>,
    pub(crate) eos_token_id: Option<u32>,
    pub(crate) token_bytes: BTreeMap<u32, Vec<u8>>,
}

impl Clone for Constraint {
    fn clone(&self) -> Self {
        Self {
            parser_dwa: self.parser_dwa.clone(),
            table: self.table.clone(),
            tokenizer: self.tokenizer.clone(),
            ignore_terminal: self.ignore_terminal,
            possible_matches: self.possible_matches.clone(),
            possible_matches_lazy: std::sync::OnceLock::new(),
            state_to_internal_tsid: self.state_to_internal_tsid.clone(),
            internal_tsid_to_states: self.internal_tsid_to_states.clone(),
            eos_token_id: self.eos_token_id,
            token_bytes: self.token_bytes.clone(),
        }
    }
}

impl Constraint {
    fn all_possible_matches(&self) -> &PossibleMatchesByState {
        if !self.possible_matches.is_empty() {
            &self.possible_matches
        } else {
            self.possible_matches_lazy.get_or_init(|| {
                build_possible_matches_from_token_bytes(&self.tokenizer, &self.token_bytes)
            })
        }
    }

    pub fn start(&self) -> ConstraintState<'_> {
        
        let initial_parser_state = 0u32;
        let initial_tok_state = self.tokenizer.initial_state();

        let mut state = BTreeMap::new();
        let gss = LeveledGSS::from_stacks(&[(vec![initial_parser_state], BTreeMap::new())]);
        state.insert(initial_tok_state, gss);

        ConstraintState { constraint: self, state }
    }

    pub fn mask_len(&self) -> usize {
        self.token_bytes
            .keys()
            .max()
            .map(|token_id| (*token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    pub(crate) fn parser_dwa(&self) -> &DWA {
        &self.parser_dwa
    }

    pub(crate) fn possible_matches_for_state(
        &self,
        tokenizer_state: u32,
    ) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
        self.all_possible_matches()
            .get(&tokenizer_state)
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn internal_tsid_for_state(&self, tokenizer_state: u32) -> u32 {
        self.state_to_internal_tsid
            .get(tokenizer_state as usize)
            .copied()
            .unwrap_or(tokenizer_state)
    }

    pub(crate) fn possible_matches_for_internal_tsid(
        &self,
        internal_tsid: u32,
    ) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
        let mut merged = BTreeMap::new();
        let Some(original_states) = self.internal_tsid_to_states.get(internal_tsid as usize) else {
            return merged;
        };

        for &tokenizer_state in original_states {
            for (terminal, token_ids) in self.possible_matches_for_state(tokenizer_state) {
                merged
                    .entry(terminal)
                    .and_modify(|existing: &mut RangeSetBlaze<u32>| {
                        *existing = existing.clone() | token_ids.clone();
                    })
                    .or_insert(token_ids);
            }
        }

        merged
    }
}
