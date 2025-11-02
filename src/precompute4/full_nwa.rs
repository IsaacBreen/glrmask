use crate::constraint::{PrecomputeNode1Index, Trie1GodWrapper};
use crate::glr::parser::GLRParser;
use crate::tokenizer::TokenizerStateID;
use crate::weighted_automata::DWA;
use std::collections::BTreeMap;

pub type Precomputed4 = BTreeMap<TokenizerStateID, DWA>;

pub fn precompute4(x: &GLRParser, x0: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>, x1: &Trie1GodWrapper) -> DWA {
    todo!()
}