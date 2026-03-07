//! NOTE: template bundle assembly is intentionally deferred.
//! Keep only this parser-NWA assembly stage boundary for now.
// SEP1_MAP: sep1 performs the nearest work inside `precompute4/parser_dwa.rs` when it assembles parser-NWA pieces after template construction; glrmask keeps that boundary as its own placeholder file.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::automata::weighted::nwa::NWA;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::templates::compile_dfa::Templates;
use crate::ds::weight::Weight;

impl Templates {
    pub(crate) fn build_bundle(
        &self,
        _terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        todo!("template bundle assembly is intentionally left as a placeholder in this cleanup pass")
    }
}