//! Template bundle assembly surface.
//!
//! This stage turns compiled template DFAs plus terminal weights into a parser
//! NWA. The algorithmic body is intentionally deferred for now.
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