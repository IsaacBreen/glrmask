




#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: `CompileDebug` and `AutomataDebug` are glrmask-only aggregation structs; sep1 exposes the nearest intermediate pieces separately via `GLRParser`, parser-DWA builders, and ad hoc debug output rather than one compiler bundle.

use std::collections::BTreeMap;

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::{EOF, AnalyzedGrammar};
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::{GrammarDef, TerminalID};
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;
use crate::compiler::stages::templates::Templates;
use crate::compiler::terminal_dwa::TerminalDWA;






#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TerminalDebug {
    
    
    pub nwa_after_build: NWA,

    
    
    pub nwa_after_collapse: NWA,

    
    
}










#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AutomataDebug {
    
    pub characterizations: BTreeMap<TerminalID, TerminalCharacterization>,

    
    pub terminal_dwa: TerminalDWA,

    
    pub terminal_debug: TerminalDebug,

    
    pub templates: Templates,

    
    pub parser_nwa_before_resolve: NWA,

    
    pub parser_nwa_after_resolve: NWA,

    
    pub parser_dwa_pre_minimize: DWA,

    
    pub parser_dwa: DWA,

    
    pub id_map: InternalIdMap,
}






























#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CompileDebug {
    

    
    
    
    
    pub grammar_def: GrammarDef,

    
    
    
    
    pub normalized_grammar_def: GrammarDef,

    
    
    pub glr_grammar: AnalyzedGrammar,

    
    
    
    pub glr_table: GLRTable,

    

    
    pub characterizations: BTreeMap<TerminalID, TerminalCharacterization>,

    
    pub terminal_dwa: TerminalDWA,

    
    pub terminal_debug: TerminalDebug,

    

    
    pub templates: Templates,

    
    pub parser_nwa_before_resolve: NWA,

    
    pub parser_nwa_after_resolve: NWA,

    
    pub parser_dwa_pre_minimize: DWA,

    
    pub parser_dwa: DWA,

    

    
    pub id_map: InternalIdMap,

    
    
    pub vocab_entries: Vec<(u32, Vec<u8>)>,

    
    pub eos_token_id: Option<u32>,
}

impl CompileDebug {
    
    pub fn from_parts(
        grammar_def: GrammarDef,
        normalized_grammar_def: GrammarDef,
        glr_grammar: AnalyzedGrammar,
        glr_table: GLRTable,
        automata: AutomataDebug,
        vocab_entries: Vec<(u32, Vec<u8>)>,
        eos_token_id: Option<u32>,
    ) -> Self {
        unimplemented!()
    }
}
