use std::collections::BTreeMap;

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::{EOF, AnalyzedGrammar};
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::{GrammarDef, TerminalID};
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;
use crate::compiler::stages::templates::Templates;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TerminalDiagnostics {
    pub nwa_after_build: NWA,
    pub nwa_after_collapse: NWA,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AutomataDiagnostics {
    pub characterizations: BTreeMap<TerminalID, TerminalCharacterization>,
    pub terminal_dwa: DWA,
    pub terminal_diagnostics: TerminalDiagnostics,
    pub templates: Templates,
    pub parser_nwa_before_resolve: NWA,
    pub parser_nwa_after_resolve: NWA,
    pub parser_dwa_pre_minimize: DWA,
    pub parser_dwa: DWA,
    pub id_map: InternalIdMap,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CompileDiagnostics {
    pub grammar_def: GrammarDef,
    pub normalized_grammar_def: GrammarDef,
    pub glr_grammar: AnalyzedGrammar,
    pub glr_table: GLRTable,
    pub characterizations: BTreeMap<TerminalID, TerminalCharacterization>,
    pub terminal_dwa: DWA,
    pub terminal_diagnostics: TerminalDiagnostics,
    pub templates: Templates,
    pub parser_nwa_before_resolve: NWA,
    pub parser_nwa_after_resolve: NWA,
    pub parser_dwa_pre_minimize: DWA,
    pub parser_dwa: DWA,
    pub id_map: InternalIdMap,
    pub vocab_entries: Vec<(u32, Vec<u8>)>,
    pub eos_token_id: Option<u32>,
}

impl CompileDiagnostics {
    pub fn from_parts(
        grammar_def: GrammarDef,
        normalized_grammar_def: GrammarDef,
        glr_grammar: AnalyzedGrammar,
        glr_table: GLRTable,
        automata: AutomataDiagnostics,
        vocab_entries: Vec<(u32, Vec<u8>)>,
        eos_token_id: Option<u32>,
    ) -> Self {
        Self {
            grammar_def,
            normalized_grammar_def,
            glr_grammar,
            glr_table,
            characterizations: automata.characterizations,
            terminal_dwa: automata.terminal_dwa,
            terminal_diagnostics: automata.terminal_diagnostics,
            templates: automata.templates,
            parser_nwa_before_resolve: automata.parser_nwa_before_resolve,
            parser_nwa_after_resolve: automata.parser_nwa_after_resolve,
            parser_dwa_pre_minimize: automata.parser_dwa_pre_minimize,
            parser_dwa: automata.parser_dwa,
            id_map: automata.id_map,
            vocab_entries,
            eos_token_id,
        }
    }
}

pub type TerminalDebug = TerminalDiagnostics;

pub type AutomataDebug = AutomataDiagnostics;

pub type CompileDebug = CompileDiagnostics;
