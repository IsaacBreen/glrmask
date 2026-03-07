


















#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This file merges the grammar-analysis and normalization responsibilities sep1 keeps in `glr/analyze.rs` into one compiler-local pass over `GrammarDef`.

use std::collections::{BTreeSet, BTreeMap};

use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Rule, Symbol, TerminalID};


pub const EOF: TerminalID = u32::MAX;


#[derive(Debug, Clone)]
pub struct AnalyzedGrammar {
    
    pub rules: Vec<Rule>,
    #[allow(dead_code)]
    
    pub start: NonterminalID,
    
    pub num_terminals: u32,
    
    pub num_nonterminals: u32,
    
    pub nullable: BTreeSet<NonterminalID>,
    
    pub first: Vec<BTreeSet<TerminalID>>,
    
    pub follow: Vec<BTreeSet<TerminalID>>,
}

impl AnalyzedGrammar {
    
    
    
    
    
    
    
    
    
    
    
    pub fn from_grammar_def(g: &GrammarDef) -> Self {
        unimplemented!()
    }

    
    pub fn first_of_seq(&self, seq: &[Symbol]) -> BTreeSet<TerminalID> {
        unimplemented!()
    }

    
    pub fn seq_is_nullable(&self, seq: &[Symbol]) -> bool {
        unimplemented!()
    }
}


























pub fn normalize_for_mask(g: &GrammarDef) -> GrammarDef {
    unimplemented!()
}





































#[allow(dead_code)]
pub(crate) fn eliminate_direct_left_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) {
    unimplemented!()
}




















pub(crate) fn eliminate_right_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) {
    unimplemented!()
}


fn max_nt_id(rules: &[Rule]) -> u32 {
    unimplemented!()
}






fn build_right_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    unimplemented!()
}




fn find_indirect_rr_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    unimplemented!()
}






fn inline_right_end(
    rules: &mut Vec<Rule>,
    from_nt: NonterminalID,
    to_nt: NonterminalID,
    nullable: &BTreeSet<NonterminalID>,
) {
    unimplemented!()
}



fn is_direct_right_recursive(rule: &Rule) -> bool {
    unimplemented!()
}

















fn resolve_direct_rr_single_nt(
    rules: &mut Vec<Rule>,
    nt: NonterminalID,
    new_nt: NonterminalID,
) {
    unimplemented!()
}






















pub(crate) fn inline_epsilon_rules(rules: &[Rule]) -> Vec<Rule> {
    unimplemented!()
}





fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {
    unimplemented!()
}





fn compute_first(
    rules: &[Rule],
    num_nt: u32,
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BTreeSet<TerminalID>> {
    unimplemented!()
}





fn compute_follow(
    rules: &[Rule],
    num_nt: u32,
    start: NonterminalID,
    first: &[BTreeSet<TerminalID>],
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BTreeSet<TerminalID>> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::model::tests::*;

    #[test]
    fn test_glr_grammar_simple() {
        let g = AnalyzedGrammar::from_grammar_def(&simple_ab_grammar());
        
        assert_eq!(g.rules.len(), 2);
        assert_eq!(g.num_nonterminals, 2); 
        assert_eq!(g.num_terminals, 2);
        assert!(g.nullable.is_empty());
        
        assert!(g.first[0].contains(&0));
        assert!(!g.first[0].contains(&1));
        
        assert!(g.follow[0].contains(&EOF));
    }

    #[test]
    fn test_glr_grammar_choice() {
        let g = AnalyzedGrammar::from_grammar_def(&choice_grammar());
        
        assert!(g.first[0].contains(&0));
        assert!(g.first[0].contains(&1));
    }

    #[test]
    fn test_glr_grammar_two_nt() {
        let g = AnalyzedGrammar::from_grammar_def(&two_nt_grammar());
        
        assert!(g.first[0].contains(&0)); 
        assert!(g.first[1].contains(&0)); 
        
        assert!(g.follow[1].contains(&1)); 
    }
}
