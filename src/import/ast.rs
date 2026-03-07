



#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: `GrammarExpr` and `NamedGrammar` are the closest glrmask analogue of sep1's grammar-import layer in `interface/interface.rs`, lowered later into compiler-local numeric IDs.

use std::collections::BTreeMap;

use crate::GlrMaskError;
use crate::compiler::grammar_def::{
    GrammarDef, NonterminalID, Rule, Symbol, Terminal, TerminalID,
};






#[derive(Debug, Clone, PartialEq)]
pub enum GrammarExpr {
    
    Ref(String),
    
    Sequence(Vec<GrammarExpr>),
    
    Choice(Vec<GrammarExpr>),
    
    Optional(Box<GrammarExpr>),
    
    Repeat(Box<GrammarExpr>),
    
    RepeatOne(Box<GrammarExpr>),
    
    Literal(Vec<u8>),
    
    
    CharClass { def: String, negate: bool },
    
    RawRegex(String),
    
    AnyByte,
}


#[derive(Debug, Clone)]
pub struct NamedGrammar {
    
    pub rules: Vec<(String, GrammarExpr)>,
    
    pub start: String,
}






struct Lowerer {
    
    rules: Vec<Rule>,
    
    terminal_map: BTreeMap<String, TerminalID>,
    
    terminals: Vec<Terminal>,
    
    nt_map: BTreeMap<String, NonterminalID>,
    
    anon_counter: u32,
}

impl Lowerer {
    fn new() -> Self {
        unimplemented!()
    }

    
    fn nt_id(&mut self, name: &str) -> NonterminalID {
        unimplemented!()
    }

    
    fn fresh_nt(&mut self, hint: &str) -> (String, NonterminalID) {
        unimplemented!()
    }

    
    fn terminal_id(&mut self, name: &str, pattern: &str) -> TerminalID {
        unimplemented!()
    }

    
    fn lower_expr(&mut self, expr: &GrammarExpr) -> Symbol {
        unimplemented!()
    }
}






fn is_terminal_name(name: &str) -> bool {
    unimplemented!()
}





fn compile_to_regex(
    expr: &GrammarExpr,
    terminal_patterns: &BTreeMap<String, String>,
) -> Result<String, GlrMaskError> {
    unimplemented!()
}





pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    unimplemented!()
}





fn escape_byte(b: u8) -> String {
    unimplemented!()
}

fn regex_escape_byte(b: u8) -> String {
    unimplemented!()
}





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_simple_sequence() {
        let g = NamedGrammar {
            rules: vec![(
                "start".into(),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Literal(b"a".to_vec()),
                    GrammarExpr::Literal(b"b".to_vec()),
                ]),
            )],
            start: "start".into(),
        };
        let gdef = lower(&g).unwrap();
        assert_eq!(gdef.start, 0);
        assert!(!gdef.rules.is_empty());
        assert_eq!(gdef.num_terminals(), 2);
    }

    #[test]
    fn test_lower_choice() {
        let g = NamedGrammar {
            rules: vec![(
                "start".into(),
                GrammarExpr::Choice(vec![
                    GrammarExpr::Literal(b"a".to_vec()),
                    GrammarExpr::Literal(b"b".to_vec()),
                ]),
            )],
            start: "start".into(),
        };
        let gdef = lower(&g).unwrap();
        
        let start_rules: Vec<_> = gdef.rules.iter().filter(|r| r.lhs == 0).collect();
        assert_eq!(start_rules.len(), 2);
    }

    #[test]
    fn test_lower_optional() {
        let g = NamedGrammar {
            rules: vec![(
                "start".into(),
                GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"a".to_vec()))),
            )],
            start: "start".into(),
        };
        let gdef = lower(&g).unwrap();
        
        assert!(gdef.rules.len() >= 2);
    }

    #[test]
    fn test_lower_repeat() {
        let g = NamedGrammar {
            rules: vec![(
                "start".into(),
                GrammarExpr::RepeatOne(Box::new(GrammarExpr::Literal(b"a".to_vec()))),
            )],
            start: "start".into(),
        };
        let gdef = lower(&g).unwrap();
        
        assert!(gdef.rules.len() >= 2);
    }

    #[test]
    fn test_lower_multi_rule() {
        let g = NamedGrammar {
            rules: vec![
                (
                    "start".into(),
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Ref("item".into()),
                        GrammarExpr::Literal(b".".to_vec()),
                    ]),
                ),
                (
                    "item".into(),
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Literal(b"a".to_vec()),
                        GrammarExpr::Literal(b"b".to_vec()),
                    ]),
                ),
            ],
            start: "start".into(),
        };
        let gdef = lower(&g).unwrap();
        assert_eq!(gdef.start, 0); 
        assert!(gdef.num_nonterminals() >= 2);
    }
}
