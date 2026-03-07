



#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: `GrammarDef`/`Rule`/`Symbol` are the closest glrmask analogue to sep1's `GrammarDefinition` plus `glr::grammar::{Production, Symbol}`, but flattened into compiler-local numeric IDs.

use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarDef {
    
    pub rules: Vec<Rule>,
    
    pub start: NonterminalID,
    
    pub terminals: Vec<Terminal>,
    
    pub terminal_patterns: Vec<String>,
}


pub type NonterminalID = u32;


pub type TerminalID = u32;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    
    pub lhs: NonterminalID,
    
    pub rhs: Vec<Symbol>,
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Symbol {
    
    Terminal(TerminalID),
    
    Nonterminal(NonterminalID),
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Terminal {
    
    pub id: TerminalID,
    
    pub name: String,
}

impl GrammarDef {
    
    pub fn num_terminals(&self) -> u32 {
        unimplemented!()
    }

    
    pub fn num_nonterminals(&self) -> u32 {
        unimplemented!()
    }

    
    pub fn terminal_pattern(&self, terminal: TerminalID) -> &str {
        let _ = terminal;
        unimplemented!()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn terminal(id: u32, name: &str) -> Terminal {
        Terminal {
            id,
            name: name.into(),
        }
    }

    
    pub fn simple_ab_grammar() -> GrammarDef {
        GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![terminal(0, "a"), terminal(1, "b")],
            terminal_patterns: vec!["a".into(), "b".into()],
        }
    }

    
    pub fn choice_grammar() -> GrammarDef {
        GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            start: 0,
            terminals: vec![terminal(0, "a"), terminal(1, "b")],
            terminal_patterns: vec!["a".into(), "b".into()],
        }
    }

    
    pub fn two_nt_grammar() -> GrammarDef {
        
        
        GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![terminal(0, "a"), terminal(1, "b")],
            terminal_patterns: vec!["a".into(), "b".into()],
        }
    }

    
    pub fn nested_nt_grammar() -> GrammarDef {
        
        
        GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Terminal(1)],
                },
            ],
            start: 0,
            terminals: vec![terminal(0, "a"), terminal(1, "b")],
            terminal_patterns: vec!["a".into(), "b".into()],
        }
    }

    
    pub fn three_terminal_grammar() -> GrammarDef {
        GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Terminal(1),
                    Symbol::Terminal(2),
                ],
            }],
            start: 0,
            terminals: vec![terminal(0, "a"), terminal(1, "b"), terminal(2, "c")],
            terminal_patterns: vec!["a".into(), "b".into(), "c".into()],
        }
    }

    
    pub fn nested_two_rhs_grammar() -> GrammarDef {
        
        
        GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
                },
            ],
            start: 0,
            terminals: vec![terminal(0, "a"), terminal(1, "b"), terminal(2, "c")],
            terminal_patterns: vec!["a".into(), "b".into(), "c".into()],
        }
    }

    #[test]
    fn test_grammar_def_basics() {
        let g = simple_ab_grammar();
        assert_eq!(g.num_terminals(), 2);
        assert_eq!(g.num_nonterminals(), 1);
    }
}
