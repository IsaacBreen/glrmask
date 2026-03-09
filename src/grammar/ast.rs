#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::GlrMaskError;
use crate::grammar::flat::{
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
        Self {
            rules: Vec::new(),
            terminal_map: BTreeMap::new(),
            terminals: Vec::new(),
            nt_map: BTreeMap::new(),
            anon_counter: 0,
        }
    }

    fn nt_id(&mut self, name: &str) -> NonterminalID {
        if let Some(&id) = self.nt_map.get(name) {
            id
        } else {
            let id = self.nt_map.len() as NonterminalID;
            self.nt_map.insert(name.to_string(), id);
            id
        }
    }

    fn fresh_nt(&mut self, hint: &str) -> (String, NonterminalID) {
        let name = format!("__{}_{}", hint, self.anon_counter);
        self.anon_counter += 1;
        let id = self.nt_id(&name);
        (name, id)
    }

    fn terminal_id(&mut self, name: &str, pattern: &str) -> TerminalID {
        if let Some(&id) = self.terminal_map.get(pattern) {
            return id;
        }
        let id = self.terminals.len() as TerminalID;
        self.terminal_map.insert(pattern.to_string(), id);
        // Decide variant: if the pattern is the same as the escaped literal of
        // the name bytes, store as Literal; otherwise store as Pattern.
        let name_bytes = name.as_bytes();
        let escaped: String = name_bytes.iter().map(|&b| regex_escape_byte(b)).collect();
        if escaped == pattern {
            self.terminals.push(Terminal::Literal {
                id,
                bytes: name_bytes.to_vec(),
            });
        } else {
            self.terminals.push(Terminal::Pattern {
                id,
                pattern: pattern.to_string(),
            });
        }
        id
    }

    fn lower_expr(&mut self, expr: &GrammarExpr) -> Symbol {
        fn emit(lowerer: &mut Lowerer, lhs: NonterminalID, expr: &GrammarExpr) -> Result<(), GlrMaskError> {
            match expr {
                GrammarExpr::Sequence(parts) => {
                    let mut rhs = Vec::new();
                    for part in parts {
                        rhs.push(lowerer.lower_expr(part));
                    }
                    lowerer.rules.push(Rule { lhs, rhs });
                }
                GrammarExpr::Choice(options) => {
                    for option in options {
                        emit(lowerer, lhs, option)?;
                    }
                }
                GrammarExpr::Optional(inner) => {
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    emit(lowerer, lhs, inner)?;
                }
                GrammarExpr::Repeat(inner) => {
                    let item = lowerer.lower_expr(inner);
                    lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                    lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(lhs), item] });
                }
                GrammarExpr::RepeatOne(inner) => {
                    let item = lowerer.lower_expr(inner);
                    lowerer.rules.push(Rule { lhs, rhs: vec![item.clone()] });
                    lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(lhs), item] });
                }
                _ => {
                    let symbol = lowerer.lower_expr_terminalish(expr)?;
                    lowerer.rules.push(Rule {
                        lhs,
                        rhs: vec![symbol],
                    });
                }
            }
            Ok(())
        }

        let (_, nt) = self.fresh_nt("expr");
        emit(self, nt, expr).expect("grammar lowering should not fail for internal expression emission");
        Symbol::Nonterminal(nt)
    }

    fn lower_expr_terminalish(&mut self, expr: &GrammarExpr) -> Result<Symbol, GlrMaskError> {
        Ok(match expr {
            GrammarExpr::Ref(name) => Symbol::Nonterminal(self.nt_id(name)),
            GrammarExpr::Literal(bytes) => {
                let pattern = bytes.iter().map(|&b| regex_escape_byte(b)).collect::<String>();
                Symbol::Terminal(self.terminal_id(&String::from_utf8_lossy(bytes), &pattern))
            }
            GrammarExpr::CharClass { def, negate } => {
                let pattern = if *negate {
                    format!("[^{def}]")
                } else {
                    format!("[{def}]")
                };
                Symbol::Terminal(self.terminal_id(&pattern, &pattern))
            }
            GrammarExpr::RawRegex(pattern) => {
                Symbol::Terminal(self.terminal_id(pattern, pattern))
            }
            GrammarExpr::AnyByte => {
                Symbol::Terminal(self.terminal_id(".", "."))
            }
            GrammarExpr::Sequence(_)
            | GrammarExpr::Choice(_)
            | GrammarExpr::Optional(_)
            | GrammarExpr::Repeat(_)
            | GrammarExpr::RepeatOne(_) => self.lower_expr(expr),
        })
    }
}

fn is_terminal_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(|ch| ch.is_ascii_uppercase() || ch == '_')
}

fn compile_to_regex(
    expr: &GrammarExpr,
    terminal_patterns: &BTreeMap<String, String>,
) -> Result<String, GlrMaskError> {
    Ok(match expr {
        GrammarExpr::Ref(name) => terminal_patterns
            .get(name)
            .cloned()
            .ok_or_else(|| GlrMaskError::GrammarParse(format!("unknown terminal '{name}'")))?,
        GrammarExpr::Sequence(parts) => parts
            .iter()
            .map(|part| compile_to_regex(part, terminal_patterns))
            .collect::<Result<Vec<_>, _>>()?
            .join(""),
        GrammarExpr::Choice(options) => options
            .iter()
            .map(|option| compile_to_regex(option, terminal_patterns))
            .collect::<Result<Vec<_>, _>>()?
            .join("|"),
        GrammarExpr::Optional(inner) => format!("(?:{})?", compile_to_regex(inner, terminal_patterns)?),
        GrammarExpr::Repeat(inner) => format!("(?:{})*", compile_to_regex(inner, terminal_patterns)?),
        GrammarExpr::RepeatOne(inner) => format!("(?:{})+", compile_to_regex(inner, terminal_patterns)?),
        GrammarExpr::Literal(bytes) => bytes.iter().map(|&b| regex_escape_byte(b)).collect(),
        GrammarExpr::CharClass { def, negate } => {
            if *negate { format!("[^{def}]") } else { format!("[{def}]") }
        }
        GrammarExpr::RawRegex(pattern) => pattern.clone(),
        GrammarExpr::AnyByte => ".".into(),
    })
}

pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    let mut lowerer = Lowerer::new();

    for (name, _) in &grammar.rules {
        lowerer.nt_id(name);
    }

    for (name, expr) in &grammar.rules {
        let lhs = lowerer.nt_id(name);
        match expr {
            GrammarExpr::Sequence(parts) => {
                let rhs = parts.iter().map(|part| lowerer.lower_expr_terminalish(part)).collect::<Result<Vec<_>, _>>()?;
                lowerer.rules.push(Rule { lhs, rhs });
            }
            GrammarExpr::Choice(options) => {
                for option in options {
                    match option {
                        GrammarExpr::Sequence(parts) => {
                            let rhs = parts.iter().map(|part| lowerer.lower_expr_terminalish(part)).collect::<Result<Vec<_>, _>>()?;
                            lowerer.rules.push(Rule { lhs, rhs });
                        }
                        _ => {
                            let symbol = lowerer.lower_expr_terminalish(option)?;
                            lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
                        }
                    }
                }
            }
            GrammarExpr::Optional(inner) => {
                lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
            GrammarExpr::Repeat(inner) => {
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: Vec::new() });
                lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(lhs), symbol] });
            }
            GrammarExpr::RepeatOne(inner) => {
                let symbol = lowerer.lower_expr_terminalish(inner)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol.clone()] });
                lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Nonterminal(lhs), symbol] });
            }
            _ => {
                let symbol = lowerer.lower_expr_terminalish(expr)?;
                lowerer.rules.push(Rule { lhs, rhs: vec![symbol] });
            }
        }
    }

    let start = lowerer.nt_id(&grammar.start);

    Ok(GrammarDef {
        rules: lowerer.rules,
        start,
        terminals: lowerer.terminals,
    })
}

fn escape_byte(b: u8) -> String {
    match b {
        b'\n' => "\\n".into(),
        b'\r' => "\\r".into(),
        b'\t' => "\\t".into(),
        b'\\' => "\\\\".into(),
        b'"' => "\\\"".into(),
        byte if byte.is_ascii_graphic() || byte == b' ' => (byte as char).to_string(),
        byte => format!("\\x{byte:02x}"),
    }
}

fn regex_escape_byte(b: u8) -> String {
    match b {
        b'.' | b'+' | b'*' | b'?' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'|' | b'^' | b'$' | b'\\' => {
            format!("\\{}", b as char)
        }
        _ => escape_byte(b),
    }
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

    /// Adapted from sep1 `test_nullability_handling_in_from_exprs`.
    #[test]
    fn test_lower_nullability_uses_epsilon_rules_not_empty_terminals() {
        let g = NamedGrammar {
            rules: vec![(
                "start".into(),
                GrammarExpr::Sequence(vec![
                    GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"x".to_vec()))),
                    GrammarExpr::Sequence(vec![]),
                    GrammarExpr::Literal(b"z".to_vec()),
                ]),
            )],
            start: "start".into(),
        };
        let gdef = lower(&g).unwrap();

        assert_eq!(gdef.terminals.len(), 2, "only the concrete x/z literals should become terminals");
        assert!(gdef.terminals.iter().any(|terminal| matches!(terminal, Terminal::Literal { bytes, .. } if bytes == b"x")));
        assert!(gdef.terminals.iter().any(|terminal| matches!(terminal, Terminal::Literal { bytes, .. } if bytes == b"z")));
        assert!(
            !gdef
                .terminals
                .iter()
                .any(|terminal| matches!(terminal, Terminal::Literal { bytes, .. } if bytes.is_empty())),
            "nullable pieces should lower through epsilon productions, not through empty terminals"
        );

        assert!(
            gdef.rules.iter().any(|rule| rule.lhs != gdef.start && rule.rhs.is_empty()),
            "lowering nullable pieces should introduce helper epsilon productions"
        );
        assert!(
            gdef.rules.iter().any(|rule| {
                rule.lhs == gdef.start
                    && rule.rhs.len() == 3
                    && matches!(rule.rhs[0], Symbol::Nonterminal(_))
                    && matches!(rule.rhs[1], Symbol::Nonterminal(_))
                    && matches!(rule.rhs[2], Symbol::Terminal(_))
            }),
            "the start rule should sequence the optional helper, the explicit epsilon helper, and the trailing literal"
        );
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
