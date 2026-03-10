#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, HashSet};

use crate::GlrMaskError;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::regex::parse_regex;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::{
    GrammarDef, NonterminalID, Rule, Symbol, Terminal, TerminalID,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GrammarExpr {
    Ref(String),
    Sequence(Vec<GrammarExpr>),
    Choice(Vec<GrammarExpr>),
    Optional(Box<GrammarExpr>),
    Repeat(Box<GrammarExpr>),
    RepeatOne(Box<GrammarExpr>),
    Literal(Vec<u8>),
    CharClass { def: String, negate: bool, utf8: bool },
    RawRegex(String),
    AnyByte,
}

#[derive(Debug, Clone)]
pub struct NamedGrammar {
    pub rules: Vec<(String, GrammarExpr)>,
    pub start: String,
    /// Rule names that are terminal definitions (as opposed to nonterminals).
    /// Set explicitly by the importer — not derived from naming conventions.
    pub terminals: HashSet<String>,
    /// Name of the terminal rule whose body should be used as the ignore pattern.
    /// Set by Lark's `%ignore` directive.
    pub ignore: Option<String>,
}

struct Lowerer {
    rules: Vec<Rule>,
    terminal_map: BTreeMap<String, TerminalID>,
    terminals: Vec<Terminal>,
    nt_map: BTreeMap<String, NonterminalID>,
    anon_counter: u32,
    terminal_names: BTreeMap<TerminalID, String>,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            terminal_map: BTreeMap::new(),
            terminals: Vec::new(),
            nt_map: BTreeMap::new(),
            anon_counter: 0,
            terminal_names: BTreeMap::new(),
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

    fn terminal_id(&mut self, name: &str, pattern: &str, utf8: bool) -> TerminalID {
        let key = format!("{}:{}", pattern, utf8);
        if let Some(&id) = self.terminal_map.get(&key) {
            return id;
        }
        let id = self.terminals.len() as TerminalID;
        self.terminal_map.insert(key, id);
        self.terminal_names.insert(id, name.to_string());
        // Decide variant: if the pattern is the same as the escaped literal of
        // the name bytes, store as Literal; otherwise store as Pattern.
        let name_bytes = name.as_bytes();
        let escaped: String = name_bytes.iter().map(|&b| regex_escape_byte(b)).collect();
        if escaped == pattern && !utf8 {
            self.terminals.push(Terminal::Literal {
                id,
                bytes: name_bytes.to_vec(),
            });
        } else {
            self.terminals.push(Terminal::Pattern {
                id,
                pattern: pattern.to_string(),
                utf8,
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
                Symbol::Terminal(self.terminal_id(&String::from_utf8_lossy(bytes), &pattern, false))
            }
            GrammarExpr::CharClass { def, negate, utf8 } => {
                let pattern = if *negate {
                    format!("[^{def}]")
                } else {
                    format!("[{def}]")
                };
                Symbol::Terminal(self.terminal_id(&pattern, &pattern, *utf8))
            }
            GrammarExpr::RawRegex(pattern) => {
                // assume utf8 true for raw regex from lark/ebnf
                Symbol::Terminal(self.terminal_id(pattern, pattern, true))
            }
            GrammarExpr::AnyByte => {
                Symbol::Terminal(self.terminal_id(".", ".", false))
            }
            GrammarExpr::Sequence(_)
            | GrammarExpr::Choice(_)
            | GrammarExpr::Optional(_)
            | GrammarExpr::Repeat(_)
            | GrammarExpr::RepeatOne(_) => self.lower_expr(expr),
        })
    }

    /// Lower an entire terminal rule body to a single Terminal::Expr.
    /// The GrammarExpr must be fully expanded (no Ref nodes).
    fn lower_terminal_rule(&mut self, name: &str, body: &GrammarExpr) -> Result<TerminalID, GlrMaskError> {
        let expr = grammar_expr_to_expr(body)?;
        // Dedup by the Expr tree itself
        for (i, t) in self.terminals.iter().enumerate() {
            if let Terminal::Expr { expr: existing, .. } = t {
                if *existing == expr {
                    return Ok(i as TerminalID);
                }
            }
        }
        let id = self.terminals.len() as TerminalID;
        self.terminal_names.insert(id, name.to_string());
        self.terminals.push(Terminal::Expr { id, expr });
        Ok(id)
    }
}

/// Convert a fully-expanded GrammarExpr (no Ref nodes) to an Expr tree.
fn grammar_expr_to_expr(ge: &GrammarExpr) -> Result<Expr, GlrMaskError> {
    Ok(match ge {
        GrammarExpr::Literal(bytes) => Expr::U8Seq(bytes.clone()),
        GrammarExpr::CharClass { def, negate, utf8 } => {
            let pattern = if *negate {
                format!("[^{def}]")
            } else {
                format!("[{def}]")
            };
            parse_regex(&pattern, *utf8)
        }
        GrammarExpr::RawRegex(pattern) => parse_regex(pattern, true),
        GrammarExpr::AnyByte => Expr::U8Class(U8Set::from_range(0, 255)),
        GrammarExpr::Sequence(parts) => {
            let exprs: Vec<Expr> = parts.iter().map(grammar_expr_to_expr).collect::<Result<_, _>>()?;
            if exprs.len() == 1 {
                exprs.into_iter().next().unwrap()
            } else {
                Expr::Seq(exprs)
            }
        }
        GrammarExpr::Choice(options) => {
            let exprs: Vec<Expr> = options.iter().map(grammar_expr_to_expr).collect::<Result<_, _>>()?;
            if exprs.len() == 1 {
                exprs.into_iter().next().unwrap()
            } else {
                Expr::Choice(exprs)
            }
        }
        GrammarExpr::Optional(inner) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner)?),
            min: 0,
            max: Some(1),
        },
        GrammarExpr::Repeat(inner) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner)?),
            min: 0,
            max: None,
        },
        GrammarExpr::RepeatOne(inner) => Expr::Repeat {
            expr: Box::new(grammar_expr_to_expr(inner)?),
            min: 1,
            max: None,
        },
        GrammarExpr::Ref(name) => {
            return Err(GlrMaskError::GrammarParse(format!(
                "unexpected Ref({name}) in terminal body — should have been expanded"
            )));
        }
    })
}

pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    let mut lowerer = Lowerer::new();

    for (name, _) in &grammar.rules {
        lowerer.nt_id(name);
    }

    for (name, expr) in &grammar.rules {
        let lhs = lowerer.nt_id(name);

        // Terminal rules: convert the entire body to a single Terminal::Expr.
        // The body should be fully expanded (no Ref nodes).
        if grammar.terminals.contains(name) {
            let tid = lowerer.lower_terminal_rule(name, expr)?;
            lowerer.rules.push(Rule { lhs, rhs: vec![Symbol::Terminal(tid)] });
            continue;
        }

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
    let nonterminal_names = lowerer
        .nt_map
        .iter()
        .filter(|(name, _)| !name.starts_with("__"))
        .map(|(name, id)| (*id, name.clone()))
        .collect();

    let ignore_terminal = if let Some(ref ignore_name) = grammar.ignore {
        // Find the terminal created for the ignore rule.
        // The ignore rule is in grammar.terminals, so it was lowered above
        // as NT → Terminal. The terminal has the ignore name in terminal_names.
        let tid = lowerer.terminal_names.iter()
            .find(|(_, name)| *name == ignore_name)
            .map(|(&id, _)| id);
        tid
    } else {
        None
    };

    Ok(GrammarDef {
        rules: lowerer.rules,
        start,
        terminals: lowerer.terminals,
        nonterminal_names,
        terminal_names: lowerer.terminal_names,
        ignore_terminal,
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
            terminals: HashSet::new(), ignore: None,
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
            terminals: HashSet::new(), ignore: None,
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
            terminals: HashSet::new(), ignore: None,
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
            terminals: HashSet::new(), ignore: None,
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
            terminals: HashSet::new(), ignore: None,
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
            terminals: HashSet::new(), ignore: None,
        };
        let gdef = lower(&g).unwrap();
        assert_eq!(gdef.start, 0); 
        assert!(gdef.num_nonterminals() >= 2);
    }

    #[test]
    fn test_lower_retains_useful_names_but_not_helper_nonterminals() {
        let g = NamedGrammar {
            rules: vec![
                (
                    "start".into(),
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Ref("named_nt".into()),
                        GrammarExpr::Literal(b"term1".to_vec()),
                    ]),
                ),
                (
                    "named_nt".into(),
                    GrammarExpr::Optional(Box::new(GrammarExpr::Literal(b"term2".to_vec()))),
                ),
            ],
            start: "start".into(),
            terminals: HashSet::new(), ignore: None,
        };

        let gdef = lower(&g).unwrap();

        let nonterminal_names: Vec<&str> = gdef
            .nonterminal_names
            .values()
            .map(|name| name.as_str())
            .collect();
        assert!(nonterminal_names.contains(&"start"));
        assert!(nonterminal_names.contains(&"named_nt"));
        assert!(!nonterminal_names.iter().any(|name| name.starts_with("__")));

        let terminal_names: Vec<&str> = gdef
            .terminal_names
            .values()
            .map(|name| name.as_str())
            .collect();
        assert!(terminal_names.contains(&"term1"));
        assert!(terminal_names.contains(&"term2"));
    }
}
