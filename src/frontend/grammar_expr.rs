//! Grammar expression AST and lowering to `GrammarDef`.
//!
//! Provides a high-level grammar IR (`GrammarExpr`) that frontends produce,
//! plus logic to flatten it into the low-level `GrammarDef` consumed by the compiler.

use std::collections::BTreeMap;

use crate::GlrMaskError;
use crate::compiler::grammar_def::{
    GrammarDef, NonterminalId, Rule, Symbol, TerminalDef, TerminalId,
};

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

/// High-level grammar expression (AST node).
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarExpr {
    /// Reference to a named rule.
    Ref(String),
    /// Sequence of sub-expressions (concatenation).
    Sequence(Vec<GrammarExpr>),
    /// Choice between alternatives (union).
    Choice(Vec<GrammarExpr>),
    /// Optional (zero or one).
    Optional(Box<GrammarExpr>),
    /// Kleene star (zero or more).
    Repeat(Box<GrammarExpr>),
    /// One or more.
    RepeatOne(Box<GrammarExpr>),
    /// Literal byte string.
    Literal(Vec<u8>),
    /// Character class: `def` is the bracket expression (e.g. `a-zA-Z`).
    /// If `negate` is true, it's `[^...]`.
    CharClass { def: String, negate: bool },
    /// Match any single byte (`.`).
    AnyByte,
}

/// A named grammar: maps rule names to their bodies. `start` is the entry rule.
#[derive(Debug, Clone)]
pub struct NamedGrammar {
    /// Rule name → expression. Insertion order is preserved.
    pub rules: Vec<(String, GrammarExpr)>,
    /// Name of the start rule (must appear in `rules`).
    pub start: String,
}

// ---------------------------------------------------------------------------
// Lowering: NamedGrammar → GrammarDef
// ---------------------------------------------------------------------------

/// State for the lowering pass.
struct Lowerer {
    /// Final rules.
    rules: Vec<Rule>,
    /// Terminal name → id.
    terminal_map: BTreeMap<String, TerminalId>,
    /// Terminal definitions.
    terminals: Vec<TerminalDef>,
    /// Rule name → nonterminal id.
    nt_map: BTreeMap<String, NonterminalId>,
    /// Counter for anonymous nonterminals.
    anon_counter: u32,
}

impl Lowerer {
    fn new() -> Self {
        Lowerer {
            rules: Vec::new(),
            terminal_map: BTreeMap::new(),
            terminals: Vec::new(),
            nt_map: BTreeMap::new(),
            anon_counter: 0,
        }
    }

    /// Get or create a nonterminal ID for a name.
    fn nt_id(&mut self, name: &str) -> NonterminalId {
        if let Some(&id) = self.nt_map.get(name) {
            id
        } else {
            let id = self.nt_map.len() as NonterminalId;
            self.nt_map.insert(name.to_string(), id);
            id
        }
    }

    /// Create an anonymous nonterminal.
    fn fresh_nt(&mut self, hint: &str) -> (String, NonterminalId) {
        let name = format!("_anon_{hint}_{}", self.anon_counter);
        self.anon_counter += 1;
        let id = self.nt_id(&name);
        (name, id)
    }

    /// Get or create a terminal for a literal/pattern.
    fn terminal_id(&mut self, name: &str, pattern: &str) -> TerminalId {
        if let Some(&id) = self.terminal_map.get(name) {
            id
        } else {
            let id = self.terminals.len() as TerminalId;
            self.terminals.push(TerminalDef {
                id,
                name: name.to_string(),
                pattern: pattern.to_string(),
            });
            self.terminal_map.insert(name.to_string(), id);
            id
        }
    }

    /// Lower a `GrammarExpr` into a single `Symbol` (creating helper rules as needed).
    fn lower_expr(&mut self, expr: &GrammarExpr) -> Symbol {
        match expr {
            GrammarExpr::Ref(name) => {
                let id = self.nt_id(name);
                Symbol::Nonterminal(id)
            }
            GrammarExpr::Literal(bytes) => {
                if bytes.len() == 1 {
                    // Single byte → single terminal.
                    let ch = bytes[0];
                    let name = format!("_lit_{}", escape_byte(ch));
                    let pattern = regex_escape_byte(ch);
                    let id = self.terminal_id(&name, &pattern);
                    Symbol::Terminal(id)
                } else {
                    // Multi-byte literal → sequence of terminals wrapped in a nonterminal.
                    let (_, nt) = self.fresh_nt("lit");
                    let rhs: Vec<Symbol> = bytes
                        .iter()
                        .map(|&b| {
                            let name = format!("_lit_{}", escape_byte(b));
                            let pattern = regex_escape_byte(b);
                            let id = self.terminal_id(&name, &pattern);
                            Symbol::Terminal(id)
                        })
                        .collect();
                    self.rules.push(Rule { lhs: nt, rhs });
                    Symbol::Nonterminal(nt)
                }
            }
            GrammarExpr::CharClass { def, negate } => {
                let pattern = if *negate {
                    format!("[^{}]", def)
                } else {
                    format!("[{}]", def)
                };
                let name = format!("_cc_{}", &pattern);
                let id = self.terminal_id(&name, &pattern);
                Symbol::Terminal(id)
            }
            GrammarExpr::AnyByte => {
                let name = "_any".to_string();
                let pattern = ".".to_string();
                let id = self.terminal_id(&name, &pattern);
                Symbol::Terminal(id)
            }
            GrammarExpr::Sequence(parts) => {
                if parts.len() == 1 {
                    return self.lower_expr(&parts[0]);
                }
                let (_, nt) = self.fresh_nt("seq");
                let rhs: Vec<Symbol> = parts.iter().map(|p| self.lower_expr(p)).collect();
                self.rules.push(Rule { lhs: nt, rhs });
                Symbol::Nonterminal(nt)
            }
            GrammarExpr::Choice(alts) => {
                if alts.len() == 1 {
                    return self.lower_expr(&alts[0]);
                }
                let (_, nt) = self.fresh_nt("choice");
                for alt in alts {
                    let sym = self.lower_expr(alt);
                    self.rules.push(Rule {
                        lhs: nt,
                        rhs: vec![sym],
                    });
                }
                Symbol::Nonterminal(nt)
            }
            GrammarExpr::Optional(inner) => {
                // A? → _anon → A | ε
                let (_, nt) = self.fresh_nt("opt");
                let sym = self.lower_expr(inner);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![sym],
                });
                // ε-production.
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![],
                });
                Symbol::Nonterminal(nt)
            }
            GrammarExpr::Repeat(inner) => {
                // A* → _anon → _anon A | ε
                let (_, nt) = self.fresh_nt("rep");
                let sym = self.lower_expr(inner);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Nonterminal(nt), sym],
                });
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![],
                });
                Symbol::Nonterminal(nt)
            }
            GrammarExpr::RepeatOne(inner) => {
                // A+ → _anon → _anon A | A
                let (_, nt) = self.fresh_nt("rep1");
                let sym = self.lower_expr(inner);
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![Symbol::Nonterminal(nt), sym.clone()],
                });
                self.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![sym],
                });
                Symbol::Nonterminal(nt)
            }
        }
    }
}

/// Lower a `NamedGrammar` to a `GrammarDef`.
pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    let mut lowerer = Lowerer::new();

    // Pre-register all named nonterminals to ensure stable IDs.
    for (name, _) in &grammar.rules {
        lowerer.nt_id(name);
    }

    // Lower each named rule.
    for (name, expr) in &grammar.rules {
        let nt = lowerer.nt_id(name);
        match expr {
            // Top-level choice → multiple rules for the same NT.
            GrammarExpr::Choice(alts) => {
                for alt in alts {
                    let rhs = match alt {
                        GrammarExpr::Sequence(parts) => {
                            parts.iter().map(|p| lowerer.lower_expr(p)).collect()
                        }
                        other => vec![lowerer.lower_expr(other)],
                    };
                    lowerer.rules.push(Rule { lhs: nt, rhs });
                }
            }
            GrammarExpr::Sequence(parts) => {
                let rhs: Vec<Symbol> = parts.iter().map(|p| lowerer.lower_expr(p)).collect();
                lowerer.rules.push(Rule { lhs: nt, rhs });
            }
            other => {
                let sym = lowerer.lower_expr(other);
                lowerer.rules.push(Rule {
                    lhs: nt,
                    rhs: vec![sym],
                });
            }
        }
    }

    let start = *lowerer.nt_map.get(&grammar.start).ok_or_else(|| {
        GlrMaskError::GrammarParse(format!("start rule '{}' not found", grammar.start))
    })?;

    Ok(GrammarDef {
        rules: lowerer.rules,
        start,
        terminals: lowerer.terminals,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn escape_byte(b: u8) -> String {
    if b.is_ascii_alphanumeric() || b == b'_' {
        String::from(b as char)
    } else {
        format!("x{:02x}", b)
    }
}

fn regex_escape_byte(b: u8) -> String {
    let ch = b as char;
    match ch {
        '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '\\' | '^' | '$' | '|' => {
            format!("\\{}", ch)
        }
        _ if b.is_ascii_graphic() || b == b' ' => String::from(ch),
        _ => format!("\\x{:02x}", b),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        // Two rules for start: one producing "a", one producing "b".
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
        // start → _anon_opt, _anon_opt → "a" | ε
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
        // start → _anon_rep1, _anon_rep1 → _anon_rep1 "a" | "a"
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
        assert_eq!(gdef.start, 0); // "start" registered first
        assert!(gdef.num_nonterminals() >= 2);
    }
}
