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
    /// Raw regex pattern (verbatim, e.g. from `/regex/` in Lark).
    RawRegex(String),
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
                // Check if this references a compiled terminal.
                if let Some(&id) = self.terminal_map.get(name) {
                    Symbol::Terminal(id)
                } else {
                    let id = self.nt_id(name);
                    Symbol::Nonterminal(id)
                }
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
            GrammarExpr::RawRegex(pattern) => {
                let name = format!("_rx_{}", pattern);
                let id = self.terminal_id(&name, pattern);
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

// ---------------------------------------------------------------------------
// Terminal rule helpers
// ---------------------------------------------------------------------------

/// Check if a rule name is a terminal name (ALL-CAPS, underscores, digits).
fn is_terminal_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
        && name.chars().any(|c| c.is_ascii_uppercase())
}

/// Compile a `GrammarExpr` into a regex pattern string.
///
/// `terminal_patterns` maps already-compiled terminal names to their regex patterns.
/// Returns `Err` if a referenced terminal is not yet compiled (for dependency resolution).
fn compile_to_regex(
    expr: &GrammarExpr,
    terminal_patterns: &BTreeMap<String, String>,
) -> Result<String, GlrMaskError> {
    match expr {
        GrammarExpr::Literal(bytes) => {
            Ok(bytes.iter().map(|&b| regex_escape_byte(b)).collect())
        }
        GrammarExpr::CharClass { def, negate } => {
            if *negate {
                Ok(format!("[^{}]", def))
            } else {
                Ok(format!("[{}]", def))
            }
        }
        GrammarExpr::Choice(alts) => {
            let parts: Vec<String> = alts
                .iter()
                .map(|a| compile_to_regex(a, terminal_patterns))
                .collect::<Result<_, _>>()?;
            if parts.len() == 1 {
                Ok(parts.into_iter().next().unwrap())
            } else {
                Ok(format!("({})", parts.join("|")))
            }
        }
        GrammarExpr::Sequence(parts) => {
            let parts: Vec<String> = parts
                .iter()
                .map(|p| compile_to_regex(p, terminal_patterns))
                .collect::<Result<_, _>>()?;
            Ok(parts.join(""))
        }
        GrammarExpr::Optional(inner) => {
            let inner = compile_to_regex(inner, terminal_patterns)?;
            Ok(format!("({})?", inner))
        }
        GrammarExpr::Repeat(inner) => {
            let inner = compile_to_regex(inner, terminal_patterns)?;
            Ok(format!("({})*", inner))
        }
        GrammarExpr::RepeatOne(inner) => {
            let inner = compile_to_regex(inner, terminal_patterns)?;
            Ok(format!("({})+", inner))
        }
        GrammarExpr::RawRegex(s) => Ok(s.clone()),
        GrammarExpr::AnyByte => Ok(".".to_string()),
        GrammarExpr::Ref(name) => {
            if is_terminal_name(name) {
                if let Some(pattern) = terminal_patterns.get(name) {
                    Ok(format!("({})", pattern))
                } else {
                    Err(GlrMaskError::GrammarParse(format!(
                        "terminal '{}' not yet compiled",
                        name
                    )))
                }
            } else {
                Err(GlrMaskError::GrammarParse(format!(
                    "terminal rule body references non-terminal '{}'",
                    name
                )))
            }
        }
    }
}

/// Lower a `NamedGrammar` to a `GrammarDef`.
///
/// Terminal rules (ALL-CAPS names) are compiled to single terminals with regex
/// patterns. Nonterminal rules (lowercase names) reference terminals directly.
pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    let mut lowerer = Lowerer::new();

    // Phase 1: Identify terminal rules (ALL-CAPS names) and compile their
    // bodies into regex patterns. Handle dependencies by iterating until all
    // terminals are compiled.
    let mut terminal_patterns: BTreeMap<String, String> = BTreeMap::new();
    let mut remaining: Vec<(&str, &GrammarExpr)> = grammar
        .rules
        .iter()
        .filter(|(name, _)| is_terminal_name(name))
        .map(|(name, expr)| (name.as_str(), expr))
        .collect();

    let max_iterations = remaining.len() + 1;
    for _ in 0..max_iterations {
        if remaining.is_empty() {
            break;
        }
        let prev_count = remaining.len();
        let mut next_remaining = Vec::new();
        for (name, expr) in remaining {
            match compile_to_regex(expr, &terminal_patterns) {
                Ok(pattern) => {
                    terminal_patterns.insert(name.to_string(), pattern);
                }
                Err(_) => {
                    next_remaining.push((name, expr));
                }
            }
        }
        if next_remaining.len() == prev_count {
            // No progress — unresolved dependency.
            let names: Vec<_> = next_remaining.iter().map(|(n, _)| *n).collect();
            return Err(GlrMaskError::GrammarParse(format!(
                "unresolved terminal dependencies: {:?}",
                names
            )));
        }
        remaining = next_remaining;
    }

    // Register compiled terminals.
    for (name, pattern) in &terminal_patterns {
        lowerer.terminal_id(name, pattern);
    }

    // Phase 2: Pre-register nonterminals (only non-terminal rules).
    for (name, _) in &grammar.rules {
        if !is_terminal_name(name) {
            lowerer.nt_id(name);
        }
    }

    // Phase 3: Lower nonterminal rules.
    for (name, expr) in &grammar.rules {
        if is_terminal_name(name) {
            continue; // Already handled as a terminal.
        }
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
