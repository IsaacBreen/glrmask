//! Grammar expression AST and lowering to `GrammarDef`.
//!
//! Provides a high-level grammar IR (`GrammarExpr`) that frontends produce,
//! plus logic to flatten it into the low-level `GrammarDef` consumed by the compiler.
#![allow(unused_imports, unused_variables, dead_code)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code)]

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
        unimplemented!("cargo-check-only stub")
    }

    /// Get or create a nonterminal ID for a name.
    fn nt_id(&mut self, name: &str) -> NonterminalId {
        unimplemented!("cargo-check-only stub")
    }

    /// Create an anonymous nonterminal.
    fn fresh_nt(&mut self, hint: &str) -> (String, NonterminalId) {
        unimplemented!("cargo-check-only stub")
    }

    /// Get or create a terminal for a literal/pattern.
    fn terminal_id(&mut self, name: &str, pattern: &str) -> TerminalId {
        unimplemented!("cargo-check-only stub")
    }

    /// Lower a `GrammarExpr` into a single `Symbol` (creating helper rules as needed).
    fn lower_expr(&mut self, expr: &GrammarExpr) -> Symbol {
        unimplemented!("cargo-check-only stub")
    }
}

// ---------------------------------------------------------------------------
// Terminal rule helpers
// ---------------------------------------------------------------------------

/// Check if a rule name is a terminal name (ALL-CAPS, underscores, digits).
fn is_terminal_name(name: &str) -> bool {
    unimplemented!("cargo-check-only stub")
}

/// Compile a `GrammarExpr` into a regex pattern string.
///
/// `terminal_patterns` maps already-compiled terminal names to their regex patterns.
/// Returns `Err` if a referenced terminal is not yet compiled (for dependency resolution).
fn compile_to_regex(
    expr: &GrammarExpr,
    terminal_patterns: &BTreeMap<String, String>,
) -> Result<String, GlrMaskError> {
    unimplemented!("cargo-check-only stub")
}

/// Lower a `NamedGrammar` to a `GrammarDef`.
///
/// Terminal rules (ALL-CAPS names) are compiled to single terminals with regex
/// patterns. Nonterminal rules (lowercase names) reference terminals directly.
pub fn lower(grammar: &NamedGrammar) -> Result<GrammarDef, GlrMaskError> {
    unimplemented!("cargo-check-only stub")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn escape_byte(b: u8) -> String {
    unimplemented!("cargo-check-only stub")
}

fn regex_escape_byte(b: u8) -> String {
    unimplemented!("cargo-check-only stub")
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
