#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]


pub use crate::grammar::surface::ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;

#[cfg(test)]
mod test_grammar_import;

#[cfg(test)]
mod test_json_schema;

pub use crate::grammar::surface::ast as grammar_expr;

use crate::compiler::debug::CompileDebug;
use crate::compiler::grammar_def::{GrammarDef, Rule, Symbol, Terminal};
use crate::compiler::{compile, compile_with_debug};
use crate::runtime::Constraint;

#[derive(Clone)]
enum SimpleSymbol {
    Nonterminal(String),
    Literal(Vec<u8>),
}

fn parse_quoted_literal(input: &str, index: &mut usize) -> crate::Result<Vec<u8>> {
    let bytes = input.as_bytes();
    let quote = bytes[*index];
    *index += 1;
    let mut out = Vec::new();
    while *index < bytes.len() {
        let byte = bytes[*index];
        *index += 1;
        if byte == quote {
            return Ok(out);
        }
        if byte == b'\\' && *index < bytes.len() {
            let escaped = bytes[*index];
            *index += 1;
            out.push(match escaped {
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                other => other,
            });
        } else {
            out.push(byte);
        }
    }
    Err(crate::GlrMaskError::GrammarParse("unterminated literal".into()))
}

fn parse_rhs_alternatives(input: &str) -> crate::Result<Vec<Vec<SimpleSymbol>>> {
    let mut alternatives = vec![Vec::new()];
    let bytes = input.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b' ' | b'\t' | b'\r' => index += 1,
            b'|' => {
                alternatives.push(Vec::new());
                index += 1;
            }
            b'"' | b'\'' => {
                let literal = parse_quoted_literal(input, &mut index)?;
                alternatives.last_mut().unwrap().push(SimpleSymbol::Literal(literal));
            }
            b if (b as char).is_ascii_alphanumeric() || b == b'_' => {
                let start = index;
                index += 1;
                while index < bytes.len() {
                    let byte = bytes[index];
                    if (byte as char).is_ascii_alphanumeric() || byte == b'_' {
                        index += 1;
                    } else {
                        break;
                    }
                }
                alternatives.last_mut().unwrap().push(SimpleSymbol::Nonterminal(input[start..index].to_string()));
            }
            other => {
                return Err(crate::GlrMaskError::GrammarParse(format!(
                    "unsupported EBNF byte '{}'",
                    other as char
                )));
            }
        }
    }
    Ok(alternatives)
}

fn parse_simple_ebnf(ebnf: &str) -> crate::Result<GrammarDef> {
    let mut productions = Vec::<(String, Vec<Vec<SimpleSymbol>>)>::new();
    for line in ebnf.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (lhs, rhs) = if let Some((lhs, rhs)) = trimmed.split_once("::=") {
            (lhs.trim(), rhs.trim())
        } else if let Some((lhs, rhs)) = trimmed.split_once(':') {
            (lhs.trim(), rhs.trim())
        } else {
            return Err(crate::GlrMaskError::GrammarParse(format!(
                "expected ::= or : in rule '{trimmed}'"
            )));
        };
        productions.push((lhs.to_string(), parse_rhs_alternatives(rhs)?));
    }

    let start_name = productions
        .first()
        .map(|(name, _)| name.clone())
        .ok_or_else(|| crate::GlrMaskError::GrammarParse("empty grammar".into()))?;

    let mut nt_map = std::collections::BTreeMap::new();
    for (index, (lhs, _)) in productions.iter().enumerate() {
        nt_map.insert(lhs.clone(), index as u32);
    }

    let mut terminal_map = std::collections::BTreeMap::<Vec<u8>, u32>::new();
    let mut terminals = Vec::<Terminal>::new();
    let mut terminal_patterns = Vec::<String>::new();
    let mut rules = Vec::<Rule>::new();

    for (lhs, alternatives) in productions {
        let lhs_id = nt_map[&lhs];
        for alternative in alternatives {
            let mut rhs = Vec::new();
            for symbol in alternative {
                match symbol {
                    SimpleSymbol::Nonterminal(name) => {
                        let Some(&nonterminal) = nt_map.get(&name) else {
                            return Err(crate::GlrMaskError::GrammarParse(format!(
                                "unknown nonterminal '{name}'"
                            )));
                        };
                        rhs.push(Symbol::Nonterminal(nonterminal));
                    }
                    SimpleSymbol::Literal(literal) => {
                        let terminal_id = if let Some(&id) = terminal_map.get(&literal) {
                            id
                        } else {
                            let id = terminals.len() as u32;
                            terminal_map.insert(literal.clone(), id);
                            terminals.push(Terminal {
                                id,
                                name: String::from_utf8_lossy(&literal).into_owned(),
                            });
                            terminal_patterns.push(String::from_utf8_lossy(&literal).into_owned());
                            id
                        };
                        rhs.push(Symbol::Terminal(terminal_id));
                    }
                }
            }
            rules.push(Rule { lhs: lhs_id, rhs });
        }
    }

    Ok(GrammarDef {
        rules,
        start: nt_map[&start_name],
        terminals,
        terminal_patterns,
    })
}


fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let grammar = ebnf::parse_ebnf(ebnf)?;
    Ok(compile(&grammar, vocab))
}


fn from_ebnf_with_debug(
    ebnf: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let grammar = ebnf::parse_ebnf(ebnf)?;
    Ok(compile_with_debug(&grammar, vocab))
}


fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (lark, vocab);
    Err(crate::GlrMaskError::Compilation(
        "Lark import is not implemented yet".into(),
    ))
}


fn from_lark_with_debug(
    lark: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let _ = (lark, vocab);
    Err(crate::GlrMaskError::Compilation(
        "Lark import with debug is not implemented yet".into(),
    ))
}


fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let grammar = json_schema::json_schema_to_grammar(schema)?;
    Ok(compile(&grammar, vocab))
}


fn from_json_schema_with_debug(
    schema: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let grammar = json_schema::json_schema_to_grammar(schema)?;
    Ok(compile_with_debug(&grammar, vocab))
}

impl Constraint {
    
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_ebnf(ebnf, vocab)
    }

    
    
    
    pub fn from_ebnf_with_debug(
        ebnf: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_ebnf_with_debug(ebnf, vocab)
    }

    
    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_lark(lark, vocab)
    }

    
    pub(crate) fn from_lark_with_debug(
        lark: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_lark_with_debug(lark, vocab)
    }

    
    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_json_schema(schema, vocab)
    }

    
    pub(crate) fn from_json_schema_with_debug(
        schema: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_json_schema_with_debug(schema, vocab)
    }
}
