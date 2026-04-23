pub use crate::grammar::ast as ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;
pub mod numeric_range;

#[cfg(test)]
mod test_grammar_import;

#[cfg(test)]
mod test_json_schema;

use crate::compiler::compile::{compile_owned_profiled, compile_profile_enabled, emit_compile_profile_summary};
use crate::compiler::compile_owned;
#[cfg(debug_assertions)]
use crate::compiler::compile::build_tokenizer;
#[cfg(debug_assertions)]
use crate::compiler::glr::analysis::{AnalyzedGrammar, EOF};
use crate::grammar::flat::GrammarDef;
#[cfg(debug_assertions)]
use crate::grammar::flat::Symbol;
use crate::grammar::factoring::factor_named_grammar;
use crate::runtime::Constraint;

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;

#[cfg(debug_assertions)]
fn debug_panic_on_ab_single_byte_overlap_follow(grammar: &GrammarDef) {
    let num_terminals = grammar.num_terminals() as usize;
    if num_terminals == 0 {
        return;
    }

    // Build terminal-follow relation from grammar production context.
    let analyzed = AnalyzedGrammar::from_grammar_def(grammar);
    let mut terminal_follow = vec![vec![false; num_terminals]; num_terminals];

    for rule in &analyzed.rules {
        let lhs = rule.lhs as usize;
        for (idx, sym) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(t0) = sym else { continue; };
            if (*t0 as usize) >= num_terminals {
                continue;
            }

            let mut suffix_nullable = true;
            for next in rule.rhs.iter().skip(idx + 1) {
                match next {
                    Symbol::Terminal(t1) => {
                        if (*t1 as usize) < num_terminals {
                            terminal_follow[*t0 as usize][*t1 as usize] = true;
                        }
                        suffix_nullable = false;
                        break;
                    }
                    Symbol::Nonterminal(nt) => {
                        if let Some(first) = analyzed.first.get(*nt as usize) {
                            for &t in first {
                                if t != EOF && (t as usize) < num_terminals {
                                    terminal_follow[*t0 as usize][t as usize] = true;
                                }
                            }
                        }
                        if !analyzed.nullable.contains(nt) {
                            suffix_nullable = false;
                            break;
                        }
                    }
                }
            }

            if suffix_nullable {
                if let Some(follow) = analyzed.follow.get(lhs) {
                    for &t in follow {
                        if t != EOF && (t as usize) < num_terminals {
                            terminal_follow[*t0 as usize][t as usize] = true;
                        }
                    }
                }
            }
        }
    }

    let tokenizer = build_tokenizer(grammar);
    let input = b"ab";
    let initial = tokenizer.initial_state();

    let mut start_states: Vec<Option<u32>> = vec![Some(initial); input.len() + 1];
    for pos in 1..=input.len() {
        start_states[pos] = tokenizer.execute_from_state_end_only(&input[..pos], initial);
    }

    for pos in 0..input.len() {
        let Some(start_state) = start_states[pos] else { continue; };
        let exec = tokenizer.execute_from_state_all_widths(&input[pos..], start_state);

        let mut width_one_terms: Vec<u32> = exec
            .matches
            .iter()
            .filter(|m| m.width == 1)
            .map(|m| m.id)
            .collect();
        width_one_terms.sort_unstable();
        width_one_terms.dedup();

        let mut involved_terminals: Vec<u32> = Vec::new();

        for &t0 in &width_one_terms {
            for &t1 in &width_one_terms {
                if (t0 as usize) < num_terminals
                    && (t1 as usize) < num_terminals
                    && terminal_follow[t0 as usize][t1 as usize]
                {
                    involved_terminals.push(t0);
                    involved_terminals.push(t1);
                }
            }
        }

        if !involved_terminals.is_empty() {
            involved_terminals.sort_unstable();
            involved_terminals.dedup();

            let terminal_names: Vec<String> = involved_terminals
                .iter()
                .map(|&t| grammar.terminal_display_name(t).to_string())
                .collect();

            panic!(
                "debug json_schema check failed on input 'ab': pos={} span=[{},{}), terminals in any conflict=[{}]",
                pos,
                pos,
                pos + 1,
                terminal_names.join(", "),
            );
        }
    }
}

fn compile_from_source(
    source: &str,
    vocab: &crate::Vocab,
    source_kind: &str,
    parse: GrammarParser,
) -> crate::Result<Constraint> {
    if compile_profile_enabled() {
        let parse_started_at = std::time::Instant::now();
        let grammar = parse(source)?;
        #[cfg(debug_assertions)]
        if source_kind == "json_schema" {
            debug_panic_on_ab_single_byte_overlap_follow(&grammar);
        }
        let import_ms = parse_started_at.elapsed().as_secs_f64() * 1000.0;
        let (constraint, profile) = compile_owned_profiled(grammar, vocab);
        emit_compile_profile_summary(Some(source_kind), Some(import_ms), &profile);
        return Ok(constraint);
    }

    let grammar = parse(source)?;
    #[cfg(debug_assertions)]
    if source_kind == "json_schema" {
        debug_panic_on_ab_single_byte_overlap_follow(&grammar);
    }
    Ok(compile_owned(grammar, vocab))
}

fn glrm_to_grammar_def(source: &str) -> crate::Result<GrammarDef> {
    let named = crate::grammar::glrm::from_glrm(source)?;
    let factored = factor_named_grammar(named);
    ast::lower(&factored)
}

impl Constraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(ebnf, vocab, "ebnf", ebnf::parse_ebnf)
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(lark, vocab, "lark", lark::parse_lark)
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(schema, vocab, "json_schema", json_schema::json_schema_to_grammar)
    }

    /// Load a grammar from the GLRM format (see [`crate::grammar::glrm`]).
    pub fn from_glrm_grammar(glrm: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(glrm, vocab, "glrm", glrm_to_grammar_def)
    }
}
