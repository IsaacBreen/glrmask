use crate::constraint::{GrammarConstraint};
use crate::debug;
use crate::finite_automata::{greedy_group, groups, ExprGroup, GroupID};
use crate::finite_automata::{Expr, Regex};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::GLRParser;
use crate::glr::table::{assign_non_terminal_ids, generate_glr_parser, generate_glr_parser_with_maps, NonTerminalID, TerminalID};
use crate::tokenizer::LLMTokenID;
use bimap::BiBTreeMap;
use kdam::tqdm;
use std::collections::{BTreeMap, HashSet, HashMap, BTreeSet};
use std::fmt::{Debug, Formatter};

type LLMToken<'a> = &'a [u8];
type LLMTokenMap = BiBTreeMap<Vec<u8>, LLMTokenID>;

#[derive(Clone)]
pub struct Grammar {
    pub productions: Vec<Production>,
    pub start_production_id: usize,
    pub literal_map: BTreeMap<String, String>,
    pub terminal_name_to_group_id: BiBTreeMap<String, usize>,
    pub terminal_expr_to_group_id: BiBTreeMap<Expr, usize>,
    pub tokenizer: Regex,
    pub glr_parser: GLRParser, // Store the generated parser
}

impl Debug for Grammar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Grammar:")?;
        writeln!(f, "  Start Production ID: {}", self.start_production_id)?;
        writeln!(f, "  Productions:")?;

        for production in &self.productions {
            write!(f, "    {} -> ", production.lhs.0)?;
            for (i, symbol) in production.rhs.iter().enumerate() {
                match symbol {
                    Symbol::Terminal(terminal) => write!(f, "{}", terminal.0)?,
                    Symbol::NonTerminal(non_terminal) => write!(f, "{}", non_terminal.0)?,
                }
                if i < production.rhs.len() - 1 {
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "  Literal Map:")?;
        for (literal, mangled_name) in &self.literal_map {
            writeln!(f, "    {:?}: {}", literal, mangled_name)?;
        }

        writeln!(f, "  Terminal Map (name to group ID):")?;
        for (name, group_id) in &self.terminal_name_to_group_id {
            writeln!(f, "    {:?}: {}", name, group_id)?;
        }

        writeln!(f, "Tokenizer:");
        writeln!(f, "{:?}", &self.tokenizer);

        Ok(())
    }
}

impl Grammar {
    fn mangle_literal(literal: &str, tokens: &BTreeMap<String, Expr>) -> String {
        let mut mangled_name = literal.to_string();
        let mut i = 0;
        while tokens.contains_key(&mangled_name) {
            mangled_name = format!("{}__literal_{}", literal, i);
            i += 1;
        }
        mangled_name
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarExpr {
    RegexExpr(Expr),
    Ref(String),
    Sequence(Vec<GrammarExpr>),
    Choice(Vec<GrammarExpr>),
    Optional(Box<GrammarExpr>),
    Repeat(Box<GrammarExpr>),
}

pub fn regex(expr: Expr) -> GrammarExpr {
    GrammarExpr::RegexExpr(expr)
}

pub fn r#ref(name: &str) -> GrammarExpr {
    GrammarExpr::Ref(name.to_string())
}

pub fn sequence(exprs: Vec<GrammarExpr>) -> GrammarExpr {
    GrammarExpr::Sequence(exprs)
}

pub fn choice(exprs: Vec<GrammarExpr>) -> GrammarExpr {
    GrammarExpr::Choice(exprs)
}

pub fn optional(expr: GrammarExpr) -> GrammarExpr {
    GrammarExpr::Optional(Box::new(expr))
}

pub fn repeat(expr: GrammarExpr) -> GrammarExpr {
    GrammarExpr::Repeat(Box::new(expr))
}

impl Grammar {
    pub fn glr_parser(&self) -> GLRParser {
        // Return a clone or reference if stored
        generate_glr_parser(&self.productions, self.start_production_id)
    }
}

impl Grammar {
    /// Constructs a `Grammar` and `Regex` tokenizer from a list of grammar expressions.
    /// The first non-terminal in the list is treated as the start symbol.
    pub fn from_exprs(exprs: Vec<(String, GrammarExpr)>) -> Self {
        let mut productions = Vec::new();
        let mut literal_map = BTreeMap::new();
        let mut terminal_name_to_group_id = BiBTreeMap::new();
        let mut terminal_expr_to_group_id = BiBTreeMap::new();
        let mut next_terminal_id = 0;

        // Add a start production.
        // make sure the start production name is not already taken by adding apostrophes to it until it's unique.
        let mut start_production_name = "start'".to_string();
        let nonterminals: HashSet<&str> = exprs.iter().map(|(name, _)| name.as_str()).collect();
        while nonterminals.contains(&start_production_name.as_str()) {
            start_production_name.push('\'');
        }
        debug!(2, "start_production_name: {:?}", start_production_name);
        productions.push(Production {
            lhs: NonTerminal(start_production_name.clone()),
            rhs: vec![Symbol::NonTerminal(NonTerminal(exprs[0].0.clone()))],
        });

        fn convert_expr(
            expr: &GrammarExpr,
            productions: &mut Vec<Production>,
            non_terminal_map: &mut BiBTreeMap<NonTerminal, NonTerminalID>,
            next_non_terminal_id: &mut usize,
            literal_map: &mut BTreeMap<String, String>,
            tokens: &mut BTreeMap<String, Expr>,
            terminal_name_to_group_id: &mut BiBTreeMap<String, usize>,
            // todo: make this `terminal_group_id_to_expr` instead
            terminal_expr_to_group_id: &mut BiBTreeMap<Expr, usize>,
            next_terminal_id: &mut usize,
        ) -> Vec<Symbol> {
            // TODO: Pre-collect all user-defined non-terminal names to ensure generated names are unique.
            // TODO: define a function that makes us a unique name for an internal rule, with an appropriate prefix.
            //  e.g. Option0, Repeat0, etc. Make sure there's no existing rule with that name (and there won't be one later either).
            //  i.e. collect all nonterminals in the grammar upfront and pass it to convert_expr.
            match expr {
                GrammarExpr::RegexExpr(regex_expr) => {
                    // TODO: what if this is already in the map (e.g. the user happens to create a rule with name `__regex_0`?
                    //  We need to generate a unique regex name.
                    if let Some(terminal_id) = terminal_expr_to_group_id.get_by_left(&regex_expr) {
                        vec![Symbol::Terminal(Terminal(format!("__regex_{}", terminal_id)))]
                    } else {
                        // Create a unique terminal name for this regex expression
                        let terminal_id = *next_terminal_id;
                        let terminal_name = format!("__regex_{}", terminal_id);
                        terminal_name_to_group_id.insert(terminal_name.clone(), terminal_id);
                        terminal_expr_to_group_id.insert(regex_expr.clone(), terminal_id);
                        tokens.insert(terminal_name.clone(), regex_expr.clone());
                        *next_terminal_id += 1;
                        vec![Symbol::Terminal(Terminal(terminal_name))]
                    }
                }
                GrammarExpr::Ref(name) => vec![Symbol::NonTerminal(NonTerminal(name.clone()))],
                GrammarExpr::Sequence(exprs) => exprs
                    .iter()
                    .flat_map(|e| {
                        convert_expr(
                            e,
                            productions,
                            non_terminal_map,
                            next_non_terminal_id,
                            literal_map,
                            tokens,
                            terminal_name_to_group_id,
                            terminal_expr_to_group_id,
                            next_terminal_id,
                        )
                    })
                    .collect(),
                GrammarExpr::Choice(exprs) => {
                    let new_nonterminal = format!("Choice{}", *next_non_terminal_id);
                    let nt = NonTerminal(new_nonterminal.clone());

                    // TODO: what if this is already in the map (e.g. the user happens to create a rule with name `Choice0`?
                    //  We need to generate a unique nonterminal name.
                    if !non_terminal_map.contains_left(&nt) {
                        non_terminal_map.insert(nt.clone(), NonTerminalID(*next_non_terminal_id));
                        *next_non_terminal_id += 1;
                    }

                    for expr in exprs {
                        let rhs = convert_expr(
                            expr,
                            productions,
                            non_terminal_map,
                            next_non_terminal_id,
                            literal_map,
                            tokens,
                            terminal_name_to_group_id,
                            terminal_expr_to_group_id,
                            next_terminal_id,
                        );
                        productions.push(Production {
                            lhs: nt.clone(),
                            rhs,
                        });
                    }

                    vec![Symbol::NonTerminal(nt)]
                }
                GrammarExpr::Optional(expr) => {
                    // TODO: name the internal rule here Option{} or something rather than Choice{}.
                    convert_expr(
                        &GrammarExpr::Choice(vec![*expr.clone(), GrammarExpr::Sequence(vec![])]),
                        productions,
                        non_terminal_map,
                        next_non_terminal_id,
                        literal_map,
                        tokens,
                        terminal_name_to_group_id,
                        terminal_expr_to_group_id,
                        next_terminal_id,
                    )
                }
                GrammarExpr::Repeat(expr) => {
                    // TODO: same as above, make sure it's unique.
                    let nonterminal_id = *next_non_terminal_id;
                    let nonterminal_name = format!("Repeat{}", nonterminal_id);
                    non_terminal_map.insert(NonTerminal(nonterminal_name.clone()), NonTerminalID(nonterminal_id));
                    *next_non_terminal_id += 1;
                    let rhs = convert_expr(
                        expr,
                        productions,
                        non_terminal_map,
                        next_non_terminal_id,
                        literal_map,
                        tokens,
                        terminal_name_to_group_id,
                        terminal_expr_to_group_id,
                        next_terminal_id,
                    );
                    productions.push(Production {
                        lhs: NonTerminal(nonterminal_name.clone()),
                        rhs,
                    });
                    vec![Symbol::NonTerminal(NonTerminal(nonterminal_name))]
                }
            }
        }

        let mut non_terminal_map = BiBTreeMap::new();
        let mut next_non_terminal_id = 0;
        let mut tokens = BTreeMap::new();

        // Process each rule definition
        for (name, expr) in tqdm!(exprs.iter()) {
            let lhs = NonTerminal(name.clone());
            // Optimization: If the top-level expression is a Choice, create multiple productions directly.
            if let GrammarExpr::Choice(choices) = expr {
                for choice_expr in choices {
                    let rhs = convert_expr(
                        choice_expr,
                        &mut productions,
                        &mut non_terminal_map,
                        &mut next_non_terminal_id,
                        &mut literal_map,
                        &mut tokens,
                        &mut terminal_name_to_group_id,
                        &mut terminal_expr_to_group_id,
                        &mut next_terminal_id,
                    );
                    productions.push(Production { lhs: lhs.clone(), rhs });
                }
            } else {
                // Otherwise, convert the expression as usual and create a single production.
                let rhs = convert_expr(
                    expr,
                    &mut productions,
                    &mut non_terminal_map,
                    &mut next_non_terminal_id,
                    &mut literal_map,
                    &mut tokens,
                    &mut terminal_name_to_group_id,
                    &mut terminal_expr_to_group_id,
                    &mut next_terminal_id,
                );
                productions.push(Production { lhs, rhs });
            }
        }

        let tokenizer_exprs_vec: Vec<ExprGroup> = tokens
            .into_iter()
            .map(|(_, expr)| greedy_group(expr))
            .collect();
        let tokenizer_expr_groups = groups(tokenizer_exprs_vec);
        debug!(2, "Building tokenizer");
        let tokenizer = tokenizer_expr_groups.clone().build();
        let glr_parser = generate_glr_parser(&productions, 0); // Generate parser once

        debug!(2, "Done defining grammar");
        Self {
            productions,
            start_production_id: 0,
            literal_map,
            terminal_name_to_group_id,
            terminal_expr_to_group_id,
            tokenizer,
            glr_parser,
        }
    }
}

impl GrammarConstraint {
    pub fn from_grammar(grammar: Grammar, llm_tokens: LLMTokenMap, eof_llm_token_id: usize, max_llm_token_id: usize) -> Self {
        debug!(2, "GrammarConstraint::from_grammar");
        let parser = grammar.glr_parser.clone(); // Use stored parser
        debug!(2, "Precomputing");
        let mut precomputed = GrammarConstraint::precompute(&grammar.tokenizer, &llm_tokens, max_llm_token_id);
        debug!(2, "precomputed.len(): {}", precomputed.len());
        debug!(2, "Done precomputing");

        Self {
            tokenizer: grammar.tokenizer,
            parser,
            precomputed,
            llm_token_map: llm_tokens,
            max_llm_token_id,
        }
    }
}

// --- Incremental Parser ---

use crate::glr::parser::GLRParserState;
use crate::tokenizer::{ExecuteResult, TokenizerStateID};

/// Manages incremental parsing against a grammar.
#[derive(Clone)]
pub struct IncrementalParser<'a> {
    grammar: &'a Grammar,
    // Maps current tokenizer state IDs to the GLR parser states reachable at that point.
    pub(crate) state: BTreeMap<TokenizerStateID, GLRParserState<'a, ()>>,
}

impl<'a> IncrementalParser<'a> {
    /// Creates a new incremental parser initialized to the start state.
    pub fn new(grammar: &'a Grammar) -> Self {
        let initial_glr_state = grammar.glr_parser.init_glr_parser::<()>();
        let initial_tokenizer_state = grammar.tokenizer.initial_state_id();
        let state = BTreeMap::from([(initial_tokenizer_state, initial_glr_state)]);
        Self { grammar, state }
    }

    /// Processes a chunk of input bytes, updating the internal state.
    pub fn feed(&mut self, bytes: &[u8]) {
        let mut next_states: BTreeMap<TokenizerStateID, GLRParserState<'a, ()>> = BTreeMap::new();
        let mut queue: BTreeMap<(usize, TokenizerStateID), GLRParserState<'a, ()>> = BTreeMap::new();

        // Initialize the queue with the current state
        for (tokenizer_state_id, glr_state) in std::mem::take(&mut self.state) {
            queue.insert((0, tokenizer_state_id), glr_state);
        }

        while let Some(((position, current_tokenizer_state_id), current_glr_state)) = queue.pop_first() {
            // Execute the tokenizer from the current state with the new bytes
            let results: ExecuteResult = self
                .grammar
                .tokenizer
                .execute_from_state(&bytes[position..], current_tokenizer_state_id);

            // Handle full token matches
            for token in results.matches {
                let grammar_token_id = TerminalID(token.id); // Assuming GroupID maps directly to TerminalID
                let mut next_glr_state = current_glr_state.clone(); // Clone before stepping
                next_glr_state.step(grammar_token_id);

                if next_glr_state.is_ok() {
                    if position + token.width == bytes.len() {
                        // Reached the end of the input, so this is a clean match
                        let next_tokenizer_state_id = self.grammar.tokenizer.initial_state_id();
                        next_states.entry(next_tokenizer_state_id)
                            .and_modify(|existing_state| existing_state.merge_with(next_glr_state.clone()))
                            .or_insert(next_glr_state.clone()); // Carry over the *original* GLR state
                    } else {
                        // After a full match, the tokenizer resets to its initial state
                        let next_tokenizer_state_id = self.grammar.tokenizer.initial_state_id();
                        queue.entry((position + token.width, next_tokenizer_state_id))
                            .and_modify(|existing_state| existing_state.merge_with(next_glr_state.clone()))
                            .or_insert(next_glr_state);
                    }
                }
            }

            // Handle partial matches (tokenizer ended mid-token)
            if let Some(end_state_id) = results.end_state {
                // Ensure at least one possible final token parses
                let possible_final_grammar_tokens: Vec<_> = self.grammar.tokenizer.tokens_accessible_from_state(TokenizerStateID(end_state_id));
                for possible_final_grammar_token in possible_final_grammar_tokens {
                    let mut final_glr_state = current_glr_state.clone(); // Clone before stepping
                    final_glr_state.step(possible_final_grammar_token);
                    if final_glr_state.is_ok() {
                        let next_tokenizer_state_id = TokenizerStateID(end_state_id);
                        next_states.entry(next_tokenizer_state_id)
                            .and_modify(|existing_state| existing_state.merge_with(current_glr_state.clone()))
                            .or_insert(current_glr_state.clone()); // Carry over the *original* GLR state
                    }
                }
            }
        }

        self.state = next_states;
    }

    /// Checks if the current state is valid (i.e., there's at least one active parse path).
    pub fn is_valid(&self) -> bool {
        self.state.values().any(|glr_state| glr_state.is_ok())
    }

    // TODO: Add is_accepting() method? Requires checking if any state can accept EOF.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finite_automata::eat_u8;
    use crate::interface::tokenizer_combinators::{eat_u8_fast, eat_u8_negation_fast, eat_u8_range_fast, repeat0_fast};
    use crate::{choice_fast, groups, seq_fast};
    use bitvec::prelude::*;
    use std::sync::{Arc, Mutex};

    fn bitvec_with_capacity_and_values(capacity: usize, values: Vec<usize>) -> BitVec {
        let mut bitvec = BitVec::new();
        bitvec.resize(capacity, false);
        for value in values {
            bitvec.set(value, true);
        }
        bitvec
    }

    #[ignore]
    #[test]
    fn test_grammar_from_exprs() {
        let exprs = vec![
            (
                "E".to_string(),
                choice(vec![
                    sequence(vec![
                        r#ref("E"),
                        regex(eat_u8(b'+')),
                        r#ref("T"),
                    ]),
                    r#ref("T"),
                ]),
            ),
            (
                "T".to_string(),
                choice(vec![
                    sequence(vec![
                        r#ref("T"),
                        regex(eat_u8(b'*')),
                        r#ref("F"),
                    ]),
                    r#ref("F"),
                ]),
            ),
            (
                "F".to_string(),
                choice(vec![
                    sequence(vec![
                        regex(eat_u8(b'(')),
                        r#ref("E"),
                        regex(eat_u8(b')')),
                    ]),
                    regex(eat_u8(b'i')),
                ]),
            ),
        ];

        let grammar = Grammar::from_exprs(exprs.clone());
        debug!(2, "{:?}", &grammar);

        let parser = grammar.glr_parser();
        debug!(2, "{:?}", &parser);

        let llm_tokens: Vec<Vec<u8>> = vec![b"i".to_vec(), b"+".to_vec(), b"*".to_vec(), b"(".to_vec(), b")".to_vec(), b"(i".to_vec(), b"+i".to_vec()];
        let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
        let eof_llm_token_id = llm_tokens.len();
        let max_llm_token_id = llm_tokens.len();
        let grammar_constraint = GrammarConstraint::from_grammar(grammar, llm_token_map.clone(), eof_llm_token_id, max_llm_token_id);
        let mut grammar_constraint_state = grammar_constraint.init();

        macro_rules! llm_token_vec {
            ($($token:expr),* $(,)?) => {
                vec![
                    $(
                        llm_token_map.get_by_left(&$token.to_vec()).unwrap().0,
                    )*
                ]
            }
        }

        // Get the mask.
        // The valid LLM tokens initially are ["i", "(", "(i"].
        let mask = grammar_constraint_state.get_mask();
        let expected_mask = bitvec_with_capacity_and_values(llm_tokens.len() + 1, llm_token_vec!(b"i", b"(", b"(i"));
        assert_eq!(mask, expected_mask);

        // Simulate generating from a LLM with the grammar constraint.
        // We may have some 'prefill' we want to pass to the parser before we generate the first new LLM token.
        // Let's say the prefill is "(i+i*i".
        // This would be best tokenized as ["(i", "+", "i", "*", "i"].
        //
        // Take note of the ambiguity in the LLM tokens; we could the prefill as ["(", "i", "+", "i", "*", "i"],
        // i.e. break the "(i" token into "(" and "i". But that's a waste of a token.
        // A good LLM tokenizer would greedily emit the longest possible token at each step.
        let prefill: Vec<_> = llm_token_vec!(b"(i", b"+i", b"*", b"i").into_iter().map(|token_id| LLMTokenID(token_id)).collect();
        grammar_constraint_state.commit_and_step_many(&prefill);

        // Get the mask.
        // The valid LLM tokens right now are ["+", "*", ")", "+i)"].
        let mask = grammar_constraint_state.get_mask();
        let expected_mask = bitvec_with_capacity_and_values(llm_tokens.len() + 1, llm_token_vec!(b"+", b"*", b")", b"+i"));
        assert_eq!(mask, expected_mask);

        // Finish it
        let terminals: Vec<_> = llm_token_vec!(b")").into_iter().map(|token_id| LLMTokenID(token_id)).collect();
        grammar_constraint_state.commit_and_step_many(&terminals);
        let mask = grammar_constraint_state.get_mask();
        let mut expected_mask = bitvec_with_capacity_and_values(llm_tokens.len() + 1, llm_token_vec!(b"+", b"*", b"+i"));
        // Add the EOF token
        expected_mask.set(llm_tokens.len(), true);
        assert_eq!(mask, expected_mask);

    }

    #[ignore]
    #[test]
    fn test_grammar_from_exprs_simple() {
        let exprs = vec![
            (
                "E".to_string(),
                sequence(vec![
                    regex(eat_u8(b'a')),
                    regex(eat_u8(b'b')),
                ]),
            ),
        ];

        let grammar = Grammar::from_exprs(exprs.clone());
        dbg!(&grammar);

        let parser = grammar.glr_parser();
        dbg!(&parser);

        let llm_tokens: Vec<Vec<u8>> = vec![b"a".to_vec(), b"b".to_vec()];
        let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
        let eof_llm_token_id = llm_tokens.len();
        let max_llm_token_id = llm_tokens.len();
        let grammar_constraint = GrammarConstraint::from_grammar(grammar, llm_token_map.clone(), eof_llm_token_id, max_llm_token_id);
        let mut grammar_constraint_state = grammar_constraint.init();

        macro_rules! llm_token_vec {
            ($($token:expr),* $(,)?) => {
                vec![
                    $(
                        llm_token_map.get_by_left(&$token.to_vec()).unwrap().0,
                    )*
                ]
            }
        }

        // Get the mask.
        let mask = grammar_constraint_state.get_mask();
        let expected_mask = bitvec_with_capacity_and_values(llm_tokens.len() + 1, llm_token_vec!(b"a"));
        assert_eq!(mask, expected_mask);

        // Commit "a"
        let terminals: Vec<_> = llm_token_vec!(b"a").into_iter().map(|token_id| LLMTokenID(token_id)).collect();
        grammar_constraint_state.commit_and_step_many(&terminals);

        // Get the mask.
        let mask = grammar_constraint_state.get_mask();
        let expected_mask = bitvec_with_capacity_and_values(llm_tokens.len() + 1, llm_token_vec!(b"b"));
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_grammar_from_exprs_very_simple() {
        let exprs = vec![
            (
                "E".to_string(),
                regex(eat_u8(b'a')),
            ),
        ];

        let grammar = Grammar::from_exprs(exprs.clone());
        dbg!(&grammar);

        let parser = grammar.glr_parser();
        dbg!(&parser);

        let llm_tokens: Vec<Vec<u8>> = vec![b"a".to_vec()];
        let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
        let eof_llm_token_id = llm_tokens.len();
        let max_llm_token_id = llm_tokens.len();
        let grammar_constraint = GrammarConstraint::from_grammar(grammar, llm_token_map.clone(), eof_llm_token_id, max_llm_token_id);
        let mut grammar_constraint_state = grammar_constraint.init();


        macro_rules! llm_token_vec {
            ($($token:expr),* $(,)?) => {
                vec![
                    $(
                        llm_token_map.get_by_left(&$token.to_vec()).unwrap().0,
                    )*
                ]
            }
        }

        // Get the mask.
        let mask = grammar_constraint_state.get_mask();
        let expected_mask = bitvec_with_capacity_and_values(llm_tokens.len() + 1, llm_token_vec!(b"a"));
        assert_eq!(mask, expected_mask);

        // Commit "a"
        let terminals: Vec<_> = llm_token_vec!(b"a").into_iter().map(|token_id| LLMTokenID(token_id)).collect();
        grammar_constraint_state.commit_and_step_many(&terminals);

        // Get the mask.
        let mask = grammar_constraint_state.get_mask();
        let mut expected_mask = bitvec_with_capacity_and_values(llm_tokens.len() + 1, llm_token_vec!());
        // Add the EOF token
        expected_mask.set(llm_tokens.len(), true);
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_precompute_for_python_name_token() {
        // ignore = rep(choice([
        //     eat_u8(ord(" ")),
        //     seq([eat_u8(ord("#")), rep(eat_u8_negation(ord("\n"))), eat_u8(ord("\n"))]),
        // ]))
        // digit = choice([eat_u8(c) for c in range(ord("0"), ord("9") + 1)])
        // alph_lower = choice([eat_u8(c) for c in range(ord("a"), ord("z") + 1)])
        // alph_upper = choice([eat_u8(c) for c in range(ord("A"), ord("Z") + 1)])
        //
        // name_start = choice([
        //     alph_lower,
        //     alph_upper,
        //     eat_u8(ord("_"))
        // ])
        // name_middle = choice([
        //     name_start,
        //     digit,
        // ])
        let ignore = repeat0_fast(choice_fast!(eat_u8_fast(b' '), seq_fast!(eat_u8_fast(b'#'), repeat0_fast(eat_u8_negation_fast(b'\n')), eat_u8_fast(b'\n'))));

        let digit = eat_u8_range_fast(b'0', b'9');
        let alph_lower = eat_u8_range_fast(b'a', b'z');
        let alph_upper = eat_u8_range_fast(b'A', b'Z');

        let name_start = choice_fast!(alph_lower, alph_upper, eat_u8_fast(b'_'));
        let name_middle = choice_fast!(name_start.clone(), digit);
        let name = seq_fast!(ignore, name_start, repeat0_fast(seq_fast!(name_middle)));

        let tokenizer = name.build();
        dbg!(&tokenizer);

        let llm_tokens: Vec<Vec<u8>> = (0..2).map(|i| format!("abcdefghijk{}", i).as_bytes().to_vec()).collect();
        let llm_tokens_slices: Vec<&[u8]> = llm_tokens.iter().map(|token| &token[..]).collect();
        let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
        let eof_llm_token_id = llm_tokens.len();
        let max_llm_token_id = llm_tokens.len();
        let precomputed = GrammarConstraint::precompute(&tokenizer, &llm_token_map, max_llm_token_id);
        // print_precomputed(&precomputed);
        println!("Done precomputing");
    }

    #[test]
    fn test_precompute_explosion() {
        let tokenizer = groups![
            eat_u8(b'a'),
            eat_u8(b'a'),
        ].build();

        let llm_tokens: Vec<Vec<u8>> = vec![b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec()];
        let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
        let eof_llm_token_id = llm_tokens.len();
        let max_llm_token_id = llm_tokens.len();
        let precomputed = GrammarConstraint::precompute(&tokenizer, &llm_token_map, max_llm_token_id);
        // print_precomputed(&precomputed);
        println!("Done precomputing");
    }
}
