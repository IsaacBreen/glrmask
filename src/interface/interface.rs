use crate::constraint::{GrammarConstraint};
use crate::debug;
use crate::finite_automata::{greedy_group, groups, ExprGroup, GroupID};
use crate::finite_automata::{Expr, Regex};
use crate::glr::grammar::{NonTerminal, Production, Symbol, Terminal};
use crate::glr::parser::GLRParser;
use crate::glr::table::{assign_non_terminal_ids, generate_glr_parser, generate_glr_parser_with_maps, NonTerminalID, TerminalID};
use crate::tokenizer::LLMTokenID;
use crate::types::TerminalID as GrammarTokenID;
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
    // pub literal_map: BTreeMap<String, String>, // Remove this line
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

        // writeln!(f, "  Literal Map:")?;
        // for (literal, mangled_name) in &self.literal_map {
        //     writeln!(f, "    {:?}: {}", literal, mangled_name)?;
        // }
        // Remove these three lines (approximately lines 40-43 in the original code)


        writeln!(f, "  Terminals:")?;
        let mut terminals = self.terminal_name_to_group_id.iter().collect::<Vec<_>>();
        terminals.sort_by_key(|(group_id, _)| *group_id);
        for (name, group_id) in terminals {
            writeln!(f, "    {:?}: {:?}", name, group_id)?;
        }

        writeln!(f, "Tokenizer:");
        writeln!(f, "{:?}", &self.tokenizer);

        Ok(())
    }
}

impl Grammar {
    // fn mangle_literal(literal: &str, tokens: &BTreeMap<String, Expr>) -> String {
    //     let mut mangled_name = literal.to_string();
    //     let mut i = 0;
    //     while tokens.contains_key(&mangled_name) {
    //         mangled_name = format!("{}__literal_{}", literal, i);
    //         i += 1;
    //     }
    //     mangled_name
    // }
    // Remove this entire function (approximately lines 56-64 in the original code)

    // Helper function to generate unique names like Base[0], Base[1], etc.
    // Or, if base itself is Base[0], then Base[0][0], Base[0][1], etc.
    fn generate_unique_indexed_name(
        base_name: &str,
        counters: &mut HashMap<String, usize>, // Key: base_name, Value: next index for this base
        all_existing_names: &mut HashSet<String>, // All names generated or defined so far (NTs and Ts)
    ) -> String {
        let idx_ref = counters.entry(base_name.to_string()).or_insert(0);
        let mut current_idx = *idx_ref;
        loop {
            let new_name = format!("{}[{}]", base_name, current_idx);
            if !all_existing_names.contains(&new_name) {
                all_existing_names.insert(new_name.clone());
                *idx_ref = current_idx + 1;
                return new_name;
            }
            current_idx += 1;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarExpr {
    RegexExpr(Expr),
    Ref(String),
    Sequence(Vec<GrammarExpr>),
    Choice(Vec<GrammarExpr>),
    Optional(Box<GrammarExpr>),
    Repeat(Box<GrammarExpr>), // Zero or more repetition
    Literal(Vec<u8>), // Add this line
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

// Add this new function
pub fn literal(bytes: Vec<u8>) -> GrammarExpr {
    GrammarExpr::Literal(bytes)
}


impl Grammar {
    pub fn glr_parser(&self) -> GLRParser {
        // Return a clone or reference if stored
        // Note: This currently regenerates the parser. If self.glr_parser is already stored,
        // perhaps return self.glr_parser.clone() instead?
        generate_glr_parser(&self.productions, self.start_production_id)
    }
}

impl Grammar {
    /// Constructs a `Grammar` and `Regex` tokenizer from a list of grammar expressions.
    /// The first non-terminal in the list is treated as the start symbol.
    pub fn from_exprs(exprs: Vec<(String, GrammarExpr)>) -> Self {
        let mut productions = Vec::new();
        // let mut literal_map = BTreeMap::new(); // Remove this line
        let mut terminal_name_to_group_id = BiBTreeMap::new();
        let mut terminal_expr_to_group_id = BiBTreeMap::new();
        let mut next_terminal_group_id = 0; // Renamed for clarity

        let mut all_names: HashSet<String> = exprs.iter().map(|(name, _)| name.clone()).collect();
        let mut per_base_counters: HashMap<String, usize> = HashMap::new();

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
        all_names.insert(start_production_name.clone()); // Ensure it's known

        fn convert_expr(
            expr: &GrammarExpr,
            current_rule_name_or_path: &str, // e.g., "S", or "S[0]" if inside an internal rule
            productions: &mut Vec<Production>,
            // literal_map: &mut BTreeMap<String, String>, // Remove if unused // This line should be removed
            terminal_string_to_expr: &mut BTreeMap<String, Expr>,
            terminal_name_to_group_id: &mut BiBTreeMap<String, usize>,
            terminal_expr_to_group_id: &mut BiBTreeMap<Expr, usize>,
            next_terminal_group_id: &mut usize,
            per_base_counters: &mut HashMap<String, usize>,
            all_names: &mut HashSet<String>, // Contains all NT and T names
        ) -> Vec<Symbol> {
            match expr {
                // Add this new arm
                GrammarExpr::Literal(bytes) => {
                    let regex_expr = Expr::U8Seq(bytes.clone());
                    if let Some(group_id) = terminal_expr_to_group_id.get_by_left(&regex_expr) {
                        // TODO: UTF8 conversion is lossy. Make sure there aren't collisions.
                        let terminal_name = String::from_utf8(bytes.clone()).expect("Internal error: bytes should be valid UTF-8");
                        vec![Symbol::Terminal(Terminal(terminal_name))]
                    } else {
                        // New terminal for this literal
                        let base_name = format!("b\"{}\"", String::from_utf8_lossy(bytes).escape_debug().to_string());
                        let terminal_name = Grammar::generate_unique_indexed_name(
                            &base_name, // Base name for the terminal based on literal content
                            per_base_counters,
                            all_names,
                        );
                        let group_id = *next_terminal_group_id;
                        terminal_name_to_group_id.insert(terminal_name.clone(), group_id);
                        terminal_expr_to_group_id.insert(regex_expr.clone(), group_id);
                        terminal_string_to_expr.insert(terminal_name.clone(), regex_expr.clone());
                        *next_terminal_group_id += 1;
                        vec![Symbol::Terminal(Terminal(terminal_name))]
                    }
                }
                GrammarExpr::RegexExpr(regex_expr) => {
                    if let Some(group_id) = terminal_expr_to_group_id.get_by_left(regex_expr) {
                        // Existing terminal, find its name
                        let terminal_name = terminal_name_to_group_id.get_by_right(group_id)
                            .expect("Internal error: group_id has no name").clone();
                        vec![Symbol::Terminal(Terminal(terminal_name))]
                    } else {
                        let terminal_name = Grammar::generate_unique_indexed_name(
                            current_rule_name_or_path, // Base name for the terminal
                            per_base_counters,
                            all_names,
                        );
                        let group_id = *next_terminal_group_id;
                        terminal_name_to_group_id.insert(terminal_name.clone(), group_id);
                        terminal_expr_to_group_id.insert(regex_expr.clone(), group_id);
                        terminal_string_to_expr.insert(terminal_name.clone(), regex_expr.clone());
                        *next_terminal_group_id += 1;
                        vec![Symbol::Terminal(Terminal(terminal_name))]
                    }
                }
                GrammarExpr::Ref(name) => {
                    // Ensure the referred name exists or will exist as a user-defined rule.
                    // This check isn't strictly necessary here for correctness but could help catch errors.
                    // For now, assume the ref is valid.
                    vec![Symbol::NonTerminal(NonTerminal(name.clone()))]
                },
                GrammarExpr::Sequence(exprs) => exprs
                    .iter()
                    .flat_map(|e| {
                        convert_expr(
                            e,
                            current_rule_name_or_path, // Pass current path
                            productions,
                            // literal_map, // if used // Remove this line
                            terminal_string_to_expr,
                            terminal_name_to_group_id,
                            terminal_expr_to_group_id,
                            next_terminal_group_id,
                            per_base_counters,
                            all_names,
                        )
                    })
                    .collect(),
                GrammarExpr::Choice(exprs) => {
                    let choice_nt_name = Grammar::generate_unique_indexed_name(
                        current_rule_name_or_path,
                        per_base_counters,
                        all_names,
                    );
                    let nt = NonTerminal(choice_nt_name.clone());
                    // all_names is updated by generate_unique_indexed_name

                    for expr in exprs {
                        let rhs = convert_expr(
                            expr,
                            &choice_nt_name, // Pass the new NT name as the base for its children
                            productions,
                            // literal_map, // if used // Remove this line
                            terminal_string_to_expr,
                            terminal_name_to_group_id,
                            terminal_expr_to_group_id,
                            next_terminal_group_id,
                            per_base_counters,
                            all_names,
                        );
                        productions.push(Production {
                            lhs: nt.clone(),
                            rhs,
                        });
                    }

                    vec![Symbol::NonTerminal(nt)]
                }
                GrammarExpr::Optional(expr_box) => {
                    // Optional(E) becomes Choice(E, epsilon)
                    convert_expr(
                        &GrammarExpr::Choice(vec![*expr_box.clone(), GrammarExpr::Sequence(vec![])]),
                        current_rule_name_or_path, // Pass current path for the Choice NT to be based on
                        productions,
                        // literal_map, // if used // Remove this line
                        terminal_string_to_expr,
                        terminal_name_to_group_id,
                        terminal_expr_to_group_id,
                        next_terminal_group_id,
                        per_base_counters,
                        all_names,
                    )
                }
                GrammarExpr::Repeat(expr_box) => {
                    // Repeat(E) becomes:
                    // RepeatNT ::= E RepeatNT
                    // RepeatNT ::= epsilon
                    let inner_expr = &*expr_box;
                    let repeat_nt_name = Grammar::generate_unique_indexed_name(
                        current_rule_name_or_path,
                        per_base_counters,
                        all_names,
                    );
                    // all_names is updated by generate_unique_indexed_name

                    // Convert the inner expression E. Terminals/NTs inside E will be named relative to repeat_nt_name.
                    let expr_symbols = convert_expr(
                        inner_expr,
                        &repeat_nt_name, // Children are named relative to this new RepeatNT
                        productions,
                        // literal_map, // if used // Remove this line
                        terminal_string_to_expr,
                        terminal_name_to_group_id,
                        terminal_expr_to_group_id,
                        next_terminal_group_id,
                        per_base_counters,
                        all_names,
                    );

                    // Production 1: RepeatNT ::= E RepeatNT
                    // Only add this if E is not an empty sequence, otherwise RepeatNT -> RepeatNT is problematic.
                    // If E can derive epsilon, this is fine. If E *is* epsilon, then RepeatNT should just be epsilon.
                    // Checking if expr_symbols is empty is a proxy for checking if inner_expr can derive epsilon *directly*
                    // as an empty sequence. This might need refinement if inner_expr is a complex structure
                    // that can derive epsilon. For now, checking for empty sequence is a simple approach.
                     if !expr_symbols.is_empty() {
                        let mut rhs1 = expr_symbols.clone();
                        rhs1.push(Symbol::NonTerminal(NonTerminal(repeat_nt_name.clone())));
                        productions.push(Production {
                            lhs: NonTerminal(repeat_nt_name.clone()),
                            rhs: rhs1,
                        });
                    } else {
                        // If expr_symbols is empty, it means inner_expr converted to an empty sequence.
                        // Repeat of epsilon is just epsilon. The RepeatNT effectively becomes an epsilon rule.
                        // No need for RepeatNT ::= RepeatNT in this specific case.
                    }


                    // Production 2: RepeatNT ::= epsilon (for zero-or-more)
                    let rhs2 = vec![]; // Epsilon production
                    productions.push(Production {
                        lhs: NonTerminal(repeat_nt_name.clone()),
                        rhs: rhs2,
                    });

                    // The Repeat(E) expression in the original rule resolves to a reference to this new RepeatNT
                    vec![Symbol::NonTerminal(NonTerminal(repeat_nt_name))]
                }
            }
        }

        let mut terminal_string_to_expr = BTreeMap::new();

        // Process each rule definition
        for (name, expr) in tqdm!(exprs.iter()) {
            let lhs = NonTerminal(name.clone());
            let lhs_name_str = &name; // This is the top-level rule name, e.g., "S"

            // Optimization: If the top-level expression is a Choice, create multiple productions directly.
            if let GrammarExpr::Choice(choices) = expr {
                for choice_expr in choices {
                    let rhs = convert_expr(
                        choice_expr,
                        lhs_name_str, // Pass current rule's name as base
                        &mut productions,
                        // &mut literal_map, // if used // Remove this line
                        &mut terminal_string_to_expr,
                        &mut terminal_name_to_group_id,
                        &mut terminal_expr_to_group_id,
                        &mut next_terminal_group_id,
                        &mut per_base_counters,
                        &mut all_names,
                    );
                    productions.push(Production { lhs: lhs.clone(), rhs });
                }
            } else {
                // Otherwise, convert the expression as usual and create a single production.
                let rhs = convert_expr(
                    expr,
                    lhs_name_str, // Pass current rule's name as base
                    &mut productions,
                    // &mut literal_map, // if used // Remove this line
                    &mut terminal_string_to_expr,
                    &mut terminal_name_to_group_id,
                    &mut terminal_expr_to_group_id,
                    &mut next_terminal_group_id,
                    &mut per_base_counters,
                    &mut all_names,
                );
                productions.push(Production { lhs, rhs });
            }
        }

        let mut tokenizer_exprs_vec: Vec<ExprGroup> = Vec::new();
        // Ensure terminals are added to the tokenizer in the order of their group IDs
        let mut sorted_terminals: Vec<_> = terminal_string_to_expr.iter().collect();
        sorted_terminals.sort_by_key(|(name, _)| terminal_name_to_group_id.get_by_left(*name).unwrap());

        for (name, expr) in sorted_terminals {
             // Use the group ID assigned during convert_expr
            let group_id = *terminal_name_to_group_id.get_by_left(name).unwrap();
            // Ensure we add the expr to the tokenizer_exprs_vec at the correct index (group_id)
             while tokenizer_exprs_vec.len() <= group_id {
                 tokenizer_exprs_vec.push(greedy_group(Expr::Epsilon));
             }
            tokenizer_exprs_vec[group_id] = greedy_group(expr.clone());
        }


        let tokenizer_expr_groups = groups(tokenizer_exprs_vec);
        debug!(2, "Building tokenizer");
        let tokenizer = tokenizer_expr_groups.clone().build();
        let glr_parser = generate_glr_parser(&productions, 0); // Generate parser once

        debug!(2, "Done defining grammar");
        Self {
            productions,
            start_production_id: 0, // Assuming the first production is the start production
            // literal_map, // Still here, but maybe unused? // Remove this line
            terminal_name_to_group_id,
            terminal_expr_to_group_id,
            tokenizer,
            glr_parser,
        }
    }
}

impl GrammarConstraint {
    pub fn from_grammar(grammar: Grammar, llm_tokens: LLMTokenMap, eof_llm_token_id: usize, max_llm_token_id: usize) -> Self {
        GrammarConstraint::new(grammar.tokenizer, grammar.glr_parser, llm_tokens, grammar.terminal_name_to_group_id, max_llm_token_id)
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
        crate::debug!(3, "Processing input bytes: {:?} with {} active tokenizer states", bytes, self.state.len());
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

            crate::debug!(4, "Processing position {} in state {}. Matches: {}", position, current_tokenizer_state_id.0, results.matches.len());
            // Handle full token matches
            for token in results.matches {
                crate::debug!(4, "Found match for token {:?} ({}) with width {}", token.id, self.grammar.terminal_name_to_group_id.get_by_right(&token.id).unwrap(), token.width);
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
                // TODO: no need to do this here unless it's needed in is_valid. Don't want to do it in is_valid because it's expensive.
                //  Would be better to put this in a lazily-initialized field in each entry in self.states, and compute it only when is_valid is called.
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
    use crate::constraint::LLMTokenBV;
    use crate::datastructures::hybrid_bitset::HybridBitset;

    fn bitvec_with_capacity_and_values(capacity: usize, values: Vec<usize>) -> HybridBitset {
        let mut bitvec = BitVec::new();
        bitvec.resize(capacity, false);
        for value in values {
            bitvec.set(value, true);
        }
        bitvec.into()
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
        let prefill: Vec<_> = llm_token_vec!(b"(i", b"+", b"i", b"*", b"i").into_iter().map(|token_id| LLMTokenID(token_id)).collect();
        grammar_constraint_state.step_with_llm_token_sequence(&prefill);

        // Get the mask.
        // The valid LLM tokens right now are ["+", "*", ")", "+i)"].
        // The prefill "(i+i*i" consumes the "i" after the second "+". The next possible tokens should be "+", "*", or ")".
        // Plus potentially "+i" if that's in the LLM vocabulary.
        let prefill_tokens: Vec<_> = llm_token_vec!(b"(i", b"+", b"i", b"*", b"i").into_iter().map(LLMTokenID).collect();
        let mut state = grammar_constraint.init();
        state.step_with_llm_token_sequence(&prefill_tokens);
        let mask = state.get_mask();
         // After "(i+i*i", expecting '+', '*', ')', or '+i'
        let expected_mask = bitvec_with_capacity_and_values(llm_tokens.len() + 1, llm_token_vec!(b"+", b"*", b")", b"+i"));
        assert_eq!(mask, expected_mask);


        // Finish it with ")"
        let terminals: Vec<_> = llm_token_vec!(b")").into_iter().map(|token_id| LLMTokenID(token_id)).collect();
        state.step_with_llm_token_sequence(&terminals); // Use the state variable modified above

        let mask = state.get_mask();

        // After "(i+i*i)", we are in a state where we expect ')'. Committing ')' reduces F -> (E).
        // After that, we are left with E + F, which expects '+' or '*'.
        // Plus EOF.
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
        grammar_constraint_state.step_with_llm_token_sequence(&terminals);

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
        let max_llm_token_id = llm_tokens.len(); // max_llm_token_id should be tokens.len() for HybridBitset
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
        grammar_constraint_state.step_with_llm_token_sequence(&terminals);

        // Get the mask.
        let mask = grammar_constraint_state.get_mask();
        // After consuming "a", the only possible next token is EOF.
        let mut expected_mask = HybridBitset::new();
        expected_mask.insert(llm_tokens.len()); // Add EOF token ID
        assert_eq!(mask, expected_mask);
    }

    #[test]
    fn test_precompute_for_python_name_token_with_names() {
        // ignore = rep(choice([
        //     eat_u8(ord(" "))),
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

        let name_start = choice_fast!(alph_lower.clone(), alph_upper.clone(), eat_u8_fast(b'_'));
        let name_middle = choice_fast!(name_start.clone(), digit.clone());
        let name = seq_fast!(ignore.clone(), name_start.clone(), repeat0_fast(seq_fast!(name_middle.clone())));

        let tokenizer = groups![
            ignore, // Group 0
            digit, // Group 1
            alph_lower, // Group 2
            alph_upper, // Group 3
            eat_u8_fast(b'_'), // Group 4
            name_start, // Group 5
            name_middle, // Group 6
            name // Group 7
        ].build();
        dbg!(&tokenizer);

        let llm_tokens: Vec<Vec<u8>> = (0..2).map(|i| format!("abcdefghijk{}", i).as_bytes().to_vec()).collect();
        let llm_tokens_slices: Vec<&[u8]> = llm_tokens.iter().map(|token| &token[..]).collect();
        let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
        let eof_llm_token_id = llm_tokens.len();
        let max_llm_token_id = llm_tokens.len();

        let mut token_name_map = BiBTreeMap::new();
        token_name_map.insert("ignore".to_string(), 0);
        token_name_map.insert("digit".to_string(), 1);
        token_name_map.insert("alph_lower".to_string(), 2);
        token_name_map.insert("alph_upper".to_string(), 3);
        token_name_map.insert("underscore".to_string(), 4);
        token_name_map.insert("name_start".to_string(), 5);
        token_name_map.insert("name_middle".to_string(), 6);
        token_name_map.insert("name".to_string(), 7);


        let precomputed = GrammarConstraint::precompute(
            &tokenizer,
            &llm_token_map,
            &token_name_map,
            max_llm_token_id,
        );
        // print_precomputed(&precomputed);
        println!("Done precomputing");
        // The test passes if it compiles and prints the token names.
    }


}
