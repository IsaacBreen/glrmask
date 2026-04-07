use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::analysis::EOF;
use super::table::{Action, GLRTable};
use crate::compiler::grammar::model::TerminalID;
use crate::ds::bitset::BitSet;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use smallvec::SmallVec;

pub type TerminalsDisallowed = BTreeMap<u32, BTreeSet<u32>>;

impl Merge for TerminalsDisallowed {
    fn merge(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        for (state, terminals) in other {
            merged
                .entry(*state)
                .or_default()
                .extend(terminals.iter().copied());
        }
        merged
    }
}

pub type ParserGSS = LeveledGSS<u32, TerminalsDisallowed>;

#[cfg(test)]
pub(crate) struct GLRParser {
    pub table: GLRTable,
    pub stack: ParserGSS,
}

#[cfg(test)]
impl GLRParser {
    pub(crate) fn new(table: GLRTable) -> Self {
        let stack = ParserGSS::from_stacks(&[(vec![0], BTreeMap::new())]);
        Self { table, stack }
    }

    pub(crate) fn step(&self, token: TerminalID) -> (Self, bool) {
        let next_stack = advance_stacks(&self.table, &self.stack, token);
        let progressed = !next_stack.is_empty();
        (
            Self {
                table: self.table.clone(),
                stack: next_stack,
            },
            progressed,
        )
    }

    pub(crate) fn valid_terminals(&self) -> Vec<TerminalID> {
        valid_terminals_for_stacks(&self.table, &self.stack)
    }
}

#[cfg(test)]
fn dedup_stacks(stacks: impl IntoIterator<Item = Vec<u32>>) -> Vec<Vec<u32>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for stack in stacks {
        if seen.insert(stack.clone()) {
            out.push(stack);
        }
    }
    out
}

fn shift_target(action: Option<&Action>) -> Option<u32> {
    match action {
        Some(Action::Shift(target)) => Some(*target),
        Some(Action::Split {
            shift: Some(target),
            ..
        }) => Some(*target),
        _ => None,
    }
}

fn stack_vectors(stack: &ParserGSS) -> Vec<Vec<u32>> {
    stack.to_stacks().into_iter().map(|(stack, _)| stack).collect()
}

fn reduce_closure_for_lookahead(
    table: &GLRTable,
    stacks: &[Vec<u32>],
    lookahead: TerminalID,
) -> Vec<Vec<u32>> {
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();

    for stack in stacks {
        if visited.insert(stack.clone()) {
            queue.push_back(stack.clone());
        }
    }

    while let Some(stack) = queue.pop_front() {
        let Some(&state) = stack.last() else {
            continue;
        };
        let Some(action) = table.action(state, lookahead) else {
            continue;
        };
        let reduce_rule_ids: &[u32] = match action {
            Action::Reduce(rule_id) => std::slice::from_ref(rule_id),
            Action::Split { reduces, .. } => reduces.as_slice(),
            Action::Shift(_) | Action::Accept => &[],
        };
        for rule_id in reduce_rule_ids {
            let rule = &table.rules[*rule_id as usize];
            if stack.len() < rule.rhs.len() + 1 {
                continue;
            }
            let keep_len = stack.len() - rule.rhs.len();
            let mut reduced = stack[..keep_len].to_vec();
            let Some(&goto_from) = reduced.last() else {
                continue;
            };
            let Some(target) = table.goto_target(goto_from, rule.lhs) else {
                continue;
            };
            reduced.push(target);
            if visited.insert(reduced.clone()) {
                queue.push_back(reduced);
            }
        }
    }

    visited.into_iter().collect()
}

#[cfg(test)]
fn advance_stack_vectors(
    table: &GLRTable,
    stacks: &[Vec<u32>],
    token: TerminalID,
) -> Vec<Vec<u32>> {
    let closure = reduce_closure_for_lookahead(table, stacks, token);
    let mut next = Vec::new();
    for stack in closure {
        let Some(&state) = stack.last() else {
            continue;
        };
        if let Some(target) = shift_target(table.action(state, token)) {
            let mut shifted = stack.clone();
            shifted.push(target);
            next.push(shifted);
        }
    }
    dedup_stacks(next)
}

fn stacks_accept(table: &GLRTable, stacks: &[Vec<u32>]) -> bool {
    reduce_closure_for_lookahead(table, stacks, EOF)
        .into_iter()
        .any(|stack| {
            stack.last().is_some_and(|state| {
                matches!(
                    table.action(*state, EOF),
                    Some(Action::Accept) | Some(Action::Split { accept: true, .. })
                )
            })
        })
}

#[cfg(test)]
fn valid_terminals_for_stack_vectors(
    table: &GLRTable,
    stacks: &[Vec<u32>],
) -> Vec<TerminalID> {
    (0..table.num_terminals)
        .filter(|&terminal| !advance_stack_vectors(table, stacks, terminal).is_empty())
        .collect()
}

pub(crate) fn advance_stacks(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> ParserGSS {
    if let Some(state) = stack.single_exclusive_top_value() {
        match table.action(state, token) {
            Some(Action::Shift(target)) => return stack.push(*target),
            Some(Action::Split {
                shift: Some(target),
                reduces,
                accept,
            }) if reduces.is_empty() && !*accept => return stack.push(*target),
            Some(Action::Reduce(_)) | Some(Action::Accept) | Some(Action::Split { .. }) => {}
            None => return ParserGSS::empty(),
        }
    }

    let frontier = stack.peek_values();
    if frontier.is_empty() {
        return ParserGSS::empty();
    }

    let mut pure_shift_targets = SmallVec::<[(u32, u32); 8]>::new();
    let mut pure_shift_only = true;
    let mut any_action = false;
    for state in frontier.iter().copied() {
        match table.action(state, token) {
            Some(Action::Shift(target)) => {
                any_action = true;
                pure_shift_targets.push((state, *target));
            }
            Some(Action::Split {
                shift: Some(target),
                reduces,
                accept,
            }) if reduces.is_empty() && !*accept => {
                any_action = true;
                pure_shift_targets.push((state, *target));
            }
            Some(Action::Reduce(_))
            | Some(Action::Accept)
            | Some(Action::Split { .. }) => {
                any_action = true;
                pure_shift_only = false;
                break;
            }
            None => {}
        }
    }
    if !any_action {
        return ParserGSS::empty();
    }
    if pure_shift_only && !pure_shift_targets.is_empty() {
        return stack.shift_top_values(pure_shift_targets);
    }

    let mut current = stack.clone();
    let mut processed = vec![false; table.num_states as usize];

    loop {
        let mut new_states = SmallVec::<[u32; 8]>::new();
        if let Some(state) = current.single_top_value() {
            if !processed[state as usize] {
                new_states.push(state);
            }
        } else {
            new_states.extend(
                current
                    .peek_values()
                    .into_iter()
                    .filter(|&state| !processed[state as usize]),
            );
        }
        if new_states.is_empty() {
            break;
        }

        let mut any_reduced = false;
        let mut pending_bases_by_target = SmallVec::<[(u32, ParserGSS); 8]>::new();
        for state in new_states {
            processed[state as usize] = true;
            let reduce_rules: &[u32] = match table.action(state, token) {
                Some(Action::Reduce(rule_id)) => std::slice::from_ref(rule_id),
                Some(Action::Split { reduces, .. }) => reduces.as_slice(),
                _ => &[],
            };
            let subtree = current.isolate(Some(state));
            let mut base_cache = SmallVec::<[((usize, u32), ParserGSS); 4]>::new();
            for &rule_id in reduce_rules {
                let rule = &table.rules[rule_id as usize];
                let rhs_len = rule.rhs.len();
                let popped = subtree.popn(rhs_len as isize);
                if popped.is_empty() {
                    continue;
                }

                let mut handle_goto_from = |goto_from: u32| {
                    if let Some(target) = table.goto_target(goto_from, rule.lhs) {
                        let base = if let Some((_, cached)) = base_cache.iter().find(
                            |((cached_rhs_len, cached_goto_from), _)| {
                                *cached_rhs_len == rhs_len && *cached_goto_from == goto_from
                            },
                        ) {
                            cached.clone()
                        } else {
                            let isolated = popped.isolate(Some(goto_from));
                            base_cache.push(((rhs_len, goto_from), isolated.clone()));
                            isolated
                        };
                        if let Some((_, existing)) = pending_bases_by_target
                            .iter_mut()
                            .find(|(existing_target, _)| *existing_target == target)
                        {
                            *existing = existing.merge(&base);
                        } else {
                            pending_bases_by_target.push((target, base));
                        }
                        any_reduced = true;
                    }
                };

                if let Some(goto_from) = popped.single_top_value() {
                    handle_goto_from(goto_from);
                } else {
                    for goto_from in popped.peek_values() {
                        handle_goto_from(goto_from);
                    }
                }
            }
        }
        if !any_reduced {
            break;
        }
        for (target, base) in pending_bases_by_target {
            current = current.absorb_push(target, &base);
        }
    }

    let mut shift_pairs = SmallVec::<[(u32, u32); 8]>::new();
    let mut handle_shift_state = |state: u32| {
        if let Some(target) = shift_target(table.action(state, token)) {
            shift_pairs.push((state, target));
        }
    };

    if let Some(state) = current.single_top_value() {
        handle_shift_state(state);
    } else {
        for state in current.peek_values() {
            handle_shift_state(state);
        }
    }
    current.shift_top_values(shift_pairs)
}

pub(crate) fn stack_may_advance_on(table: &GLRTable, stack: &ParserGSS, token: TerminalID) -> bool {
    stack.peek_values().into_iter().any(|state| {
        matches!(
            table.action(state, token),
            Some(Action::Shift(_))
                | Some(Action::Reduce(_))
                | Some(Action::Split { .. })
                | Some(Action::Accept)
        )
    })
}

/// Returns true if any terminal in the given bitset may advance the parser stack,
/// or if the parser has a Reduce/Accept action on EOF (since reductions may
/// transition to states that can then shift on future terminals).
pub(crate) fn stack_may_advance_on_any(
    table: &GLRTable,
    stack: &ParserGSS,
    terminals: &BitSet,
) -> bool {
    stack.peek_values().into_iter().any(|state| {
        if let Some(actions_for_state) = table.action.get(state as usize) {
            actions_for_state.keys().any(|&terminal| {
                let relevant = terminals.contains(terminal as usize) || terminal == EOF;
                relevant
                    && matches!(
                        actions_for_state.get(&terminal),
                        Some(Action::Shift(_))
                            | Some(Action::Reduce(_))
                            | Some(Action::Split { .. })
                            | Some(Action::Accept)
                    )
            })
        } else {
            false
        }
    })
}

pub(crate) fn stacks_finished(table: &GLRTable, stack: &ParserGSS) -> bool {
    if stack.is_empty() {
        return false;
    }

    // IELR(1) fast check: if any root state has any action for EOF,
    // the parser can accept. With IELR(1)'s precise lookahead, a reduce
    // for EOF in a state implies the reduce chain leads to accept.
    let has_eof_action = stack
        .peek_values()
        .iter()
        .any(|&state| table.action(state, EOF).is_some());

    // Debug assertion: verify the fast check matches the full check
    #[cfg(debug_assertions)]
    {
        let full_result = if has_eof_action {
            // Only pay for full check when fast check says true
            let vecs = stack_vectors(stack);
            stacks_accept(table, &vecs)
        } else {
            false
        };
        // full_result can only be true if has_eof_action is true
        // But has_eof_action true does NOT guarantee full_result true
        // If this fires, the simple IELR(1) check is insufficient
        if has_eof_action && !full_result {
            // Log but don't crash — the check is an overapproximation
            debug_assert!(
                full_result,
                "stacks_finished: IELR(1) fast check returned true but full check returned false.\n\
                 Root states with EOF action: {:?}",
                stack.peek_values().iter().filter(|&&s| table.action(s, EOF).is_some()).collect::<Vec<_>>()
            );
        }
    }

    has_eof_action
}

#[cfg(test)]
pub(crate) fn valid_terminals_for_stacks(table: &GLRTable, stack: &ParserGSS) -> Vec<TerminalID> {
    valid_terminals_for_stack_vectors(table, &stack_vectors(stack))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::tests::*;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    fn build_parser(gdef: &GrammarDef) -> GLRParser {
        let grammar = AnalyzedGrammar::from_grammar_def(gdef);
        let table = GLRTable::build(&grammar);
        GLRParser::new(table)
    }

    fn make_grammar(rules: Vec<Rule>, start: u32, terminals: Vec<Terminal>) -> GrammarDef {
        GrammarDef {
            rules,
            start,
            terminals,
            ..Default::default()
        }
    }

    fn accepts(parser: &GLRParser, input: &[TerminalID]) -> bool {
        let mut current = GLRParser {
            table: parser.table.clone(),
            stack: parser.stack.clone(),
        };
        for &token in input {
            let (next, progressed) = current.step(token);
            if !progressed {
                return false;
            }
            current = next;
        }
        stacks_finished(&current.table, &current.stack)
    }

    #[test]
    fn test_advance_stacks_preserves_accumulator_state() {
        let gdef = simple_ab_grammar();
        let grammar = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&grammar);

        let mut acc = BTreeMap::new();
        acc.insert(7, BTreeSet::from([11]));
        let gss = ParserGSS::from_stacks(&[(vec![0], acc.clone())]);

        let advanced = advance_stacks(&table, &gss, 0);
        let stacks = advanced.to_stacks();

        assert_eq!(stacks.len(), 1);
        assert_eq!(stacks[0].1, acc);
    }

    #[test]
    fn test_parse_simple_ab() {
        let gdef = simple_ab_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[1, 0])); 
        assert!(!accepts(&parser, &[])); 
    }

    #[test]
    fn test_parse_choice() {
        let gdef = choice_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0])); 
        assert!(accepts(&parser, &[1])); 
        assert!(!accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[])); 
    }

    #[test]
    fn test_parse_two_nt() {
        let gdef = two_nt_grammar(); 
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0, 1])); 
        assert!(!accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[1])); 
    }

    #[test]
    fn test_parse_ambiguous() {
        
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Nonterminal(0),
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(0),
                    ],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
            vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"+".to_vec(),
                },
            ],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0])); 
        assert!(accepts(&parser, &[0, 1, 0])); 
        assert!(accepts(&parser, &[0, 1, 0, 1, 0])); 
        assert!(!accepts(&parser, &[1])); 
        assert!(!accepts(&parser, &[0, 1])); 
    }

    #[test]
    fn test_parse_nullable() {
        
        let gdef = make_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![],
                }, 
            ],
            0,
            vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[])); 
        assert!(accepts(&parser, &[0])); 
        assert!(!accepts(&parser, &[0, 0])); 
    }

    #[test]
    fn test_valid_terminals() {
        let gdef = simple_ab_grammar(); 
        let parser = build_parser(&gdef);
        let valid = parser.valid_terminals();
        assert!(valid.contains(&0)); 
        assert!(!valid.contains(&1)); 
    }

    fn tdef(id: u32, name: &str) -> Terminal {
        Terminal::Literal { id, bytes: name.as_bytes().to_vec() }
    }

    #[test]
    fn test_glr_left_recursive() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },                          
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1]),       "\"b\" accepted");
        assert!(accepts(&parser, &[1, 0]),    "\"ba\" accepted");
        assert!(accepts(&parser, &[1, 0, 0]), "\"baa\" accepted");
        
        assert!(!accepts(&parser, &[0]),    "\"a\" rejected (must start with 'b')");
        assert!(!accepts(&parser, &[1, 1]), "\"bb\" rejected (two 'b's)");
    }

    #[test]
    fn test_glr_right_recursive() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(1)] },                          
            ],
            0,
            vec![tdef(0, "a"), tdef(1, "b")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1]),          "\"b\" accepted");
        assert!(accepts(&parser, &[0, 1]),       "\"ab\" accepted");
        assert!(accepts(&parser, &[0, 0, 1]),    "\"aab\" accepted");
        assert!(accepts(&parser, &[0, 0, 0, 1]), "\"aaab\" accepted");
        
        assert!(!accepts(&parser, &[0]),     "\"a\" rejected (must end in 'b')");
        assert!(!accepts(&parser, &[1, 0]),  "\"ba\" rejected");
        assert!(!accepts(&parser, &[1, 1]),  "\"bb\" rejected");
    }

    #[test]
    fn test_glr_expression_grammar() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(1), Symbol::Nonterminal(1)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },                                               
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(2), Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(2)] },                                               
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(3), Symbol::Nonterminal(0), Symbol::Terminal(4)] },    
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },                                                  
            ],
            0,
            vec![tdef(0, "i"), tdef(1, "+"), tdef(2, "*"), tdef(3, "("), tdef(4, ")")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[0]),                   "\"i\" accepted");
        assert!(accepts(&parser, &[0, 1, 0]),             "\"i+i\" accepted");
        assert!(accepts(&parser, &[0, 2, 0]),             "\"i*i\" accepted");
        assert!(accepts(&parser, &[0, 1, 0, 2, 0]),       "\"i+i*i\" accepted");
        assert!(accepts(&parser, &[3, 0, 1, 0, 4, 2, 0]), "\"(i+i)*i\" accepted");
        
        assert!(!accepts(&parser, &[0, 1]),       "\"i+\" rejected (incomplete)");
        assert!(!accepts(&parser, &[0, 1, 1, 0]), "\"i++i\" rejected (invalid)");
        assert!(!accepts(&parser, &[]),           "\"\" rejected (empty)");
        assert!(!accepts(&parser, &[4]),          "\")\" rejected");
        assert!(!accepts(&parser, &[3, 0]),       "\"(i\" rejected (unclosed paren)");
    }

    #[test]
    fn test_glr_reduce_reduce_conflict() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },    
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },    
            ],
            0,
            vec![tdef(0, "x")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0]),  "\"x\" accepted despite reduce/reduce conflict");
        assert!(!accepts(&parser, &[]), "\"\" rejected");
    }

    #[test]
    fn test_glr_epsilon_ambiguity() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Nonterminal(2)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },  
                Rule { lhs: 1, rhs: vec![] },                     
                Rule { lhs: 2, rhs: vec![Symbol::Terminal(0)] },  
                Rule { lhs: 2, rhs: vec![] },                     
            ],
            0,
            vec![tdef(0, "x")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[]),       "\"\" accepted (A→ε, B→ε)");
        assert!(accepts(&parser, &[0]),      "\"x\" accepted (A→x,B→ε or A→ε,B→x)");
        assert!(accepts(&parser, &[0, 0]),   "\"xx\" accepted (A→x, B→x)");
        assert!(!accepts(&parser, &[0, 0, 0]), "\"xxx\" rejected");
    }

    #[test]
    fn test_glr_highly_ambiguous() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },                             
            ],
            0,
            vec![tdef(0, "a")],
        );
        let parser = build_parser(&gdef);
        assert!(accepts(&parser, &[0]),       "\"a\" accepted");
        assert!(accepts(&parser, &[0, 0]),    "\"aa\" accepted");
        assert!(accepts(&parser, &[0, 0, 0]), "\"aaa\" accepted (many parse trees)");
        assert!(!accepts(&parser, &[]),       "\"\" rejected (S not nullable)");
    }

    #[test]
    fn test_glr_nullable_before_terminal() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] }, 
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(1)] }, 
                Rule { lhs: 1, rhs: vec![] },                    
            ],
            0,
            vec![tdef(0, "c"), tdef(1, "d")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[1, 0]), "\"dc\" accepted (A → d c)");
        assert!(accepts(&parser, &[0]),    "\"c\" accepted (A → ε c via B→ε)");
        
        assert!(!accepts(&parser, &[1]),   "\"d\" rejected (missing 'c')");
        assert!(!accepts(&parser, &[]),    "\"\" rejected (A always requires 'c')");
    }

    #[test]
    fn test_glr_ambiguous_dangling_else() {
        
        let gdef = make_grammar(
            vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2), Symbol::Nonterminal(0), Symbol::Terminal(3), Symbol::Nonterminal(0)] }, 
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(4)] }, 
            ],
            0,
            vec![tdef(0, "if"), tdef(1, "id"), tdef(2, "then"), tdef(3, "else"), tdef(4, "other")],
        );
        let parser = build_parser(&gdef);
        
        assert!(accepts(&parser, &[0, 1, 2, 0, 1, 2, 4, 3, 4]),
            "ambiguous 'if id then if id then other else other' should be accepted");
        
        assert!(accepts(&parser, &[4]),          "\"other\" accepted");
        assert!(accepts(&parser, &[0, 1, 2, 4]), "\"if id then other\" accepted");
        assert!(!accepts(&parser, &[0, 1, 2]),   "\"if id then\" rejected (incomplete)");
    }

    #[test]
    fn test_close_token_wrapper_family_remains_parseable() {
        const OPEN: u32 = 0;
        const NUM: u32 = 1;
        const COMMA: u32 = 2;
        const CLOSE: u32 = 3;

        const START: u32 = 0;
        const BODY: u32 = 1;
        const TAIL_ELEM: u32 = 2;
        const TAIL_PACK: u32 = 3;
        const FIRST_WRAP: u32 = 10;
        const WRAPPER_COUNT: usize = 24;

        let mut rules = vec![
            Rule {
                lhs: START,
                rhs: vec![
                    Symbol::Terminal(OPEN),
                    Symbol::Terminal(NUM),
                    Symbol::Nonterminal(BODY),
                    Symbol::Terminal(CLOSE),
                ],
            },
            Rule {
                lhs: BODY,
                rhs: vec![Symbol::Nonterminal(TAIL_PACK)],
            },
            Rule {
                lhs: TAIL_ELEM,
                rhs: vec![Symbol::Terminal(COMMA), Symbol::Terminal(NUM)],
            },
            Rule {
                lhs: TAIL_PACK,
                rhs: vec![Symbol::Nonterminal(TAIL_ELEM)],
            },
            Rule {
                lhs: TAIL_PACK,
                rhs: vec![
                    Symbol::Nonterminal(TAIL_ELEM),
                    Symbol::Nonterminal(TAIL_ELEM),
                ],
            },
        ];

        for i in 0..WRAPPER_COUNT {
            let wrap_nt = FIRST_WRAP + i as u32;
            rules.push(Rule {
                lhs: wrap_nt,
                rhs: vec![Symbol::Nonterminal(TAIL_PACK)],
            });
            rules.push(Rule {
                lhs: BODY,
                rhs: vec![Symbol::Nonterminal(wrap_nt)],
            });
        }

        let gdef = make_grammar(
            rules,
            START,
            vec![tdef(OPEN, "["), tdef(NUM, "n"), tdef(COMMA, ","), tdef(CLOSE, "]")],
        );
        let parser = build_parser(&gdef);

        let mut current = GLRParser {
            table: parser.table.clone(),
            stack: parser.stack.clone(),
        };
        for &token in &[OPEN, NUM, COMMA, NUM, COMMA, NUM] {
            let (next, progressed) = current.step(token);
            assert!(progressed, "prefix token {token} should progress");
            current = next;
        }

        let advanced = advance_stacks(&current.table, &current.stack, CLOSE);

        assert!(!advanced.is_empty(), "close token should remain parseable");
        assert!(
            stacks_finished(&current.table, &advanced),
            "close token should reduce the wrapper family to a finished parse"
        );
    }
}
