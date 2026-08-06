//! Slow, scalar definition of the two-state L1 terminal DWA.
//! Compute `state × token -> terminal set`, then quotient equal token columns and state rows.

use std::collections::{BTreeMap, BTreeSet};
use glrmask_lexer::__private::automata::lexer::{Lexer, tokenizer::Tokenizer};
use glrmask_vocab::Vocab;

type Signature = Vec<u32>;

#[derive(Debug)]
pub struct L1 {
    pub state_class: Vec<u32>,
    pub token_class: BTreeMap<u32, u32>,
    pub edges: BTreeMap<u32, Vec<BTreeSet<u32>>>, // terminal -> state class -> token classes
}

fn compact<T: Ord>(values: impl IntoIterator<Item = T>) -> (Vec<u32>, Vec<usize>) {
    let mut ids = BTreeMap::new();
    let mut reps = Vec::new();
    let classes = values.into_iter().enumerate().map(|(i, value)| match ids.get(&value) {
        Some(&id) => id,
        None => { let id = reps.len() as u32; ids.insert(value, id); reps.push(i); id }
    }).collect();
    (classes, reps)
}

fn signature(tokenizer: &Tokenizer, states: impl IntoIterator<Item = u32>, active: &[bool]) -> Signature {
    let mut out = BTreeSet::new();
    for q in states {
        out.extend(tokenizer.matched_terminals_iter(q).filter(|&t| active.get(t as usize) == Some(&true)));
        out.extend(tokenizer.tokens_accessible_from_state(q).iter().filter(|&t| active.get(t) == Some(&true)).map(|t| t as u32));
    }
    out.into_iter().collect()
}

pub fn build(tokenizer: &Tokenizer, vocab: &Vocab, active: &[bool]) -> L1 {
    let tokens = vocab.iter().map(|(id, b)| (id, b.to_vec())).collect::<Vec<_>>();
    let raw = (0..tokenizer.num_states()).map(|q| tokens.iter().map(|(_, b)|
        signature(tokenizer, tokenizer.execute_from_state_end_only(b, q), active)
    ).collect::<Vec<_>>()).collect::<Vec<_>>();

    let (token_classes, token_reps) = compact((0..tokens.len()).map(|v|
        raw.iter().map(|row| row[v].clone()).collect::<Vec<_>>()
    ));
    let (mut state_class, _) = compact(raw.iter().map(|row|
        token_reps.iter().map(|&v| row[v].clone()).collect::<Vec<_>>()
    ));
    let initial = tokenizer.initial_state_id() as usize;
    if state_class.iter().filter(|&&c| c == state_class[initial]).count() > 1 {
        state_class[initial] = state_class.iter().copied().max().unwrap_or(0) + 1;
    }
    let num_state_classes = state_class.iter().copied().max().map_or(0, |x| x + 1) as usize;

    let token_class = tokens.iter().zip(&token_classes).map(|((id, _), &class)| (*id, class)).collect();
    let mut edges = BTreeMap::<u32, Vec<BTreeSet<u32>>>::new();
    for (q, row) in raw.iter().enumerate() {
        for (v, terminals) in row.iter().enumerate() {
            for &t in terminals {
                edges.entry(t).or_insert_with(|| vec![BTreeSet::new(); num_state_classes])
                    [state_class[q] as usize].insert(token_classes[v]);
            }
        }
    }
    L1 { state_class, token_class, edges }
}

impl L1 {
    pub fn accepts(&self, terminal: u32, state: u32, token: u32) -> bool {
        self.edges.get(&terminal)
            .and_then(|rows| rows.get(self.state_class[state as usize] as usize))
            .is_some_and(|tokens| tokens.contains(&self.token_class[&token]))
    }
}

fn main() {
    let tokenizer = glrmask_lexer::__private::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"x".to_vec())]);
    println!("{:#?}", build(&tokenizer, &vocab, &[true, true]));
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use super::*;
    use glrmask_glr::__private::glr::analysis::AnalyzedGrammar;
    use glrmask_grammar::__private::grammar::flat::{GrammarDef, Rule, Symbol, Terminal};
    use glrmask_lexer::__private::automata::lexer::{ast::bytes, compile::build_regex};
    use glrmask_terminal_dwa::__private::terminal_dwa::{
        l1::{build_flat_transition_table, build_l1_id_map_and_terminal_dwa},
        types::TerminalColoring,
    };

    fn grammar() -> AnalyzedGrammar {
        AnalyzedGrammar::from_grammar_def(&GrammarDef {
            rules: vec![Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] }], start: 0,
            terminals: (0..2).map(|id| Terminal::Literal { id, bytes: vec![b'a' + id as u8] }).collect(),
            ..GrammarDef::default()
        })
    }

    fn check(tokenizer: Tokenizer, vocab: Vocab) {
        for active in [[true, true], [true, false], [false, true]] {
            let reference = build(&tokenizer, &vocab, &active);
            let flat = Arc::from(build_flat_transition_table(&tokenizer));
            let production = build_l1_id_map_and_terminal_dwa(
                "reference", &tokenizer, &vocab, &TerminalColoring::identity(2), false, None,
                &grammar(), &active, &flat, None, None, None, None, None,
            ).unwrap();
            for q in 0..tokenizer.num_states() {
                let tsid = production.id_map.tokenizer_states.original_to_internal[q as usize];
                for (v, token_bytes) in vocab.iter() {
                    let token = production.id_map.internal_token_for_original(v).unwrap();
                    let expected = signature(&tokenizer, tokenizer.execute_from_state_end_only(token_bytes, q), &active);
                    for t in 0..2 {
                        let weight = production.dwa.eval_word(&[t as i32]);
                        let actual = weight.token_set_for_tsid_ref(tsid).is_some_and(|set| set.contains(token));
                        assert_eq!(reference.accepts(t, q, v), expected.contains(&t), "reference active={active:?} q={q} v={v} t={t}");
                        assert_eq!(actual, expected.contains(&t), "production active={active:?} q={q} v={v} t={t}");
                    }
                }
            }
        }
    }

    #[test]
    fn epsilon_nfa_matches_production() {
        check(
            glrmask_lexer::__private::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer(),
            Vocab::new(vec![(0, vec![]), (1, b"a".to_vec()), (2, b"a".to_vec()), (3, b"aa".to_vec()), (4, b"b".to_vec()), (5, b"x".to_vec())]),
        );
    }

    #[test]
    fn deterministic_matches_production() {
        let exprs = vec![bytes(b"a"), bytes(b"ab")];
        let tokenizer = build_regex(&exprs).into_tokenizer(2, Some(Arc::from(exprs.into_boxed_slice())));
        check(tokenizer, Vocab::new(vec![(0, vec![]), (1, b"a".to_vec()), (2, b"ab".to_vec()), (3, b"aba".to_vec()), (4, b"x".to_vec())]));
    }
}
